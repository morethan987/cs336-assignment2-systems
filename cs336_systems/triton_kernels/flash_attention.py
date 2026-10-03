import einops
import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx


class FlashAttentionCtx(FunctionCtx):
    Q_TILE_SIZE: tl.constexpr
    KV_TILE_SIZE: tl.constexpr
    saved_tensors: tuple[torch.Tensor, ...]
    is_causal: bool


class FlashAttention_NoTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        FlashAttention forward
        Store: Q, K, V, O and L (logsumexp)
        Return: O
        """
        q_shape = Q.shape  # (batch, seq_len, d_model)
        ctx.is_causal = is_causal

        qtz = 32
        ktz = 32

        Q = einops.rearrange(Q, "... s d -> (...) s d")
        K = einops.rearrange(K, "... s d -> (...) s d")
        V = einops.rearrange(V, "... s d -> (...) s d")

        batch_size, _, d_model = Q.shape
        num_keys = K.shape[1]
        scale = d_model**-0.5

        # only tiling on s dim (batch_size, s, d) -> (batch_size, nqt, qtz, d)
        Q_tiles = einops.rearrange(Q, "b (nqt qtz) d -> b nqt qtz d", qtz=qtz)
        K_tiles = einops.rearrange(K, "b (nkt ktz) d -> b nkt ktz d", ktz=ktz)
        V_tiles = einops.rearrange(V, "b (nkt ktz) d -> b nkt ktz d", ktz=ktz)

        nqt = Q_tiles.shape[1]
        nkt = K_tiles.shape[1]

        O_tiles = torch.empty_like(Q_tiles)
        L_tiles = torch.empty((batch_size, nqt, qtz), device=Q.device, dtype=torch.float32)

        offs_q_base = torch.arange(qtz, device=Q.device)
        offs_k_base = torch.arange(ktz, device=Q.device)

        for b in range(batch_size):
            for q_tile_idx in range(nqt):
                q_block = Q_tiles[b, q_tile_idx]  # (qtz, d)
                o_block = torch.zeros((qtz, d_model), device=Q.device, dtype=torch.float32)
                l = torch.zeros((qtz,), device=Q.device, dtype=torch.float32)
                m = torch.full((qtz,), float("-inf"), device=Q.device, dtype=torch.float32)

                if is_causal:
                    min_q = q_tile_idx * qtz
                    num_unmasked_k_tiles = min(nkt, min_q // ktz)
                    max_k = min((q_tile_idx + 1) * qtz, num_keys)
                    num_k_tiles = min(nkt, (max_k + ktz - 1) // ktz)
                else:
                    num_unmasked_k_tiles = nkt
                    num_k_tiles = nkt

                # unmasked tiles
                for k_tile_idx in range(num_unmasked_k_tiles):
                    k_block = K_tiles[b, k_tile_idx]  # (ktz, d)
                    v_block = V_tiles[b, k_tile_idx]  # (ktz, d)

                    # logits
                    s = scale * einops.einsum(q_block, k_block, "qtz d, ktz d -> qtz ktz")

                    row_max = torch.amax(s, dim=-1)
                    m_new = torch.maximum(m, row_max)

                    p_tilde = torch.exp(s - m_new[:, None])
                    alpha = torch.exp(m - m_new)

                    l = alpha * l + torch.sum(p_tilde, dim=-1)
                    o_block = alpha[:, None] * o_block + einops.einsum(p_tilde, v_block, "qtz ktz, ktz d -> qtz d")
                    m = m_new

                if is_causal:
                    # masked tiles
                    offs_q = q_tile_idx * qtz + offs_q_base
                    for k_tile_idx in range(num_unmasked_k_tiles, num_k_tiles):
                        k_block = K_tiles[b, k_tile_idx]
                        v_block = V_tiles[b, k_tile_idx]

                        s = scale * einops.einsum(q_block, k_block, "qtz d, ktz d -> qtz ktz")

                        offs_k = k_tile_idx * ktz + offs_k_base
                        mask = offs_q[:, None] >= offs_k[None, :]
                        s = s.masked_fill(~mask, float("-inf"))

                        row_max = torch.amax(s, dim=-1)
                        m_new = torch.maximum(m, row_max)
                        p_tilde = torch.exp(s - m_new[:, None])
                        alpha = torch.exp(m - m_new)

                        l = alpha * l + torch.sum(p_tilde, dim=-1)
                        o_block = alpha[:, None] * o_block + einops.einsum(p_tilde, v_block, "qtz ktz, ktz d -> qtz d")
                        m = m_new

                o_block = o_block / l[:, None]
                l_block = m + torch.log(l)

                O_tiles[b, q_tile_idx] = o_block.to(Q.dtype)
                L_tiles[b, q_tile_idx] = l_block

        O = O_tiles.view(*q_shape)
        L = L_tiles.view(*q_shape[:-1])

        ctx.save_for_backward(Q.view(*q_shape), K.view(*q_shape), V.view(*q_shape), O, L)

        return O

    @staticmethod
    def backward(ctx, *grad_outs: torch.Tensor):
        (O_grad,) = grad_outs
        Q, K, V, O, L = ctx.saved_tensors
        Q_grad, K_grad, V_grad = _flash_backward_compiled(Q, K, V, O, O_grad, L, ctx.is_causal)
        return Q_grad, K_grad, V_grad, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,  # inputs
    O_ptr,
    L_ptr,  # outputs
    stride_qb,
    stride_qq,
    stride_qd,  # q strides
    stride_kb,
    stride_kk,
    stride_kd,  # k strides
    stride_vb,
    stride_vk,
    stride_vd,  # v strides
    stride_ob,
    stride_oq,
    stride_od,  # o strides
    stride_lb,
    stride_lq,  # l strides
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # why q is the first dim instead of b?
    # The first dim should be the fastest dim oppositing to torch
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b_idx * stride_qb,  # also caused by GPU launch order
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + b_idx * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, KV_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + b_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(KV_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + b_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + b_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    offs_q = q_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offs_k_base = tl.arange(0, KV_TILE_SIZE)

    if IS_CAUSAL:
        min_q = q_idx * Q_TILE_SIZE
        num_unmasked_k_tiles = min_q // KV_TILE_SIZE
        max_k = tl.minimum((q_idx + 1) * Q_TILE_SIZE, N_KEYS)
        num_k_tile = tl.cdiv(max_k, KV_TILE_SIZE)
    else:
        num_unmasked_k_tiles = tl.cdiv(N_KEYS, KV_TILE_SIZE)
        num_k_tile = num_unmasked_k_tiles

    # unmasked tiles
    for _ in range(num_unmasked_k_tiles):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        s = scale * tl.dot(q, k)
        row_max = tl.max(s, axis=-1)
        m_new = tl.maximum(m, row_max)
        p_tilde = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)

        l = alpha * l + tl.sum(p_tilde, axis=-1)
        o = alpha[:, None] * o + tl.dot(p_tilde, v)

        m = m_new
        K_block_ptr = tl.advance(K_block_ptr, (0, KV_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (KV_TILE_SIZE, 0))

    if IS_CAUSAL:
        # masked tiles, usually 1 or 2
        for k_idx in range(num_unmasked_k_tiles, num_k_tile):
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

            s = scale * tl.dot(q, k)
            offs_k = k_idx * KV_TILE_SIZE + offs_k_base
            mask = offs_q[:, None] >= offs_k[None, :]
            s = tl.where(mask, s, float("-inf"))

            row_max = tl.max(s, axis=-1)
            m_new = tl.maximum(m, row_max)
            p_tilde = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m - m_new)

            l = alpha * l + tl.sum(p_tilde, axis=-1)
            o = alpha[:, None] * o + tl.dot(p_tilde, v)

            m = m_new
            K_block_ptr = tl.advance(K_block_ptr, (0, KV_TILE_SIZE))
            V_block_ptr = tl.advance(V_block_ptr, (KV_TILE_SIZE, 0))

    o = o / l[:, None]
    l = m + tl.log(l)

    tl.store(O_block_ptr, o, boundary_check=(0,))
    tl.store(L_block_ptr, l, boundary_check=(0,))


@torch.compile
def _flash_backward_compiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    O_grad: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_model = Q.shape[-1]
    scale = d_model**-0.5

    S = scale * einops.einsum(Q, K, "... nq d, ... nk d -> ... nq nk")

    # Mask S BEFORE exp to prevent overflow in future tokens
    if is_causal:
        mask = torch.triu(
            torch.ones(S.shape[-2], S.shape[-1], device=S.device, dtype=torch.bool),
            diagonal=1,
        )
        S = S.masked_fill(mask, float("-inf"))

    P = torch.exp(S - L.unsqueeze(-1))

    V_grad = einops.einsum(P, O_grad, "... nq nk, ... nq d -> ... nk d")
    P_grad = einops.einsum(O_grad, V, "... nq d, ... nk d -> ... nq nk")

    D = torch.sum(O * O_grad, dim=-1, keepdim=True)
    S_grad = P * (P_grad - D)

    Q_grad = scale * einops.einsum(S_grad, K, "... nq nk, ... nk d -> ... nq d")
    K_grad = scale * einops.einsum(S_grad, Q, "... nq nk, ... nq d -> ... nk d")

    return Q_grad, K_grad, V_grad


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FlashAttentionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        FlashAttention forward
        Store: Q, K, V, O and L (logsumexp)
        Return: O
        """
        q_shape, k_shape, v_shape = Q.shape, K.shape, V.shape
        ctx.Q_TILE_SIZE = tl.constexpr(32)
        ctx.KV_TILE_SIZE = tl.constexpr(32)
        ctx.is_causal = is_causal

        Q = einops.rearrange(Q, "... nq d -> (...) nq d")
        K = einops.rearrange(K, "... nkv d -> (...) nkv d")
        V = einops.rearrange(V, "... nkv d -> (...) nkv d")

        batch_size, num_q, d_model = Q.shape
        num_k = K.shape[1]
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, num_q), device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(triton.cdiv(num_q, ctx.Q_TILE_SIZE), batch_size)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES=num_q,
            N_KEYS=num_k,
            scale=1 / (d_model**0.5),
            D=tl.constexpr(d_model),
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            KV_TILE_SIZE=ctx.KV_TILE_SIZE,
            IS_CAUSAL=tl.constexpr(ctx.is_causal),
        )

        O = O.view(*q_shape)
        ctx.save_for_backward(Q.view(*q_shape), K.view(*k_shape), V.view(*v_shape), O, L)
        return O

    @staticmethod
    def backward(ctx, *grad_outs: torch.Tensor):
        (O_grad,) = grad_outs
        Q, K, V, O, L = ctx.saved_tensors
        Q_grad, K_grad, V_grad = _flash_backward_compiled(Q, K, V, O, O_grad, L, ctx.is_causal)
        return Q_grad, K_grad, V_grad, None
