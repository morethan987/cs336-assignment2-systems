import einops
import torch
import triton.language as tl
from torch.autograd.function import FunctionCtx


class FlashAttentionCtx(FunctionCtx):
    Q_TILE_SIZE: tl.constexpr
    KV_TILE_SIZE: tl.constexpr
    is_causal: bool


class FlashAttention_NoTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FlashAttentionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        FlashAttention forward
        Store: Q, K, V, O and L (logsumexp)
        Return: O
        """
        q_shape = Q.shape  # (batch, seq_len, d_model)
        d_model = Q.shape[-1]
        ctx.Q_TILE_SIZE = tl.constexpr(32)
        ctx.KV_TILE_SIZE = tl.constexpr(32)
        ctx.is_causal = is_causal

        Q = einops.rearrange(Q, "... s d -> (...) s d")
        K = einops.rearrange(K, "... s d -> (...) s d")
        V = einops.rearrange(V, "... s d -> (...) s d")

        B = Q.shape[0]
        qtz = int(ctx.Q_TILE_SIZE)
        ktz = int(ctx.KV_TILE_SIZE)

        # only tiling on s dim (B, s, d) -> (B, nqt, qtz, d)
        Q_tiles = einops.rearrange(Q, "b (nqt qtz) d -> b nqt qtz d", qtz=qtz)
        K_tiles = einops.rearrange(K, "b (nkt ktz) d -> b nkt ktz d", ktz=ktz)
        V_tiles = einops.rearrange(V, "b (nkt ktz) d -> b nkt ktz d", ktz=ktz)

        nqt = Q_tiles.shape[1]
        nkt = K_tiles.shape[1]

        O_tiles = torch.empty_like(Q_tiles)
        L_tiles = torch.empty((B, nqt, qtz), device=Q.device, dtype=torch.float32)

        for b in range(B):
            for q_tile_idx in range(nqt):
                q_block = Q_tiles[b, q_tile_idx]  # (qtz, d)
                o_block = torch.zeros((qtz, d_model), device=Q.device, dtype=torch.float32)
                l = torch.zeros((qtz,), device=Q.device, dtype=torch.float32)
                m = torch.full((qtz,), float("-inf"), device=Q.device, dtype=torch.float32)

                for k_tile_idx in range(nkt):
                    k_block = K_tiles[b, k_tile_idx]  # (ktz, d)
                    v_block = V_tiles[b, k_tile_idx]  # (ktz, d)

                    # logits
                    s = einops.einsum(q_block, k_block, "qtz d, ktz d -> qtz ktz") / (d_model**0.5)

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
