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
        q_shape = Q.shape  # original shape
        d_model = Q.shape[-1]
        ctx.Q_TILE_SIZE = tl.constexpr(32)
        ctx.KV_TILE_SIZE = tl.constexpr(32)
        ctx.is_causal = is_causal

        # reshape to 2D
        Q = einops.rearrange(Q, "... d -> (...) d")
        K = einops.rearrange(K, "... d -> (...) d")
        V = einops.rearrange(V, "... d -> (...) d")

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous Q, K, V"

        # tiling
        Q_tiles = einops.rearrange(Q, "(nqt qtz) d -> nqt qtz d", qtz=ctx.Q_TILE_SIZE)
        K_tiles = einops.rearrange(K, "(nkt ktz) d -> nkt ktz d", ktz=ctx.KV_TILE_SIZE)
        V_tiles = einops.rearrange(V, "(nkt ktz) d -> nkt ktz d", ktz=ctx.KV_TILE_SIZE)

        nqt = Q_tiles.shape[0]

        O_tiles = torch.empty_like(Q_tiles)
        L_tiles = torch.empty((nqt, int(ctx.Q_TILE_SIZE)), device=Q.device, dtype=torch.float32)

        for q_tile_idx in range(nqt):
            q_block = Q_tiles[q_tile_idx]
            o_block = torch.zeros((int(ctx.Q_TILE_SIZE), d_model), device=Q.device, dtype=torch.float32)
            l = torch.zeros((int(ctx.Q_TILE_SIZE),), device=Q.device, dtype=torch.float32)
            m = torch.full((int(ctx.Q_TILE_SIZE),), float("-inf"), device=Q.device, dtype=torch.float32)
            for k_tile_idx in range(K_tiles.shape[0]):
                k_block, v_block = K_tiles[k_tile_idx], V_tiles[k_tile_idx]

                # logits
                s = einops.einsum(q_block, k_block, "qtz d, ktz d -> qtz ktz") / (d_model**0.5)

                row_max, _ = torch.max(s, dim=-1)
                m_new = torch.maximum(m, row_max)

                p_tilde = torch.exp(s - m_new[:, None])  # numerator
                alpha = torch.exp(m - m_new)

                l = alpha * l + torch.sum(p_tilde, dim=-1)  # denominator
                o_block = einops.einsum(p_tilde, v_block, "qtz ktz, ktz d -> qtz d") + alpha[:, None] * o_block  # without normalization
                m = m_new

            o_block = o_block / l[:, None]  # normalize
            l_block = m + torch.log(l)  # logsumexp for backward

            # write back
            O_tiles[q_tile_idx] = o_block.to(Q.dtype)
            L_tiles[q_tile_idx] = l_block

        # rearrange to origianl shape
        O = einops.rearrange(O_tiles, "nqt qtz d -> (nqt qtz) d")
        O = O.view(*q_shape)

        L = einops.rearrange(L_tiles, "nqt qtz -> (nqt qtz)")
        L = L.view(*q_shape[:-1])

        ctx.save_for_backward(Q, K, V, O, L)

        return O
