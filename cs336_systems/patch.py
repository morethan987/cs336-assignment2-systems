import ast
import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import einx
import torch
from cs336_basics.layers import softmax
from torch import nn
from torch.cuda import nvtx
from torch.utils.checkpoint import checkpoint


class Patch(enum.StrEnum):
    ATT = "att"
    RECOMPUTE = "recompute"

    def __str__(self):
        return self.value


PATCH_REGISTRY: dict[Patch, Callable] = {}


def register_patch(name: Patch):
    def decorator(fn: Callable):
        PATCH_REGISTRY[name] = fn
        return fn

    return decorator


@dataclass
class PatchSpec:
    name: Patch
    kwargs: dict[str, Any] = field(default_factory=dict)


def parse_patch_spec(raw: str) -> list[PatchSpec]:
    """
    parse patch string

    1. no-arg：'att'
    2. one-arg：'recompute:group_size=4'
    3. multi-args：'foo:bar=1:baz=True'
    4. combination：'att, recompute:group_size=4'
    """
    if not raw.strip():
        return []

    specs = []
    items = [item.strip() for item in raw.split(",") if item.strip()]

    for item in items:
        parts = item.split(":")
        patch_name = Patch(parts[0].strip())
        kwargs = {}

        for kv in parts[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    parsed_v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    parsed_v = v
                kwargs[k] = parsed_v
            else:
                raise ValueError(f"Unknown argument '{kv}' for patch '{patch_name}'")

        specs.append(PatchSpec(name=patch_name, kwargs=kwargs))

    return specs


@register_patch(Patch.ATT)
def att_patch():
    """Monkey patch scaled_dot_product_attention to add NVTX range annotations."""
    import cs336_basics.layers.multihead_self_attention as _mha

    if not hasattr(_mha, "scaled_dot_product_attention"):
        raise AttributeError("scaled_dot_product_attention not found in cs336_basics.layers.multihead_self_attention")

    def _annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = 1.0 / (Q.shape[-1] ** 0.5)

        with nvtx.range("computing attention scores"):
            # n and m all respresent seq_len, only to tag matrix shape: (n, m) or (m, n)
            scaled_dot = einx.dot("... n [d_k], ... m [d_k] -> ... n m", Q, K) * scale

        if mask is not None:
            scaled_dot = scaled_dot.masked_fill(~mask, float("-inf"))

        with nvtx.range("computing softmax"):
            sftmx = softmax(scaled_dot, -1)

        with nvtx.range("final matmul"):
            res = einx.dot("... n [m], ... [m] d_v -> ... n d_v", sftmx, V)

        return res

    _mha.scaled_dot_product_attention = _annotated_scaled_dot_product_attention  # type: ignore


def _patch_einx_for_recompute():
    """
    Patch einx to prevent it from intercepting and wrapping PyTorch's
    internal `_StopRecomputationError` into a `CallOperationError`.
    """
    try:
        import einx._src.frontend.api as einx_api
        import einx.errors

        orig_create = einx.errors.CallOperationError.create

        def patched_create(exception: BaseException, *args: Any, **kwargs: Any) -> Any:
            if exception.__class__.__name__ == "_StopRecomputationError":
                raise exception
            return orig_create(exception, *args, **kwargs)

        setattr(einx.errors.CallOperationError, "create", staticmethod(patched_create))  # noqa: B010
        if hasattr(einx_api, "CallOperationError"):
            setattr(einx_api.CallOperationError, "create", staticmethod(patched_create))  # noqa: B010

    except (ImportError, AttributeError):
        return


def _run_block_chunk(blocks: nn.ModuleList, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        x: torch.Tensor = block(x, positions)
    return x


def _make_chunk_runner(chunk_blocks: nn.ModuleList) -> Callable:
    def runner(x: torch.Tensor, positions: torch.Tensor):
        return _run_block_chunk(chunk_blocks, x, positions)

    return runner


@register_patch(Patch.RECOMPUTE)
def recompute_patch(group_size: int = 2):
    """Monkey patch TransformerLM.forward to support activation checkpointing."""
    import cs336_basics.layers.transformer_lm as _tlm

    if not hasattr(_tlm, "TransformerLM"):
        raise AttributeError("TransformerLM not found in cs336_basics.layers.transformer_lm")

    _patch_einx_for_recompute()

    def _checkpointed_forward(
        self: _tlm.TransformerLM,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence is too long:\nseq_len = {seq_len} > max_context_length = {self.context_length}")
        if token_positions is None:
            token_positions = torch.arange(0, seq_len, dtype=torch.int, device=token_ids.device)

        x = self.embed(token_ids)

        num_blocks = len(self.transformers)
        for i in range(0, num_blocks, group_size):
            chunk = self.transformers[i : i + group_size]
            runner = _make_chunk_runner(chunk)

            x = checkpoint(
                runner,
                x,
                token_positions,
                use_reentrant=False,
            )

        x = self.output_rms(x)
        return self.output_linear(x)

    _tlm.TransformerLM.forward = _checkpointed_forward
