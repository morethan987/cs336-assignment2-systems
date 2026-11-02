from __future__ import annotations

from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def is_sharded_layer(module: nn.Module) -> bool:
    """Return whether module is a supported sharded layer."""
    return isinstance(module, (nn.Linear, nn.Embedding)) or module.__class__.__name__ in {"Linear", "Embedding"}


class FSDPLinearFunction(torch.autograd.Function):
    """Linear forward with manually sharded weight-gradient reduction."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, _weight_shard: torch.Tensor, fsdp_layer: FSDPLayer) -> torch.Tensor:
        full_weight = fsdp_layer.wait_all_gather()
        ctx.fsdp_layer = fsdp_layer
        ctx.save_for_backward(x)

        x_compute = x.to(dtype=full_weight.dtype)
        bias = fsdp_layer.bias
        if bias is not None:
            bias = bias.to(dtype=full_weight.dtype)
        return F.linear(x_compute, full_weight, bias)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        fsdp_layer: FSDPLayer = ctx.fsdp_layer
        full_weight = fsdp_layer.wait_all_gather()

        x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32)
        grad_full_weight = grad_output_flat.t().matmul(x_flat)
        grad_x = grad_output.matmul(full_weight).to(dtype=x.dtype)

        if fsdp_layer.bias is not None and fsdp_layer.bias.requires_grad:
            grad_bias = grad_output_flat.sum(dim=0)
            fsdp_layer.fsdp_container._sync_replicated_grad(fsdp_layer.bias, grad_bias)

        fsdp_layer.post_backward(grad_full_weight)
        return grad_x, None, None


class FSDPEmbeddingFunction(torch.autograd.Function):
    """Embedding forward with a manually accumulated full weight gradient."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, _weight_shard: torch.Tensor, fsdp_layer: FSDPLayer) -> torch.Tensor:
        full_weight = fsdp_layer.wait_all_gather()
        ctx.fsdp_layer = fsdp_layer
        ctx.save_for_backward(x)
        return F.embedding(x, full_weight)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, None, None]:
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        fsdp_layer: FSDPLayer = ctx.fsdp_layer
        full_weight = fsdp_layer.wait_all_gather()

        grad_full_weight = torch.zeros(
            full_weight.shape,
            dtype=torch.float32,
            device=full_weight.device,
        )
        grad_full_weight.index_add_(
            0,
            x.reshape(-1),
            grad_output.reshape(-1, full_weight.shape[-1]).to(torch.float32),
        )
        fsdp_layer.post_backward(grad_full_weight)
        return None, None, None


class FSDPLayer(nn.Module):
    """A Linear/Embedding wrapper that stores only this rank's weight shard."""

    fsdp_container: FSDP

    def __init__(
        self,
        orig_module: nn.Module,
        layer_idx: int,
        fsdp_container: FSDP,
        compute_dtype: torch.dtype | None,
    ) -> None:
        # Explicitly initialize nn.Module to avoid calling base_cls.__init__
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        object.__setattr__(self, "fsdp_container", fsdp_container)
        self.compute_dtype = compute_dtype
        self.world_size = fsdp_container.world_size
        self.rank = fsdp_container.rank
        # depend on assignment 1
        self.is_linear = orig_module.__class__.__name__ == "Linear" or isinstance(orig_module, nn.Linear)
        self.is_embedding = orig_module.__class__.__name__ == "Embedding" or isinstance(orig_module, nn.Embedding)
        if not self.is_linear and not self.is_embedding:
            raise TypeError(f"Unsupported sharded layer: {type(orig_module).__name__}")

        orig_weight = getattr(orig_module, "weight", None)
        if not isinstance(orig_weight, nn.Parameter):
            raise TypeError("Target module must have a torch.nn.Parameter named 'weight'")

        self.orig_shape = tuple(orig_weight.shape)
        self.numel = orig_weight.numel()
        if self.numel % self.world_size != 0:
            raise ValueError(f"Weight elements ({self.numel}) must be divisible by world_size ({self.world_size})")
        self.shard_size = self.numel // self.world_size

        flat_weight = orig_weight.detach().flatten()
        start = self.rank * self.shard_size
        shard = flat_weight[start : start + self.shard_size].clone().to(torch.float32)
        self.weight = nn.Parameter(shard, requires_grad=orig_weight.requires_grad)

        orig_bias = getattr(orig_module, "bias", None)
        if isinstance(orig_bias, nn.Parameter):
            self.bias: nn.Parameter | None = nn.Parameter(
                orig_bias.detach().clone().to(torch.float32),
                requires_grad=orig_bias.requires_grad,
            )
        else:
            self.register_parameter("bias", None)

        self.full_weight: torch.Tensor | None = None

    def start_all_gather(self) -> None:
        if self.full_weight is not None:
            return

        target_dtype = self.compute_dtype or self.weight.dtype
        local_shard = self.weight.detach().to(dtype=target_dtype).contiguous()
        if self.world_size == 1:
            full_flat = local_shard
        else:
            full_flat = torch.empty(self.numel, dtype=target_dtype, device=self.weight.device)
            dist.all_gather_into_tensor(full_flat, local_shard)
        self.full_weight = full_flat.reshape(self.orig_shape)

    def wait_all_gather(self) -> torch.Tensor:
        if self.full_weight is None:
            self.start_all_gather()
        assert self.full_weight is not None
        return self.full_weight

    def free_full_weight(self) -> None:
        self.full_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        if self.is_linear:
            out = cast(torch.Tensor, FSDPLinearFunction.apply(x, self.weight, self))
        else:
            out = cast(torch.Tensor, FSDPEmbeddingFunction.apply(x, self.weight, self))
        self.free_full_weight()
        return out

    def post_backward(self, grad_full_weight: torch.Tensor) -> None:
        self.free_full_weight()
        flat_grad = grad_full_weight.detach().flatten().to(torch.float32).contiguous()
        start = self.rank * self.shard_size

        if self.world_size == 1:
            grad_shard = flat_grad[start : start + self.shard_size].clone()
            self.fsdp_container._accumulate_grad(self.weight, grad_shard)
            return

        grad_shard = torch.empty(self.shard_size, dtype=torch.float32, device=self.weight.device)
        handle = dist.reduce_scatter_tensor(grad_shard, flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        self.fsdp_container.grad_sync_handles.append((handle, grad_shard, self.weight))


def make_fsdp_layer(
    orig_module: nn.Module,
    layer_idx: int,
    fsdp_container: FSDP,
    compute_dtype: torch.dtype | None,
) -> FSDPLayer:
    """Create an FSDPLayer that also inherits from orig_module.__class__ for isinstance checks."""
    base_cls = orig_module.__class__

    class DynamicFSDPLayer(FSDPLayer, base_cls):  # ty: ignore[unsupported-base]
        def __init__(
            self,
            orig_module: nn.Module,
            layer_idx: int,
            fsdp_container: FSDP,
            compute_dtype: torch.dtype | None,
        ) -> None:
            FSDPLayer.__init__(
                self,
                orig_module=orig_module,
                layer_idx=layer_idx,
                fsdp_container=fsdp_container,
                compute_dtype=compute_dtype,
            )

    DynamicFSDPLayer.__name__ = base_cls.__name__
    DynamicFSDPLayer.__qualname__ = base_cls.__qualname__
    return DynamicFSDPLayer(orig_module, layer_idx, fsdp_container, compute_dtype)


class FSDP(nn.Module):
    """Minimal FSDP wrapper for Linear and Embedding layers."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.sharded_layers: list[FSDPLayer] = []
        self.grad_sync_handles: list[tuple[dist.Work, torch.Tensor, nn.Parameter]] = []
        self.bias_sync_handles: list[tuple[dist.Work, torch.Tensor, nn.Parameter]] = []
        self._wrap_layers(self.module)

        sharded_params = set()
        for layer in self.sharded_layers:
            sharded_params.update(layer.parameters())

        self.unsharded_params = [parameter for parameter in self.module.parameters() if parameter not in sharded_params]
        if self.world_size > 1:
            for parameter in self.unsharded_params:
                if parameter.requires_grad:
                    parameter.register_post_accumulate_grad_hook(self._make_unsharded_hook())

    def _wrap_layers(self, parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if is_sharded_layer(child):
                layer = make_fsdp_layer(child, len(self.sharded_layers), self, self.compute_dtype)
                setattr(parent, name, layer)
                self.sharded_layers.append(layer)
            else:
                self._wrap_layers(child)

    @staticmethod
    def _accumulate_grad(parameter: nn.Parameter, grad: torch.Tensor) -> None:
        grad = grad.to(device=parameter.device, dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = grad.contiguous()
        else:
            parameter.grad.add_(grad)

    def _sync_replicated_grad(self, parameter: nn.Parameter, grad: torch.Tensor) -> None:
        grad = grad.detach().to(device=parameter.device, dtype=torch.float32).contiguous()
        if self.world_size == 1:
            self._accumulate_grad(parameter, grad)
            return
        handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
        self.bias_sync_handles.append((handle, grad, parameter))

    def _make_unsharded_hook(self):
        def hook(parameter: nn.Parameter) -> None:
            if parameter.grad is not None and self.world_size > 1:
                grad = parameter.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                    parameter.grad = grad
                handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                self.bias_sync_handles.append((handle, grad, parameter))

        return hook

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:  # ty: ignore[invalid-mutable-override]
        return self.module(*inputs, **kwargs)

    def finish_grad_sync(self) -> None:
        for handle, grad_shard, parameter in self.grad_sync_handles:
            handle.wait()
            self._accumulate_grad(parameter, grad_shard / self.world_size)
        self.grad_sync_handles.clear()

        for handle, grad, parameter in self.bias_sync_handles:
            handle.wait()
            averaged_grad = grad / self.world_size
            if parameter.grad is None or parameter.grad is not grad:
                self._accumulate_grad(parameter, averaged_grad)
            else:
                grad.div_(self.world_size)
        self.bias_sync_handles.clear()

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Reconstruct full weights and return all parameters by their model names."""
        sharded_by_parameter_id = {id(layer.weight): layer for layer in self.sharded_layers}
        full_params: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            layer = sharded_by_parameter_id.get(id(parameter))
            if layer is None:
                full_params[name] = parameter.detach().clone()
                continue

            shard = parameter.detach().contiguous()
            if self.world_size == 1:
                full_flat = shard.clone()
            else:
                full_flat = torch.empty(layer.numel, dtype=shard.dtype, device=shard.device)
                dist.all_gather_into_tensor(full_flat, shard)
            full_params[name] = full_flat.reshape(layer.orig_shape)
        return full_params
