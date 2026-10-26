from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch import nn


class BaseDDP(nn.Module, ABC):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        self.forward = self.module.forward

    @abstractmethod
    def finish_grad_sync(self) -> None:
        """Method to sync grad"""


class NaiveDDP(BaseDDP):
    def finish_grad_sync(self) -> None:
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


class OverlappedDDP(BaseDDP):
    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.handles: list[dist.Work] = []
        self._register_hooks()

    def _register_hooks(self):
        for p in set(self.module.parameters()):
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param: torch.nn.Parameter):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)

    def finish_grad_sync(self) -> None:
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


DDP_REGISTRY: dict[str, type[BaseDDP]] = {
    "naive": NaiveDDP,
    "overlapped": OverlappedDDP,
}


def get_ddp_cls(ddp_type: str) -> type[BaseDDP]:
    name = ddp_type.lower()
    if name not in DDP_REGISTRY:
        available = list(DDP_REGISTRY.keys())
        raise ValueError(f"Unknown DDP type: '{ddp_type}'. Available types: {available}")
    return DDP_REGISTRY[name]
