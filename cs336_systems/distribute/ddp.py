import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        # broadcast params
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        self.forward = self.module.forward

        self._register_hooks()

    def _register_hooks(self):
        for p in set(self.module.parameters()):
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param: torch.nn.Parameter):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)

    def finish_grad_sync(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
