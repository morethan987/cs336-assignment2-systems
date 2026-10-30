from collections.abc import Callable
from typing import Any, overload

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class OptShard(Optimizer):
    def __init__(self, params: ParamsT, optimizer_cls: type[Optimizer], **kwargs: Any):
        """
        Args:
            params: the parameters allocated to current process
            optimizer_cls: the optimizer class going to wrap
            **kwargs: hyperparameters for optimizer_cls
        """
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.param_to_rank: dict[torch.nn.Parameter, int] = {}
        self.all_params: list[torch.nn.Parameter] = []
        self.optim: Optimizer | None = None  # local optimizer
        self.local_param_groups = []

        # add_param_group is hard coded in PyTorch Optimizer class
        # So add_param_group will be called here
        super().__init__(params, kwargs)

        self.optim = self.optimizer_cls(self.local_param_groups, **self.kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a param_group to optimizer
        Note: super().__init__ will automatically call this method
        """
        params = list(param_group["params"])
        param_group["params"] = params

        local_params = []
        for p in params:
            if p not in self.param_to_rank:  # avoid repeatedly adding
                # round-robin
                assigned_rank = len(self.param_to_rank) % self.world_size
                self.param_to_rank[p] = assigned_rank
                self.all_params.append(p)
            else:
                assigned_rank = self.param_to_rank[p]

            if assigned_rank == self.rank:
                local_params.append(p)

        # copy other settings
        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        self.local_param_groups.append(local_group)

        # maintain the PyTorch Optimizer's inner properties
        super().add_param_group(param_group)

        # for dynamic adding params in training
        # such as unfreeze some params
        if self.optim is not None:
            self.optim.add_param_group(local_group)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        assert self.optim is not None, "Optimizer has not been initialized."
        self.optim.step()

        for param in self.all_params:
            owner_rank = self.param_to_rank[param]
            dist.broadcast(param.data, src=owner_rank)

        return loss
