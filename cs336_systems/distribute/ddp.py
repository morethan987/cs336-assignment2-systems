from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class BaseDDP(nn.Module, ABC):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @abstractmethod
    def finish_grad_sync(self) -> None:
        """Method to sync grad"""


class NaiveDDP(BaseDDP):
    def finish_grad_sync(self) -> None:
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


class FlatOnlyDDP(BaseDDP):
    def finish_grad_sync(self) -> None:
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        if not grads:
            return

        flat_grad: torch.Tensor = _flatten_dense_tensors(grads)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG)

        for g, synced_g in zip(grads, _unflatten_dense_tensors(flat_grad, grads)):
            g.copy_(synced_g)


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


######## Optinal BucketDDP ########
# Overlapped DDP is the best on A100
# Even the ZeroCopyBucketDDP is much worse
# The real lecture of "the cost of mechanism".
# Optimization without considering hardware always fails
class BucketedDDP(BaseDDP):
    def __init__(self, module: nn.Module):
        super().__init__(module)
        self._register_hooks()

        self.total_bytes = sum(p.nbytes for p in self.module.parameters() if p.requires_grad)
        self.first_bucket_bytes, self.bucket_bytes = self._set_bucket_thresholds()

        self.curr_bucket: list[nn.Parameter] = []
        self.curr_buckets_bytes: int = 0
        self.is_first = True

        self.pending_syncs: list[tuple[dist.Work, torch.Tensor, list[torch.Tensor]]] = []

    def _set_bucket_thresholds(self) -> tuple[int, int]:
        """
        Get thresholds for buckets
        Return:
            tuple[int, int]: first bucket size and standard bucket size
        """
        MB: int = 1024 * 1024
        total_mb = self.total_bytes / MB
        if total_mb <= 10:
            return self.total_bytes, self.total_bytes
        elif total_mb <= 50:
            return 2 * MB, 8 * MB
        else:
            return 5 * MB, 25 * MB

    def _register_hooks(self):
        for p in set(self.module.parameters()):
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param: torch.nn.Parameter):
        self.curr_bucket.append(param)
        self.curr_buckets_bytes += param.nbytes
        threshold = self.first_bucket_bytes if self.is_first else self.bucket_bytes
        if self.curr_buckets_bytes >= threshold:
            self._flush_bucket()
            self.is_first = False

    def _flush_bucket(self):
        """
        Flat grads in bucket and all-reduce
        """
        if not self.curr_bucket:
            return

        curr_grads = [p.grad for p in self.curr_bucket if p.grad is not None]
        flat_grad: torch.Tensor = _flatten_dense_tensors(curr_grads)
        handle: dist.Work = dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG, async_op=True)
        self.pending_syncs.append((handle, flat_grad, curr_grads))

        self.curr_bucket = []
        self.curr_buckets_bytes = 0

    def finish_grad_sync(self) -> None:
        # flush last bucket
        if self.curr_bucket:
            self._flush_bucket()
        for handle, flat_grad, grads in self.pending_syncs:
            handle.wait()
            for g, synced_g in zip(grads, _unflatten_dense_tensors(flat_grad, grads)):
                g.copy_(synced_g)

        self.pending_syncs.clear()
        self.curr_bucket.clear()
        self.curr_buckets_bytes = 0
        self.is_first = True


class _Bucket:
    def __init__(self, params: list[nn.Parameter], device: torch.device):
        self.params = params
        self.total_numel = sum(p.numel() for p in params)
        self.buffer = torch.zeros(self.total_numel, dtype=params[0].dtype, device=device)
        self.pending_params = len(params)
        self.handle: dist.Work | None = None

        self.param_slices: list[tuple[nn.Parameter, int, int, torch.Size]] = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.param_slices.append((p, offset, numel, p.shape))
            offset += numel

    def reset(self):
        self.pending_params = len(self.params)
        self.handle = None

    def zero_buffer(self):
        self.buffer.zero_()

    def rebind_views(self):
        for p, offset, numel, shape in self.param_slices:
            p.grad = self.buffer[offset : offset + numel].view(shape)


class ZeroCopyBucketedDDP(BaseDDP):
    def __init__(
        self,
        module: nn.Module,
        bucket_cap_mb: float = 25.0,
        first_bucket_cap_mb: float = 5.0,
    ):
        super().__init__(module)
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)
        self.first_bucket_cap_bytes = int(first_bucket_cap_mb * 1024 * 1024)

        self.buckets: list[_Bucket] = []
        self.param_to_bucket: dict[nn.Parameter, _Bucket] = {}

        self._init_buckets()

        self._register_hooks()

        self._prepare_buckets_and_grads()

    def _init_buckets(self):
        reversed_params = [p for p in reversed(list(self.module.parameters())) if p.requires_grad]
        if not reversed_params:
            return

        device = reversed_params[0].device
        curr_params: list[nn.Parameter] = []
        curr_bytes = 0
        is_first_bucket = True

        for p in reversed_params:
            threshold = self.first_bucket_cap_bytes if is_first_bucket else self.bucket_cap_bytes

            if curr_params and p.dtype != curr_params[0].dtype:
                self._add_bucket(curr_params, device)
                curr_params = []
                curr_bytes = 0
                is_first_bucket = False
                threshold = self.bucket_cap_bytes

            curr_params.append(p)
            curr_bytes += p.nbytes

            if curr_bytes >= threshold:
                self._add_bucket(curr_params, device)
                curr_params = []
                curr_bytes = 0
                is_first_bucket = False

        if curr_params:
            self._add_bucket(curr_params, device)

    def _add_bucket(self, params: list[nn.Parameter], device: torch.device):
        bucket = _Bucket(params, device)
        self.buckets.append(bucket)
        for p in params:
            self.param_to_bucket[p] = bucket

    def _register_hooks(self):
        for p in self.param_to_bucket:
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param: torch.nn.Parameter):
        bucket = self.param_to_bucket[param]
        bucket.pending_params -= 1

        if bucket.pending_params == 0:
            bucket.handle = dist.all_reduce(
                bucket.buffer,
                op=dist.ReduceOp.AVG,
                async_op=True,
            )

    def _prepare_buckets_and_grads(self):
        for bucket in self.buckets:
            bucket.reset()
            bucket.zero_buffer()
            bucket.rebind_views()

    def __call__(self, *args, **kwargs):
        self._prepare_buckets_and_grads()
        return self.module(*args, **kwargs)

    def finish_grad_sync(self) -> None:
        for bucket in self.buckets:
            if bucket.handle is not None:
                bucket.handle.wait()
                bucket.handle = None


DDP_REGISTRY: dict[str, type[BaseDDP]] = {
    "naive": NaiveDDP,
    "flatonly": FlatOnlyDDP,
    "overlapped": OverlappedDDP,
    "bucket": BucketedDDP,
    "zerocopy": ZeroCopyBucketedDDP,
}


def get_ddp_cls(ddp_type: str) -> type[BaseDDP]:
    name = ddp_type.lower()
    if name not in DDP_REGISTRY:
        available = list(DDP_REGISTRY.keys())
        raise ValueError(f"Unknown DDP type: '{ddp_type}'. Available types: {available}")
    return DDP_REGISTRY[name]
