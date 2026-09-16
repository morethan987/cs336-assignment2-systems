"""
cuda_memory_types.py - PyTorch CUDA Memory Snapshot 静态类型定义
规范参考: https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html
"""

import pickle
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast


class Frame(TypedDict):
    """堆栈帧信息"""

    filename: str
    line: int
    name: str
    # 当开启 FX 追踪 (augment_with_fx_traces=True) 时可能包含以下字段
    fx_node_op: NotRequired[str]
    fx_node_name: NotRequired[str]
    fx_original_trace: NotRequired[str]


TraceAction = Literal[
    "alloc",  # 申请张量显存
    "free_requested",  # Python 侧请求释放显存
    "free_completed",  # 显存实际归还/跨流同步完成
    "segment_alloc",  # 分配器向驱动申请物理显存 (cudaMalloc)
    "segment_free",  # 分配器归还物理显存给驱动 (cudaFree)
    "oom",  # 发生 OOM 异常
    "snapshot",  # 快照打点事件
    "annotate",  # 用户打点元数据
]


class TraceEntry(TypedDict):
    """时间线单个事件"""

    action: TraceAction
    addr: NotRequired[int]  # OOM 事件没有 addr
    size: int  # 本次操作的字节数
    stream: NotRequired[int]
    time_us: NotRequired[int]  # 微秒时间戳
    frames: NotRequired[list[Frame]]  # 调用栈
    device_free: NotRequired[int]  # OOM 事件专用：CUDA当前仍报告空闲的字节
    pool_id: NotRequired[tuple[int, int]]
    user_metadata: NotRequired[str]  # annotate 事件专用


BlockState = Literal["active_allocated", "active_awaiting_free", "inactive"]


class Block(TypedDict):
    """Segment 内部切分的内存块"""

    size: int
    requested_size: int
    address: int
    state: BlockState
    frames: NotRequired[list[Frame]]


class Segment(TypedDict):
    """分配器向底层驱动申请的物理内存段 (cudaMalloc 返回的整块)"""

    address: int
    total_size: int
    stream: int
    segment_type: Literal["small", "large"]
    allocated_size: int
    active_size: int
    blocks: list[Block]
    segment_pool_id: NotRequired[tuple[int, int]]


class Snapshot(TypedDict):
    """torch.cuda.memory._dump_snapshot() 保存的顶层字典"""

    segments: list[Segment]
    device_traces: list[list[TraceEntry]]
    allocator_settings: NotRequired[dict[str, Any]]
    external_annotations: NotRequired[list[Any]]


def load_snapshot(file_path: str | Path) -> Snapshot:
    """带类型断言的快照加载"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return cast(Snapshot, data)
