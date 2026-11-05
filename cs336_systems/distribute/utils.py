import json
import os
import re
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from pydantic import BaseModel, Field, model_validator

from cs336_systems.distribute.ddp import DDP_REGISTRY
from cs336_systems.utils import ModelConfig

UNITS = {
    "b": 1,
    "kb": 10**3,
    "kib": 1024,
    "mb": 10**6,
    "mib": 1024**2,
    "gb": 10**9,
    "gib": 1024**3,
}


class DCConfig(BaseModel):
    world_size: int = Field(default=2, description="Number of distributed ranks")
    steps: int = Field(default=10, description="Benchmark timing iterations")
    warm_up: int = Field(default=5, description="Warmup iterations")
    backend: Literal["gloo", "nccl"] = Field(default="gloo", description="Communication backend")
    device: Literal["cuda", "cpu"] = Field(default="cpu", description="Target device")
    data_size: str = Field(default="1MB", description="e.g. 1MB, 10MB, 100MB, 1GB")
    batch_size: int = Field(default=4, description="Batch size")
    ddp_type: str = Field(default="none", description=f"Choices: {list(DDP_REGISTRY.keys())}")
    opt_shard: bool = Field(default=False, description="Use optimizer sharding")
    fsdp: bool = Field(default=False, description="Use fully-sharded data parallel")
    record_memory: bool = Field(default=False, description="Record memory with torch.profile and analyze it")
    master_addr: str = Field(default="localhost", description="Master IP address")
    master_port: str = Field(default="29500", description="Master port")
    save_base_dir: Path = Field(default=Path("distributed_res"), description="Base directory to save results")

    @model_validator(mode="after")
    def validate_config(self):
        if self.device == "cpu" and self.backend == "nccl":
            raise ValueError("NCCL backend does not support CPU tensors.")
        if self.ddp_type not in DDP_REGISTRY:
            raise ValueError(f"Invalid ddp_type: {self.ddp_type}. Available: {list(DDP_REGISTRY.keys())}")
        if self.ddp_type != "none" and self.fsdp:
            raise ValueError(f"fsdp ({self.fsdp}) and ddp_type ({self.ddp_type}) can not be set at the same time!")
        if self.opt_shard and self.fsdp:
            raise ValueError(f"fsdp ({self.fsdp}) and opt_shard ({self.opt_shard}) can not be set at the same time!")
        return self


def generate_data(model_cfg: ModelConfig, dc_cfg: DCConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tks = torch.randint(
        0,
        model_cfg.vocab_size,
        (dc_cfg.batch_size, model_cfg.context_length + 1),
        device=device,
        dtype=torch.int,
    )
    return tks[:, :-1], tks[:, 1:]


def get_tensor_size(size_str: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)$", size_str.strip())
    if not match:
        raise ValueError(f"Invalid data size string: {size_str}")

    val, unit = float(match.group(1)), match.group(2).lower()
    size = int(val * UNITS[unit]) // 4  # float32 = 4 Bytes
    return size


def setup(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "gloo"):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    device_id = None
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank)
        device_id = torch.device(f"cuda:{rank}")

    dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=device_id)


def sync(device_type: str, rank: int = 0):
    if device_type == "cuda":
        torch.cuda.synchronize(device=rank)


def summarize_metrics(gathered_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Return: min, max, mean
    """
    if not gathered_metrics:
        return {}
    all_keys = gathered_metrics[0].keys()
    return {
        k: {
            "mean": sum(m[k] for m in gathered_metrics) / len(gathered_metrics),
            "max": max(m[k] for m in gathered_metrics),
            "min": min(m[k] for m in gathered_metrics),
        }
        for k in all_keys
    }


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cur_s, cur_e in intervals[1:]:
        last_s, last_e = merged[-1]
        if cur_s <= last_e:
            merged[-1] = (last_s, max(last_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


def subtract_intervals(comm: list[tuple[float, float]], comp: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Exposed communication time"""
    exposed = []
    i = 0
    j = 0
    while i < len(comm):
        c_start, c_end = comm[i]
        while j < len(comp) and comp[j][1] <= c_start:
            j += 1

        curr_start = c_start
        k = j
        while k < len(comp) and comp[k][0] < c_end:
            comp_s, comp_e = comp[k]
            if comp_s > curr_start:
                exposed.append((curr_start, min(comp_s, c_end)))
            curr_start = max(curr_start, comp_e)
            if curr_start >= c_end:
                break
            k += 1

        if curr_start < c_end:
            exposed.append((curr_start, c_end))
        i += 1
    return exposed


def time_profile_json(trace_path: Path) -> dict[str, float]:
    """
    Analyze the json file exported by `export_chrome_trace`
    Return:
    {
        "actual_comm_time_ms": total communication time,
        "exposed_comm_time_ms": communication time that not covered by computation,
        "overlap_ratio_pct": ration of overlapping,
    }
    """
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)

    comm_raw = []
    comp_raw = []

    for event in data.get("traceEvents", []):
        if event.get("ph") == "X" and "ts" in event and "dur" in event:
            cat = event.get("cat", "")
            name = event.get("name", "")

            if cat == "kernel":
                start = float(event["ts"])
                end = start + float(event["dur"])

                if "ncclKernel" in name or "nccl" in name.lower():
                    comm_raw.append((start, end))
                else:
                    comp_raw.append((start, end))

    merged_comm = merge_intervals(comm_raw)
    merged_comp = merge_intervals(comp_raw)

    actual_comm_us = sum(e - s for s, e in merged_comm)

    exposed_comm = subtract_intervals(merged_comm, merged_comp)
    exposed_comm_us = sum(e - s for s, e in exposed_comm)

    actual_comm_ms = actual_comm_us / 1000.0
    exposed_comm_ms = exposed_comm_us / 1000.0
    overlap_ratio = ((actual_comm_ms - exposed_comm_ms) / actual_comm_ms * 100.0) if actual_comm_ms > 0 else 0.0

    return {
        "actual_comm_time_ms": actual_comm_ms,
        "exposed_comm_time_ms": exposed_comm_ms,
        "overlap_ratio_pct": overlap_ratio,
    }


def memory_profile_json(trace_path: Path) -> dict[str, float]:
    """
    Analyze the memory at:

    1. After model initialization: before FORWARD phase
    2. Directly before the optimizer step
    3. Directly after the optimizer step
    """
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    mem_timeline: list[tuple[float, float]] = []  # (ts, allocated_bytes)

    forward_starts: list[float] = []
    optimizer_spans: list[tuple[float, float]] = []

    for event in events:
        name = event.get("name", "")
        ph = event.get("ph", "")

        # get memory events
        if name == "[memory]" and ph == "i":
            args = event.get("args", {})
            if args.get("Device Type") == 1:  # CUDA
                allocated = args.get("Total Allocated") or args.get("total_allocated")
                if allocated is not None:
                    mem_timeline.append((float(event["ts"]), float(allocated)))

        # get phases
        elif event.get("cat") == "user_annotation" or ph == "X":
            if name == "FORWARD":
                forward_starts.append(float(event["ts"]))
            elif name == "OPTIMIZER_STEP":
                start = float(event["ts"])
                end = start + float(event.get("dur", 0))
                optimizer_spans.append((start, end))

    if not mem_timeline:
        return {
            "mem_before_forward_mb": 0.0,
            "mem_before_optimizer_mb": 0.0,
            "mem_after_optimizer_mb": 0.0,
        }

    # 按时间戳排序
    mem_timeline.sort(key=lambda x: x[0])
    forward_starts.sort()
    optimizer_spans.sort(key=lambda x: x[0])

    def get_memory_at(target_ts: float) -> float:
        """
        Find the nearest memory record before target_ts
        """
        curr_mem = 0.0
        for ts, mem in mem_timeline:
            if ts <= target_ts:
                curr_mem = mem
            else:
                break
        return curr_mem / (1024 * 1024)  # to MB

    # Memory before forward, model params only
    t1 = forward_starts[0] if forward_starts else mem_timeline[0][0]
    mem_before_forward = get_memory_at(t1)

    # before and after optimizer step
    target_opt = optimizer_spans[0] if optimizer_spans else (0.0, 0.0)
    mem_before_opt = get_memory_at(target_opt[0])
    mem_after_opt = get_memory_at(target_opt[1])

    return {
        "mem_before_forward_mb": mem_before_forward,
        "mem_before_optimizer_mb": mem_before_opt,
        "mem_after_optimizer_mb": mem_after_opt,
    }
