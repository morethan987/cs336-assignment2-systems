import argparse
import os
import re
from dataclasses import dataclass

import torch
import torch.distributed as dist

UNITS = {
    "b": 1,
    "kb": 10**3,
    "kib": 1024,
    "mb": 10**6,
    "mib": 1024**2,
    "gb": 10**9,
    "gib": 1024**3,
}


@dataclass
class DCConfig:
    world_size: int
    steps: int
    warm_up: int
    backend: str
    device: str
    data_size: str
    master_addr: str
    master_port: str

    def check(self):
        if self.device == "cpu" and self.backend == "nccl":
            raise ValueError("NCCL backend does not support CPU tensors.")


def parse_args() -> DCConfig:
    parser = argparse.ArgumentParser(description="Distributed Communication Benchmark")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warm_up", type=int, default=5)
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--data_size", type=str, default="1MB", help="e.g. 1MB, 10MB, 100MB, 1GB")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")

    args = parser.parse_args()

    dc_cfg = DCConfig(
        world_size=args.world_size,
        steps=args.steps,
        warm_up=args.warm_up,
        backend=args.backend,
        device=args.device,
        data_size=args.data_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    dc_cfg.check()
    return dc_cfg


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

    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank)

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def sync(device_type: str, rank: int = 0):
    if device_type == "cuda":
        torch.cuda.synchronize(device=rank)
