import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from cs336_systems.utils import ModelConfig, create_model_config, setup_model_args

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
    batch_size: int
    master_addr: str
    master_port: str
    save_base_dir: Path

    def check(self):
        if self.device == "cpu" and self.backend == "nccl":
            raise ValueError("NCCL backend does not support CPU tensors.")


def setup_dc_benchmark_args(parser: argparse.ArgumentParser):
    """Inject args of distributed communicationg"""
    group = parser.add_argument_group("Distributed Communication Benchmark Args")
    group.add_argument("--world_size", type=int, default=2)
    group.add_argument("--steps", type=int, default=10)
    group.add_argument("--warm_up", type=int, default=5)
    group.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    group.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    group.add_argument("--data_size", type=str, default="1MB", help="e.g. 1MB, 10MB, 100MB, 1GB")
    group.add_argument("--batch_size", type=int, default=4)
    group.add_argument("--master_addr", type=str, default="localhost")
    group.add_argument("--master_port", type=str, default="29500")
    group.add_argument("--save_base_dir", type=Path, default="distributed_res")


def create_dc_config(args: argparse.Namespace) -> DCConfig:
    dc_cfg = DCConfig(
        world_size=args.world_size,
        steps=args.steps,
        warm_up=args.warm_up,
        backend=args.backend,
        device=args.device,
        data_size=args.data_size,
        batch_size=args.batch_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        save_base_dir=args.save_base_dir,
    )
    dc_cfg.check()
    return dc_cfg


def generate_data(model_cfg: ModelConfig, dc_cfg: DCConfig) -> tuple[torch.Tensor, torch.Tensor]:
    tks = torch.randint(
        0,
        model_cfg.vocab_size,
        (dc_cfg.batch_size, model_cfg.context_length + 1),
        device=model_cfg.device,
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
