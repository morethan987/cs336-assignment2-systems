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
    ddp_type: str = Field(default="naive", description=f"Choices: {list(DDP_REGISTRY.keys())}")
    master_addr: str = Field(default="localhost", description="Master IP address")
    master_port: str = Field(default="29500", description="Master port")
    save_base_dir: Path = Field(default=Path("distributed_res"), description="Base directory to save results")

    @model_validator(mode="after")
    def validate_config(self):
        if self.device == "cpu" and self.backend == "nccl":
            raise ValueError("NCCL backend does not support CPU tensors.")
        if self.ddp_type not in DDP_REGISTRY:
            raise ValueError(f"Invalid ddp_type: {self.ddp_type}. Available: {list(DDP_REGISTRY.keys())}")
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
