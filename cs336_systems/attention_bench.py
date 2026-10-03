import argparse
import gc
import json
import os
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import einx
import numpy as np
import torch
from cs336_basics.layers import softmax
from cs336_basics.train_loop.utils import parse_dtype

from cs336_systems.triton_kernels import FlashAttention


# align the signature with FlashAttention
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool) -> torch.Tensor:
    # CPU scalar, reduce CUDA kernel cost
    scale = 1.0 / (Q.shape[-1] ** 0.5)

    # n and m all respresent seq_len, only to tag matrix shape: (n, m) or (m, n)
    scaled_dot = einx.dot("... n [d_k], ... m [d_k] -> ... n m", Q, K) * scale
    if is_causal:
        mask = torch.triu(
            torch.ones(Q.shape[-2], K.shape[-2], device=Q.device, dtype=torch.bool),
            diagonal=1,
        )
        scaled_dot = scaled_dot.masked_fill(mask, float("-inf"))
    return einx.dot("... n [m], ... [m] d_v -> ... n d_v", softmax(scaled_dot, -1), V)


@dataclass
class AttBenchConfig:
    steps: int
    warm_up: int
    d_models: list[int]
    seq_lens: list[int]
    compile: bool
    flash: bool
    causal_masking: bool
    batch_size: int
    device: torch.device
    dtype: torch.dtype
    show: Path | None
    res_dir: Path

    def check(self) -> None:
        if self.compile and self.flash:
            raise ValueError("Argument compile and flash should not be set at the same time!")

    def save(self, file: Path | str):
        data = {k: str(v) if isinstance(v, (torch.device, torch.dtype, Path)) else v for k, v in self.__dict__.items()}
        with open(file, "w") as f:
            json.dump(data, f, indent=2)


def _rand_qkv(batch_size: int, d_model: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors requiring gradients on the target device."""
    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


def _run_single_exp_impl(cfg: AttBenchConfig, d_model: int, seq_len: int, att_fn: Callable) -> tuple[np.ndarray, float, float]:
    """
    Execute the benchmark body. All tensors are scoped strictly as local variables.

    Returns:
        (raw_times, mem_before_bwd_mib, max_mem_mib)
    """
    fwd_times = np.zeros(cfg.steps, dtype=np.float64)
    bwd_times = np.zeros(cfg.steps, dtype=np.float64)
    mem_before_bwd_mib: float = 0.0

    Q, K, V = _rand_qkv(cfg.batch_size, d_model, seq_len, device=cfg.device, dtype=cfg.dtype)
    grad_out = torch.randn(cfg.batch_size, seq_len, d_model, device=cfg.device, dtype=cfg.dtype)

    # Warmup passes
    for _ in range(cfg.warm_up):
        Q.grad = K.grad = V.grad = None
        out = att_fn(Q, K, V, cfg.causal_masking)
        out.backward(grad_out)
    torch.cuda.synchronize(cfg.device)

    # Reset peak stats after warmup so warmup passes do not taint measurements
    torch.cuda.reset_peak_memory_stats(cfg.device)

    # Timed iterations
    for i in range(cfg.steps):
        Q.grad = K.grad = V.grad = None

        # forward
        torch.cuda.synchronize(cfg.device)
        t0 = timeit.default_timer()
        out = att_fn(Q, K, V, cfg.causal_masking)
        torch.cuda.synchronize(cfg.device)
        t1 = timeit.default_timer()

        # Record allocated memory right before the backward pass begins
        if i == 0:
            mem_before_bwd_mib = float(torch.cuda.memory_allocated(cfg.device) / (1024**2))

        # backward
        out.backward(grad_out)
        torch.cuda.synchronize(cfg.device)
        t2 = timeit.default_timer()

        fwd_times[i] = t1 - t0
        bwd_times[i] = t2 - t1

    max_mem_mib = float(torch.cuda.max_memory_allocated(cfg.device) / (1024**2))
    raw_times = np.stack([fwd_times, bwd_times], axis=0)
    return raw_times, mem_before_bwd_mib, max_mem_mib


def run_single_exp(cfg: AttBenchConfig, d_model: int, seq_len: int) -> tuple[np.ndarray | None, float | None, float | None]:
    """
    Wrapper around `_run_single_exp_impl` to trap Out-Of-Memory exceptions.

    Returns:
        (raw_times, mem_before_bwd_mib, max_mem_mib):
            All elements are None if an OOM occurs.
    """
    try:
        if cfg.compile:
            torch.compiler.reset()
            att_fn = torch.compile(scaled_dot_product_attention)
        elif cfg.flash:
            torch.compiler.reset()  # the FlashAttention may be partially implemented which contains some torch.compile
            att_fn = FlashAttention.apply
        else:
            att_fn = scaled_dot_product_attention
        return _run_single_exp_impl(cfg, d_model, seq_len, att_fn)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return None, None, None
        raise


def print_table_header() -> None:
    border = "+---------+---------+--------------------+--------------------+-----------------+-----------------+--------+"
    header = "| d_model | seq_len |   Fwd (ms) ± Std   |   Bwd (ms) ± Std   | Mem Before(MiB) |  Peak Mem (MiB) | Status |"
    print(border)
    print(header)
    print(border)


def print_table_row(
    d_model: int,
    seq_len: int,
    raw_times: np.ndarray | None,
    mem_before_mib: float | None,
    max_mem_mib: float | None,
) -> None:
    if raw_times is not None and mem_before_mib is not None and max_mem_mib is not None:
        fwd_ms = raw_times[0] * 1000.0
        bwd_ms = raw_times[1] * 1000.0
        fwd_str = f"{np.mean(fwd_ms):7.2f} ± {np.std(fwd_ms):5.2f}"
        bwd_str = f"{np.mean(bwd_ms):7.2f} ± {np.std(bwd_ms):5.2f}"
        mem_str = f"{mem_before_mib:13.2f}"
        peak_str = f"{max_mem_mib:13.2f}"
        status_str = "\033[92m  OK  \033[0m"
    else:
        fwd_str = "        N/A       "
        bwd_str = "        N/A       "
        mem_str = "          N/A"
        peak_str = "          N/A"
        status_str = "\033[91m  OOM \033[0m"

    print(f"| {d_model:^7} | {seq_len:^7} | {fwd_str:^18} | {bwd_str:^18} | {mem_str:^15} | {peak_str:^15} | {status_str} |")


def load(file_path: str | Path) -> dict[str, np.ndarray]:
    """Load benchmark results from a .npz file."""
    with np.load(file_path) as data:
        return {key: data[key] for key in data.files}


def show(data: dict[str, np.ndarray], title: str = "Attention Benchmark Results") -> None:
    """Reproduce terminal benchmark table output from loaded data."""
    border = "+---------+---------+--------------------+--------------------+-----------------+-----------------+--------+"
    print("=" * 106)
    print(f" {title}")
    print("=" * 106)
    print_table_header()

    d_models = data["d_models"]
    seq_lens = data["seq_lens"]
    raw_times = data["raw_times"]
    mem_before = data["mem_before"]
    mem_peak = data["mem_peak"]

    for d_idx, d_model in enumerate(d_models):
        for s_idx, seq_len in enumerate(seq_lens):
            if np.isnan(mem_peak[d_idx, s_idx]):
                print_table_row(int(d_model), int(seq_len), None, None, None)
            else:
                print_table_row(
                    int(d_model),
                    int(seq_len),
                    raw_times[d_idx, s_idx],
                    float(mem_before[d_idx, s_idx]),
                    float(mem_peak[d_idx, s_idx]),
                )
    print(border)


def parse_args() -> AttBenchConfig:
    parser = argparse.ArgumentParser(description="Attention Benchmark with optional torch.compile")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warm_up", type=int, default=10)
    parser.add_argument("--d_models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for attention")
    parser.add_argument("--flash", action="store_true", help="Enable flash attention")
    parser.add_argument("--causal_masking", action="store_true", help="Enable causal masking")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)

    parser.add_argument("--show", type=Path, default=None, help="Path to .npz file to display results directly")
    parser.add_argument("--res_dir", type=Path, default=Path("attention_bench_res"))
    args = parser.parse_args()

    att_bench_cfg = AttBenchConfig(
        steps=args.steps,
        warm_up=args.warm_up,
        d_models=args.d_models,
        seq_lens=args.seq_lens,
        compile=args.compile,
        flash=args.flash,
        causal_masking=args.causal_masking,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        show=args.show,
        res_dir=args.res_dir,
    )
    att_bench_cfg.check()

    return att_bench_cfg


def main() -> None:
    cfg = parse_args()

    # show mode
    if cfg.show:
        show(load(cfg.show), title=f"PyTorch Attention Benchmark ({cfg.show})")
        return

    assert torch.cuda.is_available(), "CUDA is not available on this system."
    torch.cuda.init()
    torch.cuda.set_device(cfg.device)

    mode_str = "compile" if cfg.compile else ("flash" if cfg.flash else "eager")
    timestamp = datetime.now(ZoneInfo("Asia/Chongqing")).strftime("%Y%m%d_%H%M%S")
    output_dir = cfg.res_dir / f"{mode_str}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    cfg.save(output_dir / "args.json")

    print(f"Running benchmark (Batch Size={cfg.batch_size}, Device={cfg.device}, Mode={mode_str})...")

    # Pre-allocate result grids (filled with NaN for OOM configurations)
    raw_all = np.full((len(cfg.d_models), len(cfg.seq_lens), 2, cfg.steps), np.nan, dtype=np.float64)
    mem_before_all = np.full((len(cfg.d_models), len(cfg.seq_lens)), np.nan, dtype=np.float64)
    mem_peak_all = np.full((len(cfg.d_models), len(cfg.seq_lens)), np.nan, dtype=np.float64)

    for d_idx, d_model in enumerate(cfg.d_models):
        for s_idx, seq_len in enumerate(cfg.seq_lens):
            raw_times, mem_before_mib, max_mem_mib = run_single_exp(cfg, d_model, seq_len)
            if raw_times is not None:
                raw_all[d_idx, s_idx] = raw_times
                mem_before_all[d_idx, s_idx] = mem_before_mib
                mem_peak_all[d_idx, s_idx] = max_mem_mib

    # save
    save_path = output_dir / "benchmark_results.npz"
    np.savez_compressed(
        save_path,
        raw_times=raw_all,
        mem_before=mem_before_all,
        mem_peak=mem_peak_all,
        d_models=np.array(cfg.d_models),
        seq_lens=np.array(cfg.seq_lens),
    )

    show(load(save_path), title=f"PyTorch Attention Benchmark (Batch Size={cfg.batch_size}, Device={cfg.device})")
    print(f"\n[+] Benchmark completed. All results saved to: {save_path}")


if __name__ == "__main__":
    main()
