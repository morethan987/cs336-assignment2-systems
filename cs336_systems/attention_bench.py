import argparse
import gc
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import einx
import numpy as np
import torch
import triton
import triton.testing
from cs336_basics.layers import softmax
from cs336_basics.train_loop.utils import parse_dtype

from cs336_systems.triton_kernels import FlashAttention_TileOpt


# align the signature with FlashAttention
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool) -> torch.Tensor:
    # CPU scalar, reduce CUDA kernel cost
    scale = 1.0 / (Q.shape[-1] ** 0.5)

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
    dtypes: list[torch.dtype]
    compile: bool
    flash: bool
    causal_masking: bool
    batch_size: int
    device: torch.device
    show: Path | None
    res_dir: Path

    def check(self) -> None:
        if self.compile and self.flash:
            raise ValueError("Argument compile and flash should not be set at the same time!")

    def save(self, file: Path | str):
        data = {
            k: [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], torch.dtype) else str(v) if isinstance(v, (torch.device, torch.dtype, Path)) else v
            for k, v in self.__dict__.items()
        }
        with open(file, "w") as f:
            json.dump(data, f, indent=2)


def _rand_qkv(batch_size: int, d_model: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors requiring gradients on the target device."""
    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


def _run_single_exp_impl(
    cfg: AttBenchConfig,
    d_model: int,
    seq_len: int,
    dtype: torch.dtype,
    att_fn: Callable,
) -> tuple[tuple[float, float, float], float, float]:
    """
    Execute benchmark using triton.testing.do_bench.

    Returns:
        ((fwd_ms, bwd_ms, e2e_ms), mem_before_bwd_mib, max_mem_mib)
    """
    Q, K, V = _rand_qkv(cfg.batch_size, d_model, seq_len, device=cfg.device, dtype=dtype)
    grad_out = torch.randn(cfg.batch_size, seq_len, d_model, device=cfg.device, dtype=dtype)

    def run_fwd():
        return att_fn(Q, K, V, cfg.causal_masking)

    fwd_ms = float(triton.testing.do_bench(run_fwd, warmup=cfg.warm_up, rep=cfg.steps, return_mode="median"))

    torch.cuda.reset_peak_memory_stats(cfg.device)
    out = att_fn(Q, K, V, cfg.causal_masking)
    mem_before_bwd_mib = float(torch.cuda.memory_allocated(cfg.device) / (1024**2))

    def run_bwd():
        Q.grad = K.grad = V.grad = None
        out.backward(grad_out, retain_graph=True)

    bwd_ms = float(triton.testing.do_bench(run_bwd, warmup=cfg.warm_up, rep=cfg.steps, return_mode="median"))
    max_mem_mib = float(torch.cuda.max_memory_allocated(cfg.device) / (1024**2))

    def run_e2e():
        Q.grad = K.grad = V.grad = None
        o = att_fn(Q, K, V, cfg.causal_masking)
        o.backward(grad_out)

    e2e_ms = float(triton.testing.do_bench(run_e2e, warmup=cfg.warm_up, rep=cfg.steps, return_mode="median"))

    return (fwd_ms, bwd_ms, e2e_ms), mem_before_bwd_mib, max_mem_mib


def run_single_exp(
    cfg: AttBenchConfig,
    d_model: int,
    seq_len: int,
    dtype: torch.dtype,
) -> tuple[tuple[float, float, float] | None, float | None, float | None]:
    """
    Wrapper around `_run_single_exp_impl` to trap Out-Of-Memory exceptions.
    """
    try:
        if cfg.compile:
            torch.compiler.reset()
            att_fn = torch.compile(scaled_dot_product_attention)
        elif cfg.flash:
            torch.compiler.reset()
            att_fn = FlashAttention_TileOpt.apply
        else:
            att_fn = scaled_dot_product_attention
        return _run_single_exp_impl(cfg, d_model, seq_len, dtype, att_fn)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return None, None, None
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def print_table_header() -> None:
    border = "+-------+---------+---------+----------+----------+----------+-----------------+-----------------+--------+"
    header = "| Dtype | d_model | seq_len | Fwd (ms) | Bwd (ms) | E2E (ms) | Mem Before(MiB) |  Peak Mem (MiB) | Status |"
    print(border)
    print(header)
    print(border)


def print_table_row(
    dtype_str: str,
    d_model: int,
    seq_len: int,
    lats: tuple[float, float, float] | None,
    mem_before_mib: float | None,
    max_mem_mib: float | None,
) -> None:
    border_fmt = "| {:^5} | {:^7} | {:^7} | {:^8} | {:^8} | {:^8} | {:^15} | {:^15} | {} |"
    if lats is not None and mem_before_mib is not None and max_mem_mib is not None:
        fwd_str = f"{lats[0]:8.2f}"
        bwd_str = f"{lats[1]:8.2f}"
        e2e_str = f"{lats[2]:8.2f}"
        mem_str = f"{mem_before_mib:13.2f}"
        peak_str = f"{max_mem_mib:13.2f}"
        status_str = "\033[92m  OK  \033[0m"
    else:
        fwd_str = bwd_str = e2e_str = "  N/A   "
        mem_str = peak_str = "      N/A      "
        status_str = "\033[91m  OOM \033[0m"

    print(border_fmt.format(dtype_str, d_model, seq_len, fwd_str, bwd_str, e2e_str, mem_str, peak_str, status_str))


def load(file_path: str | Path) -> dict[str, np.ndarray]:
    """Load benchmark results from a .npz file."""
    with np.load(file_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def show(data: dict[str, np.ndarray], title: str = "Attention Benchmark Results") -> None:
    """Reproduce terminal benchmark table output from loaded data."""
    border = "+-------+---------+---------+----------+----------+----------+-----------------+-----------------+--------+"
    print("=" * 106)
    print(f" {title}")
    print("=" * 106)
    print_table_header()

    dtypes = [str(d) for d in data["dtypes"]]
    d_models = data["d_models"]
    seq_lens = data["seq_lens"]
    lats = data["latencies"]  # shape: (num_dtypes, num_d_models, num_seq_lens, 3)
    mem_before = data["mem_before"]
    mem_peak = data["mem_peak"]

    for dt_idx, dtype_str in enumerate(dtypes):
        clean_dtype = dtype_str.replace("torch.", "")
        for d_idx, d_model in enumerate(d_models):
            for s_idx, seq_len in enumerate(seq_lens):
                if np.isnan(mem_peak[dt_idx, d_idx, s_idx]):
                    print_table_row(clean_dtype, int(d_model), int(seq_len), None, None, None)
                else:
                    curr_lats = (
                        float(lats[dt_idx, d_idx, s_idx, 0]),
                        float(lats[dt_idx, d_idx, s_idx, 1]),
                        float(lats[dt_idx, d_idx, s_idx, 2]),
                    )
                    print_table_row(
                        clean_dtype,
                        int(d_model),
                        int(seq_len),
                        curr_lats,
                        float(mem_before[dt_idx, d_idx, s_idx]),
                        float(mem_peak[dt_idx, d_idx, s_idx]),
                    )
    print(border)


def parse_args() -> AttBenchConfig:
    parser = argparse.ArgumentParser(description="Attention Benchmark via triton.testing.do_bench")
    parser.add_argument("--steps", type=int, default=100, help="rep steps in triton.testing.do_bench")
    parser.add_argument("--warm_up", type=int, default=25, help="warmup steps in triton.testing.do_bench")
    parser.add_argument("--d_models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--dtypes", nargs="+", type=parse_dtype, default=[torch.bfloat16, torch.float32])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for attention")
    parser.add_argument("--flash", action="store_true", help="Enable flash attention")
    parser.add_argument("--no_causal", dest="causal_masking", action="store_false", help="Disable causal masking")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--show", type=Path, default=None, help="Path to .npz file to display results directly")
    parser.add_argument("--res_dir", type=Path, default=Path("attention_bench_res"))
    args = parser.parse_args()

    att_bench_cfg = AttBenchConfig(
        steps=args.steps,
        warm_up=args.warm_up,
        d_models=args.d_models,
        seq_lens=args.seq_lens,
        dtypes=args.dtypes,
        compile=args.compile,
        flash=args.flash,
        causal_masking=args.causal_masking,
        batch_size=args.batch_size,
        device=args.device,
        show=args.show,
        res_dir=args.res_dir,
    )
    att_bench_cfg.check()
    return att_bench_cfg


def main() -> None:
    cfg = parse_args()

    # show mode
    if cfg.show:
        show(load(cfg.show), title=f"Attention Benchmark ({cfg.show})")
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
    print_table_header()

    # Pre-allocate grids (num_dtypes, num_d_models, num_seq_lens, 3) -> 3 represents [fwd, bwd, e2e]
    lats_all = np.full((len(cfg.dtypes), len(cfg.d_models), len(cfg.seq_lens), 3), np.nan, dtype=np.float64)
    mem_before_all = np.full((len(cfg.dtypes), len(cfg.d_models), len(cfg.seq_lens)), np.nan, dtype=np.float64)
    mem_peak_all = np.full((len(cfg.dtypes), len(cfg.d_models), len(cfg.seq_lens)), np.nan, dtype=np.float64)

    for dt_idx, dtype in enumerate(cfg.dtypes):
        dtype_str = str(dtype).replace("torch.", "")
        for d_idx, d_model in enumerate(cfg.d_models):
            for s_idx, seq_len in enumerate(cfg.seq_lens):
                lats, mem_before_mib, max_mem_mib = run_single_exp(cfg, d_model, seq_len, dtype)
                if lats is not None:
                    lats_all[dt_idx, d_idx, s_idx] = lats
                    mem_before_all[dt_idx, d_idx, s_idx] = mem_before_mib
                    mem_peak_all[dt_idx, d_idx, s_idx] = max_mem_mib
                print_table_row(dtype_str, d_model, seq_len, lats, mem_before_mib, max_mem_mib)

    border = "+-------+---------+---------+----------+----------+----------+-----------------+-----------------+--------+"
    print(border)

    # Save
    save_path = output_dir / "benchmark_results.npz"
    np.savez_compressed(
        save_path,
        latencies=lats_all,
        mem_before=mem_before_all,
        mem_peak=mem_peak_all,
        dtypes=np.array([str(dt) for dt in cfg.dtypes]),
        d_models=np.array(cfg.d_models),
        seq_lens=np.array(cfg.seq_lens),
    )

    print(f"\n[+] Benchmark completed. All results saved to: {save_path}")


if __name__ == "__main__":
    main()
