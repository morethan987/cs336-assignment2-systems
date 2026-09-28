import argparse
import gc
import os
import timeit
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from cs336_basics.layers import scaled_dot_product_attention

# Configurations
BATCH_SIZE = 8
DEVICE = torch.device("cuda:0")
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
NUM_WARMUP = 10
NUM_STEPS = 100
BASE_OUTPUT_DIR = "pytorch_attention_res"


def _rand_qkv(d_model: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors requiring gradients on the target device."""
    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    return Q, K, V


def _run_single_exp_impl(d_model: int, seq_len: int, att_fn: Callable) -> tuple[np.ndarray, float, float]:
    """
    Execute the benchmark body. All tensors are scoped strictly as local variables.

    Returns:
        (raw_times, mem_before_bwd_mib, max_mem_mib)
    """
    fwd_times = np.zeros(NUM_STEPS, dtype=np.float64)
    bwd_times = np.zeros(NUM_STEPS, dtype=np.float64)
    mem_before_bwd_mib: float = 0.0

    Q, K, V = _rand_qkv(d_model, seq_len)
    grad_out = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE)

    # Warmup passes
    for _ in range(NUM_WARMUP):
        Q.grad = K.grad = V.grad = None
        out = att_fn(Q, K, V)
        out.backward(grad_out)
    torch.cuda.synchronize(DEVICE)

    # Reset peak stats after warmup so warmup passes do not taint measurements
    torch.cuda.reset_peak_memory_stats(DEVICE)

    # Timed iterations
    for i in range(NUM_STEPS):
        Q.grad = K.grad = V.grad = None

        # --- Forward Pass ---
        torch.cuda.synchronize(DEVICE)
        t0 = timeit.default_timer()
        out = att_fn(Q, K, V)
        torch.cuda.synchronize(DEVICE)
        t1 = timeit.default_timer()

        # Record allocated memory right before the backward pass begins
        if i == 0:
            mem_before_bwd_mib = float(torch.cuda.memory_allocated(DEVICE) / (1024**2))

        # --- Backward Pass ---
        out.backward(grad_out)
        torch.cuda.synchronize(DEVICE)
        t2 = timeit.default_timer()

        fwd_times[i] = t1 - t0
        bwd_times[i] = t2 - t1

    max_mem_mib = float(torch.cuda.max_memory_allocated(DEVICE) / (1024**2))
    raw_times = np.stack([fwd_times, bwd_times], axis=0)
    return raw_times, mem_before_bwd_mib, max_mem_mib


def run_single_exp(d_model: int, seq_len: int, is_compiled: bool) -> tuple[np.ndarray | None, float | None, float | None]:
    """
    Wrapper around `_run_single_exp_impl` to trap Out-Of-Memory exceptions.

    Returns:
        (raw_times, mem_before_bwd_mib, max_mem_mib):
            All elements are None if an OOM occurs.
    """
    try:
        if is_compiled:
            # reset Dynamo cache and recompile counter
            torch.compiler.reset()
            att_fn = torch.compile(scaled_dot_product_attention)
        else:
            att_fn = scaled_dot_product_attention
        return _run_single_exp_impl(d_model, seq_len, att_fn)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            # Frames and local variables are unwound automatically upon return;
            # gc and empty_cache reclaim the allocator pool cleanly.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention Benchmark with optional torch.compile")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for attention")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available on this system."
    # Ensure CUDA context is explicitly initialized and set to target device
    torch.cuda.init()
    torch.cuda.set_device(DEVICE)
    torch.set_float32_matmul_precision("high")

    mode_str = "compile" if args.compile else "eager"
    timestamp = datetime.now(ZoneInfo("Asia/Chongqing")).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"{mode_str}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 106)
    print(f" PyTorch Attention Benchmark (Batch Size={BATCH_SIZE}, Device={DEVICE})")
    print("=" * 106)
    print_table_header()

    # Pre-allocate result grids (filled with NaN for OOM configurations)
    raw_all = np.full((len(D_MODELS), len(SEQ_LENS), 2, NUM_STEPS), np.nan, dtype=np.float64)
    mem_before_all = np.full((len(D_MODELS), len(SEQ_LENS)), np.nan, dtype=np.float64)
    mem_peak_all = np.full((len(D_MODELS), len(SEQ_LENS)), np.nan, dtype=np.float64)

    for d_idx, d_model in enumerate(D_MODELS):
        for s_idx, seq_len in enumerate(SEQ_LENS):
            raw_times, mem_before_mib, max_mem_mib = run_single_exp(d_model, seq_len, args.compile)
            print_table_row(d_model, seq_len, raw_times, mem_before_mib, max_mem_mib)

            if raw_times is not None:
                raw_all[d_idx, s_idx] = raw_times
                mem_before_all[d_idx, s_idx] = mem_before_mib
                mem_peak_all[d_idx, s_idx] = max_mem_mib

    border = "+---------+---------+--------------------+--------------------+-----------------+-----------------+--------+"
    print(border)

    # Persist all metrics into a compressed archive
    save_path = os.path.join(output_dir, "benchmark_results.npz")
    np.savez_compressed(
        save_path,
        raw_times=raw_all,
        mem_before=mem_before_all,
        mem_peak=mem_peak_all,
        d_models=np.array(D_MODELS),
        seq_lens=np.array(SEQ_LENS),
    )

    print(f"\n[+] Benchmark completed. All results saved to: {save_path}")


if __name__ == "__main__":
    main()
