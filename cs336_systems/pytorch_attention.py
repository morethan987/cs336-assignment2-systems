import os
import timeit
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from cs336_basics.layers import scaled_dot_product_attention

# configs
BATCH_SIZE = 8
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
NUM_WARMUP = 10
NUM_STEPS = 100
BASE_OUTPUT_DIR = "pytorch_attention_res"
TIMESTAMP = datetime.now(ZoneInfo("Asia/Chongqing")).strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, TIMESTAMP)


def _rand_qkv(d_model: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    return Q, K, V


def run_single_exp(d_model: int, seq_len: int) -> tuple[np.ndarray | None, float | None]:
    """
    Return:
        (raw_times, mem_mib):
            - raw_times: numpy ndarray with shape of (2, 100), row 0 for forward, row 1 for backward. None for OOM
            - mem_mib: memory before backward
    """
    fwd_times = np.zeros(NUM_STEPS, dtype=np.float64)
    bwd_times = np.zeros(NUM_STEPS, dtype=np.float64)
    mem_before_bwd_mib = None

    try:
        Q, K, V = _rand_qkv(d_model, seq_len)
        grad_out = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE)

        # Warmup
        for _ in range(NUM_WARMUP):
            Q.grad = K.grad = V.grad = None
            out = scaled_dot_product_attention(Q, K, V)
            out.backward(grad_out)
        torch.cuda.synchronize()

        # run 100 times
        for i in range(NUM_STEPS):
            Q.grad = K.grad = V.grad = None

            # --- Forward ---
            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            out = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            t1 = timeit.default_timer()

            # record memory
            if i == 0:
                mem_before_bwd_mib = torch.cuda.memory_allocated() / (1024**2)

            # --- Backward ---
            out.backward(grad_out)
            torch.cuda.synchronize()
            t2 = timeit.default_timer()

            # record time
            fwd_times[i] = t1 - t0
            bwd_times[i] = t2 - t1

        # stack
        raw_times = np.stack([fwd_times, bwd_times], axis=0)
        return raw_times, mem_before_bwd_mib

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            # release if OOM
            del Q, K, V, grad_out
            if "out" in locals():
                del out
            torch.cuda.empty_cache()
            return None, None
        raise


def print_table_header():
    border = "+---------+---------+--------------------+--------------------+-----------------+--------+"
    header = "| d_model | seq_len |   Fwd (ms) ± Std   |   Bwd (ms) ± Std   | Mem Before(MiB) | Status |"
    print(border)
    print(header)
    print(border)


def print_table_row(
    d_model: int,
    seq_len: int,
    raw_times: np.ndarray | None,
    mem_mib: float | None,
):
    if raw_times is not None:
        fwd_ms = raw_times[0] * 1000.0
        bwd_ms = raw_times[1] * 1000.0
        fwd_str = f"{np.mean(fwd_ms):7.2f} ± {np.std(fwd_ms):5.2f}"
        bwd_str = f"{np.mean(bwd_ms):7.2f} ± {np.std(bwd_ms):5.2f}"
        mem_str = f"{mem_mib:13.2f}"
        status_str = "\033[92m  OK  \033[0m"
    else:
        fwd_str = "        N/A         "
        bwd_str = "        N/A         "
        mem_str = "          N/A"
        status_str = "\033[91m  OOM \033[0m"

    print(f"| {d_model:^7} | {seq_len:^7} | {fwd_str:^18} | {bwd_str:^18} | {mem_str:^15} | {status_str} |")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 88)
    print(f" PyTorch Attention Benchmark (Batch Size={BATCH_SIZE}, Device={DEVICE})")
    print("=" * 88)
    print_table_header()

    # (len(d_models), len(seq_lens), 2, 100)
    # NaN if OOM
    raw_all = np.full((len(D_MODELS), len(SEQ_LENS), 2, NUM_STEPS), np.nan, dtype=np.float64)

    for d_idx, d_model in enumerate(D_MODELS):
        for s_idx, seq_len in enumerate(SEQ_LENS):
            raw_times, mem_mib = run_single_exp(d_model, seq_len)
            print_table_row(d_model, seq_len, raw_times, mem_mib)

            if raw_times is not None:
                single_path = os.path.join(OUTPUT_DIR, f"raw_d{d_model}_s{seq_len}.npy")
                np.save(single_path, raw_times)

                raw_all[d_idx, s_idx] = raw_times

    border = "+---------+---------+--------------------+--------------------+-----------------+--------+"
    print(border)

    # save raw data
    grid_path = os.path.join(OUTPUT_DIR, "raw_all.npy")
    np.save(grid_path, raw_all)

    print(f"\n[+] raw data saved to: ./{OUTPUT_DIR}/")
    print("    - single exp: raw_d<d>_s<s>.npy (shape: [2, 100])")
    print("    - data file: raw_all.npy (shape: [4, 5, 2, 100])")


if __name__ == "__main__":
    main()
