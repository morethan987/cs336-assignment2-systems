from __future__ import annotations

import sys
from pathlib import Path

from cs336_systems.nsys_analyse.nsys_io import (
    find_sqlite_files,
    is_gemm_kernel,
    read_experiment_metadata,
    run_nsys_report,
)


def extract_att_metrics(sqlite_path: Path) -> dict[str, float | int] | None:
    """Extract execution times for QK^T, Softmax, and AV phases from the nvtx_kern_sum report."""
    rows = run_nsys_report("nvtx_kern_sum", sqlite_path)
    if not rows:
        return None

    score_gemm_ms = 0.0
    final_matmul_ms = 0.0
    softmax_ms = 0.0
    softmax_calls = 0

    for row in rows:
        nvtx_range = row.get("NVTX Range", "").strip()
        k_name = row.get("Kernel Name", "").strip()

        try:
            total_ms = float(row.get("Total Time (ms)", 0.0))
            instances = int(row.get("Kern Inst", row.get("Instances", 0)))
        except ValueError:
            continue

        if not k_name or total_ms <= 0:
            continue

        # 1. Attention Scores: QK^T (filter for GEMM/MatMul kernels)
        if "computing attention scores" in nvtx_range:
            if is_gemm_kernel(k_name):
                score_gemm_ms += total_ms

        # 2. Softmax operation
        elif "computing softmax" in nvtx_range:
            softmax_ms += total_ms
            softmax_calls += instances

        # 3. Final MatMul: Softmax * V
        elif "final matmul" in nvtx_range:
            final_matmul_ms += total_ms

    return {
        "score_gemm_ms": score_gemm_ms,
        "final_matmul_ms": final_matmul_ms,
        "total_matmul_ms": score_gemm_ms + final_matmul_ms,
        "softmax_ms": softmax_ms,
        "softmax_calls": softmax_calls,
    }


def compute_attention_flops(
    batch_size: int,
    num_heads: int,
    ctx_len: int,
    d_k: int,
    num_layers: int,
    steps: int,
) -> tuple[float, float]:
    """
    Compute theoretical forward FLOPs for the Attention stage:
    1. QK^T: 2 * B * H * N^2 * d_k
    2. AV:   2 * B * H * N^2 * d_k
       -> Total MatMul = 4 * B * H * N^2 * d_k
    3. Total Softmax = 2.5 * B * H * N^2
    """
    total_calls = num_layers * steps
    matmul_flops = 4.0 * batch_size * num_heads * (ctx_len**2) * d_k * total_calls
    softmax_flops = 2.5 * batch_size * num_heads * (ctx_len**2) * total_calls
    return matmul_flops, softmax_flops


def analyze_directory(root_path: Path):
    sqlite_files = find_sqlite_files(root_path)
    if not sqlite_files:
        print(f"[!] No profile.sqlite files found under {root_path}!")
        return

    records = []

    for path in sqlite_files:
        model_cfg, bench_cfg = read_experiment_metadata(path)
        if not model_cfg or not bench_cfg:
            print(f"[!] Skipping {path}: missing or invalid args.json")
            continue

        stats = extract_att_metrics(path)
        if not stats or stats["softmax_ms"] <= 0:
            continue

        num_layers = int(model_cfg["num_layers"])
        num_heads = int(model_cfg["num_heads"])
        d_model = int(model_cfg["d_model"])
        d_k = d_model // num_heads
        ctx_len = int(model_cfg["context_length"])
        batch_size = int(bench_cfg["batch_size"])
        steps = int(bench_cfg["steps"])

        tag = f"{d_model}d_ctx{ctx_len}"

        # Compute FLOPs and throughput
        matmul_flops, softmax_flops = compute_attention_flops(
            batch_size=batch_size,
            num_heads=num_heads,
            ctx_len=ctx_len,
            d_k=d_k,
            num_layers=num_layers,
            steps=steps,
        )

        total_matmul_ms = stats["total_matmul_ms"]
        softmax_ms = stats["softmax_ms"]

        runtime_ratio = total_matmul_ms / softmax_ms if softmax_ms > 0 else 0
        flops_ratio = matmul_flops / softmax_flops if softmax_flops > 0 else 0

        matmul_tflops = (matmul_flops / (total_matmul_ms * 1e-3)) / 1e12 if total_matmul_ms > 0 else 0
        softmax_tflops = (softmax_flops / (softmax_ms * 1e-3)) / 1e12 if softmax_ms > 0 else 0
        tflops_ratio = matmul_tflops / softmax_tflops if softmax_tflops > 0 else 0

        records.append(
            {
                "tag": tag,
                "matmul_ms": total_matmul_ms,
                "softmax_ms": softmax_ms,
                "runtime_ratio": runtime_ratio,
                "flops_ratio": flops_ratio,
                "matmul_tflops": matmul_tflops,
                "softmax_tflops": softmax_tflops,
                "tflops_ratio": tflops_ratio,
            }
        )

    if not records:
        print("[!] Found profile.sqlite, but no valid attention NVTX records were detected.")
        return

    # 1. Print baseline execution time and FLOPs comparison table
    print("=" * 110)
    print(f"{'Configuration':<28} | {'MatMul Time':<12} | {'Softmax Time':<12} | {'Time Ratio':<15} | {'FLOPs Ratio':<15}")
    print("=" * 110)
    for r in records:
        print(f"{r['tag']:<28} | {r['matmul_ms']:>8.2f} ms   | {r['softmax_ms']:>8.2f} ms   | {r['runtime_ratio']:>8.2f} x         | {r['flops_ratio']:>8.2f} x")
    print("=" * 110)

    # 2. Print compute throughput comparison table
    print("\n[*] In-depth Analysis (including TFLOP/s hardware throughput):")
    print("-" * 110)
    print(f"{'Configuration':<28} | {'MatMul Throughput':<20} | {'Softmax Throughput':<20} | {'Throughput Ratio':<15}")
    print("-" * 110)
    for r in records:
        print(f"{r['tag']:<28} | {r['matmul_tflops']:>10.2f} TFLOP/s    | {r['softmax_tflops']:>10.4f} TFLOP/s    | {r['tflops_ratio']:>8.1f} x")
    print("-" * 110 + "\n")


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    analyze_directory(target)


if __name__ == "__main__":
    main()
