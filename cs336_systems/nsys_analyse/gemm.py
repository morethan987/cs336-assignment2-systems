from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from cs336_systems.nsys_analyse.nsys_io import (
    find_sqlite_files,
    is_gemm_kernel,
    run_nsys_report,
)


def extract_forward_nvtx_gemm(sqlite_path: Path):
    """Extract only forward-phase kernels based on NVTX ranges."""
    rows = run_nsys_report("nvtx_kern_sum", sqlite_path)
    if not rows:
        return None

    total_ms = 0.0
    gemm_ms = 0.0
    other_kernels = defaultdict(float)

    for row in rows:
        nvtx_range = row.get("NVTX Range", "").strip()
        k_name = row.get("Kernel Name", "").strip()

        if "forward" not in nvtx_range.lower():
            continue

        try:
            t_ms = float(row.get("Total Time (ms)", 0.0))
        except ValueError:
            continue

        if not k_name or t_ms <= 0:
            continue

        total_ms += t_ms
        if is_gemm_kernel(k_name):
            gemm_ms += t_ms
        else:
            other_kernels[k_name] += t_ms

    sorted_others = sorted(other_kernels.items(), key=lambda x: x[1], reverse=True)
    return total_ms, gemm_ms, sorted_others


def extract_train_step_full_gemm(sqlite_path: Path):
    """Extract all GPU kernels across the entire training step (including forward, backward, and optimizer)."""
    rows = run_nsys_report("cuda_gpu_kern_sum", sqlite_path)
    if not rows:
        return None

    total_ms = 0.0
    gemm_ms = 0.0
    other_kernels = defaultdict(float)

    for row in rows:
        k_name = row.get("Name", "").strip()
        try:
            t_ms = float(row.get("Total Time (ms)", 0.0))
        except ValueError:
            continue

        if not k_name or t_ms <= 0:
            continue

        total_ms += t_ms
        if is_gemm_kernel(k_name):
            gemm_ms += t_ms
        else:
            other_kernels[k_name] += t_ms

    sorted_others = sorted(other_kernels.items(), key=lambda x: x[1], reverse=True)
    return total_ms, gemm_ms, sorted_others


def print_phase_block(title: str, total_ms: float, gemm_ms: float, top_others: list[tuple[str, float]]):
    gemm_ratio = (gemm_ms / total_ms * 100) if total_ms > 0 else 0
    non_gemm_ms = total_ms - gemm_ms
    non_gemm_ratio = 100.0 - gemm_ratio

    print(f"[{title}]")
    print(f"  -> Total Time: {total_ms:.2f} ms")
    print(f"  -> GEMM Time: {gemm_ms:.2f} ms ({gemm_ratio:.1f}%)")
    print(f"  -> Non-GEMM Time: {non_gemm_ms:.2f} ms ({non_gemm_ratio:.1f}%)")
    print("  -> Top 3 Non-GEMM Kernels:")
    for k_name, k_ms in top_others[:3]:
        pct = (k_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"     * {pct:>5.1f}% | {k_ms:>6.2f} ms | {k_name[:65]}")


def analyze_single_sqlite(sqlite_path: Path):
    """Analyze a single experiment: output both Forward NVTX metrics and full-step metrics."""
    fwd_stats = extract_forward_nvtx_gemm(sqlite_path)
    train_stats = extract_train_step_full_gemm(sqlite_path)

    if not fwd_stats and not train_stats:
        return

    exp_name = sqlite_path.parent.name
    print("=" * 100)
    print(f"[Experiment Name]: {exp_name}")
    print("=" * 100)

    if fwd_stats and fwd_stats[0] > 0:
        fwd_total, fwd_gemm, fwd_others = fwd_stats
        print_phase_block("FORWARD PASS (NVTX Filtered)", fwd_total, fwd_gemm, fwd_others)

    if train_stats and train_stats[0] > 0:
        train_total, train_gemm, train_others = train_stats
        print()
        print_phase_block("COMPLETE RUN / TRAIN STEP (All GPU Kernels)", train_total, train_gemm, train_others)
    print("\n")


def analyze_paired_legacy(root_dir: Path) -> bool:
    """Compatibility check for paired legacy experiment directories (*_forward_only and *_train_step)."""
    fwd_dirs = sorted(root_dir.glob("*_forward_only"))
    if not fwd_dirs:
        return False

    has_paired = False
    for fwd_dir in fwd_dirs:
        base_name = fwd_dir.name.replace("_forward_only", "")
        train_dir = root_dir / f"{base_name}_train_step"

        fwd_sqlites = find_sqlite_files(fwd_dir)
        train_sqlites = find_sqlite_files(train_dir)

        if not fwd_sqlites or not train_sqlites:
            continue

        has_paired = True
        fwd_stats = extract_forward_nvtx_gemm(fwd_sqlites[-1])
        train_stats = extract_train_step_full_gemm(train_sqlites[-1])

        if not fwd_stats or not train_stats:
            continue

        fwd_total, fwd_gemm, fwd_others = fwd_stats
        train_total, train_gemm, train_others = train_stats

        print("=" * 100)
        print(f"[Paired Model Config]: {base_name}")
        print("=" * 100)
        print_phase_block("FORWARD ONLY (NVTX Filtered)", fwd_total, fwd_gemm, fwd_others)
        print()
        mult = (train_total / fwd_total) if fwd_total > 0 else 0
        print_phase_block(f"COMPLETE TRAIN STEP (~{mult:.2f}x Forward)", train_total, train_gemm, train_others)
        print("\n")

    return has_paired


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    # 1. Try to recognize paired legacy directory structures first
    if root_dir.is_dir() and analyze_paired_legacy(root_dir):
        return

    # 2. General single-trace / recursive directory traversal
    sqlite_files = find_sqlite_files(root_dir)
    if not sqlite_files:
        print(f"[!] No profile.sqlite files found under {root_dir}!")
        return

    for sql_file in sqlite_files:
        analyze_single_sqlite(sql_file)


if __name__ == "__main__":
    main()
