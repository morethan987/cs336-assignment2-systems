from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from cs336_systems.nsys_analyse.nsys_io import (
    find_sqlite_files,
    get_trace_mode,
    is_gemm_kernel,
    read_experiment_metadata,
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


def extract_all_gpu_kernels(sqlite_path: Path):
    """Extract all GPU kernels across the entire profiling window via cuda_gpu_kern_sum."""
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
    """Analyze a single trace with mode-accurate headers and metrics."""
    _, bench_cfg = read_experiment_metadata(sqlite_path)
    mode = get_trace_mode(sqlite_path, bench_cfg)

    fwd_stats = extract_forward_nvtx_gemm(sqlite_path)
    all_stats = extract_all_gpu_kernels(sqlite_path)

    if not fwd_stats and not all_stats:
        return

    exp_name = sqlite_path.parent.name
    print("=" * 100)
    print(f"[Experiment]: {exp_name} | [Mode: {mode.upper()}]")
    print("=" * 100)

    # 1. Forward pass NVTX breakdown
    if fwd_stats and fwd_stats[0] > 0:
        fwd_total, fwd_gemm, fwd_others = fwd_stats
        print_phase_block("FORWARD PASS (NVTX Filtered)", fwd_total, fwd_gemm, fwd_others)
        print()

    # 2. Window-wide metrics
    if all_stats and all_stats[0] > 0:
        total, gemm, others = all_stats
        if mode == "infer":
            print_phase_block("TOTAL INFERENCE PASS (All GPU Kernels)", total, gemm, others)
        elif mode == "train":
            print_phase_block("COMPLETE TRAIN STEP (All GPU Kernels: Fwd + Bwd + Opt)", total, gemm, others)
            if fwd_stats and fwd_stats[0] > 0:
                print(f"\n  -> Full Train Step / Forward Pass Ratio: ~{total / fwd_stats[0]:.2f}x")
        else:
            print_phase_block("COMPLETE RUN (Unknown Mode - All GPU Kernels)", total, gemm, others)

    print("\n")


def analyze_paired_experiment(base_name: str, infer_sqlite: Path, train_sqlite: Path):
    """Analyze a paired (infer, train) benchmark."""
    infer_stats = extract_forward_nvtx_gemm(infer_sqlite)
    train_stats = extract_all_gpu_kernels(train_sqlite)

    if not infer_stats or not train_stats:
        return

    inf_total, inf_gemm, inf_others = infer_stats
    trn_total, trn_gemm, trn_others = train_stats

    print("=" * 100)
    print(f"[Paired Benchmark Group]: {base_name}")
    print("=" * 100)
    print_phase_block("INFERENCE / FORWARD PASS", inf_total, inf_gemm, inf_others)
    print()
    mult = (trn_total / inf_total) if inf_total > 0 else 0
    print_phase_block(f"COMPLETE TRAIN STEP (~{mult:.2f}x Forward)", trn_total, trn_gemm, trn_others)
    print("\n")


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    sqlite_files = find_sqlite_files(root_dir)

    if not sqlite_files:
        print(f"[!] No profile.sqlite files found under {root_dir}!")
        return

    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    unmatched_sqlites: list[Path] = []

    for sql_file in sqlite_files:
        mode = get_trace_mode(sql_file)
        dir_name = sql_file.parent.name

        if dir_name.endswith("_train"):
            base_name = dir_name[:-6]
            groups[base_name]["train"] = sql_file
        elif dir_name.endswith("_infer"):
            base_name = dir_name[:-6]
            groups[base_name]["infer"] = sql_file
        elif mode in ("train", "infer"):
            groups[dir_name][mode] = sql_file
        else:
            unmatched_sqlites.append(sql_file)

    for base_name, modes in list(groups.items()):
        if "train" in modes and "infer" in modes:
            analyze_paired_experiment(base_name, modes["infer"], modes["train"])
        else:
            for sql_file in modes.values():
                analyze_single_sqlite(sql_file)

    for sql_file in unmatched_sqlites:
        analyze_single_sqlite(sql_file)


if __name__ == "__main__":
    main()
