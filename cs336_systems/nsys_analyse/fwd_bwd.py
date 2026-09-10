from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from cs336_systems.nsys_analyse.nsys_io import (
    find_sqlite_files,
    get_trace_mode,
    is_gemm_kernel,
    run_nsys_report,
)


def classify_phase_kernels(sqlite_path: Path, mode: str):
    """
    Classify CUDA kernels into Forward and Backward phases based on NVTX ranges.
    Respects experiment mode: skips backward classification entirely for 'infer'.
    """
    rows = run_nsys_report("nvtx_kern_sum", sqlite_path)
    if not rows:
        return None, None

    forward_kernels = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "is_gemm": False})
    backward_kernels = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "is_gemm": False})

    for row in rows:
        nvtx_range = row.get("NVTX Range", "").strip().lower()
        k_name = row.get("Kernel Name", "").strip()

        try:
            total_ms = float(row.get("Total Time (ms)", 0.0))
            count = int(row.get("Kern Inst", row.get("Instances", 0)))
        except ValueError:
            continue

        if not k_name or total_ms <= 0:
            continue

        is_gemm = is_gemm_kernel(k_name)

        if "forward" in nvtx_range:
            forward_kernels[k_name]["count"] += count
            forward_kernels[k_name]["total_ms"] += total_ms
            forward_kernels[k_name]["is_gemm"] = is_gemm
        elif mode != "infer" and "backward" in nvtx_range:
            backward_kernels[k_name]["count"] += count
            backward_kernels[k_name]["total_ms"] += total_ms
            backward_kernels[k_name]["is_gemm"] = is_gemm

    sorted_fwd = sorted(forward_kernels.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    sorted_bwd = sorted(backward_kernels.items(), key=lambda x: x[1]["total_ms"], reverse=True) if mode != "infer" else []

    return sorted_fwd, sorted_bwd


def print_kernel_table(data_list: list[tuple[str, dict]], title: str, top_k: int = 10):
    if not data_list:
        print(f"\n[!] {title}: No kernels recorded.")
        return

    total_phase_ms = sum(d["total_ms"] for _, d in data_list)
    gemm_ms = sum(d["total_ms"] for _, d in data_list if d["is_gemm"])
    gemm_ratio = (gemm_ms / total_phase_ms * 100) if total_phase_ms > 0 else 0

    print(f"\n--- {title} ---")
    print(f"  * Total Time: {total_phase_ms:.2f} ms | GEMM Time: {gemm_ms:.2f} ms ({gemm_ratio:.1f}%)")
    print(f"{'Rank':<5} {'Type':<8} {'Time (ms)':<12} {'Ratio':<8} {'Calls':<8} {'Avg (ms)':<10} Kernel Name")
    print("-" * 110)

    display_items = data_list[:top_k] if top_k else data_list
    for rank, (name, d) in enumerate(display_items, 1):
        ratio = (d["total_ms"] / total_phase_ms * 100) if total_phase_ms > 0 else 0
        avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
        k_type = "GEMM" if d["is_gemm"] else "OTHER"
        short_name = name[:60] + "..." if len(name) > 60 else name
        print(f"{rank:<5} {k_type:<8} {d['total_ms']:<12.4f} {ratio:>5.1f}%  {d['count']:<8} {avg_ms:<10.4f} {short_name}")

    if top_k and len(data_list) > top_k:
        print(f"... Omitted {len(data_list) - top_k} remaining minor kernels (full list exported to clean_kernels.csv)")


def export_clean_csv(
    output_path: Path,
    fwd_list: list[tuple[str, dict]],
    bwd_list: list[tuple[str, dict]],
    mode: str,
):
    """Export aggregated kernel breakdown with kernel type categorization."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Type", "Kernel Name", "Total Time (ms)", "Calls", "Avg Time (ms)"])

        phases = [("Forward", fwd_list)]
        if mode != "infer" and bwd_list:
            phases.append(("Backward", bwd_list))

        for phase, items in phases:
            for name, d in items:
                avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
                k_type = "GEMM" if d["is_gemm"] else "Other"
                writer.writerow([phase, k_type, name, f"{d['total_ms']:.4f}", d["count"], f"{avg_ms:.4f}"])


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    sqlite_files = find_sqlite_files(root_dir)
    if not sqlite_files:
        print(f"[!] No profile.sqlite files found under {root_dir}!")
        return

    print(f"[*] Found {len(sqlite_files)} profile.sqlite file(s). Starting phase kernel ranking...\n")

    for path in sqlite_files:
        exp_dir = path.parent
        mode = get_trace_mode(path)

        print("\n" + "=" * 110)
        print(f"  Experiment: [{exp_dir.name}] | Mode: [{mode.upper()}]")
        print("=" * 110)

        sorted_fwd, sorted_bwd = classify_phase_kernels(path, mode=mode)
        if not sorted_fwd and not sorted_bwd:
            print("[!] No forward/backward kernels matched by NVTX markers.")
            continue

        # 1. Forward Pass Ranking
        print_kernel_table(sorted_fwd, "[FORWARD PASS] Kernel Execution Ranking", top_k=top_k)

        # 2. Backward Pass Ranking (only for train/unknown)
        if mode != "infer":
            print_kernel_table(sorted_bwd, "[BACKWARD PASS] Kernel Execution Ranking", top_k=top_k)

            fwd_total = sum(d["total_ms"] for _, d in sorted_fwd) if sorted_fwd else 0
            bwd_total = sum(d["total_ms"] for _, d in sorted_bwd) if sorted_bwd else 0
            if fwd_total > 0 and bwd_total > 0:
                print(f"\n[PHASE RATIO] Backward Time / Forward Time: ~{bwd_total / fwd_total:.2f}x (Ideal: ~2.0x)")

        # 3. Export CSV without noise
        clean_csv = exp_dir / "clean_kernels.csv"
        export_clean_csv(clean_csv, sorted_fwd, sorted_bwd, mode=mode)
        print(f"\n[OK] Cleaned full dataset saved to: {clean_csv}")


if __name__ == "__main__":
    main()
