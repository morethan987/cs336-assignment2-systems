from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from cs336_systems.nsys_analyse.nsys_io import (
    find_sqlite_files,
    run_nsys_report,
)


def classify_fwd_bwd_kernels(sqlite_path: Path):
    """
    Classify CUDA kernels into Forward and Backward phases based on NVTX ranges.
    """
    rows = run_nsys_report("nvtx_kern_sum", sqlite_path)
    if not rows:
        return None, None

    forward_kernels = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    backward_kernels = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

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

        # Strictly categorize based on NVTX stage markers
        # Excludes optimizer/data preparation stages automatically
        if "forward" in nvtx_range:
            forward_kernels[k_name]["count"] += count
            forward_kernels[k_name]["total_ms"] += total_ms
        elif "backward" in nvtx_range:
            backward_kernels[k_name]["count"] += count
            backward_kernels[k_name]["total_ms"] += total_ms

    sorted_fwd = sorted(forward_kernels.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    sorted_bwd = sorted(backward_kernels.items(), key=lambda x: x[1]["total_ms"], reverse=True)

    return sorted_fwd, sorted_bwd


def print_kernel_table(data_list: list[tuple[str, dict]], title: str, top_k: int = 10):
    total_phase_ms = sum(d["total_ms"] for _, d in data_list)
    print(f"\n--- {title} (Total: {total_phase_ms:.2f} ms) ---")
    print(f"{'Rank':<5} {'Time (ms)':<12} {'Ratio':<8} {'Calls':<8} {'Avg (ms)':<10} Kernel Name")
    print("-" * 100)

    display_items = data_list[:top_k] if top_k else data_list
    for rank, (name, d) in enumerate(display_items, 1):
        ratio = (d["total_ms"] / total_phase_ms * 100) if total_phase_ms > 0 else 0
        avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
        short_name = name[:60] + "..." if len(name) > 60 else name
        print(f"{rank:<5} {d['total_ms']:<12.4f} {ratio:>5.1f}%  {d['count']:<8} {avg_ms:<10.4f} {short_name}")

    if top_k and len(data_list) > top_k:
        print(f"... Omitted {len(data_list) - top_k} remaining minor kernels (see clean_kernels.csv for full details)")


def export_clean_csv(output_path: Path, fwd_list: list[tuple[str, dict]], bwd_list: list[tuple[str, dict]]):
    """Export aggregated kernel metrics for both phases to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Kernel Name", "Total Time (ms)", "Calls", "Avg Time (ms)"])
        for phase, items in (("Forward", fwd_list), ("Backward", bwd_list)):
            for name, d in items:
                avg_ms = d["total_ms"] / d["count"] if d["count"] else 0
                writer.writerow([phase, name, f"{d['total_ms']:.4f}", d["count"], f"{avg_ms:.4f}"])


def main():
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    sqlite_files = find_sqlite_files(root_dir)
    if not sqlite_files:
        print(f"[!] No profile.sqlite files found under {root_dir}!")
        return

    print(f"[*] Found {len(sqlite_files)} profile.sqlite file(s). Starting classification...\n")

    for path in sqlite_files:
        exp_dir = path.parent
        print("\n" + "=" * 105)
        print(f"  Experiment Analysis: [{exp_dir.name}]")
        print("=" * 105)

        sorted_fwd, sorted_bwd = classify_fwd_bwd_kernels(path)
        if not sorted_fwd and not sorted_bwd:
            print("[!] No forward/backward kernels matched.")
            continue

        print_kernel_table(sorted_fwd, "[FORWARD PASS] Kernel Execution Time Ranking", top_k=top_k)
        print_kernel_table(sorted_bwd, "[BACKWARD PASS] Kernel Execution Time Ranking", top_k=top_k)

        clean_csv = exp_dir / "clean_kernels.csv"
        export_clean_csv(clean_csv, sorted_fwd, sorted_bwd)
        print(f"\n[OK] Cleaned full dataset saved to: {clean_csv}")


if __name__ == "__main__":
    main()
