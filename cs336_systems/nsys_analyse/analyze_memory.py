from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from cs336_systems.nsys_analyse.nsys_io import (
    find_sqlite_files,
    load_memory_events,
    load_nvtx_events,
    read_experiment_metadata,
)


def plot_memory_timeline(df_mem: pd.DataFrame, output_png: Path) -> None:
    """Reconstruct cumulative memory usage and save timeline plot."""
    if df_mem.empty:
        return

    # memoryOperationType: 0=Alloc, 1=Free
    df_mem = df_mem.copy()
    df_mem["delta"] = df_mem.apply(lambda r: r["bytes"] if r["memoryOperationType"] == 0 else -r["bytes"], axis=1)
    df_mem["cum_mem_MB"] = df_mem["delta"].cumsum() / (1024 * 1024)
    df_mem["time_s"] = (df_mem["start"] - df_mem["start"].iloc[0]) / 1e9

    plt.figure(figsize=(12, 5))
    plt.plot(df_mem["time_s"], df_mem["cum_mem_MB"], color="tab:blue", linewidth=1.5)
    plt.title("GPU Memory Usage Over Time (Reconstructed from SQLite)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Allocated Memory (MB)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"  -> Memory timeline plot saved to: {output_png}")


def analyze_transformer_block(
    df_mem: pd.DataFrame,
    df_nvtx: pd.DataFrame,
    preferred_block: str = "TransformerLM.transformers.10",
) -> None:
    """Analyze activation memory footprint and top memory-allocating operators inside a block."""
    # do NOT use block 0 since it's warmup block
    target_range = df_nvtx[df_nvtx["name"].str.contains(preferred_block, regex=False, na=False)]
    if target_range.empty:
        target_range = df_nvtx[df_nvtx["name"].str.contains(r"transformers\.[1-9]\d*\b", regex=True, na=False)]

    if target_range.empty:
        print("  [Warning] No suitable TransformerBlock found in NVTX ranges.")
        return

    block_info = target_range.iloc[0]
    b_start, b_end, b_name = block_info["start"], block_info["end"], block_info["name"]
    duration_ms = (b_end - b_start) / 1e6

    print(f"\n--- Transformer Block Analysis: {b_name} (Duration: {duration_ms:.2f} ms) ---")

    # collect alloc/free in block
    block_mem = df_mem[(df_mem["start"] >= b_start) & (df_mem["start"] <= b_end)].copy()
    alloc_bytes = block_mem[block_mem["memoryOperationType"] == 0]["bytes"].sum()
    free_bytes = block_mem[block_mem["memoryOperationType"] == 1]["bytes"].sum()
    saved_residuals = alloc_bytes - free_bytes

    print(f"  - Total Allocated: {alloc_bytes / (1024**2):.2f} MB")
    print(f"  - Temporarily Freed: {free_bytes / (1024**2):.2f} MB")
    print(f"  ★ Net Saved for Backward (Residuals): {saved_residuals / (1024**2):.2f} MB ({saved_residuals:,} bytes)")

    # bind to nvtx
    alloc_events = block_mem[block_mem["memoryOperationType"] == 0].copy()
    if alloc_events.empty:
        return

    sub_nvtx = df_nvtx[(df_nvtx["start"] >= b_start) & (df_nvtx["end"] <= b_end)].copy()
    sub_nvtx["duration"] = sub_nvtx["end"] - sub_nvtx["start"]
    sub_nvtx = sub_nvtx[sub_nvtx["name"] != b_name].sort_values(by="duration")

    def find_op_name(alloc_time: int) -> str:
        matched = sub_nvtx[(sub_nvtx["start"] <= alloc_time) & (sub_nvtx["end"] >= alloc_time)]
        return matched.iloc[0]["name"] if not matched.empty else "other/unlabeled"

    alloc_events["op_name"] = alloc_events["start"].apply(find_op_name)

    op_summary = alloc_events.groupby("op_name")["bytes"].sum().reset_index()
    op_summary["Size (MB)"] = op_summary["bytes"] / (1024**2)
    op_summary["Percentage"] = (op_summary["bytes"] / alloc_bytes) * 100
    op_summary = op_summary.sort_values(by="bytes", ascending=False).reset_index(drop=True)

    print("  ★ Top-5 Allocating Operators during Forward:")
    print(op_summary.head(5)[["op_name", "Size (MB)", "Percentage"]].to_string(index=False))


def analyze_backward_pass(df_mem: pd.DataFrame, df_nvtx: pd.DataFrame, model_cfg: dict[str, Any]) -> None:
    """Analyze scratchpad churn and net retained memory (gradients) during backward pass,

    deducting theoretical embedding gradients to isolate per-block footprint.
    """
    bwd_range = df_nvtx[df_nvtx["name"].str.lower().str.contains("backward", na=False)]
    if bwd_range.empty:
        print("\n  [Info] No explicit backward pass found in NVTX.")
        return

    bwd_s = bwd_range.iloc[0]["start"]
    bwd_e = bwd_range.iloc[0]["end"]
    bwd_mem = df_mem[(df_mem["start"] >= bwd_s) & (df_mem["start"] <= bwd_e)].copy()

    bwd_alloc = bwd_mem[bwd_mem["memoryOperationType"] == 0]["bytes"].sum()
    bwd_free = bwd_mem[bwd_mem["memoryOperationType"] == 1]["bytes"].sum()

    print("\n--- Backward Pass Analysis ---")
    print(f"  - Peak Memory Churn (Scratchpad Alloc): {bwd_alloc / (1024**2):.2f} MB (Frequent temporary buffers)")
    print(f"  - Total Activations Freed: {bwd_free / (1024**2):.2f} MB (Roughly matches forward activations)")

    # Determine baseline memory before forward pass to isolate newly retained gradients
    fwd_range = df_nvtx[df_nvtx["name"].str.lower().str.contains("forward", na=False)]
    if not fwd_range.empty:
        fwd_s = fwd_range.iloc[0]["start"]
        initial_model_mem = df_mem[df_mem["start"] < fwd_s]["delta"].sum()
    else:
        initial_model_mem = df_mem[df_mem["start"] < bwd_s]["delta"].sum() - (bwd_free - bwd_alloc)

    end_bwd_mem = df_mem[df_mem["start"] <= bwd_e]["delta"].sum()
    retained_grads = max(0, int(end_bwd_mem - initial_model_mem))

    print(f"  ★ Net Retained Gradients (Whole Model): {retained_grads / (1024**2):.2f} MB")

    # Extract model hyperparameters to deduct non-block parameters (Embedding & LM Head)
    layers = model_cfg.get("num_layers", 12)
    d_model = model_cfg.get("d_model", 768)
    vocab_size = model_cfg.get("vocab_size", 10000)

    # Gradient size for Token Embedding + LM Head in FP32 (4 bytes/param)
    # Factor of 2 accounts for both input embedding and output classification projection
    non_block_grad_bytes = 2 * (vocab_size * d_model * 4)

    # Deduct embedding/head gradients from whole-model gradients
    pure_blocks_grad_bytes = max(0, retained_grads - non_block_grad_bytes)
    estimated_block_grad_mb = (pure_blocks_grad_bytes / layers) / (1024**2)

    print(f"  - Non-Block Gradients Deducted (Embed + Head): {non_block_grad_bytes / (1024**2):.2f} MB")
    print(f"  ★ Estimated Pure Grad per Block ({layers} layers): {estimated_block_grad_mb:.2f} MB (Measured, Embed-deducted)")


def analyze_single_sqlite(sqlite_path: Path, save_plot: bool = True) -> None:
    print("\n=======================================================")
    print(f"Analyzing Memory Profile: {sqlite_path}")
    print("=======================================================")

    df_mem = load_memory_events(sqlite_path)
    df_nvtx = load_nvtx_events(sqlite_path)

    if df_mem.empty:
        print("[Warning] No GPU memory events found.")
        return

    model_cfg, _ = read_experiment_metadata(sqlite_path)

    # plot and save to the same dir with profile.sqlite
    if save_plot:
        plot_path = sqlite_path.parent / "reconstructed_memory_timeline.png"
        plot_memory_timeline(df_mem, plot_path)

    analyze_transformer_block(df_mem, df_nvtx)

    analyze_backward_pass(df_mem, df_nvtx, model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GPU Memory from Nsys Profile SQLite.")
    parser.add_argument(
        "target",
        type=Path,
        help="Path to profile.sqlite or directory containing profile.sqlite files.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable generating memory timeline plots.",
    )
    args = parser.parse_args()

    files = find_sqlite_files(args.target)
    if not files:
        print(f"No profile.sqlite found in: {args.target}")
        return

    for sqlite_file in files:
        analyze_single_sqlite(sqlite_file, save_plot=not args.no_plot)


if __name__ == "__main__":
    main()
