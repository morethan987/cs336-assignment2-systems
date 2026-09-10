from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

GEMM_KEYWORDS = (
    "gemm",
    "mma",
    "cublas",
    "cutlass",
    "sm80_xmma",
    "sm90_xmma",
    "wgrad",
    "dgrad",
)


def is_gemm_kernel(kernel_name: str) -> bool:
    """Select GEMM/MatMul according to kernel name"""
    name = kernel_name.lower()
    return any(kw in name for kw in GEMM_KEYWORDS)


def find_sqlite_files(target_path: Path) -> list[Path]:
    """Find all profile.sqlite files directly or recursively within the directory."""
    if target_path.is_file() and target_path.suffix == ".sqlite":
        return [target_path]
    if target_path.is_dir():
        return sorted(target_path.rglob("profile.sqlite"))
    return []


def read_experiment_metadata(sqlite_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Directly read args.json from the same directory as profile.sqlite.
    Returns: (model_cfg, bench_cfg). Fallback to ({}, {}) on failure.
    """
    args_file = sqlite_path.parent / "args.json"
    if not args_file.is_file():
        print(f"[Warning] Metadata file not found: {args_file}")
        return {}, {}

    try:
        with open(args_file, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("model_cfg", {}), meta.get("bench_cfg", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"[Warning] Failed to parse {args_file}: {e}")
        return {}, {}


def get_trace_mode(sqlite_path: Path, bench_cfg: dict[str, Any] | None = None) -> str:
    """
    Read and return normalized mode ('train', 'infer', or 'unknown').
    Can either use an existing bench_cfg dict or read args.json directly.
    """
    if bench_cfg is None:
        _, bench_cfg = read_experiment_metadata(sqlite_path)

    mode = bench_cfg.get("mode")
    if mode is not None:
        return str(mode).lower()
    return "unknown"


def run_nsys_report(report_name: str, sqlite_path: Path) -> list[dict[str, str]]:
    """Run `nsys stats -r <report_name>` and parse output directly as a CSV dictionary list."""
    cmd = [
        "nsys",
        "stats",
        "-r",
        report_name,
        "--format",
        "csv",
        "--timeunit",
        "ms",
        str(sqlite_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Error] Failed to execute nsys stats on: {sqlite_path}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"stderr:\n{e.stderr}")
        return []

    lines = res.stdout.splitlines()
    # Find the header row index (skip optional banner/metadata printed by nsys)
    header_idx = next(
        (i for i, line in enumerate(lines) if any(kw in line for kw in ("NVTX Range", "Total Time (ms)", "Kernel Name", "Name"))),
        None,
    )

    if header_idx is None:
        return []

    return list(csv.DictReader(lines[header_idx:]))
