from __future__ import annotations

from pathlib import Path

import torch

from cs336_systems.observers.base import BaseObserver


class MemoryObserver(BaseObserver):
    """
    Captures high-resolution PyTorch CUDA memory timelines and generates snapshot files
    visualizable via https://pytorch.org/memory_viz.
    """

    def __init__(self, max_entries: int = 100_000):
        self.max_entries = max_entries

    def on_window_start(self) -> None:
        if torch.cuda.is_available():
            print("[MemoryObserver] Starting PyTorch CUDA memory history recording...")
            torch.cuda.memory._record_memory_history(
                enabled="all",
                max_entries=self.max_entries,
                clear_history=True,
            )

    def on_window_end(self) -> None:
        pass  # Actual dump is deferred to finish() to ensure writing to res_dir.

    def finish(self, res_dir: Path) -> None:
        if not torch.cuda.is_available():
            return

        snapshot_path = res_dir / "memory_snapshot.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            print(f"\n[MemoryObserver] Memory snapshot successfully exported to: {snapshot_path}")
            print("Tip: Visit https://pytorch.org/memory_viz to visualize this file.")
        except (RuntimeError, OSError) as e:
            print(f"[MemoryObserver] Failed to export snapshot: {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
