from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.cuda import nvtx

from cs336_systems.observers.base import BaseObserver


class NsysObserver(BaseObserver):
    """
    Responsible for:
    1. cudaProfilerStart / Stop within the benchmark window.
    2. Process-wide NVTX annotations (negligible runtime overhead).
    3. Step-level range markers.
    """

    def __init__(self) -> None:
        self._current_step_msg = ""

    def on_window_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            print("[NsysObserver] CUDA Profiler started.")

    def on_window_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print("[NsysObserver] CUDA Profiler stopped.")

    def on_step_start(self, step: int) -> None:
        nvtx.range_push(f"step_{step}")

    def on_step_end(self, step: int) -> None:
        nvtx.range_pop()

    @contextmanager
    def stage_context(self, stage: str) -> Generator[None, None, None]:
        nvtx.range_push(stage)
        try:
            yield
        finally:
            nvtx.range_pop()

    def finish(self, res_dir: Path) -> None:
        print(f"[NsysObserver] Profiling collection complete. Please check the exported nsys report at {res_dir}.")
