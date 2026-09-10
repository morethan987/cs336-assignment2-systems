from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol, runtime_checkable

# global constants
STAGE_PREPARE = "prepare"
STAGE_FORWARD = "forward"
STAGE_BACKWARD = "backward"
STAGE_OPTIMIZER = "optimizer"
TRAIN_STAGES = (STAGE_PREPARE, STAGE_FORWARD, STAGE_BACKWARD, STAGE_OPTIMIZER)
INFER_STAGES = (STAGE_PREPARE, STAGE_FORWARD)


@runtime_checkable
class BaseObserver(Protocol):
    """
    Unified contract for all observers.
    Defaults to no-ops; subclasses only need to override the hooks they care about.
    """

    def on_window_start(self) -> None:
        """Triggered when the benchmark measurement window starts (after warm-up)."""

    def on_window_end(self) -> None:
        """Triggered when the benchmark measurement window ends."""

    def on_step_start(self, step: int) -> None:
        """Triggered at the start of each training step."""

    def on_step_end(self, step: int) -> None:
        """Triggered at the end of each training step."""

    def stage_context(self, stage: str) -> AbstractContextManager[None]:
        """Return a stage context manager (e.g., timers, NVTX markers)."""
        return nullcontext()

    def finish(self, res_dir: Path) -> None:
        """Save artifacts to disk and print summary after benchmarking completes."""
