from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from cs336_systems.observers.base import BaseObserver
from cs336_systems.observers.memory import MemoryObserver
from cs336_systems.observers.nsys import NsysObserver
from cs336_systems.observers.timing import TimingObserver

if TYPE_CHECKING:
    from cs336_systems.utils import BenchConfig, ModelConfig

OBSERVER_REGISTRY = {
    "timing": TimingObserver,
    "nsys": NsysObserver,
    "memory": MemoryObserver,
}


def create_observers(model_cfg: ModelConfig, bench_cfg: BenchConfig) -> list[BaseObserver]:
    """
    Instantiate and order observers based on the configuration.

    [Fixed Nesting Order]: Outer -> [nsys, memory] -> timing -> Innermost.
    This guarantees that TimingObserver measures closest to the compute kernel,
    excluding CPU overhead introduced by NVTX ranges and memory interceptors.
    """
    requested = [p.strip().lower() for p in bench_cfg.profilers]
    for p in requested:
        if p not in OBSERVER_REGISTRY:
            raise ValueError(f"Unknown profiler: '{p}', must select from {OBSERVER_REGISTRY}")

    # Check for Decision Point 1
    if "timing" in requested and "memory" in requested:
        warnings.warn(
            "\n[WARNING] Detected that both 'timing' and 'memory' observers are enabled!\n"
            "PyTorch memory allocation stack tracing (_record_memory_history) introduces "
            "significant CPU and runtime overhead, which will inflate wall-clock measurements! "
            "For accurate timing, please run with 'timing' alone.\n",
            stacklevel=2,
        )

    observers: list[BaseObserver] = []

    if "nsys" in requested:
        observers.append(NsysObserver())
    if "memory" in requested:
        observers.append(MemoryObserver())
    if "timing" in requested:
        observers.append(TimingObserver(device=model_cfg.device, unit_ms=bench_cfg.unit_ms))

    return observers
