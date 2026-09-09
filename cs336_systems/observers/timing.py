from __future__ import annotations

import timeit
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cs336_systems.observers.base import TRAIN_STAGES, BaseObserver
from cs336_systems.utils import export_typst


class TimingObserver(BaseObserver):
    def __init__(self, device: torch.device, unit_ms: bool = True):
        self.device = device
        self.unit_ms = unit_ms
        self.is_measuring = False

        self.records: list[list[float]] = []
        self._current_step_times: dict[str, float] = {}

    def on_window_start(self) -> None:
        self.is_measuring = True
        self.records.clear()

    def on_window_end(self) -> None:
        self.is_measuring = False

    def on_step_start(self, step: int) -> None:
        self._current_step_times = {}

    def on_step_end(self, step: int) -> None:
        if self.is_measuring:
            step_record = [self._current_step_times.get(stage, 0.0) for stage in TRAIN_STAGES]
            self.records.append(step_record)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def stage_context(self, stage: str):
        self._sync()
        t0 = timeit.default_timer()
        try:
            yield
        finally:
            self._sync()
            elapsed = timeit.default_timer() - t0
            self._current_step_times[stage] = elapsed

    def finish(self, res_dir: Path) -> None:
        if not self.records:
            return

        res = np.array(self.records, dtype=np.float64)  # (steps, num_stages)
        raw_path = res_dir / "raw_timings.npy"
        np.save(raw_path, res)
        print(f"\n[TimingObserver] raw timing data saved to: {raw_path}")

        # analysis
        mean = res.mean(axis=0)
        std = res.std(axis=0)
        total_mean = mean.sum()
        total_std = res.sum(axis=1).std()

        stages = list(TRAIN_STAGES) + ["total"]
        means = np.append(mean, total_mean)
        stds = np.append(std, total_std)

        scale = 1000.0 if self.unit_ms else 1.0
        unit = "ms" if self.unit_ms else "s"

        df = pd.DataFrame({"mean": means * scale, "std": stds * scale}, index=stages)

        # print
        self._show_results(df, unit)

        # export to typst
        df_export = df.copy()
        df_export.columns = [f"Mean ({unit})", f"Std ({unit})"]
        df_export.index = [s.capitalize() for s in df_export.index]
        table_path = res_dir / "table.typ"
        export_typst(df_export, precision=4, caption=f"Benchmarking Results ({unit})", output_path=table_path)
        print(f"[TimingObserver] Typst saved to: {table_path}")

    def _show_results(self, df: pd.DataFrame, unit: str):
        col_w_stage = 15
        col_w_data = 14
        total_width = col_w_stage + 3 + col_w_data + 3 + col_w_data

        print("\n" + "=" * total_width)
        print(f"{'Benchmarking Results':^{total_width}}")
        print("=" * total_width)
        print(f"{'Stage':<{col_w_stage}} | {f'Mean ({unit})':<{col_w_data}} | {f'Std ({unit})':<{col_w_data}}")
        print("-" * total_width)

        for stage in TRAIN_STAGES:
            m_val = df.loc[stage, "mean"]
            s_val = df.loc[stage, "std"]
            print(f"{stage.capitalize():<{col_w_stage}} | {m_val:>10.3f} {unit} | {s_val:>10.3f} {unit}")

        print("-" * total_width)
        tm_val = df.loc["total", "mean"]
        ts_val = df.loc["total", "std"]
        print(f"{'Total Step':<{col_w_stage}} | {tm_val:>10.3f} {unit} | {ts_val:>10.3f} {unit}")
        print("=" * total_width + "\n")
