from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.train_loop import AdamW, ModelConfig, gradient_clipping
from tqdm import tqdm

from cs336_systems.observers import create_observers
from cs336_systems.observers.base import (
    STAGE_BACKWARD,
    STAGE_FORWARD,
    STAGE_OPTIMIZER,
    STAGE_PREPARE,
    BaseObserver,
)
from cs336_systems.utils import BenchConfig, parse_args


class BenchmarkHarness:
    def __init__(self, model_cfg: ModelConfig, bench_cfg: BenchConfig) -> None:
        self.model_cfg = model_cfg
        self.bench_cfg = bench_cfg

        # monkey patch
        self._monkey_patch()

        # set seeds and save args
        self._set_seed()
        self._save_args()

        # model and optimizer
        self.model = TransformerLM(
            self.model_cfg.vocab_size,
            self.model_cfg.context_length,
            self.model_cfg.num_layers,
            self.model_cfg.d_model,
            self.model_cfg.num_heads,
            self.model_cfg.d_ff,
            self.model_cfg.rope_theta,
            self.model_cfg.device,
            self.model_cfg.dtype,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.bench_cfg.lr,
            weight_decay=self.bench_cfg.weight_decay,
            eps=self.bench_cfg.eps,
            betas=self.bench_cfg.betas,
        )

        self.autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.bench_cfg.use_mixed_precision and self.model_cfg.device.type == "cuda" else nullcontext()
        )

        self.observers: list[BaseObserver] = create_observers(self.model_cfg, self.bench_cfg)

    def _monkey_patch(self):
        if self.bench_cfg.att_patch:
            import cs336_basics.layers.multihead_self_attention as _mha

            from cs336_systems.utils import annotated_scaled_dot_product_attention

            if hasattr(_mha, "scaled_dot_product_attention"):
                _mha.scaled_dot_product_attention = annotated_scaled_dot_product_attention  # type: ignore
            else:
                raise AttributeError("scaled_dot_product_attention not found")

    def _set_seed(self):
        torch.manual_seed(self.bench_cfg.torch_seed)

    def _save_args(self) -> None:
        res_dir = self.bench_cfg.res_dir
        res_dir.mkdir(parents=True, exist_ok=True)

        with open(res_dir / "args.json", "w", encoding="utf-8") as f:

            def serialize(obj):
                if isinstance(obj, (torch.device, torch.dtype, Path)):
                    return str(obj)
                raise TypeError(f"Type {type(obj)} not serializable")

            json.dump({"model_cfg": asdict(self.model_cfg), "bench_cfg": asdict(self.bench_cfg)}, f, indent=4, default=serialize)

    def _save_crash_report(self, phase: str, step: int, exc: BaseException) -> None:
        crash_data = {
            "status": "failed",
            "phase": phase,
            "failed_step": step,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

        if torch.cuda.is_available() and self.model_cfg.device.type == "cuda":
            crash_data["cuda_memory"] = {
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
                "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
            }

        error_file = self.bench_cfg.res_dir / "error.json"
        try:
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(crash_data, f, indent=4)
            print(f"\n[Crash Report] Saved crash details to: {error_file}")
        except (OSError, TypeError) as dump_err:
            print(f"\n[Error] Failed to write error.json: {dump_err}")

    def generate_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        tks = torch.randint(
            0,
            self.model_cfg.vocab_size,
            (self.bench_cfg.batch_size, self.model_cfg.context_length + 1),
            device=self.model_cfg.device,
            dtype=torch.int,
        )
        return tks[:, :-1], tks[:, 1:]

    def _run_stage(self, stage: str, func: Callable):
        """Sequentially enter context with ExitStack"""
        with ExitStack() as stack:
            for obs in self.observers:
                stack.enter_context(obs.stage_context(stage))
            return func()

    def train_step(self, step: int) -> None:
        for obs in self.observers:
            obs.on_step_start(step)

        # prepare
        def prepare_op():
            return self.generate_data()

        x, targets = self._run_stage(STAGE_PREPARE, prepare_op)

        # forward
        def forward_op():
            with self.autocast_context:
                logits = self.model(x)
                loss = cross_entropy(logits, targets)
            return loss

        self.optimizer.zero_grad()
        loss = self._run_stage(STAGE_FORWARD, forward_op)

        # Backward
        def backward_op():
            loss.backward()
            gradient_clipping(self.model.parameters(), max_l2_norm=self.bench_cfg.grad_clip)

        self._run_stage(STAGE_BACKWARD, backward_op)

        # Optimizer
        self._run_stage(STAGE_OPTIMIZER, self.optimizer.step)

        for obs in self.observers:
            obs.on_step_end(step)

    def run(self) -> None:
        self.model.train()

        current_phase = "init"
        current_step = 0

        try:
            # 1. Warm-up
            if self.bench_cfg.warm_up > 0:
                current_phase = "warm_up"
                print(f"--> Warm-up steps: {self.bench_cfg.warm_up}")
                for step in tqdm(range(1, self.bench_cfg.warm_up + 1), desc="warm-up", dynamic_ncols=True):
                    current_step = step
                    self.train_step(step)

            if self.model_cfg.device.type == "cuda":
                torch.cuda.synchronize()

            # 2. Benchmarking Window Start
            current_phase = "benchmark"
            current_step = 0
            print(f"--> Start Benchmarking: {self.bench_cfg.res_dir.name} (Steps: {self.bench_cfg.steps})")

            for obs in self.observers:
                obs.on_window_start()

            # 3. Measurement Loop
            for step in tqdm(range(1, self.bench_cfg.steps + 1), desc="benchmarking", dynamic_ncols=True):
                current_step = step
                self.train_step(step)

            if self.model_cfg.device.type == "cuda":
                torch.cuda.synchronize()

        except BaseException as e:
            print(f"\n[Run Failure] Interrupted at phase='{current_phase}', step={current_step}. Error: {e}")
            self._save_crash_report(current_phase, current_step, e)
            raise

        finally:
            for obs in self.observers:
                obs.on_window_end()

            print(f"--> Finalizing observers into {self.bench_cfg.res_dir}...")
            for obs in self.observers:
                obs.finish(self.bench_cfg.res_dir)


if __name__ == "__main__":
    mcfg, bcfg = parse_args()
    harness = BenchmarkHarness(mcfg, bcfg)
    harness.run()
