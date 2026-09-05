import argparse
import json
import timeit
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.train_loop import AdamW, ModelConfig, cosine_annealing, gradient_clipping
from cs336_basics.train_loop.utils import parse_dtype
from tqdm import tqdm

from cs336_systems.utils import export_typst


@dataclass
class BenchConfig:
    name: str
    steps: int
    warm_up: int
    unit_ms: bool
    batch_size: int
    lr: float
    min_lr: float
    weight_decay: float
    eps: float
    grad_clip: float
    res_dir: Path
    torch_seed: int
    betas: tuple[float, float] = (0.9, 0.95)

    def validate(self):
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) > lr ({self.lr})")
        if self.warm_up >= self.steps:
            raise ValueError(f"warm_up ({self.warm_up}) >= max_steps ({self.steps})")


class BenchMarker:
    def __init__(self, model_cfg: ModelConfig, bench_cfg: BenchConfig) -> None:
        self.model_cfg = model_cfg
        self.bench_cfg = bench_cfg

        # seeds and dirs
        self._set_seed()
        self.res_dir = self._setup_res_dir()

        # construct model and optimizer
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

    def _set_seed(self):
        torch.manual_seed(self.bench_cfg.torch_seed)

    def _setup_res_dir(self) -> Path:
        timestamp = dt.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        res_name = f"{self.bench_cfg.name}_{timestamp}"
        res_dir = self.bench_cfg.res_dir / res_name
        res_dir.mkdir(parents=True, exist_ok=True)

        # save hyperparams
        with open(res_dir / "args.json", "w", encoding="utf-8") as f:

            def serialize(obj):
                if isinstance(obj, (torch.device, torch.dtype, Path)):
                    return str(obj)
                raise TypeError(f"Type {type(obj)} not serializable")

            json.dump({"model_cfg": asdict(self.model_cfg), "bench_cfg": asdict(self.bench_cfg)}, f, indent=4, default=serialize)

        return res_dir

    def generate_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of input sequences and next-token targets.
        Returns:
            inputs: Tensor of shape (batch_size, context_length) on device.
            targets: Tensor of shape (batch_size, context_length) on device.
        """
        tks = torch.randint(
            0,
            self.model_cfg.vocab_size,
            (self.bench_cfg.batch_size, self.model_cfg.context_length),
            device=self.model_cfg.device,
            dtype=torch.int,
        )
        return tks[:, :-1], tks[:, 1:]

    def _get_lr(self, step: int) -> float:
        return cosine_annealing(
            t=step,
            alpha_max=self.bench_cfg.lr,
            alpha_min=self.bench_cfg.min_lr,
            t_w=self.bench_cfg.warm_up,
            t_c=self.bench_cfg.steps,
        )

    def _device_sync(self):
        if self.model_cfg.device.type == "cuda":
            torch.cuda.synchronize()

    def train_step(self, step: int) -> npt.NDArray[np.float64]:
        prepare_start = timeit.default_timer()
        lr_t = self._get_lr(step)
        self.optimizer.set_lr(lr_t)

        # take data
        x, targets = self.generate_data()
        prepare = timeit.default_timer() - prepare_start

        # forward
        self.optimizer.zero_grad()
        self._device_sync()
        forward_start = timeit.default_timer()
        logits = self.model(x)
        loss = cross_entropy(logits, targets)
        self._device_sync()
        forward = timeit.default_timer() - forward_start

        # backward
        backward_start = timeit.default_timer()
        loss.backward()
        gradient_clipping(self.model.parameters(), max_l2_norm=self.bench_cfg.grad_clip)
        self._device_sync()
        backward = timeit.default_timer() - backward_start

        # optimizer
        opt_start = timeit.default_timer()
        self.optimizer.step()
        self._device_sync()
        opt = timeit.default_timer() - opt_start

        return np.array([prepare, forward, backward, opt], dtype=np.float64)

    def run(self):
        self.model.train()

        print(f"Steps for warm-up: {self.bench_cfg.warm_up}")
        for step in tqdm(range(1, self.bench_cfg.warm_up + 1), desc="warm-up", dynamic_ncols=True):
            self.train_step(step)

        print(f"Starting benchmarking: {self.res_dir.name}\nSteps for benchmarking: {self.bench_cfg.steps}")
        records = []
        for step in tqdm(range(1, self.bench_cfg.steps + 1), desc="benchmarking", dynamic_ncols=True):
            records.append(self.train_step(step))
        self.res = np.array(records)  # (step, stages)

        self.save_res()
        self.show_res(unit_ms=self.bench_cfg.unit_ms)
        self.export(unit_ms=self.bench_cfg.unit_ms)

    def save_res(self) -> None:
        path = self.res_dir / "raw_timings.npy"
        np.save(path, self.res)
        print(f"Results saved to:\n  - {path}")

    def _get_summary(self, unit_ms: bool = True) -> pd.DataFrame:
        """
        Tackle all calculations and return data frame
        """
        mean = self.res.mean(axis=0)
        std = self.res.std(axis=0)
        total_mean = mean.sum()
        total_std = self.res.sum(axis=1).std()

        stages = ["prepare", "forward", "backward", "optimizer", "total"]
        means = np.append(mean, total_mean)
        stds = np.append(std, total_std)

        scale = 1000.0 if unit_ms else 1.0

        return pd.DataFrame({"mean": means * scale, "std": stds * scale}, index=stages)

    def export(self, unit_ms: bool = True):
        unit = "ms" if unit_ms else "s"
        df = self._get_summary(unit_ms=unit_ms)
        df.columns = [f"Mean ({unit})", f"Std ({unit})"]
        df.index = [col.capitalize() for col in df.index]

        caption = f"Benchmarking Results ({unit})"
        path = self.res_dir / "table.typ"
        typst_str = export_typst(df, precision=4, caption=caption, output_path=path)
        print(typst_str)

    def show_res(self, unit_ms: bool = True) -> None:
        print("Benchmarking complete!")
        df = self._get_summary(unit_ms=unit_ms)
        unit = "ms" if unit_ms else "s"

        col_w_stage = 15
        col_w_data = 12
        header_mean = f"Mean ({unit})"
        header_std = f"Std ({unit})"
        total_width = col_w_stage + 3 + col_w_data + 3 + col_w_data  # 15 + 3 + 12 + 3 + 12 = 45

        print("\n" + "=" * total_width)
        print(f"{'Benchmarking Results':^{total_width}}")
        print("=" * total_width)
        print(f"{'Stage':<{col_w_stage}} | {header_mean:<{col_w_data}} | {header_std:<{col_w_data}}")
        print("-" * total_width)

        stages = ["prepare", "forward", "backward", "optimizer"]
        for stage in stages:
            if stage in df.index:
                mean_val = df.loc[stage, "mean"]
                std_val = df.loc[stage, "std"]
                mean_str = f"{mean_val:.3f} {unit}"
                std_str = f"{std_val:.3f} {unit}"
                print(f"{stage.capitalize():<{col_w_stage}} | {mean_str:>{col_w_data}} | {std_str:>{col_w_data}}")

        if "total" in df.index:
            total_mean = df.loc["total", "mean"]
            total_std = df.loc["total", "std"]
            total_mean_str = f"{total_mean:.3f} {unit}"
            total_std_str = f"{total_std:.3f} {unit}"
            print("-" * total_width)
            print(f"{'Total Step':<{col_w_stage}} | {total_mean_str:>{col_w_data}} | {total_std_str:>{col_w_data}}")

        print("=" * total_width + "\n")


def parse_args() -> tuple[ModelConfig, BenchConfig]:
    parser = argparse.ArgumentParser(description="Transformer Language Model BenchMark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument("--vocab_size", type=int, default=10000)
    model_group.add_argument("--context_length", type=int, default=512)
    model_group.add_argument("--num_layers", type=int, default=12)
    model_group.add_argument("--d_model", type=int, default=768)
    model_group.add_argument("--num_heads", type=int, default=12)
    model_group.add_argument("--d_ff", type=int, default=3072)
    model_group.add_argument("--rope_theta", type=float, default=10000.0)
    model_group.add_argument("--device", type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_group.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)

    # Bench
    bench_group = parser.add_argument_group("Benchmark Arguments")
    bench_group.add_argument("--name", type=str, default="bench")
    bench_group.add_argument("--steps", type=int, required=True)
    bench_group.add_argument("--warm_up", type=int, required=True)
    bench_group.add_argument("--no_unit_ms", dest="unit_ms", action="store_false", default=True)
    bench_group.add_argument("--batch_size", type=int, default=4)
    bench_group.add_argument("--lr", type=float, default=1.5e-3)
    bench_group.add_argument("--min_lr", type=float, default=1.5e-4)
    bench_group.add_argument("--weight_decay", type=float, default=0.1)
    bench_group.add_argument("--eps", type=float, default=1e-8)
    bench_group.add_argument("--betas", type=float, default=(0.9, 0.95), nargs=2)
    bench_group.add_argument("--grad_clip", type=float, default=1.0)
    bench_group.add_argument("--res_dir", type=Path, default=Path("benchmark_res"))
    bench_group.add_argument("--torch_seed", type=int, default=45)

    args = parser.parse_args()

    # dataclass
    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=args.dtype,
    )
    bench_cfg = BenchConfig(
        name=args.name,
        steps=args.steps,
        warm_up=args.warm_up,
        unit_ms=args.unit_ms,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        eps=args.eps,
        betas=tuple(args.betas),
        grad_clip=args.grad_clip,
        res_dir=args.res_dir,
        torch_seed=args.torch_seed,
    )

    # validate
    model_cfg.validate()
    bench_cfg.validate()

    return model_cfg, bench_cfg


if __name__ == "__main__":
    mcfg, bcfg = parse_args()
    benchmarker = BenchMarker(mcfg, bcfg)
    benchmarker.run()
