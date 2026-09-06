import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from datetime import timedelta, timezone
from pathlib import Path

# import cs336_basics.layers.multihead_self_attention as _mha
# from cs336_systems.utils import annotated_scaled_dot_product_attention
# # monkey patch
# _mha.scaled_dot_product_attention = annotated_scaled_dot_product_attention  # type: ignore
import torch
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.train_loop import AdamW, ModelConfig, cosine_annealing, gradient_clipping
from cs336_basics.train_loop.utils import parse_dtype
from torch.cuda import nvtx
from tqdm import tqdm


@dataclass
class BenchConfig:
    name: str
    steps: int
    warm_up: int
    unit_ms: bool
    batch_size: int
    lr: float
    lr_warm_up: int
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


class NsysBenchMarker:
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
            (self.bench_cfg.batch_size, self.model_cfg.context_length + 1),
            device=self.model_cfg.device,
            dtype=torch.int,
        )
        return tks[:, :-1], tks[:, 1:]

    def _get_lr(self, step: int) -> float:
        return cosine_annealing(
            t=step,
            alpha_max=self.bench_cfg.lr,
            alpha_min=self.bench_cfg.min_lr,
            t_w=self.bench_cfg.lr_warm_up,
            t_c=self.bench_cfg.steps,
        )

    def train_step(self, step: int):
        with nvtx.range("prepare"):
            lr_t = self._get_lr(step)
            self.optimizer.set_lr(lr_t)

            # take data
            x, targets = self.generate_data()

        # forward
        with nvtx.range("forward"):
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = cross_entropy(logits, targets)

        # backward
        with nvtx.range("backward"):
            loss.backward()
            gradient_clipping(self.model.parameters(), max_l2_norm=self.bench_cfg.grad_clip)

        # optimizer
        with nvtx.range("optimizer"):
            self.optimizer.step()

    def run(self):
        self.model.train()

        print(f"Steps for warm-up: {self.bench_cfg.warm_up}")
        for step in tqdm(range(1, self.bench_cfg.warm_up + 1), desc="warm-up", dynamic_ncols=True):
            self.train_step(step)
        torch.cuda.synchronize()

        print(f"Starting benchmarking: {self.res_dir.name}\nSteps for benchmarking: {self.bench_cfg.steps}")
        torch.cuda.cudart().cudaProfilerStart()
        for step in tqdm(range(1, self.bench_cfg.steps + 1), desc="benchmarking", dynamic_ncols=True):
            self.train_step(step)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


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
    bench_group.add_argument("--lr_warm_up", type=int, default=50)
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
        lr_warm_up=args.lr_warm_up,
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
    benchmarker = NsysBenchMarker(mcfg, bcfg)
    benchmarker.run()
