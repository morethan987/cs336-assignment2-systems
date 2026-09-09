import argparse
from dataclasses import dataclass
from pathlib import Path

import einx
import pandas as pd
import torch
from cs336_basics.layers import softmax
from cs336_basics.train_loop import ModelConfig
from cs336_basics.train_loop.utils import parse_dtype
from torch.cuda import nvtx


def export_typst(
    df: pd.DataFrame,
    precision: int | None = 3,
    caption: str | None = None,
    output_path: str | Path | None = None,
    hide_index: bool = False,
) -> str:
    styler = df.style
    if hide_index:
        styler.hide(axis="index")
    if caption:
        styler.set_caption(caption)
    styler.format(precision=precision)

    typst_code = styler.to_typst()

    if output_path:
        Path(output_path).write_text(typst_code, encoding="utf-8")

    return typst_code


def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = torch.tensor(Q.shape[-1])

    with nvtx.range("computing attention scores"):
        # n and m all respresent seq_len, only to tag matrix shape: (n, m) or (m, n)
        scaled_dot = torch.multiply(torch.rsqrt(d_k), einx.dot("... n [d_k], ... m [d_k] -> ... n m", Q, K))

    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(~mask, float("-inf"))

    with nvtx.range("computing softmax"):
        sftmx = softmax(scaled_dot, -1)

    with nvtx.range("final matmul"):
        res = einx.dot("... n [m], ... [m] d_v -> ... n d_v", sftmx, V)
    return res


MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


@dataclass
class BenchConfig:
    res_dir: Path
    profilers: list[str]
    steps: int
    warm_up: int
    unit_ms: bool
    use_mixed_precision: bool
    att_patch: bool
    batch_size: int
    lr: float
    weight_decay: float
    eps: float
    grad_clip: float
    torch_seed: int
    betas: tuple[float, float] = (0.9, 0.95)


def parse_args() -> tuple[ModelConfig, BenchConfig]:
    parser = argparse.ArgumentParser(description="Transformer Language Model BenchMark")

    # Model
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl", "10B", "10b"])
    model_group.add_argument("--vocab_size", type=int, default=10000)
    model_group.add_argument("--context_length", type=int, default=512)
    model_group.add_argument("--num_layers", type=int, default=None, help="Override num_layers")
    model_group.add_argument("--d_model", type=int, default=None, help="Override d_model")
    model_group.add_argument("--num_heads", type=int, default=None, help="Override num_heads")
    model_group.add_argument("--d_ff", type=int, default=None, help="Override d_ff")
    model_group.add_argument("--rope_theta", type=float, default=10000.0)
    model_group.add_argument("--device", type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_group.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)

    # Bench
    bench_group = parser.add_argument_group("Benchmark Arguments")
    bench_group.add_argument("--res_dir", type=Path, default=Path("benchmark_res/default_name"))
    bench_group.add_argument("--profilers", type=lambda s: [item.strip() for item in s.split(",")], default=["timing"])
    bench_group.add_argument("--steps", type=int, required=True)
    bench_group.add_argument("--warm_up", type=int, required=True)
    bench_group.add_argument("--use_mixed_precision", action="store_true", default=False)
    bench_group.add_argument("--att_patch", action="store_true", default=False)
    bench_group.add_argument("--no_unit_ms", dest="unit_ms", action="store_false", default=True)
    bench_group.add_argument("--batch_size", type=int, default=4)
    bench_group.add_argument("--lr", type=float, default=1.5e-3)
    bench_group.add_argument("--weight_decay", type=float, default=0.1)
    bench_group.add_argument("--eps", type=float, default=1e-8)
    bench_group.add_argument("--betas", type=float, default=(0.9, 0.95), nargs=2)
    bench_group.add_argument("--grad_clip", type=float, default=1.0)
    bench_group.add_argument("--torch_seed", type=int, default=45)

    args = parser.parse_args()

    # parse model size
    size_key = args.model_size.lower()
    preset = MODEL_SIZES[size_key]
    num_layers = args.num_layers if args.num_layers is not None else preset["num_layers"]
    d_model = args.d_model if args.d_model is not None else preset["d_model"]
    num_heads = args.num_heads if args.num_heads is not None else preset["num_heads"]
    d_ff = args.d_ff if args.d_ff is not None else preset["d_ff"]
    print(f"--> Using Model Size: {size_key} (layers={num_layers}, d_model={d_model}, heads={num_heads}, d_ff={d_ff})")

    # dataclass
    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=args.dtype,
    )
    bench_cfg = BenchConfig(
        res_dir=args.res_dir,
        profilers=args.profilers,
        steps=args.steps,
        warm_up=args.warm_up,
        unit_ms=args.unit_ms,
        use_mixed_precision=args.use_mixed_precision,
        att_patch=args.att_patch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.eps,
        betas=tuple(args.betas),
        grad_clip=args.grad_clip,
        torch_seed=args.torch_seed,
    )

    # validate
    model_cfg.validate()

    return model_cfg, bench_cfg
