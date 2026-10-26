import enum
from pathlib import Path

import pandas as pd
from cs336_basics.train_loop import ModelConfig
from cs336_basics.train_loop.utils import parse_dtype
from pydantic import BaseModel, Field, field_validator

from cs336_systems.patch import PatchSpec, parse_patch_spec


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


MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


class Mode(enum.StrEnum):
    TRAIN = "train"
    INFER = "infer"


class ModelArgs(BaseModel):
    """Model args and overlapping method"""

    model_size: str = Field(default="small", description="Choices: small, medium, large, xl, 10b")
    vocab_size: int = Field(default=10000)
    context_length: int = Field(default=512)
    num_layers: int | None = Field(default=None, description="Override num_layers")
    d_model: int | None = Field(default=None, description="Override d_model")
    num_heads: int | None = Field(default=None, description="Override num_heads")
    d_ff: int | None = Field(default=None, description="Override d_ff")
    rope_theta: float = Field(default=10000.0)
    dtype: str = Field(default="bfloat16", description="bfloat16, float32, float16")

    def build_model_config(self) -> ModelConfig:
        size_key = self.model_size.lower()
        preset = MODEL_SIZES[size_key]
        num_layers = self.num_layers if self.num_layers is not None else preset["num_layers"]
        d_model = self.d_model if self.d_model is not None else preset["d_model"]
        num_heads = self.num_heads if self.num_heads is not None else preset["num_heads"]
        d_ff = self.d_ff if self.d_ff is not None else preset["d_ff"]

        torch_dtype = parse_dtype(self.dtype)

        model_cfg = ModelConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=self.rope_theta,
            dtype=torch_dtype,
        )
        model_cfg.validate()
        return model_cfg


class BenchConfig(BaseModel):
    res_dir: Path = Field(default=Path("benchmark_res/default_name"))
    mode: Mode = Field(default=Mode.TRAIN)
    profilers: list[str] = Field(default_factory=lambda: ["timing"])
    patches: list[PatchSpec] = Field(default_factory=list, description="e.g. 'att, recompute:group_size=4'")
    steps: int = Field(default=10)
    warm_up: int = Field(default=5)
    jit_compile: bool = Field(default=False)
    unit_ms: bool = Field(default=True)
    use_mixed_precision: bool = Field(default=False)
    batch_size: int = Field(default=4)
    lr: float = Field(default=1.5e-3)
    weight_decay: float = Field(default=0.1)
    eps: float = Field(default=1e-8)
    betas: tuple[float, float] = Field(default=(0.9, 0.95))
    grad_clip: float = Field(default=1.0)
    torch_seed: int = Field(default=45)

    @field_validator("patches", mode="before")
    @classmethod
    def parse_patch_list(cls, v):
        return parse_patch_spec(v) if isinstance(v, str) else v
