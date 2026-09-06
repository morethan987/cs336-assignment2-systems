from pathlib import Path

import einx
import pandas as pd
import torch
from cs336_basics.layers import softmax
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


@nvtx.range("scaled dot product attention")
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
