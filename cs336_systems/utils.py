from pathlib import Path

import pandas as pd


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
