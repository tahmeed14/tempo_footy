"""Load Gradient Sports tracking data."""

from pathlib import Path

import polars as pl

COLUMN_ORDER = ()


def load_tracking(
    df_in: pl.LazyFrame,
    match_id: int | str,
    overwrite: bool = False,
) -> Path:
    """Write processed tracking data and return its output path."""
    out_path = Path(
        f"data/processed/gradient_sports/tracking/{match_id}.parquet"
    )

    if out_path.exists() and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_in.sink_parquet(out_path, compression="zstd")
    return out_path
