"""Reproduce player random-effect overlap tables for the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import xarray as xr

from tempoctrl.model.utils import (
    build_player_effects_table,
    discover_archetypes,
    summarize_archetype_overlap,
)

DEFAULT_POSTERIOR_PATH = Path("paper/results/tempo_gamma_posterior.nc")
DEFAULT_MODEL_DATA_PATH = Path("data/analysis/model_data_vFINAL.parquet")
DEFAULT_CURRENT_OVERLAP_PATH = Path(
    "paper/tables/player_archetype_overlap_alpha.csv"
)
DEFAULT_INTERPRETABLE_OVERLAP_PATH = Path(
    "paper/tables/player_archetype_overlap_cv.csv"
)

MU_PLAYER_VARIABLE = "1|player_id"
ALPHA_PLAYER_VARIABLE = "alpha_1|player_id"


def load_posterior_idata(
    path: str | Path = DEFAULT_POSTERIOR_PATH,
) -> xr.DataTree:
    """Read the fitted Gamma-model inference data."""
    idata = xr.open_datatree(path, engine="h5netcdf")
    if getattr(idata, "posterior", None) is None:
        idata.close()
        raise ValueError(
            f"{Path(path)} does not contain a posterior group. "
            "Use tempo_gamma_posterior.nc rather than the "
            "prior-predictive file."
        )
    return idata


def build_player_overlap_table(
    idata: Any,
    model_df: pl.DataFrame | pl.LazyFrame,
    *,
    hdi_prob: float = 0.94,
) -> pl.DataFrame:
    """Build the all-player overlap table on the model's raw link scales."""
    mu_table = build_player_effects_table(
        idata,
        model_df,
        variable=MU_PLAYER_VARIABLE,
        hdi_prob=hdi_prob,
    )
    alpha_table = build_player_effects_table(
        idata,
        model_df,
        variable=ALPHA_PLAYER_VARIABLE,
        hdi_prob=hdi_prob,
    )
    _, _, overlap = discover_archetypes(
        mu_table,
        alpha_table,
        mu_range=(-float("inf"), float("inf")),
        shape_range=(-float("inf"), float("inf")),
        option="value",
    )
    return overlap


def transform_overlap_to_percentage_effects(
    overlap_table: pl.DataFrame,
) -> pl.DataFrame:
    """Transform raw mu and alpha summaries to percent effects on mu and CV.

    The transformations are ``100 * (exp(u) - 1)`` for expected tempo and
    ``100 * (exp(-a / 2) - 1)`` for conditional Gamma CV. The latter is
    decreasing, so its interval bounds and sign probability are reversed.
    """
    required_columns = {
        "mu_posterior_mean",
        "mu_hdi_lower",
        "mu_hdi_upper",
        "mu_p_gt_zero",
        "mu_credible",
        "shape_posterior_mean",
        "shape_hdi_lower",
        "shape_hdi_upper",
        "shape_p_gt_zero",
        "shape_credible",
    }
    missing_columns = sorted(
        required_columns.difference(overlap_table.columns)
    )
    if missing_columns:
        raise KeyError(
            f"overlap_table is missing required columns: {missing_columns}."
        )

    return overlap_table.with_columns(
        (100 * (pl.col("mu_posterior_mean").exp() - 1)).alias(
            "mu_posterior_mean"
        ),
        (100 * (pl.col("mu_hdi_lower").exp() - 1)).alias("mu_hdi_lower"),
        (100 * (pl.col("mu_hdi_upper").exp() - 1)).alias("mu_hdi_upper"),
        (100 * ((-pl.col("shape_posterior_mean") / 2).exp() - 1)).alias(
            "shape_posterior_mean"
        ),
        (100 * ((-pl.col("shape_hdi_upper") / 2).exp() - 1)).alias(
            "shape_hdi_lower"
        ),
        (100 * ((-pl.col("shape_hdi_lower") / 2).exp() - 1)).alias(
            "shape_hdi_upper"
        ),
        (1 - pl.col("shape_p_gt_zero")).alias("shape_p_gt_zero"),
    )


def generate_current_overlap_table(
    overlap_table: pl.DataFrame,
    *,
    output_path: str | Path | None = DEFAULT_CURRENT_OVERLAP_PATH,
) -> pl.DataFrame:
    """Generate the existing alpha-sign archetype summary."""
    summary = summarize_archetype_overlap(overlap_table)
    _write_csv(summary, output_path)
    return summary


def generate_interpretable_overlap_table(
    overlap_table: pl.DataFrame,
    *,
    output_path: str | Path | None = DEFAULT_INTERPRETABLE_OVERLAP_PATH,
) -> pl.DataFrame:
    """Generate the archetype summary using percent effects on mu and CV."""
    transformed = transform_overlap_to_percentage_effects(overlap_table)
    summary = summarize_archetype_overlap(
        transformed,
        dispersion_scale="conditional_cv",
    )
    summary = summary.rename(
        {
            column: column.replace("alpha", "conditional_cv")
            for column in summary.columns
            if "alpha" in column
        }
    )
    _write_csv(summary, output_path)
    return summary


def reproduce_overlap_tables(
    *,
    posterior_path: str | Path = DEFAULT_POSTERIOR_PATH,
    model_data_path: str | Path = DEFAULT_MODEL_DATA_PATH,
    current_output_path: str | Path | None = DEFAULT_CURRENT_OVERLAP_PATH,
    interpretable_output_path: str | Path | None = (
        DEFAULT_INTERPRETABLE_OVERLAP_PATH
    ),
    hdi_prob: float = 0.94,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load fitted inputs and generate both overlap summary tables."""
    model_df = pl.read_parquet(
        model_data_path,
        columns=["player_id", "playername", "teamname"],
    )
    idata = load_posterior_idata(posterior_path)
    try:
        overlap = build_player_overlap_table(
            idata,
            model_df,
            hdi_prob=hdi_prob,
        )
        current = generate_current_overlap_table(
            overlap,
            output_path=current_output_path,
        )
        interpretable = generate_interpretable_overlap_table(
            overlap,
            output_path=interpretable_output_path,
        )
    finally:
        idata.close()

    return current, interpretable


def _write_csv(table: pl.DataFrame, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(destination)


def main() -> None:
    """Generate the paper's raw-alpha and interpretable-CV overlap tables."""
    reproduce_overlap_tables()


if __name__ == "__main__":
    main()
