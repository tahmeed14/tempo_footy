"""Reproduce the player tempo-archetype quadrant figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tempoctrl.reproduce_re_and_case_studies import (
    DEFAULT_MODEL_DATA_PATH,
    DEFAULT_POSTERIOR_PATH,
    build_player_overlap_table,
    load_posterior_idata,
    transform_overlap_to_percentage_effects,
)

DEFAULT_FIGURES_DIRECTORY = Path("paper/figures")
DEFAULT_FIGURE_NAME = "player_archetypes_tempo"
DEFAULT_DIRECTIONAL_FIGURE_NAME = "player_archetypes_tempo_directional_support"
DEFAULT_FIGSIZE = (7.0, 4.5)


def plot_player_archetypes(
    transformed_overlap: pl.DataFrame,
    *,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    highlight_label: str = "Credible for both",
) -> tuple[Figure, Axes]:
    """Plot player effects on expected tempo and conditional CV.

    ``transformed_overlap`` must contain percentage-scale effects produced by
    :func:`transform_overlap_to_percentage_effects`. Players whose HDIs exclude
    zero for both effects are highlighted in gold.
    """
    required_columns = (
        "mu_posterior_mean",
        "shape_posterior_mean",
        "mu_credible",
        "shape_credible",
    )
    missing_columns = sorted(
        set(required_columns).difference(transformed_overlap.columns)
    )
    if missing_columns:
        raise KeyError(
            "transformed_overlap is missing required columns: "
            f"{missing_columns}."
        )

    points = (
        transformed_overlap.select(required_columns)
        .drop_nulls(["mu_posterior_mean", "shape_posterior_mean"])
        .with_columns(
            (
                (pl.col("mu_credible") == "Credible")
                & (pl.col("shape_credible") == "Credible")
            ).alias("credible_both")
        )
    )
    if points.is_empty():
        raise ValueError("transformed_overlap contains no plottable players.")

    other_players = points.filter(~pl.col("credible_both"))
    credible_players = points.filter(pl.col("credible_both"))

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        ax.scatter(
            other_players["mu_posterior_mean"],
            other_players["shape_posterior_mean"],
            s=28,
            color="skyblue",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.65,
            label=f"Other players (n={other_players.height})",
            zorder=2,
        )
        ax.scatter(
            credible_players["mu_posterior_mean"],
            credible_players["shape_posterior_mean"],
            s=42,
            color="gold",
            edgecolors="darkgoldenrod",
            linewidths=0.8,
            alpha=0.95,
            label=f"{highlight_label} (n={credible_players.height})",
            zorder=3,
        )

        ax.axvline(0, color="gray", linewidth=1, zorder=1)
        ax.axhline(0, color="gray", linewidth=1, zorder=1)
        ax.set_axisbelow(True)
        ax.grid(True, color="#d9d9d9", linewidth=0.7)
        ax.set_title("Player Archetypes for Tempo", fontsize="large", pad=20)
        ax.set_xlabel("Player effect on expected tempo (% Δμ)", fontsize=11)
        ax.set_ylabel(
            "Player effect on conditional tempo variability (% ΔCV)",
            fontsize=11,
        )

        quadrant_labels = (
            (0.98, 0.98, "High Tempo, High Variability", "right", "top"),
            (0.98, 0.02, "High Tempo, Low Variability", "right", "bottom"),
            (0.02, 0.98, "Low Tempo, High Variability", "left", "top"),
            (0.02, 0.02, "Low Tempo, Low Variability", "left", "bottom"),
        )
        for (
            x,
            y,
            label,
            horizontal_alignment,
            vertical_alignment,
        ) in quadrant_labels:
            ax.text(
                x,
                y,
                label,
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                horizontalalignment=horizontal_alignment,
                verticalalignment=vertical_alignment,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=4,
            )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
            ncol=2,
        )

    return fig, ax


def plot_player_archetypes_directional_support(
    transformed_overlap: pl.DataFrame,
    *,
    probability_threshold: float = 0.90,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> tuple[Figure, Axes]:
    """Highlight players with directional support for their quadrant.

    Each effect must exceed ``probability_threshold`` in the direction shown
    by its plotted posterior mean. Because ``shape_p_gt_zero`` has already
    been transformed to represent ``P(ΔCV > 0)``, the same quadrant-aware
    calculation applies to both axes.
    """
    if (
        isinstance(probability_threshold, bool)
        or not isinstance(probability_threshold, (int, float))
        or not 0.5 < probability_threshold < 1
    ):
        raise ValueError("probability_threshold must be between 0.5 and 1.")

    probability_columns = {"mu_p_gt_zero", "shape_p_gt_zero"}
    missing_columns = sorted(
        probability_columns.difference(transformed_overlap.columns)
    )
    if missing_columns:
        raise KeyError(
            "transformed_overlap is missing required columns: "
            f"{missing_columns}."
        )

    lower_threshold = 1 - probability_threshold
    directionally_supported = transformed_overlap.with_columns(
        pl.when(pl.col("mu_posterior_mean") > 0)
        .then(pl.col("mu_p_gt_zero") > probability_threshold)
        .when(pl.col("mu_posterior_mean") < 0)
        .then(pl.col("mu_p_gt_zero") < lower_threshold)
        .otherwise(False)
        .alias("_mu_direction_supported"),
        pl.when(pl.col("shape_posterior_mean") > 0)
        .then(pl.col("shape_p_gt_zero") > probability_threshold)
        .when(pl.col("shape_posterior_mean") < 0)
        .then(pl.col("shape_p_gt_zero") < lower_threshold)
        .otherwise(False)
        .alias("_cv_direction_supported"),
    ).with_columns(
        pl.when(pl.col("_mu_direction_supported"))
        .then(pl.lit("Credible"))
        .otherwise(pl.lit("Not Credible"))
        .alias("mu_credible"),
        pl.when(pl.col("_cv_direction_supported"))
        .then(pl.lit("Credible"))
        .otherwise(pl.lit("Not Credible"))
        .alias("shape_credible"),
    )

    threshold_percent = probability_threshold * 100
    return plot_player_archetypes(
        directionally_supported,
        figsize=figsize,
        highlight_label=(
            f">{threshold_percent:g}% directional support for both"
        ),
    )


def generate_player_archetypes_figure(
    *,
    posterior_path: str | Path = DEFAULT_POSTERIOR_PATH,
    model_data_path: str | Path = DEFAULT_MODEL_DATA_PATH,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
    filename: str = DEFAULT_FIGURE_NAME,
    hdi_prob: float = 0.95,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Load results and save the archetype plot as PDF and PNG."""
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
    finally:
        idata.close()

    transformed = transform_overlap_to_percentage_effects(overlap)
    fig, _ = plot_player_archetypes(transformed, figsize=figsize)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / filename
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)

    return pdf_path, png_path


def generate_directional_support_archetypes_figure(
    *,
    posterior_path: str | Path = DEFAULT_POSTERIOR_PATH,
    model_data_path: str | Path = DEFAULT_MODEL_DATA_PATH,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
    filename: str = DEFAULT_DIRECTIONAL_FIGURE_NAME,
    hdi_prob: float = 0.95,
    probability_threshold: float = 0.90,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save the quadrant-aware directional-support plot as PDF and PNG."""
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
    finally:
        idata.close()

    transformed = transform_overlap_to_percentage_effects(overlap)
    fig, _ = plot_player_archetypes_directional_support(
        transformed,
        probability_threshold=probability_threshold,
        figsize=figsize,
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / filename
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)

    return pdf_path, png_path


def main() -> None:
    """Generate both player tempo-archetype figures for the paper."""
    generate_player_archetypes_figure()
    generate_directional_support_archetypes_figure()


if __name__ == "__main__":
    main()
