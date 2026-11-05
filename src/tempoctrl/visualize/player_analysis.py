"""Player-level posterior visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes


def plot_archetypes(
    mu_table: pl.DataFrame,
    alpha_table: pl.DataFrame,
    *,
    player_id_column: str = "player_id",
) -> Axes:
    """Plot player mu and alpha posterior point estimates against each other.

    The inputs should be tables returned by
    :func:`tempoctrl.model.utils.build_player_effects_table`, with ``mu_table``
    built from ``"1|player_id"`` and ``alpha_table`` built from
    ``"alpha_1|player_id"``. Only players present in both tables are plotted.

    Parameters
    ----------
    mu_table
        Player-level posterior summary for the mu effect.
    alpha_table
        Player-level posterior summary for the alpha effect.
    player_id_column
        Player identifier used to align the two tables.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the scatter plot.

    """
    _validate_effect_table(mu_table, "mu_table", player_id_column)
    _validate_effect_table(alpha_table, "alpha_table", player_id_column)

    points = (
        mu_table.select(
            player_id_column,
            pl.col("posterior_mean").alias("mu_estimate"),
        )
        .join(
            alpha_table.select(
                player_id_column,
                pl.col("posterior_mean").alias("alpha_estimate"),
            ),
            on=player_id_column,
            how="inner",
        )
        .drop_nulls(["mu_estimate", "alpha_estimate"])
    )
    if points.is_empty():
        raise ValueError(
            "mu_table and alpha_table have no plottable players in common."
        )

    with plt.style.context("default"):
        _, ax = plt.subplots(figsize=(10, 6), dpi=100)
        sns.scatterplot(
            x=points["mu_estimate"].to_numpy(),
            y=points["alpha_estimate"].to_numpy(),
            color="skyblue",
            edgecolor="white",
            alpha=0.7,
            ax=ax,
        )

        ax.axvline(0, color="gray", linewidth=1)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlabel("Mu posterior mean", fontsize=12)
        ax.set_ylabel("Alpha posterior mean", fontsize=12)
        ax.set_title(
            "Player posterior point estimates",
            fontsize="large",
            pad=20,
        )

        quadrant_labels = (
            (0.98, 0.98, "Faster tempo, lower variance", "right", "top"),
            (0.02, 0.98, "Slower tempo, lower variance", "left", "top"),
            (0.02, 0.02, "Slower tempo, higher variance", "left", "bottom"),
            (0.98, 0.02, "Higher tempo, higher variance", "right", "bottom"),
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
                fontsize=9,
                horizontalalignment=horizontal_alignment,
                verticalalignment=vertical_alignment,
            )

    return ax


def _validate_effect_table(
    table: pl.DataFrame,
    table_name: str,
    player_id_column: str,
) -> None:
    if not isinstance(table, pl.DataFrame):
        raise TypeError(f"{table_name} must be a Polars DataFrame.")

    required_columns = [player_id_column, "posterior_mean"]
    missing_columns = [
        column for column in required_columns if column not in table.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{table_name} is missing required columns: {missing_columns}."
        )
