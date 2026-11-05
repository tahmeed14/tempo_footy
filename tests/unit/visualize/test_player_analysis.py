"""Test player-analysis visualizations."""

import matplotlib.pyplot as plt
import polars as pl
import pytest

from tempoctrl.visualize.player_analysis import plot_archetypes


def test_plot_archetypes_plots_players_present_in_both_tables() -> None:
    """Plot shared players while excluding those absent from either
    table.
    """
    mu_table = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "posterior_mean": [-0.2, 0.1, 0.5],
        }
    )
    alpha_table = pl.DataFrame(
        {
            "player_id": [2, 3, 4],
            "posterior_mean": [0.4, -0.3, 0.8],
        }
    )

    ax = plot_archetypes(mu_table, alpha_table)

    plotted_points = {
        tuple(point) for point in ax.collections[0].get_offsets()
    }
    assert plotted_points == {(0.1, 0.4), (0.5, -0.3)}
    assert ax.get_xlabel() == "Mu posterior mean"
    assert ax.get_ylabel() == "Alpha posterior mean"
    assert ax.get_title() == "Player posterior point estimates"
    assert len(ax.lines) == 2
    assert {text.get_text() for text in ax.texts} == {
        "Faster tempo, lower variance",
        "Slower tempo, lower variance",
        "Slower tempo, higher variance",
        "Higher tempo, higher variance",
    }
    assert all(text.get_fontsize() == 9 for text in ax.texts)
    assert ax.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert all(line.get_visible() for line in ax.get_xgridlines())
    assert all(line.get_visible() for line in ax.get_ygridlines())

    plt.close(ax.figure)


def test_plot_archetypes_rejects_missing_columns() -> None:
    """Reject plot inputs missing required player or effect columns."""
    valid_table = pl.DataFrame({"player_id": [1], "posterior_mean": [0.1]})

    with pytest.raises(KeyError, match="posterior_mean"):
        plot_archetypes(
            valid_table,
            pl.DataFrame({"player_id": [1]}),
        )


def test_plot_archetypes_rejects_no_common_players() -> None:
    """Reject effect tables without any shared plottable players."""
    mu_table = pl.DataFrame({"player_id": [1], "posterior_mean": [0.1]})
    alpha_table = pl.DataFrame({"player_id": [2], "posterior_mean": [0.2]})

    with pytest.raises(ValueError, match="no plottable players in common"):
        plot_archetypes(mu_table, alpha_table)
