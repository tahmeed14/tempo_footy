"""Test archetype-figure reproduction."""

import matplotlib.pyplot as plt
import polars as pl

from tempoctrl.reproduce_archetypes import (
    plot_player_archetypes,
    plot_player_archetypes_directional_support,
)


def test_plot_player_archetypes_highlights_players_and_labels_quadrants() -> (
    None
):
    """Highlight known credible players and label all four quadrants."""
    overlap = pl.DataFrame(
        {
            "mu_posterior_mean": [10.0, -5.0, 3.0, -2.0],
            "shape_posterior_mean": [4.0, 2.0, -3.0, -1.0],
            "mu_credible": [
                "Credible",
                "Credible",
                "Credible",
                "Not Credible",
            ],
            "shape_credible": [
                "Credible",
                "Not Credible",
                "Credible",
                "Credible",
            ],
        }
    )

    fig, ax = plot_player_archetypes(overlap)

    assert ax.get_title() == "Player Archetypes for Tempo"
    assert len(ax.collections) == 2
    assert [
        len(collection.get_offsets()) for collection in ax.collections
    ] == [2, 2]
    assert [text.get_text() for text in ax.texts] == [
        "High Tempo, High Variability",
        "High Tempo, Low Variability",
        "Low Tempo, High Variability",
        "Low Tempo, Low Variability",
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Other players (n=2)",
        "Credible for both (n=2)",
    ]

    plt.close(fig)


def test_directional_support_is_evaluated_for_each_quadrant() -> None:
    """Use each quadrant sign when applying directional support cutoffs.
    """
    overlap = pl.DataFrame(
        {
            "mu_posterior_mean": [5.0, 5.0, -5.0, -5.0, 5.0],
            "shape_posterior_mean": [5.0, -5.0, 5.0, -5.0, 5.0],
            "mu_p_gt_zero": [0.95, 0.95, 0.05, 0.05, 0.80],
            "shape_p_gt_zero": [0.95, 0.05, 0.95, 0.05, 0.95],
            "mu_credible": ["Not Credible"] * 5,
            "shape_credible": ["Not Credible"] * 5,
        }
    )

    fig, ax = plot_player_archetypes_directional_support(overlap)

    assert [
        len(collection.get_offsets()) for collection in ax.collections
    ] == [1, 4]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Other players (n=1)",
        ">90% directional support for both (n=4)",
    ]

    plt.close(fig)
