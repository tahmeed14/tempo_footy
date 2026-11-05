"""Test model utility functions."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import xarray as xr

from tempoctrl.model.utils import (
    build_player_effects_table,
    discover_archetypes,
    filter_archetype_players,
    summarize_archetype_overlap,
    summarize_player_effects,
)


def _idata() -> SimpleNamespace:
    """Build two players with known posterior draws for summary tests.
    """
    effect = xr.DataArray(
        [
            [[-2.0, 1.0], [-1.0, 2.0], [1.0, 3.0], [2.0, 4.0]],
        ],
        dims=("chain", "draw", "player_id__factor_dim"),
        coords={"player_id__factor_dim": [10, 20]},
    )
    return SimpleNamespace(posterior=xr.Dataset({"alpha_1|player_id": effect}))


def test_summarize_player_effects_returns_ranked_statistics() -> None:
    """Recover known player means, deviations, and sign probabilities.
    """
    ranking = summarize_player_effects(_idata(), hdi_prob=0.5)

    assert ranking.columns.tolist() == [
        "player_id",
        "posterior_mean",
        "posterior_sd",
        "hdi_lower",
        "hdi_upper",
        "p_gt_zero",
        "shape_credible",
    ]
    assert ranking["player_id"].tolist() == [10, 20]
    assert ranking["posterior_mean"].tolist() == pytest.approx([0.0, 2.5])
    assert ranking["posterior_sd"].tolist() == pytest.approx(
        [np.sqrt(2.5), np.sqrt(1.25)]
    )
    assert ranking["p_gt_zero"].tolist() == pytest.approx([0.5, 1.0])
    assert (ranking["hdi_lower"] <= ranking["hdi_upper"]).all()


def test_summarize_player_effects_labels_credible_intervals() -> None:
    """Distinguish an interval spanning zero from a positive interval.
    """
    ranking = summarize_player_effects(_idata())

    assert ranking["shape_credible"].tolist() == [
        "Not Credible",
        "Credible",
    ]


def test_player_effects_use_mu_credibility_for_mu_effect() -> None:
    """Label location-effect credibility independently of shape effects.
    """
    idata = _idata()
    idata.posterior = idata.posterior.rename_vars(
        {"alpha_1|player_id": "1|player_id"}
    )

    ranking = summarize_player_effects(idata, variable="1|player_id")

    assert "mu_credible" in ranking.columns


def test_summarize_player_effects_rejects_unknown_variable() -> None:
    """Reject a posterior variable absent from the synthetic dataset."""
    with pytest.raises(KeyError, match="was not found"):
        summarize_player_effects(_idata(), variable="missing")


@pytest.mark.parametrize("hdi_prob", [0.0, 1.0, -0.1, 1.1])
def test_summarize_player_effects_rejects_invalid_hdi_probability(
    hdi_prob: float,
) -> None:
    """Reject interval probabilities at or outside the open unit range.
    """
    with pytest.raises(ValueError, match="between 0 and 1"):
        summarize_player_effects(_idata(), hdi_prob=hdi_prob)


def test_build_player_effects_table_adds_metadata_and_model_row_counts() -> (
    None
):
    """Join player metadata and known row counts without changing ranks.
    """
    model_df = pl.DataFrame(
        {
            "player_id": pl.Series([10, 10, 20], dtype=pl.Int32),
            "playername": ["Low", "Low", "High"],
            "teamname": ["A", "A", "B"],
            "response": [1.0, 2.0, 3.0],
        }
    )

    table = build_player_effects_table(_idata(), model_df, hdi_prob=0.5)

    assert table.columns == [
        "rank",
        "player_id",
        "posterior_mean",
        "posterior_sd",
        "hdi_lower",
        "hdi_upper",
        "p_gt_zero",
        "shape_credible",
        "playername",
        "teamname",
        "count",
    ]
    assert table["player_id"].dtype == pl.Int32
    assert table.select(
        "rank", "player_id", "playername", "teamname", "count"
    ).to_dicts() == [
        {
            "rank": 1,
            "player_id": 20,
            "playername": "High",
            "teamname": "B",
            "count": 1,
        },
        {
            "rank": 2,
            "player_id": 10,
            "playername": "Low",
            "teamname": "A",
            "count": 2,
        },
    ]


def test_build_player_effects_table_rejects_missing_metadata_columns() -> None:
    """Reject metadata missing the team name needed for player labels.
    """
    with pytest.raises(KeyError, match="teamname"):
        build_player_effects_table(
            _idata(),
            pl.DataFrame({"player_id": [10], "playername": ["Low"]}),
        )


def _effect_table(values: list[float]) -> pl.DataFrame:
    """Build ranked effects for four players with stable metadata."""
    return (
        pl.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "posterior_mean": values,
                "posterior_sd": [0.1] * 4,
                "hdi_lower": [value - 0.2 for value in values],
                "hdi_upper": [value + 0.2 for value in values],
                "p_gt_zero": [0.9] * 4,
                "playername": ["A", "B", "C", "D"],
                "teamname": ["T1", "T1", "T2", "T2"],
                "count": [10, 20, 30, 40],
            }
        )
        .sort(["posterior_mean", "player_id"], descending=[True, False])
        .with_row_index("rank", offset=1)
    )


def test_discover_archetypes_filters_rank_ranges_and_intersection() -> None:
    """Select explicit player ranks and their shared archetype members.
    """
    mu_matches, shape_matches, both_matches = discover_archetypes(
        _effect_table([0.9, 0.7, 0.4, 0.1]),
        _effect_table([0.2, 0.8, 0.6, 0.3]),
        mu_range=(1, 2),
        shape_range=(1, 2),
        option="rank",
    )

    assert mu_matches.select("player_id", "rank").rows() == [(1, 1), (2, 2)]
    assert shape_matches.select("player_id", "rank").rows() == [(2, 1), (3, 2)]
    assert both_matches["player_id"].to_list() == [2]
    assert both_matches.columns[:4] == [
        "player_id",
        "playername",
        "teamname",
        "count",
    ]
    assert both_matches.select("mu_rank", "shape_rank").row(0) == (2, 1)
    assert "mu_posterior_mean" in both_matches.columns
    assert "shape_posterior_mean" in both_matches.columns


def test_discover_archetypes_filters_value_ranges_and_intersection() -> None:
    """Select explicit effect ranges and their shared player members."""
    mu_matches, shape_matches, both_matches = discover_archetypes(
        _effect_table([0.9, 0.7, 0.4, 0.1]),
        _effect_table([0.2, 0.8, 0.6, 0.3]),
        mu_range=(0.35, 0.75),
        shape_range=(0.25, 0.65),
        option="value",
    )

    assert mu_matches["player_id"].to_list() == [2, 3]
    assert shape_matches["player_id"].to_list() == [3, 4]
    assert both_matches["player_id"].to_list() == [3]


def test_discover_archetypes_retains_credibility_columns_in_overlap() -> None:
    """Preserve both effect credibility labels through archetype joins.
    """
    mu_table = _effect_table([0.9, 0.7, 0.4, 0.1]).with_columns(
        pl.lit("Credible").alias("mu_credible")
    )
    shape_table = _effect_table([0.2, 0.8, 0.6, 0.3]).with_columns(
        pl.lit("Credible").alias("shape_credible")
    )

    _, _, overlap = discover_archetypes(
        mu_table,
        shape_table,
        mu_range=(1, 4),
        shape_range=(1, 4),
    )

    assert "mu_credible" in overlap.columns
    assert "shape_credible" in overlap.columns


def test_summarize_archetype_overlap_counts_categories_and_evidence() -> None:
    """Count known quadrant members and their supporting evidence."""
    overlap = pl.DataFrame(
        {
            "mu_posterior_mean": [1.0, 1.0, -1.0, -1.0],
            "shape_posterior_mean": [-1.0, 1.0, -1.0, 1.0],
            "mu_credible": [
                "Credible",
                "Not Credible",
                "Credible",
                "Not Credible",
            ],
            "shape_credible": [
                "Credible",
                "Credible",
                "Not Credible",
                "Not Credible",
            ],
            "mu_p_gt_zero": [0.95, 0.91, 0.09, 0.20],
            "shape_p_gt_zero": [0.05, 0.95, 0.05, 0.91],
        }
    )

    summary = summarize_archetype_overlap(overlap)

    assert summary["category"].to_list() == [
        "High Tempo (+mu), High Variance (-alpha)",
        "High Tempo (+mu), Low Variance (+alpha)",
        "Low Tempo (-mu), High Variance (-alpha)",
        "Low Tempo (-mu), Low Variance (+alpha)",
    ]
    assert summary["number_of_players"].to_list() == [1, 1, 1, 1]
    assert summary["number_of_players_credible_mu"].to_list() == [1, 0, 1, 0]
    assert summary["percent_of_players_credible_mu"].to_list() == [
        100.0,
        0.0,
        100.0,
        0.0,
    ]
    assert summary["number_of_players_credible_alpha"].to_list() == [
        1,
        1,
        0,
        0,
    ]
    assert summary["number_of_players_probability_supported_mu"].to_list() == [
        1,
        1,
        1,
        0,
    ]
    assert summary[
        "number_of_players_probability_supported_alpha"
    ].to_list() == [
        1,
        1,
        1,
        1,
    ]
    assert summary["not_strict_num_players_credible_both"].to_list() == [
        1,
        0,
        0,
        0,
    ]
    assert summary["not_strict_percent_players_credible_both"].to_list() == [
        100.0,
        0.0,
        0.0,
        0.0,
    ]
    assert summary["strict_num_players_credible_both"].to_list() == [
        1,
        0,
        0,
        0,
    ]
    assert summary["strict_percent_players_credible_both"].to_list() == [
        100.0,
        0.0,
        0.0,
        0.0,
    ]


def test_archetype_overlap_supports_custom_probability_thresholds() -> None:
    """Apply a custom sign-probability threshold to quadrant evidence.
    """
    overlap = pl.DataFrame(
        {
            "mu_posterior_mean": [1.0],
            "shape_posterior_mean": [-1.0],
            "mu_credible": ["Credible"],
            "shape_credible": ["Credible"],
            "mu_p_gt_zero": [0.85],
            "shape_p_gt_zero": [0.15],
        }
    )

    default_summary = summarize_archetype_overlap(overlap)
    summary = summarize_archetype_overlap(
        overlap,
        positive_probability_threshold=0.80,
        negative_probability_threshold=0.20,
    )

    assert (
        summary.row(0, named=True)[
            "number_of_players_probability_supported_mu"
        ]
        == 1
    )
    assert (
        summary.row(0, named=True)[
            "number_of_players_probability_supported_alpha"
        ]
        == 1
    )
    assert (
        default_summary.row(0, named=True)[
            "not_strict_num_players_credible_both"
        ]
        == 1
    )
    assert (
        default_summary.row(0, named=True)["strict_num_players_credible_both"]
        == 0
    )
    assert (
        default_summary.row(0, named=True)[
            "not_strict_percent_players_credible_both"
        ]
        == 100.0
    )
    assert (
        default_summary.row(0, named=True)[
            "strict_percent_players_credible_both"
        ]
        == 0.0
    )
    assert summary.row(0, named=True)["strict_num_players_credible_both"] == 1
    assert (
        summary.row(0, named=True)["strict_percent_players_credible_both"]
        == 100.0
    )


@pytest.mark.parametrize(
    (
        "tempo",
        "variance",
        "mu",
        "alpha",
        "mu_probability",
        "alpha_probability",
        "category",
    ),
    [
        (
            "high",
            "high",
            1.0,
            -1.0,
            0.95,
            0.05,
            "High Tempo (+mu), High Variance (-alpha)",
        ),
        (
            "high",
            "low",
            1.0,
            1.0,
            0.95,
            0.95,
            "High Tempo (+mu), Low Variance (+alpha)",
        ),
        (
            "low",
            "high",
            -1.0,
            -1.0,
            0.05,
            0.05,
            "Low Tempo (-mu), High Variance (-alpha)",
        ),
        (
            "low",
            "low",
            -1.0,
            1.0,
            0.05,
            0.95,
            "Low Tempo (-mu), Low Variance (+alpha)",
        ),
    ],
)
def test_filter_archetype_players_filters_each_category(
    tempo: str,
    variance: str,
    mu: float,
    alpha: float,
    mu_probability: float,
    alpha_probability: float,
    category: str,
) -> None:
    """Select only the explicitly requested archetype and its labels."""
    overlap = pl.DataFrame(
        {
            "player_id": [1],
            "playername": ["Player One"],
            "mu_posterior_mean": [mu],
            "shape_posterior_mean": [alpha],
            "mu_credible": ["Credible"],
            "shape_credible": ["Credible"],
            "mu_p_gt_zero": [mu_probability],
            "shape_p_gt_zero": [alpha_probability],
        }
    )

    players = filter_archetype_players(
        overlap,
        tempo=tempo,
        variance=variance,
    )

    assert players.columns == ["player_archetype", *overlap.columns]
    assert players["player_archetype"].to_list() == [category]
    assert players["playername"].to_list() == ["Player One"]


def test_filter_archetype_players_credibility_flags_disable_filters() -> None:
    """Include unsupported members only when credibility filters are
    off.
    """
    overlap = pl.DataFrame(
        {
            "playername": ["Not HDI credible", "Low probability"],
            "mu_posterior_mean": [1.0, 1.0],
            "shape_posterior_mean": [-1.0, -1.0],
            "mu_credible": ["Not Credible", "Credible"],
            "shape_credible": ["Not Credible", "Credible"],
            "mu_p_gt_zero": [0.95, 0.80],
            "shape_p_gt_zero": [0.05, 0.20],
        }
    )

    default_players = filter_archetype_players(
        overlap,
        tempo="high",
        variance="high",
    )
    players_without_credibility_filters = filter_archetype_players(
        overlap,
        tempo="high",
        variance="high",
        mu_credible=False,
        alpha_credible=False,
    )

    assert default_players.is_empty()
    assert players_without_credibility_filters["playername"].to_list() == [
        "Not HDI credible"
    ]


def test_discover_archetypes_validates_option_and_range_types() -> None:
    """Reject unsupported filtering modes and malformed effect ranges.
    """
    table = _effect_table([0.9, 0.7, 0.4, 0.1])

    with pytest.raises(ValueError, match="option"):
        discover_archetypes(
            table,
            table,
            mu_range=(1, 2),
            shape_range=(1, 2),
            option="percentile",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="integers"):
        discover_archetypes(
            table,
            table,
            mu_range=(1.0, 2.0),
            shape_range=(1, 2),
            option="rank",
        )
