"""Test random-effects table reproduction."""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pytest

from tempoctrl.reproduce_random_effects_table import (
    ALPHA_PLAYER_COLUMNS,
    GROUP_SD_COLUMNS,
    MU_OPPONENT_COLUMNS,
    MU_PLAYER_COLUMNS,
    generate_alpha_player_random_effects_table,
    generate_mu_opponent_random_effects_table,
    generate_mu_player_random_effects_table,
    generate_random_effect_standard_deviations_table,
)


def _posterior() -> az.InferenceData:
    """Build synthetic posterior draws with known parameter values."""
    rng = np.random.default_rng(8)
    return az.from_dict(
        {
            "posterior": {
                "Intercept": rng.normal(2, 0.1, (2, 50)),
                "1|player_id_sigma": rng.exponential(0.2, (2, 50)),
                "1|player_id": rng.normal(
                    np.array([0.2, -0.2]), 0.05, (2, 50, 2)
                ),
                "alpha_1|player_id_sigma": rng.exponential(0.2, (2, 50)),
                "alpha_1|player_id": rng.normal(
                    np.array([0.3, -0.3]), 0.05, (2, 50, 2)
                ),
                "1|opponent_id_sigma": rng.exponential(0.2, (2, 50)),
                "1|opponent_id": rng.normal(
                    np.array([0.1, -0.1]), 0.05, (2, 50, 2)
                ),
            }
        },
        coords={
            "player_id__factor_dim": ["1", "2"],
            "opponent_id__factor_dim": ["10", "20"],
        },
        dims={
            "1|player_id": ["player_id__factor_dim"],
            "alpha_1|player_id": ["player_id__factor_dim"],
            "1|opponent_id": ["opponent_id__factor_dim"],
        },
    )


def _metadata(path: Path, *, ambiguous: bool = False) -> None:
    """Write explicit player metadata into a temporary Parquet file."""
    player_names = ["One", "One", "Two", "Two"]
    if ambiguous:
        player_names[1] = "Different One"
    pl.DataFrame(
        {
            "game_id": [1, 2, 1, 2],
            "player_id": [1, 1, 2, 2],
            "player_name": player_names,
            "team_id": [10, 10, 20, 20],
            "team_name": ["Team A", "Team A", "Team B", "Team B"],
        }
    ).write_parquet(path)


def test_random_effect_generators_extract_join_transform_and_rank(
    tmp_path: Path,
) -> None:
    """Recover known player and opponent effects, metadata, and ranks.
    """
    metadata_path = tmp_path / "lookup.parquet"
    _metadata(metadata_path)
    idata = _posterior()

    mu_players = generate_mu_player_random_effects_table(
        idata, metadata_lookup_path=metadata_path, output_path=None
    )
    alpha_players = generate_alpha_player_random_effects_table(
        idata, metadata_lookup_path=metadata_path, output_path=None
    )
    mu_opponents = generate_mu_opponent_random_effects_table(
        idata, metadata_lookup_path=metadata_path, output_path=None
    )
    group_sds = generate_random_effect_standard_deviations_table(
        idata,
        output_path=None,
    )

    assert mu_players.columns == list(MU_PLAYER_COLUMNS)
    assert alpha_players.columns == list(ALPHA_PLAYER_COLUMNS)
    assert mu_opponents.columns == list(MU_OPPONENT_COLUMNS)
    assert group_sds.columns == list(GROUP_SD_COLUMNS)
    assert (
        mu_players.height == alpha_players.height == mu_opponents.height == 2
    )
    assert mu_players["playername"].to_list() == ["One", "Two"]
    assert mu_opponents["opponent_teamname"].to_list() == ["Team A", "Team B"]
    assert np.allclose(mu_players["p_gt_zero"] + mu_players["p_lt_zero"], 1)
    assert np.allclose(
        alpha_players["conditional_cv_ratio_hdi_lower"],
        np.exp(-0.5 * alpha_players["hdi_upper"]),
    )
    assert np.allclose(
        mu_players["expected_tempo_pct_change"],
        100 * (np.exp(mu_players["estimate"]) - 1),
    )
    assert np.allclose(
        alpha_players["conditional_cv_pct_change"],
        100 * (np.exp(-0.5 * alpha_players["estimate"]) - 1),
    )
    assert "group_sd_estimate" not in mu_players.columns
    assert "group_sd_estimate" not in alpha_players.columns
    assert "group_sd_estimate" not in mu_opponents.columns
    assert group_sds["group_sd_variable"].to_list() == [
        "1|player_id_sigma",
        "alpha_1|player_id_sigma",
        "1|opponent_id_sigma",
    ]
    assert group_sds.row(0, named=True)["estimate"] == pytest.approx(
        idata.posterior["1|player_id_sigma"].mean().item()
    )
    assert mu_players["rank"].to_list() == [1, 2]
    assert alpha_players["shape_rank"].to_list() == [1, 2]
    assert alpha_players.sort("variability_rank")["player_id"].to_list() == [
        2,
        1,
    ]


def test_player_metadata_ambiguity_is_rejected(tmp_path: Path) -> None:
    """Reject conflicting metadata rather than duplicating player
    effects.
    """
    metadata_path = tmp_path / "ambiguous.parquet"
    _metadata(metadata_path, ambiguous=True)

    with pytest.raises(ValueError, match="multiple names or teams"):
        generate_mu_player_random_effects_table(
            _posterior(),
            metadata_lookup_path=metadata_path,
            output_path=None,
        )
