"""Test distribution-figure reproduction."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tempoctrl.reproduce_distribution_visuals import (
    _extract_player_random_effect_draws,
    _prepare_player_effect_plot_data,
    plot_top_bottom_alpha_player_effects,
    plot_top_bottom_mu_player_effects,
)


def _posterior() -> az.InferenceData:
    """Build synthetic posterior draws with known parameter values."""
    offsets = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    alpha_offsets = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    sample_noise = np.linspace(-0.02, 0.02, 40).reshape(2, 20, 1)
    return az.from_dict(
        {
            "posterior": {
                "1|player_id_sigma": np.ones((2, 20)),
                "1|player_id": offsets + sample_noise,
                "alpha_1|player_id_sigma": np.ones((2, 20)),
                "alpha_1|player_id": alpha_offsets + sample_noise,
            }
        },
        coords={"player_id__factor_dim": ["1", "2", "3", "4", "5"]},
        dims={
            "1|player_id": ["player_id__factor_dim"],
            "alpha_1|player_id": ["player_id__factor_dim"],
        },
    )


def _metadata(path: Path, *, ambiguous: bool = False) -> None:
    """Write explicit player metadata into a temporary Parquet file."""
    names = ["One", "Two", "Three", "Four", "Five"]
    teams = ["A", "B", "C", "D", "E"]
    player_ids = [1, 2, 3, 4, 5]
    if ambiguous:
        player_ids.append(1)
        names.append("Different One")
        teams.append("A")
    pd.DataFrame(
        {"player_id": player_ids, "player_name": names, "team_name": teams}
    ).to_parquet(path)


def test_extracts_effects_without_selecting_sigma_hyperparameters() -> None:
    """Select player effects while excluding similarly named scale
    terms.
    """
    idata = _posterior()

    mu_name, player_ids, mu_draws = _extract_player_random_effect_draws(
        idata, "mu"
    )
    alpha_name, _, alpha_draws = _extract_player_random_effect_draws(
        idata, "alpha"
    )

    assert mu_name == "1|player_id"
    assert alpha_name == "alpha_1|player_id"
    assert player_ids.tolist() == ["1", "2", "3", "4", "5"]
    assert mu_draws.shape == alpha_draws.shape == (40, 5)


def test_selection_ranking_transform_and_alpha_display_order(
    tmp_path: Path,
) -> None:
    """Preserve known player rankings and transformed shape display
    order.
    """
    metadata_path = tmp_path / "players.parquet"
    _metadata(metadata_path)

    _, mu_data, mu_draws = _prepare_player_effect_plot_data(
        _posterior(),
        component="mu",
        metadata_path=metadata_path,
        top_n=2,
        bottom_n=2,
    )
    _, alpha_data, alpha_draws = _prepare_player_effect_plot_data(
        _posterior(),
        component="alpha",
        metadata_path=metadata_path,
        top_n=2,
        bottom_n=2,
    )

    assert mu_data["player_id"].tolist() == ["1", "2", "4", "5"]
    assert alpha_data["player_id"].tolist() == ["5", "4", "2", "1"]
    assert len(mu_data) == len(alpha_data) == 4
    assert mu_data["player_id"].is_unique
    assert alpha_data["player_id"].is_unique
    assert np.allclose(
        mu_draws[:, 0], 100 * (np.exp(-0.4 + np.linspace(-0.02, 0.02, 40)) - 1)
    )
    assert np.allclose(
        alpha_draws[:, 0],
        100 * (np.exp(-0.5 * (0.8 + np.linspace(-0.02, 0.02, 40))) - 1),
    )
    assert alpha_data["transformed_posterior_mean"].is_monotonic_increasing


def test_plots_save_and_metadata_ambiguity_is_rejected(tmp_path: Path) -> None:
    """Save synthetic effect plots and reject ambiguous player metadata.
    """
    metadata_path = tmp_path / "players.parquet"
    _metadata(metadata_path)
    mu_output = tmp_path / "mu.png"
    alpha_output = tmp_path / "alpha.pdf"

    mu_figure = plot_top_bottom_mu_player_effects(
        _posterior(),
        metadata_path=metadata_path,
        top_n=2,
        bottom_n=2,
        output_path=mu_output,
    )
    alpha_figure = plot_top_bottom_alpha_player_effects(
        _posterior(),
        metadata_path=metadata_path,
        top_n=2,
        bottom_n=2,
        output_path=alpha_output,
    )

    assert mu_output.is_file()
    assert alpha_output.is_file()
    assert len(mu_figure.axes[0].get_yticklabels()) == 4
    assert len(alpha_figure.axes[0].get_yticklabels()) == 4
    plt.close(mu_figure)
    plt.close(alpha_figure)

    _metadata(metadata_path, ambiguous=True)
    with pytest.raises(ValueError, match="multiple names or teams"):
        plot_top_bottom_mu_player_effects(
            _posterior(),
            metadata_path=metadata_path,
            top_n=2,
            bottom_n=2,
        )
