"""Reproduce posterior-distribution figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde

from tempoctrl.visualize.possessions import (
    plot_dev_players_to_pass,
    plot_dev_possession_movement,
    plot_dev_start_frame,
)

FIGSIZE_FULL = (7.0, 4.5)
FIGSIZE_WIDE = (7.0, 3.5)
FIGSIZE_SQUARE = (5.5, 5.5)
POSS_EXAMPLE = "src/tempoctrl/possession_example.parquet"
FIGURES_PATH = "paper/figures"
DEFAULT_POSTERIOR_PATH = Path("paper/results/tempo_gamma_posterior.nc")
DEFAULT_PLAYER_LOOKUP = Path(
    "data/curated/gradient_sports/metadata_lookup/player_game_lookup.parquet"
)
DEFAULT_MU_PLAYER_FIGURE = Path(
    "paper/figures/top_bottom_mu_player_random_effects.png"
)
DEFAULT_ALPHA_PLAYER_FIGURE = Path(
    "paper/figures/top_bottom_alpha_player_random_effects.png"
)


def _load_idata(idata_or_path: Any) -> Any:
    """Return an InferenceData-like object with a posterior group."""
    if isinstance(idata_or_path, (str, Path)):
        path = Path(idata_or_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Posterior NetCDF file was not found: {path}."
            )
        idata = az.from_netcdf(path)
    else:
        idata = idata_or_path

    if getattr(idata, "posterior", None) is None:
        raise ValueError(
            "idata_or_path must contain an ArviZ posterior group."
        )
    return idata


def _find_player_random_effect(
    posterior: Any,
    component: Literal["mu", "alpha"],
) -> tuple[str, str]:
    """Identify one Bambi player deviation, excluding its SD hyperparameter."""
    level_dimension = "player_id__factor_dim"
    candidates: list[str] = []

    for name, values in posterior.data_vars.items():
        if (component == "alpha") != name.startswith("alpha_"):
            continue
        if "|player_id" not in name or name.endswith("_sigma"):
            continue
        non_sample_dimensions = [
            dimension
            for dimension in values.dims
            if dimension not in {"chain", "draw"}
        ]
        if non_sample_dimensions == [level_dimension]:
            candidates.append(name)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {component} player-level random effect; "
            f"found {candidates}. Posterior variables were "
            f"{list(posterior.data_vars)}."
        )
    return candidates[0], level_dimension


def _extract_player_random_effect_draws(
    idata_or_path: Any,
    component: Literal["mu", "alpha"],
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return variable name, player IDs, and flattened posterior draws."""
    idata = _load_idata(idata_or_path)
    variable_name, level_dimension = _find_player_random_effect(
        idata.posterior,
        component,
    )
    posterior = idata.posterior[variable_name]
    required_dimensions = {"chain", "draw", level_dimension}
    if set(posterior.dims) != required_dimensions:
        raise ValueError(
            f"{variable_name!r} must have dimensions {required_dimensions}; "
            f"found {posterior.dims}."
        )

    player_ids = np.asarray(posterior.coords[level_dimension].values).astype(
        str
    )
    if len(np.unique(player_ids)) != len(player_ids):
        raise ValueError("Posterior player coordinates contain duplicate IDs.")

    draws = posterior.transpose("chain", "draw", level_dimension).values
    draws = np.asarray(draws, dtype=float).reshape(-1, len(player_ids))
    if not np.isfinite(draws).all():
        raise ValueError(
            f"{variable_name!r} contains non-finite posterior draws."
        )
    return variable_name, player_ids, draws


def _build_player_lookup(metadata_path: str | Path) -> pd.DataFrame:
    """Build a one-row-per-player lookup, rejecting ambiguous metadata."""
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Player metadata lookup was not found: {path}."
        )

    metadata = pd.read_parquet(path)
    required_columns = ["player_id", "player_name", "team_name"]
    missing_columns = [
        column for column in required_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Player metadata is missing columns: {missing_columns}."
        )

    lookup = metadata.loc[:, required_columns].dropna().drop_duplicates()
    conflicts = lookup.groupby("player_id", dropna=False).agg(
        player_name_count=("player_name", "nunique"),
        team_name_count=("team_name", "nunique"),
    )
    conflicts = conflicts.loc[
        (conflicts["player_name_count"] > 1)
        | (conflicts["team_name_count"] > 1)
    ]
    if not conflicts.empty:
        raise ValueError(
            "Metadata contains player IDs mapped to multiple names or teams: "
            f"{conflicts.index.tolist()}."
        )

    lookup = lookup.drop_duplicates(subset="player_id").rename(
        columns={"player_name": "playername", "team_name": "teamname"}
    )
    lookup["player_id"] = lookup["player_id"].astype(str)
    if lookup["player_id"].duplicated().any():
        raise ValueError(
            "String-normalized metadata contains duplicate player IDs."
        )
    return lookup


def _select_top_bottom_players(
    player_ids: np.ndarray,
    raw_draws: np.ndarray,
    *,
    top_n: int,
    bottom_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select non-overlapping extremes by raw posterior mean."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer.")
    if (
        isinstance(bottom_n, bool)
        or not isinstance(bottom_n, int)
        or bottom_n < 0
    ):
        raise ValueError("bottom_n must be a non-negative integer.")
    if top_n + bottom_n == 0:
        raise ValueError("At least one player must be requested.")

    raw_means = raw_draws.mean(axis=0)
    order = np.lexsort((player_ids, raw_means))
    bottom_count = min(bottom_n, len(order))
    bottom_indices = order[:bottom_count]
    remaining = order[bottom_count:]
    top_count = min(top_n, len(remaining))
    top_indices = (
        remaining[-top_count:] if top_count else np.array([], dtype=int)
    )
    selected = np.concatenate([bottom_indices, top_indices])

    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("Top and bottom player selections overlap.")
    groups = np.array(
        ["bottom"] * len(bottom_indices) + ["top"] * len(top_indices)
    )
    return selected, groups


def _compute_density(draws: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Estimate a stable density, including for nearly constant draws."""
    draws = np.asarray(draws, dtype=float)
    spread = float(np.ptp(draws))
    scale = max(float(np.std(draws, ddof=1)), spread / 6, 1e-8)
    if scale <= 1e-7 * max(1.0, abs(float(np.mean(draws)))):
        bandwidth = max(float(np.ptp(grid)) / 200, 1e-6)
        z = (grid - float(np.mean(draws))) / bandwidth
        return np.exp(-0.5 * z**2) / (bandwidth * np.sqrt(2 * np.pi))

    try:
        density = gaussian_kde(draws)(grid)
    except (ValueError, np.linalg.LinAlgError):
        z = (grid - float(np.mean(draws))) / scale
        density = np.exp(-0.5 * z**2) / (scale * np.sqrt(2 * np.pi))
    return np.asarray(density)


def _posterior_hdi(draws: np.ndarray, hdi_prob: float) -> np.ndarray:
    """Calculate an ArviZ HDI across supported ArviZ API versions."""
    try:
        interval = az.hdi(draws, prob=hdi_prob)
    except TypeError:
        interval = az.hdi(draws, hdi_prob=hdi_prob)
    return np.asarray(interval, dtype=float).reshape(-1)


def _plot_posterior_ridges(
    plot_data: pd.DataFrame,
    transformed_draws: np.ndarray,
    *,
    hdi_prob: float,
    title: str,
    subtitle: str,
    xlabel: str,
    direction_label: str,
    lower_group_label: str,
    upper_group_label: str,
    output_path: str | Path | None,
) -> Figure:
    """Plot posterior ridges, transformed summaries, and a shared x-axis."""
    if not 0 < hdi_prob < 1:
        raise ValueError("hdi_prob must be strictly between 0 and 1.")
    if len(plot_data) != transformed_draws.shape[1]:
        raise ValueError("Plot metadata and posterior draws are misaligned.")

    finite_draws = transformed_draws.reshape(-1)
    lower, upper = np.quantile(finite_draws, [0.001, 0.999])
    lower = min(float(lower), 0.0)
    upper = max(float(upper), 0.0)
    span = upper - lower
    if span <= 0:
        span = max(abs(lower), 1.0)
    grid = np.linspace(lower - 0.08 * span, upper + 0.08 * span, 512)

    first_group = plot_data["selection_group"].iloc[0]
    first_group_count = int(
        (plot_data["selection_group"] == first_group).sum()
    )
    has_two_groups = plot_data["selection_group"].nunique() == 2
    y_positions = np.arange(len(plot_data), dtype=float)
    if has_two_groups:
        y_positions[first_group_count:] += 1.0

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    colors = {"bottom": "#4472A5", "top": "#B45A3C"}
    for index, row in plot_data.reset_index(drop=True).iterrows():
        y = y_positions[index]
        draws = transformed_draws[:, index]
        density = _compute_density(draws, grid)
        density_height = 0.58 * density / max(float(density.max()), 1e-12)
        color = colors[row["selection_group"]]
        ax.fill_between(
            grid,
            y,
            y + density_height,
            color=color,
            alpha=0.32,
            linewidth=0,
        )
        ax.plot(grid, y + density_height, color=color, linewidth=1.2)
        hdi_lower, hdi_upper = _posterior_hdi(draws, hdi_prob)
        posterior_mean = float(np.mean(draws))
        interval_y = y + 0.12
        ax.plot(
            [hdi_lower, hdi_upper],
            [interval_y, interval_y],
            color="#252525",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            posterior_mean,
            interval_y,
            s=24,
            color="#252525",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )

    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=0)
    if has_two_groups:
        separator_y = y_positions[first_group_count - 1] + 1.0
        ax.axhline(separator_y, color="#D0D0D0", linewidth=0.8)
        label_transform = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.99,
            separator_y - 0.12,
            lower_group_label,
            transform=label_transform,
            ha="right",
            va="top",
            fontsize=9,
            color="#666666",
        )
        ax.text(
            0.99,
            separator_y + 0.12,
            upper_group_label,
            transform=label_transform,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#666666",
        )

    labels = plot_data["playername"].astype(str).tolist()
    duplicated_labels = pd.Series(labels).duplicated(keep=False).to_numpy()
    labels = [
        f"{label} ({player_id})" if duplicate else label
        for label, player_id, duplicate in zip(
            labels,
            plot_data["player_id"],
            duplicated_labels,
            strict=True,
        )
    ]
    ax.set_yticks(y_positions, labels)
    ax.set_ylim(-0.25, y_positions[-1] + 0.85)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.975)
    ax.set_title(subtitle, fontsize=10.5, color="#555555", pad=14)
    ax.text(
        0.5,
        -0.11,
        direction_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.88, bottom=0.16)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def _prepare_player_effect_plot_data(
    idata_or_path: Any,
    *,
    component: Literal["mu", "alpha"],
    metadata_path: str | Path,
    top_n: int,
    bottom_n: int,
) -> tuple[str, pd.DataFrame, np.ndarray]:
    """Extract, select, label, transform, and naturally order player draws."""
    variable_name, player_ids, raw_draws = _extract_player_random_effect_draws(
        idata_or_path,
        component,
    )
    selected, groups = _select_top_bottom_players(
        player_ids,
        raw_draws,
        top_n=top_n,
        bottom_n=bottom_n,
    )
    selected_ids = player_ids[selected]
    selected_raw_draws = raw_draws[:, selected]
    if component == "mu":
        transformed_draws = 100 * (np.exp(selected_raw_draws) - 1)
    else:
        transformed_draws = 100 * (np.exp(-0.5 * selected_raw_draws) - 1)

    plot_data = pd.DataFrame(
        {
            "player_id": selected_ids,
            "raw_posterior_mean": selected_raw_draws.mean(axis=0),
            "transformed_posterior_mean": transformed_draws.mean(axis=0),
            "selection_group": groups,
        }
    )
    lookup = _build_player_lookup(metadata_path)
    original_rows = len(plot_data)
    plot_data = plot_data.merge(
        lookup,
        on="player_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(plot_data) != original_rows:
        raise ValueError("Metadata join changed the number of posterior rows.")
    missing = plot_data.loc[
        plot_data[["playername", "teamname"]].isna().any(axis=1),
        "player_id",
    ]
    if not missing.empty:
        raise ValueError(
            f"Metadata is missing for modeled player IDs: {missing.tolist()}."
        )

    semantic_group_order = (
        {"bottom": 0, "top": 1}
        if component == "mu"
        else {"top": 0, "bottom": 1}
    )
    group_order = (
        plot_data["selection_group"].map(semantic_group_order).to_numpy()
    )
    order = np.lexsort(
        (
            plot_data["player_id"].to_numpy(),
            plot_data["transformed_posterior_mean"].to_numpy(),
            group_order,
        )
    )
    plot_data = plot_data.iloc[order].reset_index(drop=True)
    transformed_draws = transformed_draws[:, order]
    return variable_name, plot_data, transformed_draws


def plot_top_bottom_mu_player_effects(
    idata_or_path: Any,
    metadata_path: str | Path = DEFAULT_PLAYER_LOOKUP,
    top_n: int = 5,
    bottom_n: int = 5,
    hdi_prob: float = 0.95,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot extreme player effects on conditional expected tempo."""
    _, plot_data, transformed_draws = _prepare_player_effect_plot_data(
        idata_or_path,
        component="mu",
        metadata_path=metadata_path,
        top_n=top_n,
        bottom_n=bottom_n,
    )
    return _plot_posterior_ridges(
        plot_data,
        transformed_draws,
        hdi_prob=hdi_prob,
        title="Player heterogeneity in expected ball-speed tempo",
        subtitle=(
            "Posterior distributions for the five highest and five lowest "
            "player random effects"
        ),
        xlabel="Player effect on expected tempo (%)",
        direction_label=(
            "Lower expected tempo ←    0    → Higher expected tempo"
        ),
        lower_group_label="Lowest expected-tempo effects",
        upper_group_label="Highest expected-tempo effects",
        output_path=output_path,
    )


def plot_top_bottom_alpha_player_effects(
    idata_or_path: Any,
    metadata_path: str | Path = DEFAULT_PLAYER_LOOKUP,
    top_n: int = 5,
    bottom_n: int = 5,
    hdi_prob: float = 0.95,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot extreme shape-model player effects on conditional Gamma CV."""
    _, plot_data, transformed_draws = _prepare_player_effect_plot_data(
        idata_or_path,
        component="alpha",
        metadata_path=metadata_path,
        top_n=top_n,
        bottom_n=bottom_n,
    )
    return _plot_posterior_ridges(
        plot_data,
        transformed_draws,
        hdi_prob=hdi_prob,
        title="Player heterogeneity in conditional tempo variability",
        subtitle=(
            "Posterior distributions for the five highest and five lowest "
            "shape-model player effects"
        ),
        xlabel="Player effect on conditional tempo variability (% CV)",
        direction_label="Lower variability ←    0    → Higher variability",
        lower_group_label="Lower conditional variability",
        upper_group_label="Higher conditional variability",
        output_path=output_path,
    )


def save_quarto_figure(
    fig: Figure,
    filename: str,
    output_dir: str = FIGURES_PATH,
    dpi: int = 300,
) -> None:
    """Save a figure in PDF and PNG formats for Quarto."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_dir / filename

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )


def possession_example() -> None:
    """Save figures illustrating an example possession."""
    start_frame, start_ax = plot_dev_start_frame(
        data_path=POSS_EXAMPLE,
        game_id=10517,
        possession_ids="10517_Argentina_1531_403",
        frame_id=65972,
        home_team_name="Argentina",
        away_team_name="France",
        home_color="skyblue",
        away_color="darkblue",
        ball_color="white",
        title="Argentina vs France | Player Possession Start",
        # figsize = FIGSIZE_SQUARE
    )

    pass_fig, pass_ax = plot_dev_players_to_pass(
        data_path=POSS_EXAMPLE,
        game_id=10517,
        possession_ids="10517_Argentina_1531_403",
        start_frame=65972,
        home_team_name="Argentina",
        away_team_name="France",
        home_color="skyblue",
        away_color="darkblue",
        ball_color="white",
        ball_start_color="white",
        ball_end_color="gray",
        start_player_size=55,
        title="Argentina vs France | Player Attempts Pass",
        # figsize = FIGSIZE_SQUARE
    )

    possession_fig, possession_ax = plot_dev_possession_movement(
        data_path=POSS_EXAMPLE,
        game_id=10517,
        possession_ids="10517_Argentina_1531_403",
        home_team_name="Argentina",
        away_team_name="France",
        home_color="skyblue",
        away_color="darkblue",
        ball_start_color="white",
        ball_end_color="gray",
        ball_trajectory_color="darkorange",
        title="Argentina vs France | Pass Successful to Teammate",
        # figsize = FIGSIZE_FULL
    )

    save_quarto_figure(start_frame, filename="player_possession_start")
    save_quarto_figure(pass_fig, filename="player_possession_pass_attempt")
    save_quarto_figure(
        possession_fig, filename="player_possession_pass_success"
    )


def main() -> None:
    """Reproduce all posterior-distribution figures."""
    possession_example()


if __name__ == "__main__":
    main()
