"""Generate machine-readable random-effects tables used by the paper."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Literal

import arviz as az
import pandas as pd
import polars as pl

DEFAULT_POSTERIOR_PATH = Path("paper/results/tempo_gamma_posterior.nc")
DEFAULT_METADATA_LOOKUP_PATH = Path(
    "data/curated/gradient_sports/metadata_lookup/player_game_lookup.parquet"
)
DEFAULT_MU_PLAYER_OUTPUT_PATH = Path(
    "paper/tables/model_mu_player_random_effects.csv"
)
DEFAULT_ALPHA_PLAYER_OUTPUT_PATH = Path(
    "paper/tables/model_alpha_player_random_effects.csv"
)
DEFAULT_MU_OPPONENT_OUTPUT_PATH = Path(
    "paper/tables/model_mu_opponent_random_effects.csv"
)
DEFAULT_GROUP_SD_OUTPUT_PATH = Path(
    "paper/tables/model_random_effect_standard_deviations.csv"
)

MU_PLAYER_COLUMNS = (
    "model",
    "group",
    "player_id",
    "playername",
    "teamname",
    "estimate",
    "posterior_sd",
    "hdi_prob",
    "hdi_lower",
    "hdi_upper",
    "p_gt_zero",
    "p_lt_zero",
    "r_hat",
    "ess_bulk",
    "ess_tail",
    "tempo_ratio",
    "tempo_ratio_hdi_lower",
    "tempo_ratio_hdi_upper",
    "tempo_pct_difference",
    "tempo_pct_difference_lower",
    "tempo_pct_difference_upper",
    "expected_tempo_pct_change",
    "rank",
)

ALPHA_PLAYER_COLUMNS = (
    "model",
    "group",
    "player_id",
    "playername",
    "teamname",
    "estimate",
    "posterior_sd",
    "hdi_prob",
    "hdi_lower",
    "hdi_upper",
    "p_gt_zero",
    "p_lt_zero",
    "r_hat",
    "ess_bulk",
    "ess_tail",
    "shape_ratio",
    "shape_ratio_hdi_lower",
    "shape_ratio_hdi_upper",
    "conditional_cv_ratio",
    "conditional_cv_ratio_hdi_lower",
    "conditional_cv_ratio_hdi_upper",
    "conditional_cv_pct_difference",
    "conditional_cv_pct_difference_lower",
    "conditional_cv_pct_difference_upper",
    "conditional_cv_pct_change",
    "conditional_variance_ratio",
    "conditional_variance_ratio_hdi_lower",
    "conditional_variance_ratio_hdi_upper",
    "conditional_variance_pct_difference",
    "conditional_variance_pct_difference_lower",
    "conditional_variance_pct_difference_upper",
    "shape_rank",
    "variability_rank",
)

MU_OPPONENT_COLUMNS = (
    "model",
    "group",
    "opponent_id",
    "opponent_teamname",
    "estimate",
    "posterior_sd",
    "hdi_prob",
    "hdi_lower",
    "hdi_upper",
    "p_gt_zero",
    "p_lt_zero",
    "r_hat",
    "ess_bulk",
    "ess_tail",
    "tempo_ratio",
    "tempo_ratio_hdi_lower",
    "tempo_ratio_hdi_upper",
    "tempo_pct_difference",
    "tempo_pct_difference_lower",
    "tempo_pct_difference_upper",
    "expected_tempo_pct_change",
    "rank",
)

GROUP_SD_COLUMNS = (
    "model",
    "group",
    "effect_variable",
    "group_sd_variable",
    "estimate",
    "posterior_sd",
    "hdi_prob",
    "hdi_lower",
    "hdi_upper",
    "r_hat",
    "ess_bulk",
    "ess_tail",
)


def _summary_with_95_hdi(
    idata: Any, variable_names: list[str]
) -> pd.DataFrame:
    """Call az.summary with an explicit 95% HDI across ArviZ APIs."""
    parameters = inspect.signature(az.summary).parameters
    kwargs: dict[str, Any] = {
        "var_names": variable_names,
        "filter_vars": None,
        "group": "posterior",
        "kind": "all",
        "round_to": "none",
    }
    if "ci_prob" in parameters:
        kwargs["ci_prob"] = 0.95
        kwargs["ci_kind"] = "hdi"
    else:
        kwargs["hdi_prob"] = 0.95

    summary = az.summary(idata, **kwargs)
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("az.summary() must return a pandas DataFrame.")
    return summary


def _find_hdi_columns(columns: pd.Index) -> tuple[str, str]:
    """Find lower and upper HDI columns across ArviZ naming conventions."""
    hdi_columns = [
        str(column)
        for column in columns
        if str(column).lower().startswith("hdi")
    ]
    if len(hdi_columns) != 2:
        raise ValueError(
            "az.summary() must return exactly two HDI columns; found "
            f"{hdi_columns}."
        )
    lower = [
        column
        for column in hdi_columns
        if "lower" in column.lower()
        or column.lower().endswith("_lb")
        or "2.5" in column
    ]
    upper = [
        column
        for column in hdi_columns
        if "upper" in column.lower()
        or column.lower().endswith("_ub")
        or "97.5" in column
    ]
    if len(lower) == 1 and len(upper) == 1:
        return lower[0], upper[0]
    return hdi_columns[0], hdi_columns[1]


def _posterior_sign_probabilities(
    posterior: Any,
    variable_name: str,
    summary_terms: pd.Index,
) -> pd.DataFrame:
    """Calculate sign probabilities directly from chain and draw samples."""
    values = posterior[variable_name]
    p_gt_zero = (values > 0).mean(dim=("chain", "draw")).values.reshape(-1)
    p_lt_zero = (values < 0).mean(dim=("chain", "draw")).values.reshape(-1)
    if len(summary_terms) != len(p_gt_zero):
        raise ValueError(
            f"Could not align posterior draws for {variable_name!r} with "
            "its az.summary() rows."
        )
    return pd.DataFrame(
        {
            "term": summary_terms,
            "p_gt_zero": p_gt_zero,
            "p_lt_zero": p_lt_zero,
        }
    ).set_index("term")


def generate_mu_player_random_effects_table(
    idata: Any,
    *,
    metadata_lookup_path: str | Path = DEFAULT_METADATA_LOOKUP_PATH,
    output_path: str | Path | None = DEFAULT_MU_PLAYER_OUTPUT_PATH,
) -> pl.DataFrame:
    """Generate player deviations from the Gamma conditional-mean component."""
    effects = _summarize_group_specific_effect(
        idata, component="mu", group_name="player_id", id_column="player_id"
    )
    effects = _join_metadata_without_expansion(
        effects,
        _load_player_metadata(metadata_lookup_path),
        id_column="player_id",
        metadata_columns=("playername", "teamname"),
    )
    effects = _add_tempo_transformations(effects)
    effects = _add_deterministic_rank(effects, "estimate", "player_id", "rank")
    table = _finalize_table(effects, MU_PLAYER_COLUMNS)
    _write_table(table, output_path, DEFAULT_MU_PLAYER_OUTPUT_PATH.name)
    return table


def generate_alpha_player_random_effects_table(
    idata: Any,
    *,
    metadata_lookup_path: str | Path = DEFAULT_METADATA_LOOKUP_PATH,
    output_path: str | Path | None = DEFAULT_ALPHA_PLAYER_OUTPUT_PATH,
) -> pl.DataFrame:
    """Generate player deviations from the Gamma shape component."""
    effects = _summarize_group_specific_effect(
        idata, component="alpha", group_name="player_id", id_column="player_id"
    )
    effects = _join_metadata_without_expansion(
        effects,
        _load_player_metadata(metadata_lookup_path),
        id_column="player_id",
        metadata_columns=("playername", "teamname"),
    )
    effects = _add_alpha_transformations(effects)
    effects = _add_deterministic_rank(
        effects, "estimate", "player_id", "shape_rank"
    )
    variability_ranks = _rank_mapping(
        effects,
        "conditional_cv_pct_difference",
        "player_id",
        "variability_rank",
    )
    effects = effects.join(variability_ranks, on="player_id", how="left")
    table = _finalize_table(effects, ALPHA_PLAYER_COLUMNS)
    _write_table(table, output_path, DEFAULT_ALPHA_PLAYER_OUTPUT_PATH.name)
    return table


def generate_mu_opponent_random_effects_table(
    idata: Any,
    *,
    metadata_lookup_path: str | Path = DEFAULT_METADATA_LOOKUP_PATH,
    output_path: str | Path | None = DEFAULT_MU_OPPONENT_OUTPUT_PATH,
) -> pl.DataFrame:
    """Generate opposition-team deviations from the Gamma mean component."""
    effects = _summarize_group_specific_effect(
        idata,
        component="mu",
        group_name="opponent_id",
        id_column="opponent_id",
    )
    effects = _join_metadata_without_expansion(
        effects,
        _load_opponent_metadata(metadata_lookup_path),
        id_column="opponent_id",
        metadata_columns=("opponent_teamname",),
    )
    effects = _add_tempo_transformations(effects)
    effects = _add_deterministic_rank(
        effects, "estimate", "opponent_id", "rank"
    )
    table = _finalize_table(effects, MU_OPPONENT_COLUMNS)
    _write_table(table, output_path, DEFAULT_MU_OPPONENT_OUTPUT_PATH.name)
    return table


def generate_random_effect_standard_deviations_table(
    idata: Any,
    *,
    output_path: str | Path | None = DEFAULT_GROUP_SD_OUTPUT_PATH,
) -> pl.DataFrame:
    """Generate one posterior-summary row per random-effect SD term."""
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group.")

    specifications: tuple[tuple[Literal["mu", "alpha"], str], ...] = (
        ("mu", "player_id"),
        ("alpha", "player_id"),
        ("mu", "opponent_id"),
    )
    summaries = []
    for component, group_name in specifications:
        effect_variable, _ = _find_group_specific_variable(
            posterior,
            component=component,
            group_name=group_name,
        )
        summaries.append(
            _summarize_group_sd(
                idata,
                effect_variable,
                component=component,
                group="player" if group_name == "player_id" else "opponent",
            )
        )

    table = _finalize_table(pl.concat(summaries), GROUP_SD_COLUMNS)
    _write_table(table, output_path, DEFAULT_GROUP_SD_OUTPUT_PATH.name)
    return table


def _find_group_specific_variable(
    posterior: Any,
    *,
    component: Literal["mu", "alpha"],
    group_name: str,
) -> tuple[str, str]:
    """Find one Bambi group deviation and its factor coordinate dimension."""
    candidates: list[tuple[str, str]] = []
    expected_dimension = f"{group_name}__factor_dim"
    for name, values in posterior.data_vars.items():
        if (component == "alpha") != name.startswith("alpha_"):
            continue
        if "|" not in name or name.endswith("_sigma"):
            continue
        level_dimensions = [
            dimension
            for dimension in values.dims
            if dimension not in {"chain", "draw"}
        ]
        if level_dimensions == [expected_dimension]:
            candidates.append((name, expected_dimension))

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {component} group-specific effect for "
            f"{group_name!r}; found {[name for name, _ in candidates]}."
        )
    return candidates[0]


def _summarize_group_specific_effect(
    idata: Any,
    *,
    component: Literal["mu", "alpha"],
    group_name: str,
    id_column: str,
) -> pl.DataFrame:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group.")

    variable_name, level_dimension = _find_group_specific_variable(
        posterior, component=component, group_name=group_name
    )
    summary = _summary_with_95_hdi(idata, [variable_name])
    required_columns = {"mean", "sd", "r_hat", "ess_bulk", "ess_tail"}
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        raise ValueError(
            "az.summary() did not return required columns: "
            f"{sorted(missing_columns)}."
        )
    hdi_lower_column, hdi_upper_column = _find_hdi_columns(summary.columns)

    terms = summary.index.astype(str)
    levels = posterior[variable_name].coords[level_dimension].values
    term_to_level = {f"{variable_name}[{level}]": level for level in levels}
    if len(terms) != len(levels) or any(
        term not in term_to_level for term in terms
    ):
        raise ValueError(
            f"Could not align {variable_name!r} summary rows with "
            "posterior levels."
        )

    probabilities = _posterior_sign_probabilities(
        posterior, variable_name, terms
    )
    result = pd.DataFrame(
        {
            "model": "alpha" if component == "alpha" else "mu",
            "group": "player" if group_name == "player_id" else "opponent",
            id_column: [term_to_level[term] for term in terms],
            "estimate": pd.to_numeric(summary["mean"]).to_numpy(),
            "posterior_sd": pd.to_numeric(summary["sd"]).to_numpy(),
            "hdi_prob": 0.95,
            "hdi_lower": pd.to_numeric(summary[hdi_lower_column]).to_numpy(),
            "hdi_upper": pd.to_numeric(summary[hdi_upper_column]).to_numpy(),
            "p_gt_zero": probabilities.loc[terms, "p_gt_zero"].to_numpy(),
            "p_lt_zero": probabilities.loc[terms, "p_lt_zero"].to_numpy(),
            "r_hat": pd.to_numeric(summary["r_hat"]).to_numpy(),
            "ess_bulk": pd.to_numeric(summary["ess_bulk"]).to_numpy(),
            "ess_tail": pd.to_numeric(summary["ess_tail"]).to_numpy(),
        }
    )
    table = pl.from_pandas(result).with_columns(
        pl.col(id_column).cast(pl.Int64, strict=True)
    )
    if table.height != len(levels) or table[id_column].n_unique() != len(
        levels
    ):
        raise ValueError(f"Expected one posterior row per {group_name} level.")
    return table


def _summarize_group_sd(
    idata: Any,
    effect_variable_name: str,
    *,
    component: Literal["mu", "alpha"],
    group: Literal["player", "opponent"],
) -> pl.DataFrame:
    """Summarize one group-level random-effect SD posterior."""
    posterior = idata.posterior
    sigma_variable = f"{effect_variable_name}_sigma"
    if sigma_variable not in posterior:
        raise KeyError(
            "Posterior random-effect SD variable "
            f"{sigma_variable!r} was not found."
        )
    if set(posterior[sigma_variable].dims) != {"chain", "draw"}:
        raise ValueError(
            "Posterior random-effect SD variable "
            f"{sigma_variable!r} must be scalar "
            "apart from chain and draw dimensions."
        )

    summary = _summary_with_95_hdi(idata, [sigma_variable])
    if summary.shape[0] != 1:
        raise ValueError(
            f"Expected one summary row for {sigma_variable!r}; found "
            f"{summary.shape[0]}."
        )
    required_columns = {"mean", "sd", "r_hat", "ess_bulk", "ess_tail"}
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        raise ValueError(
            "az.summary() did not return required group-SD columns: "
            f"{sorted(missing_columns)}."
        )
    hdi_lower_column, hdi_upper_column = _find_hdi_columns(summary.columns)
    row = summary.iloc[0]

    return pl.DataFrame(
        {
            "model": [component],
            "group": [group],
            "effect_variable": [effect_variable_name],
            "group_sd_variable": [sigma_variable],
            "estimate": [float(row["mean"])],
            "posterior_sd": [float(row["sd"])],
            "hdi_prob": [0.95],
            "hdi_lower": [float(row[hdi_lower_column])],
            "hdi_upper": [float(row[hdi_upper_column])],
            "r_hat": [float(row["r_hat"])],
            "ess_bulk": [float(row["ess_bulk"])],
            "ess_tail": [float(row["ess_tail"])],
        }
    )


def _read_metadata(metadata_lookup_path: str | Path) -> pl.DataFrame:
    path = Path(metadata_lookup_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata lookup was not found: {path}.")
    return pl.read_parquet(path)


def _load_player_metadata(metadata_lookup_path: str | Path) -> pl.DataFrame:
    lookup = _read_metadata(metadata_lookup_path)
    required_columns = ["player_id", "player_name", "team_name"]
    missing_columns = [
        column for column in required_columns if column not in lookup.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Metadata lookup is missing columns: {missing_columns}."
        )

    clean = lookup.select(required_columns).drop_nulls()
    conflicts = (
        clean.group_by("player_id")
        .agg(
            pl.col("player_name").n_unique().alias("player_name_count"),
            pl.col("team_name").n_unique().alias("team_name_count"),
        )
        .filter(
            (pl.col("player_name_count") > 1) | (pl.col("team_name_count") > 1)
        )
    )
    if not conflicts.is_empty():
        raise ValueError(
            "Metadata contains player IDs mapped to multiple names or teams: "
            f"{conflicts['player_id'].to_list()}."
        )

    return clean.unique(subset="player_id").rename(
        {"player_name": "playername", "team_name": "teamname"}
    )


def _load_opponent_metadata(metadata_lookup_path: str | Path) -> pl.DataFrame:
    lookup = _read_metadata(metadata_lookup_path)
    required_columns = ["team_id", "team_name"]
    missing_columns = [
        column for column in required_columns if column not in lookup.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Metadata lookup is missing columns: {missing_columns}."
        )

    clean = lookup.select(required_columns).drop_nulls()
    conflicts = (
        clean.group_by("team_id")
        .agg(pl.col("team_name").n_unique().alias("team_name_count"))
        .filter(pl.col("team_name_count") > 1)
    )
    if not conflicts.is_empty():
        raise ValueError(
            "Metadata contains team IDs mapped to multiple names: "
            f"{conflicts['team_id'].to_list()}."
        )

    return clean.unique(subset="team_id").rename(
        {"team_id": "opponent_id", "team_name": "opponent_teamname"}
    )


def _join_metadata_without_expansion(
    effects: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    id_column: str,
    metadata_columns: tuple[str, ...],
) -> pl.DataFrame:
    original_height = effects.height
    joined = effects.join(metadata, on=id_column, how="left")
    if joined.height != original_height:
        raise ValueError("Metadata join changed the number of posterior rows.")

    missing = joined.filter(
        pl.any_horizontal(
            [pl.col(column).is_null() for column in metadata_columns]
        )
    )
    if not missing.is_empty():
        raise ValueError(
            f"Metadata is missing for modeled {id_column} values: "
            f"{missing[id_column].to_list()}."
        )
    return joined


def _add_tempo_transformations(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_columns(
        pl.col("estimate").exp().alias("tempo_ratio"),
        pl.col("hdi_lower").exp().alias("tempo_ratio_hdi_lower"),
        pl.col("hdi_upper").exp().alias("tempo_ratio_hdi_upper"),
    ).with_columns(
        ((pl.col("tempo_ratio") - 1) * 100).alias("tempo_pct_difference"),
        ((pl.col("tempo_ratio_hdi_lower") - 1) * 100).alias(
            "tempo_pct_difference_lower"
        ),
        ((pl.col("tempo_ratio_hdi_upper") - 1) * 100).alias(
            "tempo_pct_difference_upper"
        ),
        ((pl.col("tempo_ratio") - 1) * 100).alias("expected_tempo_pct_change"),
    )


def _add_alpha_transformations(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_columns(
        pl.col("estimate").exp().alias("shape_ratio"),
        pl.col("hdi_lower").exp().alias("shape_ratio_hdi_lower"),
        pl.col("hdi_upper").exp().alias("shape_ratio_hdi_upper"),
        (-0.5 * pl.col("estimate")).exp().alias("conditional_cv_ratio"),
        (-0.5 * pl.col("hdi_upper"))
        .exp()
        .alias("conditional_cv_ratio_hdi_lower"),
        (-0.5 * pl.col("hdi_lower"))
        .exp()
        .alias("conditional_cv_ratio_hdi_upper"),
        (-pl.col("estimate")).exp().alias("conditional_variance_ratio"),
        (-pl.col("hdi_upper"))
        .exp()
        .alias("conditional_variance_ratio_hdi_lower"),
        (-pl.col("hdi_lower"))
        .exp()
        .alias("conditional_variance_ratio_hdi_upper"),
    ).with_columns(
        ((pl.col("conditional_cv_ratio") - 1) * 100).alias(
            "conditional_cv_pct_difference"
        ),
        ((pl.col("conditional_cv_ratio_hdi_lower") - 1) * 100).alias(
            "conditional_cv_pct_difference_lower"
        ),
        ((pl.col("conditional_cv_ratio_hdi_upper") - 1) * 100).alias(
            "conditional_cv_pct_difference_upper"
        ),
        ((pl.col("conditional_cv_ratio") - 1) * 100).alias(
            "conditional_cv_pct_change"
        ),
        ((pl.col("conditional_variance_ratio") - 1) * 100).alias(
            "conditional_variance_pct_difference"
        ),
        ((pl.col("conditional_variance_ratio_hdi_lower") - 1) * 100).alias(
            "conditional_variance_pct_difference_lower"
        ),
        ((pl.col("conditional_variance_ratio_hdi_upper") - 1) * 100).alias(
            "conditional_variance_pct_difference_upper"
        ),
    )


def _rank_mapping(
    table: pl.DataFrame,
    value_column: str,
    id_column: str,
    rank_column: str,
) -> pl.DataFrame:
    """Create deterministic ordinal ranks using the group ID to break ties."""
    return (
        table.select(id_column, value_column)
        .sort([value_column, id_column], descending=[True, False])
        .with_row_index(rank_column, offset=1)
        .select(id_column, rank_column)
    )


def _add_deterministic_rank(
    table: pl.DataFrame,
    value_column: str,
    id_column: str,
    rank_column: str,
) -> pl.DataFrame:
    ranks = _rank_mapping(table, value_column, id_column, rank_column)
    return table.join(ranks, on=id_column, how="left").sort(rank_column)


def _finalize_table(
    table: pl.DataFrame,
    expected_columns: tuple[str, ...],
) -> pl.DataFrame:
    missing_columns = [
        column for column in expected_columns if column not in table.columns
    ]
    if missing_columns:
        raise RuntimeError(
            f"Generated table is missing columns: {missing_columns}."
        )
    result = table.select(expected_columns)
    if result.columns != list(expected_columns):
        raise RuntimeError(
            "Generated table does not match the required schema."
        )
    return result


def _write_table(
    table: pl.DataFrame,
    output_path: str | Path | None,
    default_filename: str,
) -> None:
    if output_path is None:
        return
    destination = Path(output_path)
    if not destination.suffix:
        destination = destination / default_filename
    elif destination.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV path or a directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(destination)


def main() -> None:
    """Regenerate all paper random-effects summary tables."""
    if not DEFAULT_POSTERIOR_PATH.is_file():
        raise FileNotFoundError(
            f"Posterior file was not found: {DEFAULT_POSTERIOR_PATH}. "
            "Run this script from the repository root."
        )

    idata = az.from_netcdf(DEFAULT_POSTERIOR_PATH)
    mu_players = generate_mu_player_random_effects_table(idata)
    alpha_players = generate_alpha_player_random_effects_table(idata)
    mu_opponents = generate_mu_opponent_random_effects_table(idata)
    group_sds = generate_random_effect_standard_deviations_table(idata)
    print(
        f"Wrote {mu_players.height} mu player effects to "
        f"{DEFAULT_MU_PLAYER_OUTPUT_PATH}"
    )
    print(
        f"Wrote {alpha_players.height} alpha player effects to "
        f"{DEFAULT_ALPHA_PLAYER_OUTPUT_PATH}"
    )
    print(
        f"Wrote {mu_opponents.height} mu opponent effects to "
        f"{DEFAULT_MU_OPPONENT_OUTPUT_PATH}"
    )
    print(
        f"Wrote {group_sds.height} random-effect SD summaries to "
        f"{DEFAULT_GROUP_SD_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
