"""Common data, priors and mean-aligned hierarchical BST distributions.

These are research models, independent of the CMSAC reproduction workflow.
PyMC is used directly because the arithmetic mean must have the same regression
structure in all three families, including distributional regression for spread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import brentq
from scipy.special import gammaln

from tempoctrl.model.load import read_model_data
from tempoctrl.model.transform import (
    BALL_HEIGHT_ORDER,
    DEFENDER_CHALLENGES_ORDER,
    PERIOD_ORDER,
    PITCH_THIRD_ORDER,
    POSITION_GROUP_ORDER,
    PRESSURE_ORDER,
    add_log_sequence,
    polars_enum_to_ordered_pandas,
)
from tempoctrl.reproduce_tables import (
    CATEGORICAL_PREDICTORS,
    CONTINUOUS_PREDICTORS,
    OUTCOME_VARIABLE,
)

FAMILIES = ("gamma", "lognormal", "weibull")
PRIOR_NAMES = ("baseline", "stronger", "weaker")
OBSERVATION_KEY = "match_team_player_possession_id"
DISPERSION_PREDICTORS = (
    "first_touch_defender_pressure_type", "defender_num_challenges",
    "starting_pitch_third", "player_position_group",
)
REFERENCE_LEVELS = {
    "first_touch_ballheight": BALL_HEIGHT_ORDER.categories.to_list(),
    "first_touch_defender_pressure_type": PRESSURE_ORDER.categories.to_list(),
    "defender_num_challenges": DEFENDER_CHALLENGES_ORDER.categories.to_list(),
    "starting_pitch_third": PITCH_THIRD_ORDER.categories.to_list(),
    "player_position_group": POSITION_GROUP_ORDER.categories.to_list(),
    "game_period": PERIOD_ORDER.categories.to_list(),
}


def repository_root(start: str | Path | None = None) -> Path:
    """Resolve paths from root, an investigation directory, or an interactive cell."""
    for origin in (Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parent):
        for candidate in (origin, *origin.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "src/tempoctrl").is_dir():
                return candidate
    raise FileNotFoundError("Could not locate the tempoctrl repository.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def specification_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass
class PreparedData:
    frame: pd.DataFrame
    exclusions: pd.DataFrame
    mean_design: pd.DataFrame
    dispersion_design: pd.DataFrame
    player_index: np.ndarray
    opponent_index: np.ndarray
    players: list[str]
    opponents: list[str]
    audit: dict


def design_matrix(frame: pd.DataFrame, predictors: tuple[str, ...]) -> pd.DataFrame:
    """Treatment contrasts with explicit, repository-defined reference levels."""
    columns = {}
    for name in predictors:
        if name in REFERENCE_LEVELS:
            # Unobserved levels have no likelihood information; omit their all-zero
            # columns. The first repository level remains the reference.
            for level in REFERENCE_LEVELS[name][1:]:
                if (frame[name] == level).any():
                    columns[f"{name}[{level}]"] = (frame[name] == level).astype(float)
        else:
            columns[name] = frame[name].astype(float)
    return pd.DataFrame(columns, index=frame.index)


def prepare_data(path: str | Path) -> PreparedData:
    """Read a fresh processed snapshot and freeze one cohort for every model.

    No upstream outlier rule (including the CMSAC 25 m/s cutoff) is reapplied.
    Unusable rows are excluded with explicit reasons. Unknown categories and
    ambiguous observational keys fail rather than silently changing contrasts.
    The outcome is assumed to be m/s, as in the production tempo metric.
    """
    path = Path(path).resolve()
    source_hash = file_sha256(path)
    raw = read_model_data(path)
    if isinstance(raw, pl.LazyFrame):
        raw = raw.collect()
    required = (
        OBSERVATION_KEY, "game_id", "player_id", "opponent_id", OUTCOME_VARIABLE,
        "game_state_goal_diff", "player_possession_sequence_number",
        *CATEGORICAL_PREDICTORS,
    )
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Processed data is missing required columns: {missing}")
    keys = raw[OBSERVATION_KEY]
    if keys.null_count() or keys.n_unique() != raw.height:
        raise ValueError(f"{OBSERVATION_KEY} must be unique and non-null.")
    # Selecting modeled columns also makes unrelated nullable metadata harmless.
    frame = raw.select(required).to_pandas()
    reasons = pd.Series("", index=frame.index)
    for name in required:
        invalid = frame[name].isna()
        if name in (OUTCOME_VARIABLE, "game_state_goal_diff", "player_possession_sequence_number"):
            invalid |= ~np.isfinite(pd.to_numeric(frame[name], errors="coerce"))
        if name in (OUTCOME_VARIABLE, "player_possession_sequence_number"):
            invalid |= frame[name] <= 0
        reasons.loc[invalid] += f"{name}:missing_or_invalid;"
    exclusions = frame.loc[reasons != "", [OBSERVATION_KEY, "game_id"]].copy()
    exclusions["reason"] = reasons[reasons != ""]
    frame = frame.loc[reasons == ""].copy()
    if len(frame) < 2:
        raise ValueError("At least two valid observations are required.")
    for name, levels in REFERENCE_LEVELS.items():
        unknown = set(frame[name].astype(str)) - set(levels)
        if unknown:
            raise ValueError(f"Unknown levels in {name}: {sorted(unknown)}")
    frame[OBSERVATION_KEY] = frame[OBSERVATION_KEY].astype(str)
    frame = frame.sort_values(OBSERVATION_KEY).reset_index(drop=True)
    for name in ("player_id", "opponent_id"):
        frame[name] = frame[name].astype(str)
    # Recompute after exclusions: never trust stale z scores from a prior snapshot.
    transformed = pl.from_pandas(frame).with_columns(
        *(pl.col(name).cast(pl.String).cast(pl.Enum(levels)) for name, levels in REFERENCE_LEVELS.items())
    )
    logged = np.log1p(frame["player_possession_sequence_number"].to_numpy(dtype=float))
    sequence_sd = float(logged.std(ddof=1))
    if np.ptp(logged) == 0:
        raise ValueError("Possession sequence has zero variance; cannot standardize.")
    transformed = add_log_sequence(transformed)
    frame = polars_enum_to_ordered_pandas(transformed, columns=list(REFERENCE_LEVELS))
    players = sorted(frame["player_id"].unique())
    opponents = sorted(frame["opponent_id"].unique())
    mean_design = design_matrix(frame, (*CONTINUOUS_PREDICTORS, *CATEGORICAL_PREDICTORS))
    dispersion_design = design_matrix(frame, DISPERSION_PREDICTORS)
    # Include modeled values, IDs and ordering; a source hash also detects upstream
    # changes to metadata not used by the likelihood.
    cohort_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    audit = {
        "source_path": str(path), "source_sha256": source_hash,
        "cohort_sha256": cohort_hash, "input_rows": raw.height,
        "included_rows": len(frame), "excluded_rows": len(exclusions),
        "players": len(players), "opponents": len(opponents),
        "matches": int(frame["game_id"].nunique()), "units": "m/s (upstream assumption)",
        "outcome_min": float(frame[OUTCOME_VARIABLE].min()),
        "outcome_max": float(frame[OUTCOME_VARIABLE].max()),
        "sequence_log_mean": float(logged.mean()), "sequence_log_sd": sequence_sd,
        "reference_levels": REFERENCE_LEVELS,
        "mean_columns": list(mean_design), "dispersion_columns": list(dispersion_design),
        "filter_policy": "Exclude nonfinite/nonpositive outcomes, invalid sequence and missing model inputs; no extra outlier cutoff.",
    }
    if file_sha256(path) != source_hash:
        raise RuntimeError("Input changed during preparation; rerun on a stable snapshot.")
    return PreparedData(
        frame, exclusions, mean_design, dispersion_design,
        pd.Categorical(frame["player_id"], categories=players).codes,
        pd.Categorical(frame["opponent_id"], categories=opponents).codes,
        players, opponents, audit,
    )


def log_cv(family: str, log_dispersion: np.ndarray | float) -> np.ndarray:
    """Natural dispersion is alpha (Gamma), sigma (log-normal), or k (Weibull)."""
    value = np.exp(np.asarray(log_dispersion))
    if family == "gamma":
        return -0.5 * np.asarray(log_dispersion)
    if family == "lognormal":
        return 0.5 * np.log(np.expm1(value**2))
    if family == "weibull":
        ratio = gammaln(1 + 2 / value) - 2 * gammaln(1 + 1 / value)
        return 0.5 * np.log(np.expm1(ratio))
    raise ValueError(f"Unknown family: {family}")


@dataclass(frozen=True)
class PriorSpec:
    family: str
    name: str
    mean_intercept_center: float
    mean_intercept_sd: float
    mean_effect_sd: float
    mean_group_sd: float
    dispersion_intercept_center: float
    dispersion_intercept_sd: float
    dispersion_effect_sd: float
    dispersion_group_sd: float
    log_cv_slope: float

    def as_dict(self) -> dict:
        return asdict(self)


def prior_spec(family: str, name: str = "baseline") -> PriorSpec:
    """Proposed widths matched locally on log(CV), pending predictive calibration.

    Centers come from the existing CMSAC Gamma priors: mean 7, alpha 3.
    This is a baseline-informed sensitivity specification, not independent
    external elicitation. All configurations hold those centers fixed.
    """
    if family not in FAMILIES or name not in PRIOR_NAMES:
        raise ValueError(f"Expected family in {FAMILIES} and prior in {PRIOR_NAMES}.")
    target_log_cv = -0.5 * np.log(3.0)
    center = brentq(lambda x: float(log_cv(family, x)) - target_log_cv, -2, 3)
    delta = 1e-5
    slope = float((log_cv(family, center + delta) - log_cv(family, center - delta)) / (2 * delta))
    # Gamma baseline maps exactly to intercept SD .5 and effect/group SD .3.
    mean_i, mean_e, cv_i, cv_e = {
        "stronger": (0.25, 0.15, 0.15, 0.075),
        "baseline": (0.40, 0.30, 0.25, 0.150),
        "weaker": (0.60, 0.45, 0.35, 0.225),
    }[name]
    return PriorSpec(
        family, name, float(np.log(7)), mean_i, mean_e, mean_e,
        center, cv_i / abs(slope), cv_e / abs(slope), cv_e / abs(slope), slope,
    )


def conditional_moments(family: str, mean: np.ndarray, dispersion: np.ndarray) -> dict[str, np.ndarray]:
    """Derive moments draw by draw on the original m/s scale."""
    cv = np.exp(log_cv(family, np.log(dispersion)))
    return {"mean": np.asarray(mean), "sd": mean * cv, "cv": cv}


def build_model(data: PreparedData, priors: PriorSpec):
    """Build a noncentered PyMC model without fitting or saving large deterministics."""
    import pymc as pm
    import pytensor.tensor as pt

    coords = {
        "observation": data.frame[OBSERVATION_KEY].to_list(),
        "mean_term": list(data.mean_design), "dispersion_term": list(data.dispersion_design),
        "player": data.players, "opponent": data.opponents,
    }
    with pm.Model(coords=coords) as model:
        mean_intercept = pm.Normal("mean_intercept", priors.mean_intercept_center, priors.mean_intercept_sd)
        mean_beta = pm.Normal("mean_beta", 0, priors.mean_effect_sd, dims="mean_term")
        dispersion_intercept = pm.Normal("dispersion_intercept", priors.dispersion_intercept_center, priors.dispersion_intercept_sd)
        dispersion_beta = pm.Normal("dispersion_beta", 0, priors.dispersion_effect_sd, dims="dispersion_term")

        def group_effect(name, dim, scale):
            sd = pm.HalfNormal(f"{name}_sd", scale)
            z = pm.Normal(f"{name}_z", 0, 1, dims=dim)
            return pm.Deterministic(name, z * sd, dims=dim)

        mean_player = group_effect("mean_player", "player", priors.mean_group_sd)
        mean_opponent = group_effect("mean_opponent", "opponent", priors.mean_group_sd)
        dispersion_player = group_effect("dispersion_player", "player", priors.dispersion_group_sd)
        eta = (
            mean_intercept + pt.dot(data.mean_design.to_numpy(), mean_beta)
            + mean_player[data.player_index] + mean_opponent[data.opponent_index]
        )
        zeta = (
            dispersion_intercept + pt.dot(data.dispersion_design.to_numpy(), dispersion_beta)
            + dispersion_player[data.player_index]
        )
        mean, spread = pt.exp(eta), pt.exp(zeta)
        y = data.frame[OUTCOME_VARIABLE].to_numpy()
        if priors.family == "gamma":
            pm.Gamma(OUTCOME_VARIABLE, alpha=spread, beta=spread / mean, observed=y, dims="observation")
        elif priors.family == "lognormal":
            pm.LogNormal(OUTCOME_VARIABLE, mu=eta - spread**2 / 2, sigma=spread, observed=y, dims="observation")
        else:
            scale = pt.exp(eta - pt.gammaln(1 + 1 / spread))
            pm.Weibull(OUTCOME_VARIABLE, alpha=spread, beta=scale, observed=y, dims="observation")
    return model
