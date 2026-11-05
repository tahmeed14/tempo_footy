"""Generate machine-readable result tables used by the paper."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import polars as pl

MU_FIXED_EFFECTS_COLUMNS = (
    "model",
    "parameter_component",
    "term",
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
    "exp_estimate",
    "exp_hdi_lower",
    "exp_hdi_upper",
    "expected_tempo_pct_change",
    "expected_tempo_pct_change_lower",
    "expected_tempo_pct_change_upper",
)
ALPHA_FIXED_EFFECTS_COLUMNS = (
    "model",
    "parameter_component",
    "term",
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
    "conditional_cv_pct_change",
    "conditional_cv_pct_change_lower",
    "conditional_cv_pct_change_upper",
    "conditional_variance_ratio",
    "conditional_variance_ratio_hdi_lower",
    "conditional_variance_ratio_hdi_upper",
    "conditional_variance_pct_change",
    "conditional_variance_pct_change_lower",
    "conditional_variance_pct_change_upper",
)
DEFAULT_MU_FIXED_EFFECTS_PATH = Path("paper/tables/model_mu_fixed_effects.csv")
DEFAULT_ALPHA_FIXED_EFFECTS_PATH = Path(
    "paper/tables/model_alpha_fixed_effects.csv"
)
DEFAULT_POSTERIOR_PATH = Path("paper/results/tempo_gamma_posterior.nc")
_SAMPLE_DIMS = ("chain", "draw")


def generate_mu_fixed_effects_table(
    idata: Any,
    output_path: str | Path | None = DEFAULT_MU_FIXED_EFFECTS_PATH,
) -> pl.DataFrame:
    """Generate the Gamma mean-model fixed-effects results table.

    Parameters
    ----------
    idata
        Fitted ArviZ ``InferenceData`` from the distributional Gamma model.
    output_path
        CSV destination. A directory path receives the default filename.
        Pass ``None`` to return the table without writing it.

    Returns
    -------
    polars.DataFrame
        One row per population-level coefficient, with posterior summaries,
        diagnostics, sign probabilities, and log-link transformations.

    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group.")

    fixed_effect_names = _mu_fixed_effect_names(posterior)
    if not fixed_effect_names:
        raise ValueError(
            "No mu fixed effects were found in the posterior group."
        )

    result = _fixed_effects_summary_frame(
        idata,
        posterior,
        fixed_effect_names,
        model="mu",
        parameter_component="mean",
    )

    result["exp_estimate"] = np.exp(result["estimate"])
    result["exp_hdi_lower"] = np.exp(result["hdi_lower"])
    result["exp_hdi_upper"] = np.exp(result["hdi_upper"])
    result["expected_tempo_pct_change"] = 100 * (result["exp_estimate"] - 1)
    result["expected_tempo_pct_change_lower"] = 100 * (
        result["exp_hdi_lower"] - 1
    )
    result["expected_tempo_pct_change_upper"] = 100 * (
        result["exp_hdi_upper"] - 1
    )

    intercept = result["term"] == "Intercept"
    result.loc[
        intercept,
        [
            "expected_tempo_pct_change",
            "expected_tempo_pct_change_lower",
            "expected_tempo_pct_change_upper",
        ],
    ] = np.nan

    table = pl.from_pandas(result.loc[:, MU_FIXED_EFFECTS_COLUMNS])
    if table.columns != list(MU_FIXED_EFFECTS_COLUMNS):
        raise RuntimeError(
            "The generated table does not match the required schema."
        )

    if output_path is not None:
        destination = _resolve_output_path(
            output_path,
            DEFAULT_MU_FIXED_EFFECTS_PATH.name,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(destination)

    return table


def generate_alpha_fixed_effects_table(
    idata: Any,
    output_path: str | Path | None = DEFAULT_ALPHA_FIXED_EFFECTS_PATH,
) -> pl.DataFrame:
    """Generate the Gamma shape-model fixed-effects results table.

    The CV and variance transformations describe conditional dispersion through
    alpha, conceptually holding the modeled conditional mean component
    separate.
    Pass ``None`` as ``output_path`` to return the table without writing it.
    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group.")

    fixed_effect_names = _alpha_fixed_effect_names(posterior)
    if not fixed_effect_names:
        raise ValueError(
            "No alpha fixed effects were found in the posterior group."
        )

    result = _fixed_effects_summary_frame(
        idata,
        posterior,
        fixed_effect_names,
        model="alpha",
        parameter_component="shape",
    )

    result["shape_ratio"] = np.exp(result["estimate"])
    result["shape_ratio_hdi_lower"] = np.exp(result["hdi_lower"])
    result["shape_ratio_hdi_upper"] = np.exp(result["hdi_upper"])

    result["conditional_cv_ratio"] = np.exp(-0.5 * result["estimate"])
    result["conditional_cv_ratio_hdi_lower"] = np.exp(
        -0.5 * result["hdi_upper"]
    )
    result["conditional_cv_ratio_hdi_upper"] = np.exp(
        -0.5 * result["hdi_lower"]
    )
    result["conditional_cv_pct_change"] = 100 * (
        result["conditional_cv_ratio"] - 1
    )
    result["conditional_cv_pct_change_lower"] = 100 * (
        result["conditional_cv_ratio_hdi_lower"] - 1
    )
    result["conditional_cv_pct_change_upper"] = 100 * (
        result["conditional_cv_ratio_hdi_upper"] - 1
    )

    result["conditional_variance_ratio"] = np.exp(-result["estimate"])
    result["conditional_variance_ratio_hdi_lower"] = np.exp(
        -result["hdi_upper"]
    )
    result["conditional_variance_ratio_hdi_upper"] = np.exp(
        -result["hdi_lower"]
    )
    result["conditional_variance_pct_change"] = 100 * (
        result["conditional_variance_ratio"] - 1
    )
    result["conditional_variance_pct_change_lower"] = 100 * (
        result["conditional_variance_ratio_hdi_lower"] - 1
    )
    result["conditional_variance_pct_change_upper"] = 100 * (
        result["conditional_variance_ratio_hdi_upper"] - 1
    )

    contrast_columns = [
        "conditional_cv_ratio",
        "conditional_cv_ratio_hdi_lower",
        "conditional_cv_ratio_hdi_upper",
        "conditional_cv_pct_change",
        "conditional_cv_pct_change_lower",
        "conditional_cv_pct_change_upper",
        "conditional_variance_ratio",
        "conditional_variance_ratio_hdi_lower",
        "conditional_variance_ratio_hdi_upper",
        "conditional_variance_pct_change",
        "conditional_variance_pct_change_lower",
        "conditional_variance_pct_change_upper",
    ]
    result.loc[result["term"] == "alpha_Intercept", contrast_columns] = np.nan

    table = pl.from_pandas(result.loc[:, ALPHA_FIXED_EFFECTS_COLUMNS])
    if table.columns != list(ALPHA_FIXED_EFFECTS_COLUMNS):
        raise RuntimeError(
            "The generated table does not match the required schema."
        )

    if output_path is not None:
        destination = _resolve_output_path(
            output_path,
            DEFAULT_ALPHA_FIXED_EFFECTS_PATH.name,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(destination)

    return table


def _mu_fixed_effect_names(posterior: Any) -> list[str]:
    """Identify Bambi mu fixed-effect variables from posterior metadata."""
    names: list[str] = []
    for name, values in posterior.data_vars.items():
        if name == "alpha" or name.startswith("alpha_"):
            continue
        if "|" in name or name.endswith("_sigma"):
            continue
        if not set(_SAMPLE_DIMS).issubset(values.dims):
            continue

        coefficient_dims = [
            dim for dim in values.dims if dim not in _SAMPLE_DIMS
        ]
        if coefficient_dims and coefficient_dims != [f"{name}_dim"]:
            continue
        names.append(name)

    return names


def _alpha_fixed_effect_names(posterior: Any) -> list[str]:
    """Identify Bambi alpha fixed-effect variables from posterior metadata."""
    names: list[str] = []
    for name, values in posterior.data_vars.items():
        if not name.startswith("alpha_"):
            continue
        if "|" in name or name.endswith("_sigma"):
            continue
        if not set(_SAMPLE_DIMS).issubset(values.dims):
            continue

        coefficient_dims = [
            dim for dim in values.dims if dim not in _SAMPLE_DIMS
        ]
        if coefficient_dims and coefficient_dims != [f"{name}_dim"]:
            continue
        names.append(name)

    return names


def _fixed_effects_summary_frame(
    idata: Any,
    posterior: Any,
    fixed_effect_names: list[str],
    *,
    model: str,
    parameter_component: str,
) -> pd.DataFrame:
    """Build the posterior-summary columns shared by fixed-effect tables."""
    summary = _summary_with_95_hdi(idata, fixed_effect_names)
    required_summary_columns = {"mean", "sd", "r_hat", "ess_bulk", "ess_tail"}
    missing_summary_columns = required_summary_columns.difference(
        summary.columns
    )
    if missing_summary_columns:
        raise ValueError(
            "az.summary() did not return required columns: "
            f"{sorted(missing_summary_columns)}."
        )
    hdi_lower_column, hdi_upper_column = _find_hdi_columns(summary.columns)

    terms = summary.index.astype(str)
    probabilities = _posterior_sign_probabilities(
        posterior,
        fixed_effect_names,
        terms,
    )
    return pd.DataFrame(
        {
            "model": model,
            "parameter_component": parameter_component,
            "term": terms,
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
    variable_names: list[str],
    summary_terms: pd.Index,
) -> pd.DataFrame:
    """Calculate sign probabilities directly from chain and draw samples."""
    records: list[dict[str, float | str]] = []
    for name in variable_names:
        matching_terms = [
            term
            for term in summary_terms
            if term == name or term.startswith(f"{name}[")
        ]
        values = posterior[name]
        p_gt_zero = (values > 0).mean(dim=_SAMPLE_DIMS).values.reshape(-1)
        p_lt_zero = (values < 0).mean(dim=_SAMPLE_DIMS).values.reshape(-1)
        if len(matching_terms) != len(p_gt_zero):
            raise ValueError(
                f"Could not align posterior draws for fixed effect {name!r} "
                "with its az.summary() rows."
            )

        records.extend(
            {
                "term": term,
                "p_gt_zero": float(gt_zero),
                "p_lt_zero": float(lt_zero),
            }
            for term, gt_zero, lt_zero in zip(
                matching_terms,
                p_gt_zero,
                p_lt_zero,
                strict=True,
            )
        )

    probabilities = pd.DataFrame.from_records(records).set_index("term")
    missing_terms = summary_terms.difference(probabilities.index)
    if not missing_terms.empty:
        raise ValueError(
            "Could not calculate sign probabilities for summary terms: "
            f"{missing_terms.tolist()}."
        )
    return probabilities


def _resolve_output_path(
    output_path: str | Path, default_filename: str
) -> Path:
    destination = Path(output_path)
    if not destination.suffix:
        destination = destination / default_filename
    elif destination.suffix.lower() != ".csv":
        raise ValueError("output_path must be a CSV path or a directory.")
    return destination


def main() -> None:
    """Regenerate the paper's machine-readable fixed-effects tables."""
    if not DEFAULT_POSTERIOR_PATH.is_file():
        raise FileNotFoundError(
            f"Posterior file was not found: {DEFAULT_POSTERIOR_PATH}. "
            "Run this script from the repository root."
        )

    idata = az.from_netcdf(DEFAULT_POSTERIOR_PATH)
    mu_table = generate_mu_fixed_effects_table(idata)
    alpha_table = generate_alpha_fixed_effects_table(idata)

    print(
        f"Wrote {mu_table.height} mu fixed effects to "
        f"{DEFAULT_MU_FIXED_EFFECTS_PATH}"
    )
    print(
        f"Wrote {alpha_table.height} alpha fixed effects to "
        f"{DEFAULT_ALPHA_FIXED_EFFECTS_PATH}"
    )


if __name__ == "__main__":
    main()
