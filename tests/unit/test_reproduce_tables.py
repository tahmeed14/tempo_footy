"""Test fixed-effects table reproduction."""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pytest

from tempoctrl.reproduce_fixed_effects_tables import (
    ALPHA_FIXED_EFFECTS_COLUMNS,
    MU_FIXED_EFFECTS_COLUMNS,
    generate_alpha_fixed_effects_table,
    generate_mu_fixed_effects_table,
)


def _synthetic_gamma_posterior() -> az.InferenceData:
    """Build known Gamma-model draws for fixed-effect summaries."""
    rng = np.random.default_rng(42)
    return az.from_dict(
        {
            "posterior": {
                "Intercept": rng.normal(2.0, 0.1, size=(2, 100)),
                "continuous_predictor": rng.normal(0.2, 0.1, size=(2, 100)),
                "category": rng.normal(0.0, 0.1, size=(2, 100, 2)),
                "1|player_id_sigma": rng.exponential(0.2, size=(2, 100)),
                "1|player_id": rng.normal(0.0, 0.2, size=(2, 100, 2)),
                "alpha_Intercept": rng.normal(1.0, 0.1, size=(2, 100)),
                "alpha_category": rng.normal(
                    loc=np.array([0.4, -0.4]),
                    scale=0.05,
                    size=(2, 100, 2),
                ),
            }
        },
        coords={
            "category_dim": ["B", "C"],
            "player_id__factor_dim": [10, 20],
            "alpha_category_dim": ["B", "C"],
        },
        dims={
            "category": ["category_dim"],
            "1|player_id": ["player_id__factor_dim"],
            "alpha_category": ["alpha_category_dim"],
        },
    )


def test_generate_mu_fixed_effects_table_has_required_schema_and_values(
    tmp_path: Path,
) -> None:
    """Recover known location effects and their transformed intervals.
    """
    output_path = tmp_path / "model_mu_fixed_effects.csv"

    table = generate_mu_fixed_effects_table(
        _synthetic_gamma_posterior(),
        output_path=output_path,
    )

    assert table.columns == list(MU_FIXED_EFFECTS_COLUMNS)
    assert table["term"].to_list() == [
        "Intercept",
        "continuous_predictor",
        "category[B]",
        "category[C]",
    ]
    assert table["model"].unique().to_list() == ["mu"]
    assert table["parameter_component"].unique().to_list() == ["mean"]
    assert table["hdi_prob"].unique().to_list() == [0.95]
    assert not any(
        "|" in term or term.startswith("alpha_") for term in table["term"]
    )
    assert np.allclose(
        table["p_gt_zero"] + table["p_lt_zero"],
        1.0,
    )
    assert np.allclose(table["exp_estimate"], np.exp(table["estimate"]))
    assert np.allclose(table["exp_hdi_lower"], np.exp(table["hdi_lower"]))
    assert np.allclose(table["exp_hdi_upper"], np.exp(table["hdi_upper"]))
    assert table.row(0, named=True)["expected_tempo_pct_change"] is None
    assert table.row(0, named=True)["expected_tempo_pct_change_lower"] is None
    assert table.row(0, named=True)["expected_tempo_pct_change_upper"] is None
    assert table.row(1, named=True)[
        "expected_tempo_pct_change"
    ] == pytest.approx(100 * (table.row(1, named=True)["exp_estimate"] - 1))
    assert output_path.is_file()
    assert pl.read_csv(output_path).columns == list(MU_FIXED_EFFECTS_COLUMNS)


def test_generate_mu_fixed_effects_table_can_skip_writing() -> None:
    """Return fixed-effect summaries without writing an output file."""
    table = generate_mu_fixed_effects_table(
        _synthetic_gamma_posterior(),
        output_path=None,
    )

    assert table.height == 4


def test_generate_alpha_fixed_effects_table_has_required_schema_and_values(
    tmp_path: Path,
) -> None:
    """Recover known shape effects and conditional variability changes.
    """
    output_path = tmp_path / "model_alpha_fixed_effects.csv"

    table = generate_alpha_fixed_effects_table(
        _synthetic_gamma_posterior(),
        output_path=output_path,
    )

    assert table.columns == list(ALPHA_FIXED_EFFECTS_COLUMNS)
    assert table["term"].to_list() == [
        "alpha_Intercept",
        "alpha_category[B]",
        "alpha_category[C]",
    ]
    assert table["model"].unique().to_list() == ["alpha"]
    assert table["parameter_component"].unique().to_list() == ["shape"]
    assert table["hdi_prob"].unique().to_list() == [0.95]
    assert all(
        term.startswith("alpha_") and "|" not in term for term in table["term"]
    )
    assert np.allclose(table["p_gt_zero"] + table["p_lt_zero"], 1.0)
    assert np.allclose(table["shape_ratio"], np.exp(table["estimate"]))
    assert np.allclose(
        table["shape_ratio_hdi_lower"], np.exp(table["hdi_lower"])
    )
    assert np.allclose(
        table["shape_ratio_hdi_upper"], np.exp(table["hdi_upper"])
    )
    assert np.allclose(
        table["conditional_cv_ratio_hdi_lower"][1:],
        np.exp(-0.5 * table["hdi_upper"][1:]),
    )
    assert np.allclose(
        table["conditional_cv_ratio_hdi_upper"][1:],
        np.exp(-0.5 * table["hdi_lower"][1:]),
    )
    assert np.allclose(
        table["conditional_variance_ratio_hdi_lower"][1:],
        np.exp(-table["hdi_upper"][1:]),
    )
    assert np.allclose(
        table["conditional_variance_ratio_hdi_upper"][1:],
        np.exp(-table["hdi_lower"][1:]),
    )

    intercept = table.row(0, named=True)
    assert intercept["shape_ratio"] is not None
    contrast_columns = ALPHA_FIXED_EFFECTS_COLUMNS[16:]
    assert all(intercept[column] is None for column in contrast_columns)

    positive_effect = table.row(1, named=True)
    negative_effect = table.row(2, named=True)
    assert positive_effect["estimate"] > 0
    assert positive_effect["conditional_cv_ratio"] < 1
    assert positive_effect["conditional_variance_ratio"] < 1
    assert negative_effect["estimate"] < 0
    assert negative_effect["conditional_cv_ratio"] > 1
    assert negative_effect["conditional_variance_ratio"] > 1
    assert output_path.is_file()
    assert pl.read_csv(output_path).columns == list(
        ALPHA_FIXED_EFFECTS_COLUMNS
    )
