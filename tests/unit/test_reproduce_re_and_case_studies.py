"""Test random-effects and case-study reproduction."""

import math

import polars as pl
import pytest

from tempoctrl.model.utils import summarize_archetype_overlap
from tempoctrl.reproduce_re_and_case_studies import (
    transform_overlap_to_percentage_effects,
)


def _overlap() -> pl.DataFrame:
    """Build four archetype quadrants with explicit effect estimates."""
    return pl.DataFrame(
        {
            "mu_posterior_mean": [math.log(2)],
            "mu_hdi_lower": [math.log(1.5)],
            "mu_hdi_upper": [math.log(2.5)],
            "mu_p_gt_zero": [0.95],
            "mu_credible": ["Credible"],
            "shape_posterior_mean": [math.log(4)],
            "shape_hdi_lower": [math.log(2)],
            "shape_hdi_upper": [math.log(8)],
            "shape_p_gt_zero": [0.98],
            "shape_credible": ["Credible"],
        }
    )


def test_transform_overlap_to_percentage_effects_reverses_cv_direction() -> (
    None
):
    """Convert known shape effects with the inverse CV relationship."""
    transformed = transform_overlap_to_percentage_effects(_overlap()).row(
        0,
        named=True,
    )

    assert transformed["mu_posterior_mean"] == pytest.approx(100)
    assert transformed["mu_hdi_lower"] == pytest.approx(50)
    assert transformed["mu_hdi_upper"] == pytest.approx(150)
    assert transformed["shape_posterior_mean"] == pytest.approx(-50)
    assert transformed["shape_hdi_lower"] == pytest.approx(
        100 / math.sqrt(8) - 100
    )
    assert transformed["shape_hdi_upper"] == pytest.approx(
        100 / math.sqrt(2) - 100
    )
    assert transformed["shape_p_gt_zero"] == pytest.approx(0.02)


def test_summarize_archetype_overlap_labels_conditional_cv_directly() -> None:
    """Label explicit quadrant counts by conditional variability."""
    transformed = transform_overlap_to_percentage_effects(_overlap())

    summary = summarize_archetype_overlap(
        transformed,
        dispersion_scale="conditional_cv",
    )

    assert summary["category"].to_list() == [
        "High Tempo (+%Δμ), High Variability (+%ΔCV)",
        "High Tempo (+%Δμ), Low Variability (-%ΔCV)",
        "Low Tempo (-%Δμ), High Variability (+%ΔCV)",
        "Low Tempo (-%Δμ), Low Variability (-%ΔCV)",
    ]
    assert summary["number_of_players"].to_list() == [0, 1, 0, 0]
