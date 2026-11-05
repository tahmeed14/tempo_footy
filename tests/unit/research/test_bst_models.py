"""Scientific and reproducibility invariants for the BST comparison workflow."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tempoctrl.research.bst_models import (
    FAMILIES, OBSERVATION_KEY, OUTCOME_VARIABLE, REFERENCE_LEVELS,
    conditional_moments, log_cv, prepare_data, prior_spec,
)
from tempoctrl.research.bst_evaluation import compare_saved_runs, predictive_checks
from tempoctrl.research.bst_workflow import RunConfig, evaluate_run, memory_estimate, run_family


@pytest.fixture
def processed_data(tmp_path):
    """Small observed-design cohort with recurring players and varying sequence."""
    rng = np.random.default_rng(13)
    n = 60
    frame = pd.DataFrame({
        OBSERVATION_KEY: [f"possession-{i:03d}" for i in range(n)],
        "game_id": np.arange(n) % 4,
        "player_id": np.arange(n) % 6,
        "opponent_id": np.arange(n) % 3,
        OUTCOME_VARIABLE: rng.gamma(3, 7 / 3, n),
        "player_possession_sequence_number": 1 + np.arange(n) % 7,
        "game_state_goal_diff": np.arange(n) % 3 - 1,
        # Deliberately stale preprocessing must never be reused.
        "player_possession_sequence_log_z": 999.0,
    })
    for name, levels in REFERENCE_LEVELS.items():
        frame[name] = [levels[i % len(levels)] for i in range(n)]
    path = tmp_path / "processed.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_preparation_freezes_rows_preserves_contrasts_and_recomputes_sequence(processed_data):
    prepared = prepare_data(processed_data)
    frame = prepared.frame
    assert frame[OBSERVATION_KEY].is_monotonic_increasing
    assert frame.player_possession_sequence_log_z.mean() == pytest.approx(0, abs=1e-14)
    assert frame.player_possession_sequence_log_z.std() == pytest.approx(1)
    for name, levels in REFERENCE_LEVELS.items():
        assert frame[name].cat.categories.to_list() == levels
        assert f"{name}[{levels[0]}]" not in prepared.mean_design
    assert prepared.audit["cohort_sha256"] == prepare_data(processed_data).audit["cohort_sha256"]


def test_invalid_rows_are_logged_without_reapplying_an_outlier_cutoff(processed_data):
    frame = pd.read_parquet(processed_data)
    frame.loc[0, OUTCOME_VARIABLE] = 35
    frame.loc[1, OUTCOME_VARIABLE] = 0
    frame.loc[2, OUTCOME_VARIABLE] = np.inf
    frame.loc[3, "first_touch_ballheight"] = None
    frame.to_parquet(processed_data, index=False)
    prepared = prepare_data(processed_data)
    assert prepared.audit["included_rows"] == len(frame) - 3
    assert prepared.audit["excluded_rows"] == 3
    assert prepared.frame[OUTCOME_VARIABLE].max() == 35
    assert prepared.exclusions.reason.str.len().min() > 0


def test_updated_snapshot_changes_identity(processed_data):
    before = prepare_data(processed_data)
    frame = pd.read_parquet(processed_data)
    frame.loc[0, OUTCOME_VARIABLE] += .1
    frame.to_parquet(processed_data, index=False)
    after = prepare_data(processed_data)
    assert before.audit["source_sha256"] != after.audit["source_sha256"]
    assert before.audit["cohort_sha256"] != after.audit["cohort_sha256"]


@pytest.mark.parametrize("problem", ["duplicate_key", "unknown_category", "constant_sequence"])
def test_ambiguous_inputs_fail_explicitly(processed_data, problem):
    frame = pd.read_parquet(processed_data)
    if problem == "duplicate_key":
        frame.loc[0, OBSERVATION_KEY] = frame.loc[1, OBSERVATION_KEY]
    elif problem == "unknown_category":
        frame.loc[0, "first_touch_ballheight"] = "Unknown"
    else:
        frame.player_possession_sequence_number = 1
    frame.to_parquet(processed_data, index=False)
    with pytest.raises(ValueError):
        prepare_data(processed_data)


@pytest.mark.parametrize("family", FAMILIES)
def test_prior_centers_and_local_widths_have_comparable_cv_meanings(family):
    strong, base, weak = [prior_spec(family, name) for name in ("stronger", "baseline", "weaker")]
    for spec in (strong, base, weak):
        assert np.exp(log_cv(family, spec.dispersion_intercept_center)) == pytest.approx(1 / np.sqrt(3))
        assert spec.mean_intercept_center == base.mean_intercept_center
        assert spec.dispersion_intercept_center == base.dispersion_intercept_center
    assert base.dispersion_intercept_sd * abs(base.log_cv_slope) == pytest.approx(.25)
    assert strong.dispersion_group_sd < base.dispersion_group_sd < weak.dispersion_group_sd


@pytest.mark.parametrize("family", FAMILIES)
def test_drawwise_original_scale_moments_match_independent_scipy_distributions(family):
    mean = np.array([3.0, 7.0, 12.0])
    dispersion = np.array([.8, 1.5, 3.0])
    if family == "gamma":
        reference = stats.gamma(dispersion, scale=mean / dispersion)
    elif family == "lognormal":
        reference = stats.lognorm(dispersion, scale=mean * np.exp(-dispersion**2 / 2))
    else:
        from scipy.special import gamma
        reference = stats.weibull_min(dispersion, scale=mean / gamma(1 + 1 / dispersion))
    calculated = conditional_moments(family, mean, dispersion)
    np.testing.assert_allclose(calculated["mean"], reference.mean())
    np.testing.assert_allclose(calculated["sd"], reference.std(), rtol=1e-12)
    np.testing.assert_allclose(calculated["cv"], reference.std() / reference.mean(), rtol=1e-12)


def test_gamma_mean_sd_interface_preserves_the_manuscript_mean_shape_likelihood():
    """Changing the displayed Gamma convention must not change its density."""
    import pymc as pm
    import pytensor
    from pytensor.compile.mode import Mode

    mean = np.array([2.0, 7.0, 12.0, 4.0])
    shape = np.array([.6, 3.0, 8.0, 1.0])
    y = np.array([.1, 6.0, 20.0, 4.0])
    sd = mean / np.sqrt(shape)
    manuscript = pm.Gamma.dist(mu=mean, sigma=sd)
    shape_rate = pm.Gamma.dist(alpha=shape, beta=shape / mean)
    evaluate = pytensor.function(
        [], [pm.logp(manuscript, y), pm.logp(shape_rate, y)],
        mode=Mode(linker="py", optimizer="fast_compile"),
    )
    actual, previous = evaluate()
    expected = stats.gamma.logpdf(y, a=shape, scale=mean / shape)
    np.testing.assert_allclose(actual, previous, rtol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(sd**2, mean**2 / shape)


def test_predictive_checks_detect_shape_and_numeric_errors(processed_data):
    frame = prepare_data(processed_data).frame
    with pytest.raises(ValueError, match="shape"):
        predictive_checks(frame, np.ones((5, len(frame) - 1)))
    with pytest.raises(ValueError, match="non-finite"):
        predictive_checks(frame, np.full((5, len(frame)), np.inf))
    checks, coverage = predictive_checks(frame, np.tile(frame[OUTCOME_VARIABLE], (10, 1)))
    assert (coverage.in_sample_coverage == 1).all()
    assert checks.query("statistic == 'mean' and variable == 'overall'").observed.iloc[0] == pytest.approx(frame[OUTCOME_VARIABLE].mean())


def test_preparation_is_opt_in_and_new_runs_do_not_overwrite(processed_data, tmp_path):
    config = RunConfig(family="gamma", prior="baseline", data=processed_data, output=tmp_path / "runs")
    first = run_family(config)[0]
    second = run_family(config)[0]
    assert first != second
    assert not (first / "posterior.nc").exists()
    assert json.loads((first / "manifest.json").read_text())["status"] == "prepared"
    assert (first / "methods.md").exists()
    low_memory = replace(config, mode="fit", max_working_gib=1e-10)
    with pytest.raises(MemoryError):
        run_family(low_memory)
    assert memory_estimate(100, config)["log_likelihood_gib"] > 0


def test_evaluation_rejects_a_changed_saved_snapshot(processed_data, tmp_path):
    config = RunConfig(family="gamma", prior="baseline", data=processed_data, output=tmp_path / "runs")
    directory = run_family(config)[0]
    frozen = pd.read_parquet(directory / "observations.parquet")
    frozen.loc[0, OUTCOME_VARIABLE] += 1
    frozen.to_parquet(directory / "observations.parquet", index=False)
    with pytest.raises(ValueError, match="snapshot changed"):
        evaluate_run(directory)


def _saved_comparison_run(directory: Path, *, cohort="same", shift=0.0):
    directory.mkdir()
    manifest = {
        "status": "complete", "family": "gamma", "prior": {"name": "baseline"},
        "data": {"source_sha256": "source", "cohort_sha256": cohort},
        "implementation_sha256": "code",
        "evaluation": {"elpd_loo": -8.0 + 4 * shift, "sampling_diagnostics_pass": True, "loo_diagnostics_pass": True},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame({OBSERVATION_KEY: ["a", "b", "c", "d"], "game_id": [1, 1, 2, 2],
                  "player_id": [1, 2, 1, 2], "elpd_loo": np.full(4, -2 + shift)}).to_csv(directory / "loo_pointwise.csv", index=False)


def test_saved_comparison_uses_paired_scores_and_rejects_mismatched_data(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    _saved_comparison_run(first)
    _saved_comparison_run(second, shift=.5)
    result = compare_saved_runs([first, second], tmp_path / "comparison")
    assert result.elpd_difference_vs_first.to_list() == [0, 2]
    assert result.paired_observation_se.to_list() == [0, 0]
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["data"]["cohort_sha256"] = "different"
    (second / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differ in data"):
        compare_saved_runs([first, second], tmp_path / "bad_comparison")
