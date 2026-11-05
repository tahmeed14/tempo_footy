"""Keep sequential teaching models aligned with the reusable research models."""

from dataclasses import replace
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from pytensor.compile.mode import Mode
import pytest

from tempoctrl.research.bst_interactive import (
    interactive_model_source_hash, load_interactive_snapshot, start_interactive_run,
)
from tempoctrl.research.bst_models import (
    FAMILIES, OBSERVATION_KEY, OUTCOME_VARIABLE, REFERENCE_LEVELS,
    build_model, prepare_data, repository_root,
)
from tempoctrl.research.bst_workflow import RunConfig


@pytest.fixture
def teaching_data(tmp_path):
    count = 30
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({
        OBSERVATION_KEY: [f"p-{number:03d}" for number in range(count)],
        "game_id": np.arange(count) % 3,
        "player_id": np.arange(count) % 6,
        "opponent_id": np.arange(count) % 3,
        OUTCOME_VARIABLE: rng.gamma(3, 7 / 3, count),
        "player_possession_sequence_number": 1 + np.arange(count) % 7,
        "game_state_goal_diff": np.arange(count) % 3 - 1,
    })
    for name, levels in REFERENCE_LEVELS.items():
        frame[name] = [levels[number % len(levels)] for number in range(count)]
    path = tmp_path / "teaching.parquet"
    frame.to_parquet(path, index=False)
    return path


def _walkthrough(family, data, output):
    path = repository_root() / "notebooks/research_investigations" / f"{family}_model_comparison_interactive.py"
    namespace = {"__name__": "__main__"}
    for cell in path.read_text().split("# %%"):
        if not cell.strip() or cell.startswith(" [markdown]"):
            continue
        exec(compile(cell, str(path), "exec"), namespace)
        if "CONFIG = RunConfig(" in cell:
            namespace["CONFIG"] = replace(namespace["CONFIG"], data=data, output=output)
    return namespace


@pytest.mark.parametrize("family", FAMILIES)
def test_teaching_cells_build_the_same_likelihood_and_joint_density_without_sampling(
    teaching_data, tmp_path, monkeypatch, family,
):
    def unexpected_sampling(*args, **kwargs):
        pytest.fail("The default teaching walkthrough must not start sampling.")

    for name in ("sample", "sample_prior_predictive", "sample_posterior_predictive"):
        monkeypatch.setattr(pm, name, unexpected_sampling)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = _walkthrough(family, teaching_data, tmp_path / "runs")
    teaching_model = namespace["model"]
    reference_model = build_model(namespace["prepared"], namespace["priors"])
    # Exercise both graphs at a nontrivial point. Python mode avoids requiring
    # a working local C compiler for this scientific equivalence check.
    with pytensor.config.change_flags(cxx=""):
        point = reference_model.initial_point(random_seed=0)
        rng = np.random.default_rng(9)
        point = {key: value + rng.normal(0, .08, size=value.shape) for key, value in point.items()}
        mode = Mode(linker="py", optimizer="fast_compile")
        for observed_only in (False, True):
            reference = reference_model.compile_logp(
                vars=reference_model.observed_RVs if observed_only else None,
                mode=mode,
            )
            teaching = teaching_model.compile_logp(
                vars=teaching_model.observed_RVs if observed_only else None,
                mode=mode,
            )
            np.testing.assert_allclose(teaching(point), reference(point), rtol=1e-12)
    assert namespace["idata"] is None
    assert namespace["prior_idata"] is None
    manifest = json.loads((namespace["RUN_DIRECTORY"] / "manifest.json").read_text())
    assert manifest["workflow"] == "sequential_interactive"
    assert len(manifest["interactive_sources"]) == 4
    plt.close("all")


def test_saved_walkthrough_restores_its_original_snapshot(teaching_data, tmp_path):
    prepared = prepare_data(teaching_data)
    config = RunConfig(family="gamma", prior="baseline", data=teaching_data, output=tmp_path / "runs")
    directory = start_interactive_run(prepared, config)
    # Replacing the current input must not affect saved-fit exploration.
    frame = pd.read_parquet(teaching_data)
    frame.loc[0, OUTCOME_VARIABLE] += 10
    frame.to_parquet(teaching_data, index=False)
    restored, settings = load_interactive_snapshot(directory, "gamma")
    assert settings.prior == "baseline"
    assert restored.audit["cohort_sha256"] == prepared.audit["cohort_sha256"]
    np.testing.assert_allclose(restored.mean_design, prepared.mean_design)
    with pytest.raises(ValueError, match="Saved family"):
        load_interactive_snapshot(directory, "weibull")
    assert config.data == teaching_data


def test_sequential_walkthrough_rejects_an_all_priors_loop(teaching_data, tmp_path):
    config = RunConfig(family="gamma", data=teaching_data, output=tmp_path / "runs")
    with pytest.raises(ValueError, match="Choose one prior"):
        start_interactive_run(prepare_data(teaching_data), config)


def test_editing_prior_selection_preserves_comparability_but_changing_likelihood_does_not(tmp_path):
    source = repository_root() / "notebooks/research_investigations/gamma_model_comparison_interactive.py"
    edited = tmp_path / source.name
    baseline = source.read_text()
    edited.write_text(baseline.replace('PRIOR_NAME = "baseline"', 'PRIOR_NAME = "stronger"'))
    assert interactive_model_source_hash(source) == interactive_model_source_hash(edited)
    edited.write_text(baseline.replace("sigma=mu / pt.sqrt(alpha)", "sigma=mu * pt.sqrt(alpha)"))
    assert interactive_model_source_hash(source) != interactive_model_source_hash(edited)
