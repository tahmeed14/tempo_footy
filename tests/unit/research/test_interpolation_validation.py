"""Scientific regressions for simultaneous masking and policy comparison."""

from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tempoctrl.gradient_sports.ball_metrics import add_ball_metrics
from tempoctrl.gradient_sports.possessions_transform import label_pitch_thirds
from tempoctrl.gradient_sports.tempo_metrics import aggregate_possession_tempo
from tempoctrl.research.interpolation_data import GapCase, PLAYER, TEAM
from tempoctrl.research.interpolation_evaluation import evaluate_cases
from tempoctrl.research.interpolation_models import Interpolator, reconstruct, reconstruct_blocks
from tempoctrl.research.interpolation_validation import (
    DecisionCriteria, ValidationConfig, assess_confirmation, decision_frontier, freeze_confirmation,
    load_policy_validation, policy_grid, policy_selection, run_confirmation, run_policy_validation,
    scoped_policy_metrics, validation_match_metrics,
)
from tempoctrl.research.interpolation_workflow import InterpolationConfig, start_study
from tempoctrl.research.bst_models import file_sha256, specification_hash


def joint_case():
    frames = np.arange(50)
    xy = np.column_stack([.004 * frames**2, np.sin(frames / 10)])
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    return GapCase("joint", 1, 1, "p", "t", 101, 4, 10., frames, xy,
                   np.r_[8:11, 25:29], distance, 4.9, distance, 4.9,
                   "PA", "VISIBLE", 4., .1, "all_context_visible", True,
                   "joint_gaps", "2_blocks_lengths_1_to_5", 5)


def test_every_block_conditions_on_identical_fully_masked_input(monkeypatch):
    from tempoctrl.research import interpolation_models as module

    case = joint_case()
    masked = case.reference_xy.copy()
    masked[case.target] = np.nan
    calls = []
    original = module.reconstruct

    def spy(spec, frames, xy, target, fps):
        assert np.isnan(xy[case.target]).all()
        calls.append(target.copy())
        return original(spec, frames, xy, target, fps)

    monkeypatch.setattr(module, "reconstruct", spy)
    reconstruct_blocks(Interpolator("pchip", "pchip"), case.frames, masked, case.target, case.fps)
    assert len(calls) == 2
    np.testing.assert_array_equal(np.concatenate(calls), case.target)
    assert np.isnan(masked[case.target]).all()


@pytest.mark.parametrize("family", ["linear", "pchip", "hybrid", "missing"])
def test_joint_tempo_matches_full_production_recomputation(family):
    case = joint_case()
    scores, predictions = evaluate_cases([case], (Interpolator(family, family),))
    xy = case.reference_xy.copy()
    xy[case.target] = np.asarray(predictions.iloc[0].predicted_xy)
    data = pl.DataFrame({"game_id": [1] * len(xy), "framenum": case.frames,
                         PLAYER: ["p"] * len(xy), TEAM: ["t"] * len(xy),
                         "balls_smooth": [{"x": float(x) if np.isfinite(x) else None,
                                           "y": float(y) if np.isfinite(y) else None} for x, y in xy]})
    metrics = data.lazy().pipe(label_pitch_thirds).pipe(add_ball_metrics, TEAM, frame_rate=case.fps)
    for level in ("player", "team"):
        direct = aggregate_possession_tempo(metrics, level).collect()[f"ball_speed_tempo_{level}"][0]
        assert scores.iloc[0][f"{level}_tempo"] == pytest.approx(direct, abs=1e-12)
    assert scores.n_blocks.iloc[0] == 2
    assert scores.hidden_frames.iloc[0] == 7
    if family != "missing":
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * case.fps
        reference = np.linalg.norm(np.diff(case.reference_xy, axis=0), axis=1) * case.fps
        hidden = np.isin(np.arange(len(xy)), case.target)
        affected = hidden[:-1] | hidden[1:]
        assert scores.speed_mae.iloc[0] == pytest.approx(np.abs(speed[affected] - reference[affected]).mean())


@pytest.mark.parametrize("length,expected", [(1, "linear"), (4, "pchip")])
def test_hybrid_rule_depends_only_on_observable_gap_length(length, expected):
    case = joint_case()
    target = np.arange(8, 8 + length)
    masked = case.reference_xy.copy()
    masked[target] = np.nan
    hybrid = reconstruct(Interpolator("h", "hybrid", 1.), case.frames, masked, target, case.fps)
    reference = reconstruct(Interpolator("r", expected, 1.), case.frames, masked, target, case.fps)
    np.testing.assert_allclose(hybrid.coordinates, reference.coordinates)


def test_each_block_has_its_own_gap_limit_and_partial_fills_are_explicit():
    case = joint_case()
    masked = case.reference_xy.copy()
    masked[case.target] = np.nan
    result = reconstruct_blocks(Interpolator("linear_3", "linear", max_gap=3),
                                case.frames, masked, case.target, case.fps)
    assert result.status == "partially_filled"
    assert np.isfinite(result.coordinates[:3]).all()
    assert np.isnan(result.coordinates[3:]).all()
    scores, _ = evaluate_cases([case], (Interpolator("linear_3", "linear", max_gap=3),))
    assert scores.ade.isna().all()
    assert scores.fill_fraction.iloc[0] == pytest.approx(3 / 7)


def test_possession_weighting_and_missing_tempo_denominator_are_visible():
    case = joint_case()
    scores, _ = evaluate_cases([case], (Interpolator("linear_5", "linear"),))
    data = pd.concat([scores] * 3, ignore_index=True)
    data["case_id"] = ["a", "b", "c"]
    data["player_possession"] = ["p1", "p1", "p2"]
    data["player_tempo_abs_error"] = [0., 2., 3.]
    data["player_tempo_bias"] = [0., -2., 3.]
    data["player_tempo_squared_error"] = [0., 4., 9.]
    row = validation_match_metrics(data).iloc[0]
    assert row.player_tempo_abs_error == pytest.approx(2.)  # mean(1,3)
    assert row.player_tempo_bias == pytest.approx(1.)  # mean(-1,3)
    assert row.player_tempo_rmse == pytest.approx(np.sqrt(5.5))
    assert row.player_tempo_p95 == 3.
    data.loc[2, "player_tempo_abs_error"] = np.nan
    assert validation_match_metrics(data).player_tempo_available_fraction.iloc[0] == pytest.approx(2 / 3)


def test_scoped_comparison_uses_declared_joint_design_cap():
    case = joint_case()
    specs = policy_grid(replace(ValidationConfig(), limits=(5, 10)))
    scores, _ = evaluate_cases([case], specs)
    # A short realized mask sampled under a wider design does not silently
    # enter the five-frame cohort just because this draw happened to be easy.
    scores["scenario_cap"] = 10
    summary = scoped_policy_metrics(scores, specs)
    assert set(summary.gap_limit) == {10}


def test_policy_selection_cannot_use_held_out_match_errors():
    rows = []
    for game in range(4):
        for experiment in ("single_gap", "joint_gaps"):
            for method, ade in (("linear_5", 2.), ("pchip_5", 1.)):
                rows.append(dict(game_id=game, method=method, gap_limit=5, experiment=experiment, ade=ade))
    metrics = pd.DataFrame(rows)
    splits = pd.DataFrame(dict(game_id=range(4), split="development", fold=[0, 0, 1, 1]))
    choices, _ = policy_selection(metrics, splits)
    metrics.loc[(metrics.game_id < 2) & (metrics.method == "pchip_5"), "ade"] = 100
    changed, _ = policy_selection(metrics, splits)
    assert choices.query("fold == 0").method.item() == "pchip_5"
    assert changed.query("fold == 0").method.item() == "pchip_5"


def test_frontier_preserves_coordinate_tempo_tradeoff():
    rows = []
    for method, coordinate, tempo in (("linear_5", 2., 1.), ("pchip_5", 1., 2.), ("bad_5", 3., 3.)):
        rows.append(dict(gap_limit=5, experiment="single_gap", method=method, ade=coordinate,
                         speed_mae=coordinate, player_tempo_abs_error=tempo, team_tempo_abs_error=tempo,
                         player_tempo_p95=tempo, team_tempo_p95=tempo))
    frontier = decision_frontier(pd.DataFrame(rows)).set_index("method")
    assert not frontier.loc["linear_5", "dominated"]
    assert not frontier.loc["pchip_5", "dominated"]
    assert frontier.loc["bad_5", "dominated"]


def test_confirmation_requires_explicit_scientific_limits_and_untampered_protocol(tmp_path):
    criteria = DecisionCriteria(*([None] * 11))
    with pytest.raises(ValueError, match="explicitly"):
        criteria.validate()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"candidate": "pchip_5", "protocol_sha256": "changed"}))
    with pytest.raises(ValueError, match="protocol changed"):
        run_confirmation(protocol)


def test_new_development_runner_cannot_open_reserved_matches(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src/tempoctrl").mkdir(parents=True)
    for folder in ("integrated", "metadata"):
        (tmp_path / folder).mkdir()
    for game in range(8):
        (tmp_path / "integrated" / f"{game}.parquet").touch()
        (tmp_path / "metadata" / f"{game}.json").write_text('[{"week":1}]')
    study = replace(InterpolationConfig(), integrated="integrated", metadata="metadata", output="study")
    split = start_study(study, root=tmp_path)
    test_id = int(split.query("split == 'test'").game_id.iloc[0])
    with pytest.raises(ValueError, match="test matches remain locked"):
        run_policy_validation(study=study, root=tmp_path, match_ids=[test_id])


def test_pilot_cannot_freeze_final_confirmation(monkeypatch, tmp_path):
    from tempoctrl.research import interpolation_validation as module

    monkeypatch.setattr(module, "load_policy_validation", lambda _: {
        "manifest": {"phase": "validation", "pilot_subset": True}})
    with pytest.raises(ValueError, match="not a pilot"):
        freeze_confirmation(tmp_path, "pchip_5", DecisionCriteria(*([1.] * 11)))


def test_confirmation_is_consumed_once_and_records_review_requirement(monkeypatch, tmp_path):
    from tempoctrl.research import interpolation_validation as module

    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src/tempoctrl").mkdir(parents=True)
    output = tmp_path / "study"
    output.mkdir()
    (output / "split_manifest.json").write_text("{}")
    development = tmp_path / "development"
    development.mkdir()
    (development / "manifest.json").write_text("{}")
    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "manifest.json").write_text(json.dumps({"artifacts_sha256": {}}))
    study = replace(InterpolationConfig(), output="study")
    protocol = {
        "study": asdict(study), "config": asdict(ValidationConfig()),
        "candidate": "pchip_5", "criteria": asdict(DecisionCriteria(*([1.] * 11))),
        "implementation_sha256": {}, "development_directory": str(development),
        "development_manifest_sha256": file_sha256(development / "manifest.json"),
        "split_manifest_sha256": file_sha256(output / "split_manifest.json"),
    }
    protocol["protocol_sha256"] = specification_hash(protocol)
    path = output / "confirmation_protocol.json"
    path.write_text(json.dumps(protocol))
    calls = []

    def fake_run(config, **kwargs):
        calls.append(kwargs["_protocol"]["candidate"])
        return completed

    monkeypatch.setattr(module, "load_policy_validation", lambda _: {})
    monkeypatch.setattr(module, "run_policy_validation", fake_run)
    monkeypatch.setattr(module, "assess_confirmation", lambda *args: pd.DataFrame({"passed": [True]}))
    assert run_confirmation(path, root=tmp_path) == completed
    state = json.loads((output / "confirmation_state.json").read_text())
    assert state["diagnostic_review_required"]
    assert not state["production_changed"]
    with pytest.raises(FileExistsError):
        run_confirmation(path, root=tmp_path)
    assert calls == ["pchip_5"]


def test_confirmation_rejects_one_downstream_metric_failure(monkeypatch, tmp_path):
    from tempoctrl.research import interpolation_validation as module

    scoped, paired = [], []
    metrics = ("ade", "p95_error", "speed_mae", "player_tempo_abs_error", "team_tempo_abs_error",
               "player_tempo_p95", "team_tempo_p95")
    for experiment in ("single_gap", "joint_gaps"):
        for game in (1, 2):
            scoped.append({"game_id": game, "method": "pchip_5", "gap_limit": 5,
                           "experiment": experiment, "fill_fraction": 1.,
                           "player_tempo_available_fraction": 1., "team_tempo_available_fraction": 1.,
                           **{m: .1 for m in metrics}})
        for metric in metrics:
            paired.append({"method": "pchip_5", "gap_limit": 5, "experiment": experiment,
                           "metric": metric, "upper_95": -.1})
    paired[-1]["upper_95"] = 2.  # Joint-gap team-tempo tail degradation.
    monkeypatch.setattr(module, "load_policy_validation", lambda _: {
        "scoped": pd.DataFrame(scoped), "paired": pd.DataFrame(paired), "manifest": {"matches": [1, 2]}})
    pd.DataFrame({"sampled": [1], "requested": [1]}).to_csv(tmp_path / "mask_support.csv", index=False)
    protocol = {"criteria": asdict(DecisionCriteria(*([1.] * 11))), "candidate": "pchip_5",
                "policies": [{"name": "pchip_5", "max_gap": 5}],
                "study": {"split_seed": 17}, "config": {"bootstrap_draws": 100}}
    # Set the requested coordinate gain to zero; all checks except the
    # deliberately worsened joint team-tempo tail should pass.
    protocol["criteria"]["minimum_coordinate_gain_m"] = 0.
    assessment = assess_confirmation(tmp_path, protocol)
    failures = assessment[~assessment.passed]
    assert failures.check.tolist() == ["joint_gaps:paired:team_tempo_p95"]


@pytest.mark.parametrize("problem,message", [
    ("joint", "own joint-mask scenario"),
    ("code", "implementation changed"),
    ("support", "unsupported requested development masks"),
])
def test_freezing_requires_tested_cap_code_identity_and_mask_support(monkeypatch, tmp_path, problem, message):
    from tempoctrl.research import interpolation_validation as module

    config = asdict(ValidationConfig())
    if problem == "joint":
        config["joint_length_caps"] = (15,)
    source = "interpolation_models.py"
    digest = file_sha256(module.Path(module.__file__).parent / source)
    manifest = {"phase": "validation", "pilot_subset": False,
                "policies": [asdict(Interpolator("pchip_5", "pchip"))],
                "config": config, "implementation_sha256": {source: "changed" if problem == "code" else digest}}
    monkeypatch.setattr(module, "load_policy_validation", lambda _: {"manifest": manifest})
    pd.DataFrame({"experiment": ["single_gap", "joint_gaps"], "gap_length": [5, np.nan],
                  "scenario_cap": [np.nan, 5], "sampled": [1, 2], "requested": [2, 2]}).to_csv(
                      tmp_path / "mask_support.csv", index=False)
    with pytest.raises(ValueError, match=message):
        freeze_confirmation(tmp_path, "pchip_5", DecisionCriteria(*([1.] * 11)))


def test_saved_validation_refuses_changed_masks_or_predictions(tmp_path):
    artifact = tmp_path / "cases.parquet"
    artifact.write_text("reference and masks")
    manifest = {"status": "validation_complete", "artifacts_sha256": {artifact.name: file_sha256(artifact)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    artifact.write_text("changed mask")
    with pytest.raises(ValueError, match="artifact changed"):
        load_policy_validation(tmp_path)


def test_freezing_writes_immutable_matched_cap_protocol_without_opening_test(monkeypatch, tmp_path):
    from tempoctrl.research import interpolation_validation as module

    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src/tempoctrl").mkdir(parents=True)
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "split_manifest.json").write_text("{}")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.json").write_text("{}")
    pd.DataFrame({"experiment": ["single_gap", "joint_gaps"], "gap_length": [5, np.nan],
                  "scenario_cap": [np.nan, 5], "sampled": [2, 2], "requested": [2, 2]}).to_csv(
                      evidence / "mask_support.csv", index=False)
    source = "interpolation_models.py"
    manifest = {"phase": "validation", "pilot_subset": False,
                "config": asdict(ValidationConfig()), "study": asdict(replace(InterpolationConfig(), output="study")),
                "policies": [asdict(s) for s in policy_grid()],
                "implementation_sha256": {source: file_sha256(module.Path(module.__file__).parent / source)},
                "split_manifest_sha256": file_sha256(study_dir / "split_manifest.json")}
    monkeypatch.setattr(module, "load_policy_validation", lambda _: {"manifest": manifest})
    criteria = DecisionCriteria(*([1.] * 11))
    path = freeze_confirmation(evidence, "hybrid_5", criteria, root=tmp_path)
    protocol = json.loads(path.read_text())
    signature = protocol.pop("protocol_sha256")
    assert specification_hash(protocol) == signature
    assert protocol["config"]["lengths"] == [1, 2, 3, 4, 5]
    assert protocol["config"]["joint_length_caps"] == [5]
    assert [p["name"] for p in protocol["policies"]] == ["linear_5", "hybrid_5"]
    assert not (study_dir / "confirmation_state.json").exists()
    assert freeze_confirmation(evidence, "hybrid_5", criteria, root=tmp_path) == path
    with pytest.raises(ValueError, match="different immutable"):
        freeze_confirmation(evidence, "hybrid_5", replace(criteria, maximum_ade_m=2.), root=tmp_path)
