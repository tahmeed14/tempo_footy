"""Scientific invariants for hidden-block reconstruction and paired evaluation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tempoctrl.gradient_sports.ball_metrics import add_ball_metrics
from tempoctrl.gradient_sports.possessions_transform import label_pitch_thirds
from tempoctrl.gradient_sports.tempo_metrics import aggregate_possession_tempo
from tempoctrl.research.interpolation_data import (
    GapCase, KEY, PLAYER, TEAM, build_gap_cases, mask_physical_frames,
    prepare_match, split_matches,
)
from tempoctrl.research.interpolation_evaluation import (
    evaluate_cases, grouped_selection, length_weights, match_summary, paired_bootstrap,
)
from tempoctrl.research.interpolation_models import (
    Interpolator, candidate_interpolators, production_linear, reconstruct, transition,
)
from tempoctrl.research.interpolation_workflow import (
    InterpolationConfig, run_development, start_study,
)


def make_case(length=3, game=1):
    frames = np.arange(20)
    xy = np.column_stack([.01 * frames**2, .1 * frames])
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    return GapCase(f"{game}:1:8:{length}:11", game, 1, "player", "team", 11, length,
                   10., frames, xy, np.arange(8, 8 + length), distance, 1.9,
                   distance, 1.9, "PA", "VISIBLE", 4., 0.)


def test_split_is_match_disjoint_reproducible_and_metadata_only():
    inventory = pd.DataFrame({"game_id": np.arange(64), "week": [str(i % 4) for i in range(64)]})
    split = split_matches(inventory)
    pd.testing.assert_frame_equal(split, split_matches(inventory.sample(frac=1, random_state=3)))
    assert (split.split == "test").sum() == 13
    assert set(split.loc[split.split == "test", "fold"]) == {-1}
    assert set(split.loc[split.split == "development", "fold"]) == set(range(5))
    assert not split.game_id.duplicated().any()


def test_mask_removes_every_physical_copy_including_synthetic_and_other_possession():
    data = pl.DataFrame({
        "game_id": [1] * 4, "period": [1, 1, 1, 2], "framenum": [10, 10, 11, 10],
        PLAYER: ["a", "b", "a", "a"], "is_synthetic_pass_end": [False, True, False, False],
        "balls_smooth": [{"x": 1., "y": 2., "z": 3., "visibility": "VISIBLE"}] * 4,
    })
    masked = mask_physical_frames(data, pl.DataFrame({"game_id": [1], "period": [1], "framenum": [10]}))
    assert masked["balls_smooth"].struct.field("x").to_list() == [None, None, 1., 1.]
    assert masked["balls_smooth"].struct.field("z").to_list() == [None, None, 3., 3.]
    assert masked["balls_smooth"].struct.field("visibility").to_list()[:2] == [None, None]


@pytest.mark.parametrize("spec", candidate_interpolators((.5,)))
def test_candidates_preserve_inputs_and_use_only_masked_context(spec):
    case = make_case()
    masked = case.reference_xy.copy()
    masked[case.target] = np.nan
    before = masked.copy()
    result = reconstruct(spec, case.frames, masked, case.target, case.fps)
    np.testing.assert_array_equal(masked, before)
    assert result.coordinates.shape == (3, 2)
    if spec.family != "missing" and spec.max_gap >= 3:
        assert np.isfinite(result.coordinates).all()
    else:
        assert np.isnan(result.coordinates).all()


def test_hidden_target_leakage_and_boundary_extrapolation_are_rejected():
    case = make_case()
    with pytest.raises(ValueError, match="masked"):
        reconstruct(Interpolator("p", "pchip"), case.frames, case.reference_xy, case.target, case.fps)
    masked = case.reference_xy.copy()
    masked[:3] = np.nan
    with pytest.raises(ValueError, match="Two-sided"):
        reconstruct(Interpolator("p", "pchip"), case.frames, masked, np.arange(3), case.fps)


def test_linear_calls_production_with_numerical_frame_distance():
    frames = np.array([0, 1, 4])
    xy = np.array([[0., 0.], [np.nan, np.nan], [8., 4.]])
    np.testing.assert_allclose(production_linear(frames, xy, [1], max_gap=3), [[2., 1.]])
    assert np.isnan(production_linear(frames, xy, [1], max_gap=2)).all()


@pytest.mark.parametrize("d", [2, 3])
def test_exact_process_transition_composes_in_time(d):
    a, qa = transition(.1, d, 2.)
    b, qb = transition(.2, d, 2.)
    c, qc = transition(.3, d, 2.)
    np.testing.assert_allclose(c, b @ a, atol=1e-14)
    np.testing.assert_allclose(qc, b @ qa @ b.T + qb, atol=1e-14)
    assert np.linalg.eigvalsh(qc).min() > 0


def test_full_possession_delta_equals_direct_production_recalculation():
    case = make_case()
    spec = Interpolator("linear_5", "linear")
    scores, predictions = evaluate_cases([case], (spec, Interpolator("leave_missing", "missing")))
    coordinates = case.reference_xy.copy()
    coordinates[case.target] = np.asarray(predictions.iloc[0].predicted_xy)
    direct = pl.DataFrame({
        "game_id": [1] * 20, "framenum": case.frames, PLAYER: ["p"] * 20, TEAM: ["t"] * 20,
        "balls_smooth": [{"x": float(x), "y": float(y)} for x, y in coordinates],
    }).lazy().pipe(label_pitch_thirds).pipe(add_ball_metrics, TEAM, frame_rate=10.)
    actual = aggregate_possession_tempo(direct, "player").collect()["ball_speed_tempo_player"][0]
    row = scores[scores.method == "linear_5"].iloc[0]
    assert row.player_tempo == pytest.approx(actual, abs=1e-12)
    missing = scores[scores.method == "leave_missing"].iloc[0]
    assert np.isnan(missing.ade)
    assert missing.fill_fraction == 0
    assert missing.player_time_coverage < 1


def test_absence_of_natural_gaps_is_explicit_and_incomplete_support_cannot_win():
    gaps = pd.DataFrame({"game_id": [1], "gap_length": [2]})
    assert all(np.isnan(list(length_weights(gaps, [2]).values())))
    assert length_weights(gaps, [1])[2] == 1
    case = make_case()
    scores, _ = evaluate_cases([case], (Interpolator("linear_5", "linear"),))
    summary = match_summary(scores)
    assert not summary.complete_primary_support.any()
    assert summary.balanced_ade.isna().all()


def test_grouped_selection_never_uses_held_out_outcomes_for_tuning():
    records = []
    for match in range(4):
        for length in range(1, 6):
            for name in ("a", "b"):
                records.append({"game_id": match, "gap_length": length, "method": name,
                                "family": "pchip", "fill_fraction": 1., "status": "filled",
                                "ade": 1. if name == "a" else 2.})
    scores = pd.DataFrame(records)
    split = pd.DataFrame({"game_id": range(4), "split": "development", "fold": [0, 0, 1, 1]})
    gaps = pd.DataFrame({"game_id": range(4), "gap_length": [1] * 4})
    selected, _ = grouped_selection(scores, split, gaps)
    changed = scores.copy()
    changed.loc[(changed.game_id < 2) & (changed.method == "a"), "ade"] = 100
    selected_changed, _ = grouped_selection(changed, split, gaps)
    assert selected.loc[selected.fold == 0, "selected_method"].item() == "a"
    assert selected_changed.loc[selected_changed.fold == 0, "selected_method"].item() == "a"


def test_paired_bootstrap_requires_same_match_and_preserves_difference():
    frame = pd.DataFrame({"game_id": [1, 2, 3] * 2, "method": ["linear_5"] * 3 + ["pchip"] * 3,
                          "balanced_ade": [1., 2., 3., .5, 1.5, 2.5]})
    row = paired_bootstrap(frame, draws=100).query("method == 'pchip'").iloc[0]
    assert row.difference == pytest.approx(-.5)
    assert row.lower_95 == pytest.approx(-.5)
    assert row.paired_matches == 3


def test_period_resets_and_conflicting_duplicates_cannot_leak_into_masks(tmp_path):
    import json

    metadata, events = tmp_path / "metadata", tmp_path / "events"
    metadata.mkdir()
    events.mkdir()
    (metadata / "1.json").write_text(json.dumps([{"fps": 20.}]))
    pl.DataFrame({"match_team_possession_id": ["t"], "attacking_team_direction": ["R"]}).write_parquet(events / "1.parquet")
    rows = []
    for period in (1, 2):
        for frame in range(60):
            rows.append({"game_id": 1, "period": period, "framenum": frame,
                         "match_team_possession_id": "t", "match_team_player_possession_id": "p",
                         "successful_pass_or_cross": False, "possession_event_type": "PA",
                         "game_event_type": "PLAY",
                         "balls_smooth": {"x": frame / 10, "y": period * 1., "z": 0., "visibility": "VISIBLE"}})
    rows.append({**rows[20], "balls_smooth": {"x": 99., "y": 1., "z": 0., "visibility": "VISIBLE"}})
    path = tmp_path / "1.parquet"
    pl.DataFrame(rows).write_parquet(path)
    prepared = prepare_match(path, metadata, events)
    assert prepared.audit["conflicting_coordinate_keys"] == 1
    assert set(prepared.panel[PLAYER].drop_nulls()) == {"1|p", "2|p"}
    cases, support = build_gap_cases(prepared, seeds=(11, 23), lengths=(1, 2, 3, 4, 5), per_length=2)
    assert len(cases) == 20
    for seed in (11, 23):
        used = set()
        for case in (c for c in cases if c.seed == seed):
            keys = {(case.period, int(f)) for f in case.frames[case.target]}
            assert not used & keys
            assert (1, 20) not in keys
            assert np.all(np.diff(case.frames) == 1)
            used |= keys
    assert (support.sampled == 2).all()


def test_locked_test_matches_cannot_be_requested_by_development_runner(tmp_path):
    import json

    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src/tempoctrl").mkdir(parents=True)
    integrated, metadata = tmp_path / "integrated", tmp_path / "metadata"
    integrated.mkdir()
    metadata.mkdir()
    for game in range(8):
        (integrated / f"{game}.parquet").touch()
        (metadata / f"{game}.json").write_text(json.dumps([{"week": 1}]))
    config = replace(InterpolationConfig(), integrated="integrated", metadata="metadata", output="study")
    split = start_study(config, root=tmp_path)
    test_id = int(split.loc[split.split == "test", "game_id"].iloc[0])
    with pytest.raises(ValueError, match="Test matches remain locked"):
        run_development(config, root=tmp_path, match_ids=[test_id, 1])
    changed = replace(config, split_seed=config.split_seed + 1)
    with pytest.raises(ValueError, match="Existing locked split differs"):
        start_study(changed, root=tmp_path)
