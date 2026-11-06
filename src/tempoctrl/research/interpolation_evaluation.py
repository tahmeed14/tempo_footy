"""Paired coordinate and production-metric evaluation of independently masked gaps."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm, chi2

from tempoctrl.gradient_sports.ball_metrics import add_ball_metrics
from tempoctrl.gradient_sports.possessions_transform import label_pitch_thirds
from tempoctrl.gradient_sports.tempo_metrics import aggregate_possession_tempo
from tempoctrl.research.interpolation_data import GapCase, PLAYER, TEAM
from tempoctrl.research.interpolation_models import Interpolator, reconstruct_blocks


def _coordinate_scores(reference, prediction, variance):
    finite = np.isfinite(prediction).all(axis=1)
    result = {"fill_fraction": float(finite.mean()), "ade": np.nan,
              "rmse": np.nan, "p95_error": np.nan, "max_error": np.nan,
              "bias_x": np.nan, "bias_y": np.nan}
    if not finite.all():
        # No accuracy score on a favorable successful subset.
        return result
    residual = prediction - reference
    error = np.linalg.norm(residual, axis=1)
    result.update(ade=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                  p95_error=float(np.quantile(error, .95)), max_error=float(error.max()),
                  bias_x=float(residual[:, 0].mean()), bias_y=float(residual[:, 1].mean()))
    result.update({f"within_{threshold:g}m": float((error <= threshold).mean())
                   for threshold in (.25, .5, 1., 2.)})
    if variance is not None:
        variance = np.asarray(variance)
        if variance.shape != residual.shape or not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("Predictive observation variances must be finite and positive.")
        z2 = residual**2 / variance
        result["observation_nlpd"] = float(np.mean(.5 * np.sum(np.log(2 * np.pi * variance) + z2, axis=1)))
        for probability in (.5, .8, .95):
            cutoff = norm.ppf((1 + probability) / 2)
            for j, axis in enumerate(("x", "y")):
                result[f"coverage_{axis}_{probability:g}"] = float((z2[:, j] <= cutoff**2).mean())
                result[f"width_{axis}_{probability:g}"] = float((2 * cutoff * np.sqrt(variance[:, j])).mean())
            result[f"coverage_joint_{probability:g}"] = float((z2.sum(axis=1) <= chi2.ppf(probability, 2)).mean())
    return result


def evaluate_cases(cases: list[GapCase], specs: tuple[Interpolator, ...]):
    """Score all attempts; batch the existing metric/aggregation implementations.

    Each sample perturbs one or several interior gaps together. The production distance and
    covered-time changes are added to original full-possession totals, so these
    are actual affected-player/team tempo changes, not just bridge average speed.
    This equivalence holds because targets are strictly inside one possession
    and never synthetic/conflicting/duplicate physical observations.
    """
    records, predictions, bridge_rows = [], [], []
    case_lookup = {case.case_id: case for case in cases}
    if len(case_lookup) != len(cases):
        raise ValueError("Case identifiers must be unique.")
    if len({c.fps for c in cases}) > 1:
        raise ValueError("Evaluate one FPS batch at a time.")
    for case in cases:
        masked = case.reference_xy.copy()
        masked[case.target] = np.nan
        left, right = case.target[0] - 1, case.target[-1] + 1
        local_reference = case.reference_xy[left:right + 1]
        local_frames = case.frames[left:right + 1]
        dt = np.diff(local_frames) / case.fps
        hidden = np.zeros(len(local_frames), dtype=bool)
        hidden[case.target - left] = True
        affected = hidden[:-1] | hidden[1:]
        reference_velocity = np.diff(local_reference, axis=0) / dt[:, None]
        reference_speed = np.linalg.norm(reference_velocity, axis=1)
        # A reference bridge is scored through exactly the same production calls.
        for method, local in [("__reference", local_reference)]:
            for frame, position in zip(local_frames, local, strict=True):
                bridge_rows.append(_bridge_row(case, method, frame, position))
        for spec in specs:
            start = perf_counter()
            try:
                result = reconstruct_blocks(spec, case.frames, masked, case.target, case.fps)
                prediction, variance, status = result.coordinates, result.observation_variance, result.status
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                prediction, variance, status = np.full((len(case.target), 2), np.nan), None, f"failed:{type(error).__name__}:{error}"
            elapsed = perf_counter() - start
            if prediction.shape != (len(case.target), 2):
                raise ValueError("Unexpected target prediction shape.")
            local = local_reference.copy()
            local[case.target - left] = prediction
            for frame, position in zip(local_frames, local, strict=True):
                bridge_rows.append(_bridge_row(case, spec.name, frame, position))
            velocity = np.diff(local, axis=0) / dt[:, None]
            speed = np.linalg.norm(velocity, axis=1)
            valid = np.isfinite(speed[affected])
            record = {
                "case_id": case.case_id, "game_id": case.game_id, "period": case.period,
                "seed": case.seed, "gap_length": case.gap_length,
                "experiment": case.experiment, "scenario": case.scenario,
                "scenario_cap": case.scenario_cap or case.gap_length,
                "player_possession": case.player_possession, "team_possession": case.team_possession,
                "hidden_frames": len(case.target),
                "n_blocks": int(1 + (np.diff(case.target) > 1).sum()),
                "missing_seconds": len(case.target) / case.fps,
                "anchor_seconds": (case.frames[right] - case.frames[left]) / case.fps,
                "method": spec.name, "family": spec.family, "status": status,
                "context_seconds": spec.context_seconds, "max_gap": spec.max_gap,
                "inference_seconds": elapsed, "event_type": case.event_type,
                "visibility": case.visibility, "max_reference_speed": case.max_reference_speed,
                "turn_radians": case.turn_radians,
                "reference_quality": case.reference_quality,
                "all_targets_visible": case.all_targets_visible,
                "speed_mae": float(np.mean(np.abs(speed[affected] - reference_speed[affected]))) if valid.all() else np.nan,
                "speed_rmse": float(np.sqrt(np.mean((speed[affected] - reference_speed[affected])**2))) if valid.all() else np.nan,
                "velocity_mae": float(np.linalg.norm(velocity[affected] - reference_velocity[affected], axis=1).mean()) if valid.all() else np.nan,
                "fixed_segment_coverage": float(valid.mean()),
                "outside_105x68_fraction": float(((np.abs(prediction[:, 0]) > 52.5) | (np.abs(prediction[:, 1]) > 34)).mean()) if np.isfinite(prediction).all() else np.nan,
                **_coordinate_scores(case.reference_xy[case.target], prediction, variance),
            }
            records.append(record)
            predictions.append({"case_id": case.case_id, "method": spec.name,
                                "predicted_xy": prediction.tolist(),
                                "observation_variance": None if variance is None else variance.tolist()})
    if not records:
        raise ValueError("No supported evaluation cases; inspect the audit and support table.")
    bridge = pl.DataFrame(bridge_rows, schema_overrides={
        "balls_smooth": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
    })
    metrics = bridge.lazy().pipe(label_pitch_thirds).pipe(add_ball_metrics, TEAM, frame_rate=cases[0].fps).collect()
    # Both aggregations are reused; no independently redefined tempo estimator.
    player = aggregate_possession_tempo(metrics.lazy(), "player").collect()
    team = aggregate_possession_tempo(metrics.lazy(), "team").collect()
    player_totals = {r["match_team_player_possession_id"]: r for r in player.iter_rows(named=True)}
    team_totals = {r["match_team_possession_id"]: r for r in team.iter_rows(named=True)}
    pitch = {r["bridge_id"]: r["pitch_third"] for r in metrics.group_by("bridge_id", maintain_order=True)
             .agg(pl.col("pitch_third").slice(1, pl.len() - 2)).iter_rows(named=True)}
    for record in records:
        case = case_lookup[record["case_id"]]
        left = int(case.target[0] - 1)
        ref_key, pred_key = f"{case.case_id}|__reference", f"{case.case_id}|{record['method']}"
        for level, totals in (("player", player_totals), ("team", team_totals)):
            ref_local, pred_local = totals[ref_key], totals[pred_key]
            reference_distance = getattr(case, f"{level}_distance")
            reference_seconds = getattr(case, f"{level}_seconds")
            distance = reference_distance - (ref_local["total_ball_displacement"] or 0) + (pred_local["total_ball_displacement"] or 0)
            seconds = reference_seconds - (ref_local["elapsed_frames"] or 0) / case.fps + (pred_local["elapsed_frames"] or 0) / case.fps
            tempo = distance / seconds if seconds > 1e-10 else np.nan
            reference_tempo = reference_distance / reference_seconds
            record[f"{level}_tempo_reference"] = reference_tempo
            record[f"{level}_tempo"] = tempo
            record[f"{level}_tempo_bias"] = tempo - reference_tempo
            record[f"{level}_tempo_abs_error"] = abs(tempo - reference_tempo)
            record[f"{level}_tempo_squared_error"] = (tempo - reference_tempo)**2
            record[f"{level}_time_coverage"] = seconds / reference_seconds
            record[f"{level}_distance_coverage"] = distance / reference_distance if reference_distance > 0 else np.nan
        record["path_length_bias"] = (player_totals[pred_key]["total_ball_displacement"] or 0) - player_totals[ref_key]["total_ball_displacement"]
        record["path_length_abs_error"] = abs(record["path_length_bias"])
        record["tempo_range_inclusion_changed"] = (0 < record["player_tempo_reference"] < 25) != (0 < record["player_tempo"] < 25)
        record["pitch_third_disagreement"] = float(np.mean([
            pitch[ref_key][i - left - 1] != pitch[pred_key][i - left - 1]
            for i in case.target
        ]))
    return pd.DataFrame(records), pd.DataFrame(predictions)


def _bridge_row(case, method, frame, position):
    key = f"{case.case_id}|{method}"
    return {"bridge_id": key, "game_id": case.game_id, "framenum": int(frame),
            PLAYER: key, TEAM: key,
            "balls_smooth": {c: float(position[j]) if np.isfinite(position[j]) else None
                             for j, c in enumerate(("x", "y"))}}


def length_weights(gaps: pd.DataFrame, matches, lengths=(1, 2, 3, 4, 5)) -> dict:
    """Training-match natural-gap frequencies; no silent uniform substitution."""
    selected = gaps[gaps.game_id.isin(matches) & gaps.gap_length.isin(lengths)]
    counts = selected.gap_length.value_counts()
    total = int(counts.sum())
    return {int(length): float(counts.get(length, 0) / total) if total else np.nan for length in lengths}


def match_summary(scores: pd.DataFrame, weights=None, *, primary=True):
    """Equal gap weight within length, then fixed length weights within match.

    Completeness is explicit: a missing positive-weight length or unfilled target
    invalidates ADE for that match/method rather than dropping its contribution.
    Other diagnostics are balanced across lengths; P95 is the mean gap-level P95,
    not a pooled frame percentile. This is named explicitly in reports.
    """
    data = scores[scores.gap_length <= 5] if primary else scores
    columns = [c for c in data.select_dtypes(include=["number", "bool"]).columns
               if c not in ("game_id", "period", "seed", "gap_length")]
    groups = data.groupby(["game_id", "method", "family", "gap_length"], observed=True)
    strata = groups[columns].mean().reset_index()
    out = []
    for (game_id, method, family), group in strata.groupby(["game_id", "method", "family"]):
        row = {"game_id": game_id, "method": method, "family": family}
        row.update(group[columns].mean().to_dict())
        selected = data[(data.game_id == game_id) & (data.method == method)]
        complete = (set(group.gap_length) == set(range(1, 6))) if primary else True
        complete &= bool((selected.fill_fraction == 1).all())
        row["balanced_ade"] = float(group.ade.mean()) if complete else np.nan
        row["complete_primary_support"] = complete
        row["attempted_gaps"] = len(selected)
        row["failed_fraction"] = float(selected.status.str.startswith("failed").mean())
        row["fallback_fraction"] = float((selected.status == "linear_fallback").mean())
        row["weighted_ade"] = np.nan
        if weights and all(np.isfinite(list(weights.values()))):
            present = group.set_index("gap_length").ade
            needed = [k for k, w in weights.items() if w > 0]
            if complete and all(k in present and np.isfinite(present[k]) for k in needed):
                row["weighted_ade"] = sum(weights[k] * present[k] for k in needed)
        out.append(row)
    return pd.DataFrame(out)


def paired_bootstrap(match_scores: pd.DataFrame, metric="balanced_ade", *, baseline="linear_5", draws=2000, seed=42):
    """Paired match-cluster differences, never frame-level bootstrap replicates."""
    base = match_scores[match_scores.method == baseline].set_index("game_id")[metric]
    rows = []
    rng = np.random.default_rng(seed)
    for method, group in match_scores.groupby("method"):
        paired = pd.concat([base.rename("baseline"), group.set_index("game_id")[metric].rename("candidate")], axis=1).dropna()
        delta = (paired.candidate - paired.baseline).to_numpy()
        if len(delta) < 2:
            continue
        boot = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
        rows.append({"method": method, "metric": metric, "paired_matches": len(delta),
                     "difference": float(delta.mean()), "lower_95": float(np.quantile(boot, .025)),
                     "upper_95": float(np.quantile(boot, .975)),
                     "baseline_mean": float(paired.baseline.mean())})
    return pd.DataFrame(rows)


def grouped_selection(scores, splits, natural_gaps):
    """Tune each family using training matches; assess on disjoint fold matches.

    Candidate local fits condition on visible validation anchors; they learn no
    global parameters. Hyperparameter selection and natural weights use only the
    fold's training matches. Family comparisons remain development comparisons.
    """
    available = set(scores.game_id)
    dev = splits[(splits.split == "development") & splits.game_id.isin(available)]
    selections, assessed = [], []
    for fold in sorted(dev.fold.unique()):
        train = dev.loc[dev.fold != fold, "game_id"].tolist()
        valid = dev.loc[dev.fold == fold, "game_id"].tolist()
        if not train or not valid:
            continue
        weights = length_weights(natural_gaps, train)
        training = match_summary(scores[scores.game_id.isin(train)], weights)
        validation = match_summary(scores[scores.game_id.isin(valid)], weights)
        metric = "weighted_ade" if np.isfinite(list(weights.values())).all() else "balanced_ade"
        for family, group in training.groupby("family"):
            candidates = group.groupby("method").agg(value=(metric, "mean"), n=(metric, "count"))
            candidates = candidates[(candidates.n == len(train)) & candidates.value.notna()]
            if candidates.empty:
                continue
            chosen = candidates.sort_values(["value", "method"]).index[0]
            selections.append({"fold": int(fold), "family": family, "selected_method": chosen,
                               "training_matches": train, "validation_matches": valid,
                               "selection_metric": metric, "training_score": float(candidates.loc[chosen, "value"]),
                               "training_length_weights": weights})
            rows = validation[validation.method == chosen].copy()
            rows["selected_configuration"] = chosen
            rows["method"] = f"tuned_{family}"
            rows["fold"] = fold
            assessed.append(rows)
    return pd.DataFrame(selections), pd.concat(assessed, ignore_index=True) if assessed else pd.DataFrame()
