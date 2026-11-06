"""Decision-focused interpolation policies, joint masks and match-level evidence.

All decisions based on reference motion are diagnostics, never deployment rules.
The existing split manifest owns the test boundary throughout this follow-up.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import polars as pl

from tempoctrl.research.bst_models import file_sha256, repository_root, specification_hash
from tempoctrl.research.bst_evaluation import markdown_table
from tempoctrl.research.interpolation_data import (
    GapCase, PLAYER, TEAM, build_gap_cases, case_record, prepare_match, reference_segments,
)
from tempoctrl.research.interpolation_evaluation import evaluate_cases, paired_bootstrap
from tempoctrl.research.interpolation_models import Interpolator
from tempoctrl.research.interpolation_workflow import InterpolationConfig, _json, start_study


@dataclass(frozen=True)
class ValidationConfig:
    """Declared before execution; dense lengths separate interpolation from gap caps."""

    lengths: tuple[int, ...] = tuple(range(1, 46))
    limits: tuple[int, ...] = (5, 10, 15, 20, 30, 45)
    seeds: tuple[int, ...] = (101, 211, 307)
    per_length: int = 1
    joint_blocks: tuple[int, ...] = (2, 4)
    joint_length_caps: tuple[int, ...] = (5, 10, 15, 20, 30, 45)
    joint_per_scenario: int = 3
    context_seconds: float = .25
    bootstrap_draws: int = 2000


def policy_grid(config=ValidationConfig()):
    """Hybrid uses linear for one hidden frame, otherwise PCHIP with fallback."""
    return tuple(
        Interpolator(f"{family}_{limit}", family, config.context_seconds, limit)
        for limit in config.limits for family in ("linear", "pchip", "hybrid")
    ) + (Interpolator("leave_missing", "missing"),)


def build_joint_cases(prepared, config=ValidationConfig()):
    """Hide multiple bounded blocks in ONE continuous player/team segment.

    Each scenario is a stress design with uniformly sampled block lengths, not
    an estimate of the natural missingness process. Segments are sampled without
    replacement within scenario/seed; unsupported cells remain in the audit.
    """
    segments = reference_segments(prepared)
    game = int(prepared.panel["game_id"][0])
    cases, support = [], []
    for seed in config.seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, game, 702]))
        for cap in config.joint_length_caps:
            for count in config.joint_blocks:
                scenario = f"{count}_blocks_lengths_1_to_{cap}"
                selected = 0
                for seg_id in rng.permutation(len(segments)):
                    group, frames, xy = segments[seg_id]
                    safe = ((group["physical_copies"].to_numpy() == 1)
                            & ~group["is_synthetic_pass_end"].to_numpy())
                    lengths = rng.integers(1, cap + 1, size=count)
                    blocks = []
                    available = safe.copy()
                    available[[0, -1]] = False
                    for length in lengths:
                        starts = np.flatnonzero(np.convolve(
                            available.astype(int), np.ones(int(length), dtype=int), "valid",
                        ) == length) if length <= len(available) else np.array([], dtype=int)
                        if not len(starts):
                            break
                        start = int(rng.choice(starts))
                        block = np.arange(start, start + length)
                        blocks.append(block)
                        # Preserve at least one observed frame between blocks.
                        available[max(0, start - 1):min(len(available), start + length + 1)] = False
                    if len(blocks) != count:
                        continue
                    target = np.sort(np.concatenate(blocks))
                    player, team = group[PLAYER][0], group[TEAM][0]
                    pdist, psec = prepared.player_totals[player]
                    tdist, tsec = prepared.team_totals[team]
                    if psec is None or tsec is None or min(psec, tsec) <= 0:
                        continue
                    # Retain the full segment, including all unhidden context.
                    visibility = group["balls_smooth"].struct.field("visibility").to_numpy()
                    lo, hi = target[0] - 1, target[-1] + 1
                    velocities = np.diff(xy[lo:hi + 1], axis=0) * prepared.frame_rate
                    norms = np.linalg.norm(velocities, axis=1)
                    cosine = np.sum(velocities[:-1] * velocities[1:], axis=1) / np.maximum(norms[:-1] * norms[1:], 1e-12)
                    hidden = np.isin(np.arange(lo, hi + 1), target)
                    affected = hidden[:-1] | hidden[1:]
                    affected_turn = affected[:-1] | affected[1:]
                    cases.append(GapCase(
                        f"{game}:joint:{seed}:{cap}:{count}:{selected}", game,
                        int(group["period"][0]), player, team, seed, int(max(lengths)),
                        prepared.frame_rate, frames.copy(), xy.copy(), target,
                        float(pdist), float(psec), float(tdist), float(tsec),
                        str(group["possession_event_type"][int(target[0])]), str(visibility[target[0]]),
                        float(norms[affected].max()), float(np.arccos(np.clip(cosine[affected_turn], -1, 1)).max()),
                        "all_context_visible" if np.all(visibility == "VISIBLE") else "includes_nonvisible_context",
                        bool(np.all(visibility[target] == "VISIBLE")), "joint_gaps", scenario, cap,
                    ))
                    selected += 1
                    if selected == config.joint_per_scenario:
                        break
                support.append({"game_id": game, "seed": seed, "scenario": scenario, "scenario_cap": cap,
                                "requested": config.joint_per_scenario, "sampled": selected})
    return cases, pd.DataFrame(support)


def validation_match_metrics(scores):
    """One match/method/experiment row; case percentiles are explicitly labelled.

    Average masks within player possession first, then possessions within a
    scenario, then scenarios equally. Team errors analogously average within
    affected team possessions. Tail metrics are the scenario-balanced mean of
    within-match possession-realization P95s, not a pooled global percentile.
    """
    data = scores.reset_index(drop=True).copy()
    match_keys = ["game_id", "method", "experiment"]
    keys = [*match_keys, "__cell"]
    data["__cell"] = np.where(data.experiment == "single_gap", data.gap_length.astype(str), data.scenario)
    data["complete_case_fraction"] = (data.fill_fraction == 1).astype(float)
    data["failure_fraction"] = data.status.str.startswith("failed").astype(float)
    data["fallback_case_fraction"] = (data.status == "linear_fallback").astype(float)
    accuracy = ["ade", "rmse", "p95_error", "speed_mae", "path_length_abs_error"]
    cells = data.groupby(keys)[[*accuracy, "fill_fraction", "complete_case_fraction",
                               "failure_fraction", "fallback_case_fraction"]].mean()
    for level in ("player", "team"):
        possession = f"{level}_possession"
        columns = [f"{level}_{suffix}" for suffix in
                   ("tempo_abs_error", "tempo_bias", "tempo_squared_error", "time_coverage")]
        # Vectorized two-stage averaging gives every possession equal weight.
        possession_means = data.groupby([*keys, possession])[columns].mean()
        cells = cells.join(possession_means.groupby(keys).mean())
        error = f"{level}_tempo_abs_error"
        counts = data.groupby([*keys, possession])["case_id"].transform("count")
        finite = data[error].notna()
        valid = data.loc[finite, [*keys, error]].copy()
        valid["__weight"] = 1 / counts[finite]
        valid = valid.sort_values(error)
        if len(valid):
            cumulative = valid.groupby(keys)["__weight"].cumsum()
            total = valid.groupby(keys)["__weight"].transform("sum")
            quantiles = valid[cumulative >= .95 * total].drop_duplicates(keys).set_index(keys)[error]
            cells[f"{level}_tempo_p95"] = quantiles
        else:
            cells[f"{level}_tempo_p95"] = np.nan
        data[f"{level}_tempo_available_fraction"] = finite.astype(float)
        cells[f"{level}_tempo_available_fraction"] = data.groupby(keys)[f"{level}_tempo_available_fraction"].mean()
    result = cells.groupby(match_keys).mean()
    incomplete = data.groupby(match_keys).fill_fraction.min() < 1
    result.loc[incomplete, accuracy] = np.nan
    for level in ("player", "team"):
        result[f"{level}_tempo_rmse"] = np.sqrt(result[f"{level}_tempo_squared_error"])
    result["cases"] = data.groupby(match_keys).case_id.nunique()
    result["scenario_cells"] = cells.groupby(match_keys).size()
    return result.reset_index()


def scoped_policy_metrics(scores, specs):
    """Matched cohorts for each cap; all policies at that cap see identical cases.

    Across-cap coverage is assessed separately on the complete common design.
    A lower-cap policy never wins by quietly dropping harder attempted cases.
    """
    rows = []
    for limit in sorted({spec.max_gap for spec in specs if spec.family != "missing"}):
        methods = [s.name for s in specs if s.max_gap == limit and s.family != "missing"]
        eligible = scores[(scores.scenario_cap <= limit) & scores.method.isin(methods)]
        if len(eligible):
            summary = validation_match_metrics(eligible)
            summary["gap_limit"] = limit
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def paired_policy_metrics(scoped, *, draws=2000, seed=20261105):
    rows = []
    for (limit, experiment), group in scoped.groupby(["gap_limit", "experiment"]):
        for metric in ("ade", "p95_error", "speed_mae", "player_tempo_abs_error",
                       "team_tempo_abs_error", "player_tempo_p95", "team_tempo_p95"):
            paired = paired_bootstrap(group, metric, baseline=f"linear_{limit}", draws=draws, seed=seed)
            paired["gap_limit"], paired["experiment"] = limit, experiment
            rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def policy_selection(scoped, splits):
    """Choose family at each FIXED cap using training matches only.

    Equal weight to single and joint experiment ADE. Caps remain a scientific
    choice needing absolute error tolerances; smaller caps have easier cohorts.
    """
    choices, held = [], []
    dev = splits[splits.split == "development"]
    for fold in sorted(dev.fold.unique()):
        train = set(dev.loc[dev.fold != fold, "game_id"])
        valid = set(dev.loc[dev.fold == fold, "game_id"])
        present_train = train & set(scoped.game_id)
        for limit, group in scoped.groupby("gap_limit"):
            training = group[group.game_id.isin(train)]
            table = training.groupby("method").agg(ade=("ade", "mean"), cells=("ade", "count"))
            expected = len(present_train) * 2
            table = table[(table.cells == expected) & table.ade.notna()]
            if not expected or table.empty:
                continue
            chosen = table.sort_values(["ade", "method"]).index[0]
            choices.append({"fold": int(fold), "gap_limit": int(limit), "method": chosen,
                            "training_ade": float(table.loc[chosen, "ade"]),
                            "training_matches": sorted(present_train),
                            "validation_matches": sorted(valid & set(group.game_id))})
            selected = group[group.game_id.isin(valid) & (group.method == chosen)].copy()
            selected["selected_configuration"] = chosen
            selected["method"] = f"selected_{limit}"
            held.append(selected)
    return pd.DataFrame(choices), pd.concat(held, ignore_index=True) if held else pd.DataFrame()


def diagnostic_strata(scores, specs):
    """Overlapping reference-only strata; thresholds are diagnostic, never rules."""
    subsets = {
        "all": np.ones(len(scores), dtype=bool),
        "all_context_visible": scores.reference_quality == "all_context_visible",
        "all_targets_visible": scores.all_targets_visible,
        "reference_fast_over_15mps": scores.max_reference_speed > 15,
        "reference_turn_over_45deg": scores.turn_radians > np.pi / 4,
    }
    rows = []
    for label, selected in subsets.items():
        if selected.any():
            result = scoped_policy_metrics(scores[selected], specs)
            result["stratum"] = label
            rows.append(result)
    # Events are provider possession labels, not independently annotated touches.
    for event, group in scores.groupby("event_type"):
        result = scoped_policy_metrics(group, specs)
        result["stratum"] = f"event:{event}"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _validate_config(config):
    if (not config.lengths or len(set(config.lengths)) != len(config.lengths)
            or min(config.lengths) < 1 or not config.limits or min(config.limits) < 1
            or 5 not in config.limits or not config.seeds
            or len(set(config.seeds)) != len(config.seeds)
            or config.per_length < 1 or config.joint_per_scenario < 1
            or not config.joint_blocks or min(config.joint_blocks) < 2
            or not config.joint_length_caps or min(config.joint_length_caps) < 1
            or config.context_seconds <= 0 or config.bootstrap_draws < 2):
        raise ValueError("Require positive lengths, counts, contexts, unique seeds and a five-frame baseline.")
    if not set(range(1, max(config.limits) + 1)).issubset(config.lengths):
        raise ValueError("Test every integer gap length up to the largest policy cap.")
    if max(config.joint_length_caps) > max(config.limits):
        raise ValueError("Joint scenario caps cannot exceed the largest studied policy cap.")


def run_policy_validation(config=ValidationConfig(), *, study=InterpolationConfig(), root=None,
                          match_ids=None, progress=print, _protocol=None):
    """Run development follow-up, or the exact frozen confirmation internally."""
    _validate_config(config)
    root = repository_root(root)
    splits = start_study(study, root=root)
    phase = "validation" if _protocol is None else "confirmation"
    allowed = splits[splits.split == ("development" if _protocol is None else "test")]
    if match_ids is not None:
        if not set(match_ids).issubset(set(allowed.game_id)):
            raise ValueError("Requested matches cross the study boundary; test matches remain locked.")
        allowed = allowed[allowed.game_id.isin(match_ids)]
    if len(allowed) < 2:
        raise ValueError("Require at least two matches.")
    specs = policy_grid(config) if _protocol is None else tuple(Interpolator(**s) for s in _protocol["policies"])
    directory = root / study.output / datetime.now(timezone.utc).strftime(f"{phase}_%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True)
    snapshots = directory / "implementation"
    snapshots.mkdir()
    sources = list(Path(__file__).parent.glob("interpolation_*.py")) + [root / "notebooks/research_investigations/ball_interpolation_model_comparison.py"]
    for path in sources:
        shutil.copy2(path, snapshots / path.name)
    splits.to_csv(directory / "splits.csv", index=False)
    manifest = {"status": "running", "phase": phase, "config": asdict(config), "study": asdict(study),
                "policies": [asdict(s) for s in specs], "matches": allowed.game_id.tolist(),
                "pilot_subset": len(allowed) != int((splits.split == ("development" if _protocol is None else "test")).sum()),
                "final_test_evaluated": _protocol is not None,
                "split_manifest_sha256": file_sha256(root / study.output / "split_manifest.json"),
                "implementation_sha256": {p.name: file_sha256(p) for p in sources},
                "packages": {name: version(name) for name in ("numpy", "scipy", "polars", "pandas")},
                "reference": "provider-smoothed coordinates, not independent physical truth"}
    _json(directory / "manifest.json", manifest)
    scores_all, gaps_all, audit_all, support_all = [], [], [], []
    try:
        for index, match in enumerate(allowed.itertuples(index=False), 1):
            if progress:
                progress(f"{phase.title()} match {index}/{len(allowed)}: {match.game_id}", flush=True)
            prepared = prepare_match(Path(match.path), root / study.metadata, root / study.events)
            single, support = build_gap_cases(prepared, seeds=config.seeds, lengths=config.lengths,
                                              per_length=config.per_length, context_seconds=config.context_seconds)
            joint, joint_support = build_joint_cases(prepared, config)
            support["experiment"], joint_support["experiment"] = "single_gap", "joint_gaps"
            cases = single + joint
            match_dir = directory / "matches" / str(match.game_id)
            match_dir.mkdir(parents=True)
            pl.DataFrame([case_record(c) for c in cases]).write_parquet(match_dir / "cases.parquet")
            scores, predictions = evaluate_cases(cases, specs)
            scores.to_parquet(match_dir / "scores.parquet", index=False)
            pl.DataFrame(predictions.to_dict("records"), infer_schema_length=None).write_parquet(match_dir / "predictions.parquet")
            scores_all.append(scores)
            audit_all.append(prepared.audit)
            gaps_all.append(prepared.gaps)
            support_all.extend([support, joint_support])
        scores = pd.concat(scores_all, ignore_index=True)
        scores.to_parquet(directory / "scores.parquet", index=False)
        pd.DataFrame(audit_all).to_csv(directory / "audit.csv", index=False)
        pd.concat(gaps_all, ignore_index=True).to_csv(directory / "natural_gaps.csv", index=False)
        pd.concat(support_all, ignore_index=True).to_csv(directory / "mask_support.csv", index=False)
        all_metrics = validation_match_metrics(scores)
        all_metrics.to_csv(directory / "common_design_match_metrics.csv", index=False)
        scoped = scoped_policy_metrics(scores, specs)
        scoped.to_csv(directory / "scoped_match_metrics.csv", index=False)
        paired = paired_policy_metrics(scoped, draws=config.bootstrap_draws, seed=study.split_seed)
        paired.to_csv(directory / "paired_policy_differences.csv", index=False)
        strata = diagnostic_strata(scores, specs)
        strata.to_csv(directory / "diagnostic_strata.csv", index=False)
        if _protocol is None:
            selections, oof = policy_selection(scoped, splits)
            selections.to_json(directory / "fold_policy_choices.json", orient="records", indent=2)
            oof.to_csv(directory / "selected_policy_oof.csv", index=False)
            if len(oof):
                baseline = scoped[scoped.method == "linear_" + scoped.gap_limit.astype(str)]
                paired_policy_metrics(pd.concat([baseline, oof], ignore_index=True),
                                      draws=config.bootstrap_draws, seed=study.split_seed).to_csv(
                    directory / "paired_selected_policy_oof.csv", index=False)
        leaderboard = scoped.groupby(["gap_limit", "method", "experiment"]).mean(numeric_only=True).reset_index()
        leaderboard.to_csv(directory / "policy_leaderboard.csv", index=False)
        decision_frontier(scoped).to_csv(directory / "decision_frontier.csv", index=False)
        report = ["# Interpolation policy validation", "",
                  f"Phase: {phase}; matches: {len(allowed)}; mask realizations: {scores.case_id.nunique()}.",
                  "Coordinate/speed scores require every target in the reported cohort to be filled. Across-cap comparisons have different eligible cases; consult common-design coverage separately.",
                  "Tempo scores compare reconstructed and reference FULL affected possessions. RMSE and scenario-balanced within-match possession-realization P95 complement MAE and signed bias. Errors with unavailable tempo are reported with explicit availability fractions.",
                  "Joint scenarios hide all blocks before any fit. They perturb one player possession at a time; team errors describe that affected team possession, not simultaneous corruption across all players in the team possession.",
                  "Seeds and repeated masks are dependent within matches. Intervals resample paired whole matches. Diagnostic subsets and selected policies remain exploratory in development.",
                  markdown_table(leaderboard.drop(columns=["game_id"], errors="ignore")),
                  "Test tracking was not opened." if _protocol is None else "This is the predeclared confirmation cohort."]
        (directory / "results.md").write_text("\n\n".join(report) + "\n")
        # Hash masks and predictions as well as combined summaries.
        artifacts = [p for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"]
        manifest.update(status=f"{phase}_complete", cases=scores.case_id.nunique(),
                        score_rows=len(scores), artifacts_sha256={str(p.relative_to(directory)): file_sha256(p) for p in artifacts})
        _json(directory / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(directory / "manifest.json", manifest)
        raise
    return directory


def load_policy_validation(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] not in ("validation_complete", "confirmation_complete"):
        raise ValueError("Require a completed policy validation.")
    for name, digest in manifest["artifacts_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Saved artifact changed: {name}")
    return {"manifest": manifest, "scores": pd.read_parquet(directory / "scores.parquet"),
            "leaderboard": pd.read_csv(directory / "policy_leaderboard.csv"),
            "scoped": pd.read_csv(directory / "scoped_match_metrics.csv"),
            "paired": pd.read_csv(directory / "paired_policy_differences.csv"),
            "common": pd.read_csv(directory / "common_design_match_metrics.csv"),
            "strata": pd.read_csv(directory / "diagnostic_strata.csv")}


def decision_frontier(scoped):
    """Descriptive Pareto choices WITHIN a fixed cap, never across easier cohorts."""
    metrics = ["ade", "speed_mae", "player_tempo_abs_error", "team_tempo_abs_error",
               "player_tempo_p95", "team_tempo_p95"]
    table = scoped.groupby(["gap_limit", "experiment", "method"])[metrics].mean().reset_index()
    table["dominated"] = False
    for _, group in table.groupby(["gap_limit", "experiment"]):
        for index, row in group.iterrows():
            if row[metrics].isna().any():
                table.loc[index, "dominated"] = True
                continue
            others = group[group.method != row.method][metrics]
            table.loc[index, "dominated"] = bool(((others <= row[metrics]).all(axis=1)
                                                   & (others < row[metrics]).any(axis=1)).any())
    return table


@dataclass(frozen=True)
class DecisionCriteria:
    """Scientific limits must be supplied explicitly; there are no inferred defaults.

    Absolute limits apply to equal-match summaries. P95 quantities are means of
    within-match/scenario percentiles, as defined in validation_match_metrics.
    Noninferiority compares paired match means to linear at the SAME gap cap.
    """

    minimum_coordinate_gain_m: float
    maximum_ade_m: float
    maximum_mean_mask_coordinate_p95_m: float
    maximum_speed_mae_mps: float
    maximum_player_tempo_mae_mps: float
    maximum_team_tempo_mae_mps: float
    maximum_player_tempo_p95_mps: float
    maximum_team_tempo_p95_mps: float
    tempo_mae_noninferiority_margin_mps: float
    tempo_p95_noninferiority_margin_mps: float
    speed_mae_noninferiority_margin_mps: float

    def validate(self):
        if not all(v is not None and np.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Every scientific limit must be explicitly supplied, finite and nonnegative.")


def freeze_confirmation(directory, candidate, criteria, *, root=None):
    """Freeze one policy and matched-cap linear comparator before opening test data.

    This only writes a protocol. It never reads test tracking or changes production.
    A pilot cannot freeze the final protocol, and an existing protocol is immutable.
    """
    root = repository_root(root)
    directory = Path(directory).resolve()
    results = load_policy_validation(directory)
    manifest = results["manifest"]
    if manifest["phase"] != "validation" or manifest["pilot_subset"]:
        raise ValueError("Freeze from a completed full-development validation, not a pilot or confirmation.")
    criteria.validate()
    policies = {s["name"]: s for s in manifest["policies"]}
    if candidate not in policies or policies[candidate]["family"] not in ("pchip", "hybrid"):
        raise ValueError("Choose a studied PCHIP or hybrid candidate explicitly.")
    selected = policies[candidate]
    limit = selected["max_gap"]
    if limit not in manifest["config"]["joint_length_caps"]:
        raise ValueError("Candidate cap needs its own joint-mask scenario before confirmation.")
    for name, digest in manifest["implementation_sha256"].items():
        if name == "ball_interpolation_model_comparison.py":
            continue  # Replay settings/Markdown do not change numerical helpers.
        if file_sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"Numerical implementation changed after development: {name}")
    support = pd.read_csv(directory / "mask_support.csv")
    eligible_support = support[((support.experiment == "single_gap") & (support.gap_length <= limit))
                               | ((support.experiment == "joint_gaps") & (support.scenario_cap <= limit))]
    if not len(eligible_support) or (eligible_support.sampled < eligible_support.requested).any():
        raise ValueError("Candidate has unsupported requested development masks; review support before freezing.")
    config = ValidationConfig(**{**manifest["config"],
        "lengths": tuple(range(1, limit + 1)), "limits": tuple(sorted({5, limit})),
        "joint_length_caps": tuple(c for c in manifest["config"]["joint_length_caps"] if c <= limit)})
    if not config.joint_length_caps:
        raise ValueError("The candidate needs supported joint corruption scenarios.")
    study = InterpolationConfig(**manifest["study"])
    study_dir = root / study.output
    if file_sha256(study_dir / "split_manifest.json") != manifest["split_manifest_sha256"]:
        raise ValueError("Study split changed after development.")
    specs = [policies[name] for name in dict.fromkeys(["linear_5", f"linear_{limit}", candidate])]
    protocol = {"development_directory": str(directory),
                "development_manifest_sha256": file_sha256(directory / "manifest.json"),
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "config": asdict(config), "study": asdict(study), "policies": specs,
                "candidate": candidate, "criteria": asdict(criteria),
                "implementation_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("interpolation_*.py")},
                "acceptance_scope": "single_gap and joint_gaps overall cohorts; VISIBLE-only and event strata are mandatory diagnostics"}
    if (study_dir / "confirmation_state.json").exists():
        raise ValueError("The reserved test has already been opened; it cannot be refrozen.")
    protocol["protocol_sha256"] = specification_hash(protocol)
    path = study_dir / "confirmation_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("A different immutable confirmation protocol already exists.")
    else:
        _json(path, protocol)
    return path


def assess_confirmation(directory, protocol):
    """Require complete support and paired improvement/noninferiority in BOTH tasks.

    Absolute guardrails use the one-sided 95% bootstrap upper mean. Paired
    guardrails use the upper end of the saved two-sided 95% match interval.
    Neither interval corrects for dependence between matches sharing teams.
    """
    results = load_policy_validation(directory)
    criteria = DecisionCriteria(**protocol["criteria"])
    criteria.validate()
    candidate = protocol["candidate"]
    limit = next(s["max_gap"] for s in protocol["policies"] if s["name"] == candidate)
    rows = []
    mapping = {"ade": "maximum_ade_m", "p95_error": "maximum_mean_mask_coordinate_p95_m",
               "speed_mae": "maximum_speed_mae_mps", "player_tempo_abs_error": "maximum_player_tempo_mae_mps",
               "team_tempo_abs_error": "maximum_team_tempo_mae_mps", "player_tempo_p95": "maximum_player_tempo_p95_mps",
               "team_tempo_p95": "maximum_team_tempo_p95_mps"}
    rng = np.random.default_rng(protocol["study"]["split_seed"])
    support = pd.read_csv(Path(directory) / "mask_support.csv")
    rows.append({"check": "all_requested_mask_cells_supported", "passed": bool((support.sampled == support.requested).all())})
    for experiment in ("single_gap", "joint_gaps"):
        match = results["scoped"].query("method == @candidate and gap_limit == @limit and experiment == @experiment")
        complete = (len(match) == len(results["manifest"]["matches"])
                    and (match.fill_fraction == 1).all()
                    and (match.player_tempo_available_fraction == 1).all()
                    and (match.team_tempo_available_fraction == 1).all())
        rows.append({"check": f"{experiment}:complete_matches_and_predictions", "passed": bool(complete)})
        for metric, criterion in mapping.items():
            values = match[metric].dropna().to_numpy()
            upper = np.inf
            if len(values) == len(results["manifest"]["matches"]):
                boot = values[rng.integers(0, len(values), size=(protocol["config"]["bootstrap_draws"], len(values)))].mean(axis=1)
                upper = float(np.quantile(boot, .95))
            bound = getattr(criteria, criterion)
            rows.append({"check": f"{experiment}:absolute:{metric}", "upper_bound": upper,
                         "limit": bound, "passed": upper <= bound})
        paired = results["paired"].query("method == @candidate and gap_limit == @limit and experiment == @experiment")
        for metric in ("ade", "speed_mae", "player_tempo_abs_error", "team_tempo_abs_error", "player_tempo_p95", "team_tempo_p95"):
            bound = (-criteria.minimum_coordinate_gain_m if metric == "ade" else
                     criteria.speed_mae_noninferiority_margin_mps if metric == "speed_mae" else
                     criteria.tempo_p95_noninferiority_margin_mps if metric.endswith("p95") else
                     criteria.tempo_mae_noninferiority_margin_mps)
            selected = paired[paired.metric == metric]
            upper = float(selected.upper_95.iloc[0]) if len(selected) == 1 else np.inf
            rows.append({"check": f"{experiment}:paired:{metric}", "upper_bound": upper,
                         "limit": bound, "passed": upper <= bound})
    return pd.DataFrame(rows)


def run_confirmation(protocol_path, *, root=None, progress=print):
    """Consume a frozen protocol once. A failed opening is recorded, never retried silently."""
    root = repository_root(root)
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    signature = protocol.pop("protocol_sha256")
    if specification_hash(protocol) != signature:
        raise ValueError("Frozen confirmation protocol changed.")
    protocol["protocol_sha256"] = signature
    DecisionCriteria(**protocol["criteria"]).validate()
    for name, digest in protocol["implementation_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"Implementation changed after freezing: {name}")
    study = InterpolationConfig(**protocol["study"])
    if file_sha256(root / study.output / "split_manifest.json") != protocol["split_manifest_sha256"]:
        raise ValueError("Split changed after freezing.")
    development = Path(protocol["development_directory"])
    if file_sha256(development / "manifest.json") != protocol["development_manifest_sha256"]:
        raise ValueError("Development evidence changed after freezing.")
    load_policy_validation(development)
    state_path = protocol_path.parent / "confirmation_state.json"
    # Exclusive creation prevents two kernels from opening the holdout together.
    with state_path.open("x") as handle:
        json.dump({"status": "started", "protocol_sha256": signature}, handle)
    try:
        directory = run_policy_validation(ValidationConfig(**protocol["config"]), study=study,
                                          root=root, progress=progress, _protocol=protocol)
        assessment = assess_confirmation(directory, protocol)
        assessment.to_csv(directory / "confirmation_assessment.csv", index=False)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts_sha256"]["confirmation_assessment.csv"] = file_sha256(directory / "confirmation_assessment.csv")
        manifest["confirmation_protocol_sha256"] = signature
        _json(manifest_path, manifest)
        _json(state_path, {"status": "complete", "directory": str(directory),
                           "protocol_sha256": signature, "passed_statistical_criteria": bool(assessment.passed.all()),
                           "diagnostic_review_required": True, "production_changed": False})
    except Exception as error:
        _json(state_path, {"status": "failed_after_opening", "protocol_sha256": signature,
                           "error": f"{type(error).__name__}: {error}"})
        raise
    return directory
