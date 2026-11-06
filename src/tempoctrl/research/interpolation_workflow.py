"""Saved, reproducible development-only ball-interpolation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import polars as pl

from tempoctrl.research.bst_models import file_sha256, repository_root, specification_hash
from tempoctrl.research.bst_evaluation import markdown_table
from tempoctrl.research.interpolation_data import (
    build_gap_cases, case_record, discover_matches, prepare_match, split_matches,
)
from tempoctrl.research.interpolation_evaluation import (
    evaluate_cases, grouped_selection, length_weights, match_summary, paired_bootstrap,
)
from tempoctrl.research.interpolation_models import candidate_interpolators


@dataclass(frozen=True)
class InterpolationConfig:
    integrated: str = "data/integrated/gradient_sports"
    metadata: str = "data/raw/gradient_sports/metadata"
    events: str = "data/processed/gradient_sports/events"
    output: str = "data/investigate/ball_interpolation"
    split_seed: int = 20261105
    test_fraction: float = .2
    folds: int = 5
    mask_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    lengths: tuple[int, ...] = (1, 2, 3, 4, 5, 8, 15, 30, 45)
    gaps_per_length: int = 3
    contexts: tuple[float, ...] = (.25, .5, 1.)
    bootstrap_draws: int = 2000
    # None deliberately blocks confirmatory promotion; discussed with the user
    # after inspecting development tradeoffs rather than assumed to be .1 m/s.
    tempo_tolerance: float | None = None


def _json(path, item):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        return value
    path.write_text(json.dumps(clean(item), indent=2, sort_keys=True, allow_nan=False) + "\n")


def start_study(config: InterpolationConfig = InterpolationConfig(), *, root=None):
    """Create an immutable split manifest before reading any tracking values.

    Metadata may describe all matches. Test frame data are not opened. An
    existing split cannot silently change when files or metadata change.
    """
    if config.gaps_per_length < 1 or not config.contexts or min(config.contexts) <= 0:
        raise ValueError("Positive gap counts and context durations are required.")
    if not config.mask_seeds or len(set(config.mask_seeds)) != len(config.mask_seeds):
        raise ValueError("Require unique mask seeds.")
    if not set(range(1, 6)).issubset(config.lengths) or min(config.lengths) < 1:
        raise ValueError("Include primary lengths 1–5; gap lengths must be positive.")
    if config.bootstrap_draws < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    root = repository_root(root)
    output = root / config.output
    output.mkdir(parents=True, exist_ok=True)
    inventory = discover_matches(root / config.integrated, root / config.metadata)
    expected = split_matches(inventory, seed=config.split_seed,
                             test_fraction=config.test_fraction, folds=config.folds)
    signature = specification_hash({"seed": config.split_seed, "fraction": config.test_fraction,
                                    "folds": config.folds, "inventory": inventory.to_dict("records")})
    split_path = output / "split_manifest.json"
    if split_path.exists():
        previous = json.loads(split_path.read_text())
        if previous["signature"] != signature or previous["matches"] != expected.to_dict("records"):
            raise ValueError("Existing locked split differs. Use a new study output directory explicitly.")
    else:
        _json(split_path, {"signature": signature, "matches": expected.to_dict("records"),
                           "test_status": "tracking values unopened by this study",
                           "stratification": "metadata week; proportional held-out match allocation"})
    return expected


def run_development(config: InterpolationConfig = InterpolationConfig(), *, root=None,
                    match_ids=None, progress=print) -> Path:
    """Run named local candidates on development matches; never read test frames.

    Supplying a development subset is an explicitly labelled pilot. A unique run
    directory preserves every previous experiment, including failures.
    """
    root = repository_root(root)
    splits = start_study(config, root=root)
    dev = splits[splits.split == "development"]
    if match_ids is not None:
        if not set(match_ids).issubset(set(dev.game_id)):
            raise ValueError("Only development match IDs can be evaluated here. Test matches remain locked.")
        dev = dev[dev.game_id.isin(match_ids)]
    if len(dev) < 2:
        raise ValueError("Need at least two development matches; full folds may require more.")
    directory = root / config.output / datetime.now(timezone.utc).strftime("development_%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "matches").mkdir()
    specs = candidate_interpolators(config.contexts)
    implementation = {path.name: file_sha256(path) for path in Path(__file__).parent.glob("interpolation_*.py")}
    snapshots = directory / "implementation"
    snapshots.mkdir()
    for path in Path(__file__).parent.glob("interpolation_*.py"):
        (snapshots / path.name).write_text(path.read_text())
    notebook = root / "notebooks/research_investigations/ball_interpolation_model_comparison.py"
    if notebook.exists():
        implementation[notebook.name] = file_sha256(notebook)
        (snapshots / notebook.name).write_text(notebook.read_text())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False).stdout.strip()
    patch = subprocess.run(["git", "diff", "--binary"], cwd=root, capture_output=True, text=True, check=False).stdout
    (directory / "working_tree.patch").write_text(patch)
    splits.to_csv(directory / "splits.csv", index=False)
    manifest = {"status": "running", "phase": "development", "config": asdict(config),
                "git_revision": revision, "implementation_sha256": implementation,
                "working_tree_patch_sha256": file_sha256(directory / "working_tree.patch"),
                "packages": {p: version(p) for p in ("numpy", "scipy", "polars", "pandas")},
                "candidate_grid": [asdict(s) for s in specs],
                "development_matches": dev.game_id.tolist(),
                "pilot_subset": len(dev) != int((splits.split == "development").sum()),
                "final_test_evaluated": False,
                "promotion_status": "not assessed; scientific tolerances and final-test protocol must be frozen",
                "supported_task": "bounded fully-missing 2D gaps; isolated interior single-gap perturbations",
                "deferred": ["joint multi-gap possession corruption", "partial-axis recovery", "absent-row insertion",
                             "boundary extrapolation", "independent video truth", "domain-transfer holdouts",
                             "learned player-context models", "production deployment"]}
    _json(directory / "manifest.json", manifest)
    audit_rows, gap_tables, supports, score_tables = [], [], [], []
    try:
        for index, item in enumerate(dev.itertuples(index=False), 1):
            if progress:
                progress(f"Development match {index}/{len(dev)}: {item.game_id}", flush=True)
            prepared = prepare_match(Path(item.path), root / config.metadata, root / config.events)
            cases, support = build_gap_cases(
                prepared, seeds=config.mask_seeds, lengths=config.lengths,
                per_length=config.gaps_per_length, context_seconds=max(config.contexts),
            )
            audit_rows.append(prepared.audit)
            gap_tables.append(prepared.gaps)
            supports.append(support)
            match_dir = directory / "matches" / str(item.game_id)
            match_dir.mkdir()
            _json(match_dir / "audit.json", prepared.audit)
            prepared.gaps.to_csv(match_dir / "natural_gaps.csv", index=False)
            support.to_csv(match_dir / "mask_support.csv", index=False)
            if not cases:
                raise ValueError(f"Match {item.game_id} has no supported masks; review the audit before redefining cohort.")
            pl.DataFrame([case_record(case) for case in cases]).write_parquet(match_dir / "cases.parquet")
            scores, predictions = evaluate_cases(cases, specs)
            scores.to_parquet(match_dir / "scores.parquet", index=False)
            pl.DataFrame(predictions.to_dict("records"), infer_schema_length=None).write_parquet(match_dir / "predictions.parquet")
            score_tables.append(scores)
            _json(directory / "progress.json", {"completed_matches": [r["game_id"] for r in audit_rows]})
        audit = pd.DataFrame(audit_rows)
        gaps = pd.concat(gap_tables, ignore_index=True)
        scores = pd.concat(score_tables, ignore_index=True)
        support = pd.concat(supports, ignore_index=True)
        audit.to_csv(directory / "audit.csv", index=False)
        gaps.to_csv(directory / "natural_gaps.csv", index=False)
        support.to_csv(directory / "mask_support.csv", index=False)
        scores.to_parquet(directory / "scores.parquet", index=False)
        weights = length_weights(gaps, dev.game_id)
        # All-development weights below are descriptive only. Fold selection and
        # validation use distinct training-only weights in grouped_selection.
        summary = match_summary(scores, weights)
        summary.to_csv(directory / "match_metrics.csv", index=False)
        selections, oof = grouped_selection(scores, splits, gaps)
        selections.to_json(directory / "fold_selections.json", orient="records", indent=2)
        if len(oof):
            oof.to_csv(directory / "selected_family_oof.csv", index=False)
            comparison = pd.concat([summary[summary.method == "linear_5"], oof], ignore_index=True)
            paired_bootstrap(comparison, draws=config.bootstrap_draws, seed=config.split_seed).to_csv(
                directory / "paired_selected_family_differences.csv", index=False)
        paired = paired_bootstrap(summary, draws=config.bootstrap_draws, seed=config.split_seed)
        paired.to_csv(directory / "paired_development_differences.csv", index=False)
        leaderboard = summary.groupby(["method", "family"], observed=True).agg(
            matches=("game_id", "nunique"), complete_matches=("balanced_ade", "count"),
            balanced_ade=("balanced_ade", "mean"), weighted_ade=("weighted_ade", "mean"),
            fill_fraction=("fill_fraction", "mean"), gap_p95_error=("p95_error", "mean"),
            speed_mae=("speed_mae", "mean"), player_tempo_mae=("player_tempo_abs_error", "mean"),
            player_tempo_bias=("player_tempo_bias", "mean"), team_tempo_mae=("team_tempo_abs_error", "mean"),
            player_time_coverage=("player_time_coverage", "mean"),
            fallback_fraction=("fallback_fraction", "mean"), failed_fraction=("failed_fraction", "mean"),
        ).reset_index().sort_values("balanced_ade", na_position="last")
        leaderboard.to_csv(directory / "development_leaderboard.csv", index=False)
        quality = scores[scores.gap_length <= 5].groupby(
            ["reference_quality", "method", "game_id", "gap_length"], observed=True,
        ).agg(ade=("ade", "mean"), gaps=("case_id", "count"),
              player_tempo_abs_error=("player_tempo_abs_error", "mean")).reset_index()
        quality.to_csv(directory / "quality_stratum_metrics.csv", index=False)
        missing_support = support[support.sampled < support.requested]
        report = [
            "# Ball interpolation: development comparison", "",
            f"Development matches: {len(dev)}; independent masks: {scores.case_id.nunique()}; candidate configurations: {len(specs)}.",
            "This is a pilot subset." if manifest["pilot_subset"] else "All designated development matches were evaluated.",
            "Final test matches were not opened. Scores compare reconstruction of provider-smoothed coordinates, not physical ground truth.", "",
            "## Primary scope and interpretation", "",
            "Primary scores use 1–5-frame gaps. Equal gaps within length and equal lengths within match precede equal-match averaging. weighted_ade uses development natural-gap frequencies descriptively; selection folds use training-only frequencies. Missing positive-weight support and abstentions are not scored as zero error. gap_p95_error is the average within-gap P95, not a pooled frame percentile.",
            "Each case modifies a single interior physical gap in isolation. Full affected-possession tempo is recomputed by adding the production bridge distance/time changes to the original production possession totals. Cumulative multi-gap possession corruption and boundary-target reconstruction remain required sensitivities before promotion.", "",
            "reference_quality distinguishes windows whose entire visible input/reference context has provider visibility VISIBLE from windows containing another visibility state. Both cohorts are retained; quality_stratum_metrics.csv records support and stratum results. Neither label establishes independent physical truth.", "",
            markdown_table(leaderboard), "", "## Paired differences versus current linear_5", "",
            "Negative differences favor the candidate. Intervals resample matches, preserving paired gaps and mask seeds. Hyperparameter comparisons on development data are exploratory; they do not replace final testing.",
            markdown_table(paired), "", "## Fold-isolated hyperparameter selection", "",
            "fold_selections.json records each family's choice using only training-match masks. selected_family_oof.csv assesses those choices on held-out development matches. No global trajectory parameters are learned from hidden validation targets. Folds with no valid supported family are omitted explicitly in that file; inspect support before interpreting a pooled result.", "",
            f"Requested mask cells below target sample count: {len(missing_support)}; see mask_support.csv.",
            "Naturally absent rows, partial coordinates, unknown direction and conflicting duplicates appear in the audit or excluded-cohort diagnostics; no unsupported recovery task is combined into primary scores.", "",
            "## Decision status", "",
            "No production change is recommended automatically. The absolute coordinate tolerance, allowable tempo degradation, tail guardrails, independent confirmation and outstanding stress tests must be resolved before promotion. The research script includes equations, plots and the plan-to-implementation scope table.",
        ]
        (directory / "results.md").write_text("\n\n".join(report) + "\n")
        manifest.update(status="development_complete", cases=int(scores.case_id.nunique()),
                        score_rows=len(scores), descriptive_length_weights=weights,
                        artifacts_sha256={name: file_sha256(directory / name) for name in
                                          ("scores.parquet", "audit.csv", "natural_gaps.csv", "mask_support.csv", "splits.csv")})
        _json(directory / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(directory / "manifest.json", manifest)
        raise
    return directory


def load_development_run(directory: str | Path):
    """Read a named, completed run; refuse changed evaluation snapshots."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "development_complete":
        raise ValueError("Run is not a completed development experiment.")
    for name, digest in manifest["artifacts_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Saved artifact changed: {name}")
    return {"manifest": manifest, "scores": pd.read_parquet(directory / "scores.parquet"),
            "leaderboard": pd.read_csv(directory / "development_leaderboard.csv"),
            "audit": pd.read_csv(directory / "audit.csv"),
            "support": pd.read_csv(directory / "mask_support.csv"),
            "paired": pd.read_csv(directory / "paired_development_differences.csv")}
