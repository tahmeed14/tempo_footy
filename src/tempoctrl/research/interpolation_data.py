"""Auditable source-only trajectory preparation, grouped splits and block masks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from tempoctrl.gradient_sports.ball_metrics import add_ball_metrics
from tempoctrl.gradient_sports.frame_rates import resolve_gradient_sports_frame_rates
from tempoctrl.gradient_sports.possessions_transform import (
    POSSESSION_COLUMNS, TRACKING_COLUMNS, create_synthetic_final_pass_frame,
    fill_bounded_possessions_id, label_pitch_thirds, normalize_ball_coordinates,
    propagate_successful_delivery_possession,
)
from tempoctrl.gradient_sports.tempo_metrics import aggregate_possession_tempo
from tempoctrl.research.bst_models import file_sha256, specification_hash

PLAYER = "dev_match_team_player_possession_id"
TEAM = "dev_match_team_possession_id"
KEY = ["game_id", "period", "framenum"]


@dataclass
class PreparedMatch:
    panel: pl.DataFrame
    player_totals: dict
    team_totals: dict
    audit: dict
    gaps: pd.DataFrame
    frame_rate: float


@dataclass
class GapCase:
    case_id: str
    game_id: int
    period: int
    player_possession: str
    team_possession: str
    seed: int
    gap_length: int
    fps: float
    frames: np.ndarray
    reference_xy: np.ndarray
    target: np.ndarray
    player_distance: float
    player_seconds: float
    team_distance: float
    team_seconds: float
    event_type: str
    visibility: str
    max_reference_speed: float
    turn_radians: float
    reference_quality: str = "unclassified"
    all_targets_visible: bool = False
    experiment: str = "single_gap"
    scenario: str = "single_gap"
    scenario_cap: int = 0


def discover_matches(integrated: Path, metadata: Path) -> pd.DataFrame:
    """Filename and metadata inventory only; no test tracking values are opened."""
    rows = []
    for path in sorted(integrated.glob("*.parquet"), key=lambda p: int(p.stem)):
        meta_path = metadata / f"{path.stem}.json"
        item = {}
        if meta_path.exists():
            payload = json.loads(meta_path.read_text())
            item = payload[0] if isinstance(payload, list) else payload
        rows.append({
            "game_id": int(path.stem), "path": str(path.resolve()),
            "metadata_path": str(meta_path.resolve()),
            "metadata_sha256": file_sha256(meta_path) if meta_path.exists() else None,
            "week": str(item.get("week", "unknown")), "date": item.get("date"),
            "stadium": item.get("stadium", {}).get("name"),
            "home_team": item.get("homeTeam", {}).get("id"),
            "away_team": item.get("awayTeam", {}).get("id"),
        })
    if not rows:
        raise FileNotFoundError(f"No integrated matches in {integrated}")
    return pd.DataFrame(rows)


def split_matches(inventory: pd.DataFrame, *, seed=20261105, test_fraction=.2, folds=5):
    """Reserve metadata-stratified whole matches; folds balance match counts.

    Largest-remainder allocation by metadata week keeps at least one development
    match per stratum where possible. Small datasets use leave-one-match-out
    development folds. No tracking outcomes influence this allocation.
    """
    if inventory.game_id.duplicated().any() or len(inventory) < 4:
        raise ValueError("Need at least four distinct matches for a held-out comparison.")
    if not 0 < test_fraction < .5 or folds < 2:
        raise ValueError("Require 0 < test_fraction < .5 and at least two folds.")
    frame = inventory.sort_values("game_id").copy()
    rng = np.random.default_rng(seed)
    count = min(len(frame) - 3, int(np.ceil(len(frame) * test_fraction)))
    strata = [(str(k), rng.permutation(g.game_id.to_numpy()).tolist())
              for k, g in frame.groupby("week", sort=True)]
    allocations = {k: 0 for k, _ in strata}
    for _ in range(count):
        available = [(len(ids) * count / len(frame) - allocations[k], k)
                     for k, ids in strata if allocations[k] < max(0, len(ids) - 1)]
        if not available:
            available = [(len(ids) * count / len(frame) - allocations[k], k)
                         for k, ids in strata if allocations[k] < len(ids)]
        allocations[max(available)[1]] += 1
    test = {i for k, ids in strata for i in ids[:allocations[k]]}
    dev = rng.permutation([i for i in frame.game_id if i not in test])
    n_folds = min(folds, len(dev)) if len(frame) >= 20 else len(dev)
    mapping = {int(match): int(index % n_folds) for index, match in enumerate(dev)}
    frame["split"] = np.where(frame.game_id.isin(test), "test", "development")
    frame["fold"] = frame.game_id.map(mapping).fillna(-1).astype(int)
    return frame


def mask_physical_frames(data: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Mask all copies of each physical key, regardless of possession or row type."""
    marked = data.join(targets.select(KEY).unique().with_columns(pl.lit(True).alias("__hide")),
                       on=KEY, how="left", maintain_order="left")
    fields = data.schema["balls_smooth"].fields
    return marked.with_columns(pl.struct([
        pl.when(pl.col("__hide").fill_null(False)).then(None)
        .otherwise(pl.col("balls_smooth").struct.field(field.name)).alias(field.name)
        for field in fields
    ]).alias("balls_smooth")).drop("__hide")


def _totals(frame: pl.DataFrame, level: str):
    key = "match_team_player_possession_id" if level == "player" else "match_team_possession_id"
    result = aggregate_possession_tempo(frame.lazy(), level).collect()
    return {row[key]: (row["total_ball_displacement"],
                       row["elapsed_frames"] / row["frame_rate"] if row["elapsed_frames"] is not None and row["frame_rate"] else None)
            for row in result.iter_rows(named=True)}


def prepare_match(path: Path, metadata_dir: Path, events_dir: Path) -> PreparedMatch:
    """Reuse possession/metric functions on source data, with period boundaries.

    This controlled preparation explicitly isolates periods and namespaces the
    derived IDs. It is not an assertion that production's whole preprocessing
    has been validated. Candidate linear reconstruction still calls the exact
    current production interpolator.
    """
    game_id = int(path.stem)
    resolution = resolve_gradient_sports_frame_rates(path, metadata_dir)[0]
    fps = resolution.frame_rate
    scan = pl.scan_parquet(path)
    ball_type = scan.collect_schema()["balls_smooth"]
    if "is_imputed" in {f.name for f in ball_type.fields}:
        raise ValueError("Use original pre-imputation integrated data, not reconstructed coordinates.")
    columns = [*KEY, *POSSESSION_COLUMNS, "balls_smooth", "successful_pass_or_cross",
               "possession_event_type", "game_event_type"]
    raw = scan.select(columns).collect()
    if raw["game_id"].null_count() or raw["game_id"].n_unique() != 1 or raw["game_id"][0] != game_id:
        raise ValueError("Integrated file must contain its filename's single non-null match ID.")
    xy = raw.select(pl.col("balls_smooth").struct.field("x").alias("x"),
                    pl.col("balls_smooth").struct.field("y").alias("y"))
    x, y = xy["x"].to_numpy(), xy["y"].to_numpy()
    # Reject conflicting duplicates as reference targets instead of choosing a
    # favorable coordinate. Exact duplicates are retained in the raw audit.
    physical = raw.with_columns(xy).group_by(KEY).agg(
        pl.len().alias("source_copies"),
        pl.struct("x", "y").n_unique().alias("coordinate_versions"),
    )
    audit = {
        "game_id": game_id, "source_rows": raw.height,
        "source_sha256": file_sha256(path), "fps": fps, "fps_source": resolution.source,
        "null_period_rows": raw["period"].null_count(),
        "null_frame_rows": raw["framenum"].null_count(),
        "both_null_rows": int((xy["x"].is_null() & xy["y"].is_null()).sum()),
        "partial_null_rows": int((xy["x"].is_null() ^ xy["y"].is_null()).sum()),
        "nan_or_infinite_nonnull_values": int(((~np.isfinite(x)) & ~xy["x"].is_null().to_numpy()).sum()
                                                  + ((~np.isfinite(y)) & ~xy["y"].is_null().to_numpy()).sum()),
        "duplicate_keys": physical.filter(pl.col("source_copies") > 1).height,
        "conflicting_coordinate_keys": physical.filter(pl.col("coordinate_versions") > 1).height,
        "outside_105x68_rows": int(((np.abs(x) > 52.5) | (np.abs(y) > 34)).sum()),
        "missing_raw_player_ids": raw["match_team_player_possession_id"].null_count(),
        "reference": "provider-smoothed coordinates; not independent physical ground truth",
        "visibility_counts": raw["balls_smooth"].struct.field("visibility").value_counts().to_dicts(),
        "finite_height_rows": int(raw["balls_smooth"].struct.field("z").is_finite().fill_null(False).sum()),
    }
    directions_path = events_dir / f"{game_id}.parquet"
    directions = pl.scan_parquet(directions_path).select(
        "match_team_possession_id", "attacking_team_direction",
    ).drop_nulls().unique().collect()
    audit["events_sha256"] = file_sha256(directions_path)
    if directions["match_team_possession_id"].n_unique() != directions.height:
        raise ValueError("Conflicting attacking direction for a team possession.")
    parts = []
    for period in sorted(raw["period"].drop_nulls().unique().to_list()):
        source = raw.filter((pl.col("period") == period) & pl.col("framenum").is_not_null())
        prepared = (source.lazy()
            .with_columns(*(pl.lit(None).alias(c) for c in TRACKING_COLUMNS if c != "balls_smooth"))
            .pipe(fill_bounded_possessions_id, POSSESSION_COLUMNS)
            .pipe(propagate_successful_delivery_possession)
            .pipe(create_synthetic_final_pass_frame)
            .with_columns(pl.lit(period).alias("period"))
            .join(directions.lazy(), left_on=TEAM, right_on="match_team_possession_id",
                  how="left", validate="m:1")
            .pipe(normalize_ball_coordinates)
            .with_columns(*(pl.concat_str(pl.lit(f"{period}|"), pl.col(c)).alias(c) for c in (PLAYER, TEAM)))
            .drop("home_players_smooth", "away_players_smooth")
            .collect())
        parts.append(prepared)
    if not parts:
        raise ValueError("No valid period/frame keys.")
    panel = pl.concat(parts).sort(["period", "framenum", "is_synthetic_pass_end"], descending=[False, False, True])
    panel = (panel.join(physical, on=KEY, how="left")
             .with_columns(pl.len().over(KEY).alias("physical_copies")))
    # Non-finite source values are not metric/reference observations. This
    # quality rule is applied equally, before candidate comparisons.
    panel = panel.with_columns(pl.struct([
        pl.when(pl.col("balls_smooth").struct.field(c).is_finite())
        .then(pl.col("balls_smooth").struct.field(c)).otherwise(None).alias(c)
        for c in ("x", "y")
    ]).alias("__metric_ball"))
    metric_panel = panel.drop("balls_smooth").rename({"__metric_ball": "balls_smooth"})
    metrics = (metric_panel.lazy().pipe(label_pitch_thirds)
               .pipe(add_ball_metrics, TEAM, frame_rate=fps).collect())
    audit["derived_player_id_rows"] = panel[PLAYER].len() - panel[PLAYER].null_count()
    audit["unknown_direction_rows"] = panel.filter(
        ~pl.col("attacking_team_direction").is_in(["L", "R"]).fill_null(False)).height
    audit["unknown_direction_in_possession_rows"] = panel.filter(
        pl.col(PLAYER).is_not_null() & ~pl.col("attacking_team_direction").is_in(["L", "R"]).fill_null(False)).height
    canonical = panel.unique(subset=["period", PLAYER, "framenum"], keep="first", maintain_order=True)
    gaps = []
    for group in canonical.drop_nulls([PLAYER, TEAM]).partition_by(["period", PLAYER], maintain_order=True):
        f = group["framenum"].to_numpy()
        bx = group["balls_smooth"].struct.field("x")
        by = group["balls_smooth"].struct.field("y")
        complete = np.isfinite(bx.to_numpy()) & np.isfinite(by.to_numpy())
        both_null = bx.is_null().to_numpy() & by.is_null().to_numpy()
        i = 0
        while i < len(f):
            if not both_null[i]:
                i += 1
                continue
            start = i
            while i < len(f) and both_null[i]:
                i += 1
            if start and i < len(f) and complete[start - 1] and complete[i]:
                gaps.append({"game_id": game_id, "period": int(group["period"][0]),
                             "gap_length": int(f[i] - f[start - 1] - 1),
                             "missing_rows": int(i - start),
                             "missing_seconds": float((f[i] - f[start - 1] - 1) / fps)})
    audit["bounded_both_null_gaps"] = len(gaps)
    audit["absent_frame_intervals"] = sum(
        int((np.diff(g["framenum"].unique().sort().to_numpy()) > 1).sum())
        for g in raw.drop_nulls(KEY).partition_by("period")
    )
    return PreparedMatch(panel, _totals(metrics, "player"), _totals(metrics, "team"),
                         audit, pd.DataFrame(gaps, columns=["game_id", "period", "gap_length", "missing_rows", "missing_seconds"]), fps)


def reference_segments(prepared: PreparedMatch):
    """Finite continuous, unambiguous, oriented player/team reference segments."""
    panel = prepared.panel
    segments = []
    canonical = panel.unique(subset=["period", PLAYER, "framenum"], keep="first", maintain_order=True)
    for group in canonical.drop_nulls([PLAYER, TEAM]).partition_by(["period", PLAYER, TEAM], maintain_order=True):
        xy = np.column_stack([group["balls_smooth"].struct.field(c).to_numpy() for c in ("x", "y")])
        frames = group["framenum"].to_numpy()
        valid = np.isfinite(xy).all(axis=1) & group["attacking_team_direction"].is_in(["L", "R"]).fill_null(False).to_numpy()
        valid &= group["coordinate_versions"].to_numpy() == 1
        breaks = np.r_[True, (np.diff(frames) != 1) | ~valid[:-1] | ~valid[1:]]
        for indices in np.split(np.arange(len(group)), np.flatnonzero(breaks)[1:]):
            indices = indices[valid[indices]]
            if len(indices) >= 3:
                segments.append((group[indices], frames[indices], xy[indices]))
    return segments


def build_gap_cases(prepared: PreparedMatch, *, seeds=(11, 23, 37, 53, 71),
                    lengths=(1, 2, 3, 4, 5, 8, 15, 30, 45), per_length=3,
                    context_seconds=1.0) -> tuple[list[GapCase], pd.DataFrame]:
    """Sample isolated interior gaps; no target overlap within each mask seed."""
    game_id = int(prepared.panel["game_id"][0])
    segments = reference_segments(prepared)
    candidates = {length: [] for length in lengths}
    context_frames = max(1, int(np.ceil(context_seconds * prepared.frame_rate)))
    for seg_id, (group, frames, xy) in enumerate(segments):
        safe = ((group["physical_copies"].to_numpy() == 1)
                & ~group["is_synthetic_pass_end"].to_numpy())
        unsafe_prefix = np.r_[0, np.cumsum(~safe)]
        for length in lengths:
            for start in range(1, len(frames) - length):
                if unsafe_prefix[start + length] == unsafe_prefix[start]:
                    candidates[length].append((seg_id, start))
    cases, support = [], []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), game_id]))
        used = set()
        for length in rng.permutation(lengths):
            selected = 0
            pool = candidates[int(length)]
            for candidate_index in rng.permutation(len(pool)):
                seg_id, start = pool[candidate_index]
                group, frames, xy = segments[seg_id]
                target_frames = frames[start:start + length]
                period = int(group["period"][0])
                keys = {(period, int(f)) for f in target_frames}
                if used & keys:
                    continue
                used |= keys
                lo, hi = max(0, start - context_frames), min(len(frames), start + length + context_frames)
                player, team = group[PLAYER][0], group[TEAM][0]
                pdist, psec = prepared.player_totals[player]
                tdist, tsec = prepared.team_totals[team]
                if psec is None or tsec is None or psec <= 0 or tsec <= 0:
                    continue
                local = xy[start - 1:start + length + 1]
                velocities = np.diff(local, axis=0) * prepared.frame_rate
                if len(velocities) > 1:
                    norms = np.linalg.norm(velocities, axis=1)
                    cos = np.sum(velocities[:-1] * velocities[1:], axis=1) / np.maximum(norms[:-1] * norms[1:], 1e-12)
                    turn = float(np.max(np.arccos(np.clip(cos, -1, 1))))
                else:
                    turn = 0.
                case_id = f"{game_id}:{period}:{int(target_frames[0])}:{length}:{seed}"
                visibility = group["balls_smooth"].struct.field("visibility").to_numpy()
                all_context_visible = bool(np.all(visibility[lo:hi] == "VISIBLE"))
                all_targets_visible = bool(np.all(visibility[start:start + length] == "VISIBLE"))
                cases.append(GapCase(
                    case_id, game_id, period, player, team, int(seed), int(length), prepared.frame_rate,
                    frames[lo:hi].copy(), xy[lo:hi].copy(), np.arange(start - lo, start - lo + length),
                    float(pdist), float(psec), float(tdist), float(tsec),
                    str(group["possession_event_type"][start]),
                    str(group["balls_smooth"].struct.field("visibility")[start]),
                    float(np.linalg.norm(velocities, axis=1).max()), turn,
                    "all_context_visible" if all_context_visible else "includes_nonvisible_context",
                    all_targets_visible,
                ))
                selected += 1
                if selected == per_length:
                    break
            support.append({"game_id": game_id, "seed": int(seed), "gap_length": int(length),
                            "available_starts": len(pool), "requested": per_length, "sampled": selected})
    return cases, pd.DataFrame(support)


def case_record(case: GapCase):
    """Parquet-compatible case serialization; hashes bind reference and mask."""
    result = dict(vars(case))
    for column in ("frames", "reference_xy", "target"):
        result[column] = result[column].tolist()
    result["case_sha256"] = specification_hash(result)
    return result
