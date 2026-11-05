"""Build frame-, team-, and player-level possession datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from tempoctrl.gradient_sports.frame_rates import FrameRateSpec
from tempoctrl.gradient_sports.ingest import scan_processed_files
from tempoctrl.gradient_sports.possessions_transform import (
    transform_possessions,
)
from tempoctrl.gradient_sports.tempo_metrics import (
    PossessionLevel,
    aggregate_possession_tempo,
)

FRAME_LEVEL_ORDER = (
    "match_team_possession_id",
    "match_team_player_possession_id",
    "dev_match_team_player_possession_id",
    "dev_match_team_possession_id",
    "game_id",
    "game_event_id",
    "possession_event_id",
    "possession_event_type",
    "successful_pass_or_cross",
    "player_id",
    "attacking_team_direction",
    "game_event_type",
    "framenum",
    "formattedgameclock",
    "is_synthetic_pass_end",
    "pitch_third",
    "balls_smooth",
    "delta_x",
    "delta_y",
    "delta_frame",
    "ball_displacement",
    "ball_speed",
    "frame_rate",
    "away_players_smooth",
    "home_players_smooth",
)


_TEMPO_INPUT_COLUMNS = (
    "game_id",
    "dev_match_team_possession_id",
    "dev_match_team_player_possession_id",
    "framenum",
    "pitch_third",
    "ball_displacement",
    "delta_frame",
    "frame_rate",
)
_POSSESSION_LEVELS: tuple[PossessionLevel, ...] = ("team", "player")


def _build_output_paths(
    output_dir: Path,
    output_name: str,
) -> tuple[Path, ...]:
    """Build frame, team, and player output paths in publication order."""
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be a non-empty file name.")

    return (
        output_dir / output_name,
        output_dir / f"team_{output_name}",
        output_dir / f"player_{output_name}",
    )


def _validate_staged_outputs(paths: tuple[Path, ...]) -> None:
    """Validate staged parquet footers and schemas without reading rows."""
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Staged output is missing or empty: {path}")
        pl.scan_parquet(path).collect_schema()


def _publish_staged_outputs(
    staged_paths: tuple[Path, ...],
    target_paths: tuple[Path, ...],
) -> None:
    """Publish staged files and restore prior outputs after failure."""
    backup_dir = staged_paths[0].parent / "backups"
    backup_dir.mkdir()
    backups: dict[Path, Path] = {}
    published_paths: list[Path] = []

    try:
        for staged_path, target_path in zip(
            staged_paths,
            target_paths,
            strict=True,
        ):
            if target_path.exists():
                backup_path = backup_dir / target_path.name
                target_path.replace(backup_path)
                backups[target_path] = backup_path

            staged_path.replace(target_path)
            published_paths.append(target_path)
    except Exception:
        for published_path in reversed(published_paths):
            if published_path.exists():
                published_path.unlink()
        for target_path, backup_path in reversed(backups.items()):
            if backup_path.exists():
                backup_path.replace(target_path)
        raise


def write_possession_outputs(
    frame_df: pl.LazyFrame,
    output_name: str,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Build and transactionally publish all possession outputs.

    The frame parquet is a materialization boundary. Team and player
    aggregations scan only their required columns from that file, so the
    upstream transformation graph executes once rather than once per
    output. Files are staged and validated before existing outputs are
    replaced. A publication failure restores the previous files.

    Args:
        frame_df: Fully transformed frame-level possession rows.
        output_name: File name for the frame-level parquet.
        output_dir: Directory receiving all three parquet files.

    Returns:
        Frame-, team-, and player-level output paths, in that order.

    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    target_paths = _build_output_paths(
        output_directory,
        output_name,
    )

    with TemporaryDirectory(
        dir=output_directory,
        prefix=".possession-outputs-",
    ) as staging_name:
        staging_directory = Path(staging_name)
        staged_paths = _build_output_paths(
            staging_directory,
            output_name,
        )
        frame_path = staged_paths[0]

        frame_df.sink_parquet(frame_path, compression="zstd")
        tempo_input = pl.scan_parquet(frame_path).select(_TEMPO_INPUT_COLUMNS)
        for level, output_path in zip(
            _POSSESSION_LEVELS,
            staged_paths[1:],
            strict=True,
        ):
            (
                tempo_input.pipe(
                    aggregate_possession_tempo,
                    level,
                ).sink_parquet(output_path, compression="zstd")
            )

        _validate_staged_outputs(staged_paths)
        _publish_staged_outputs(staged_paths, target_paths)

    return target_paths


def possessions_load(
    df_path: str | Path | Sequence[str | Path],
    output_name: str,
    *,
    output_dir: str | Path,
    frame_rate: FrameRateSpec,
) -> tuple[Path, ...]:
    """Load and write possession datasets, returning output paths."""
    frame_df = (
        scan_processed_files(df_path=df_path)
        .pipe(transform_possessions, frame_rate=frame_rate)
        .select(FRAME_LEVEL_ORDER)
    )

    return write_possession_outputs(
        frame_df,
        output_name,
        output_dir=output_dir,
    )
