"""Possession-level tempo metrics derived from ball movement."""

from __future__ import annotations

from typing import Literal

import polars as pl

from tempoctrl.gradient_sports.frame_rates import FRAME_RATE_COLUMN

PossessionLevel = Literal["team", "player"]

_GAME_COLUMN = "game_id"
_FRAME_COLUMN = "framenum"
_DISPLACEMENT_COLUMN = "ball_displacement"
_DELTA_FRAME_COLUMN = "delta_frame"
_PITCH_THIRD_COLUMN = "pitch_third"
_TEAM_POSSESSION_COLUMN = "dev_match_team_possession_id"
_PLAYER_POSSESSION_COLUMN = "dev_match_team_player_possession_id"
_OUTPUT_TEAM_POSSESSION_COLUMN = "match_team_possession_id"
_OUTPUT_PLAYER_POSSESSION_COLUMN = "match_team_player_possession_id"
_UNIQUE_PLAYER_POSSESSION_COUNT = "unique_player_possession_count"
_TEAM_START_FRAME_COLUMN = "__team_possession_start_frame"
_STARTING_PITCH_THIRD_COLUMN = "starting_pitch_third"
_POSSESSION_COLUMNS: dict[PossessionLevel, str] = {
    "team": _TEAM_POSSESSION_COLUMN,
    "player": _PLAYER_POSSESSION_COLUMN,
}


def _validate_tempo_inputs(
    df: pl.LazyFrame,
    level: PossessionLevel,
) -> str:
    """Validate tempo inputs and return the possession ID column."""
    if level not in _POSSESSION_COLUMNS:
        raise ValueError("level must be either 'team' or 'player'.")
    possession_column = _POSSESSION_COLUMNS[level]
    required_columns = [
        _GAME_COLUMN,
        possession_column,
        _FRAME_COLUMN,
        _DISPLACEMENT_COLUMN,
        _DELTA_FRAME_COLUMN,
        _PITCH_THIRD_COLUMN,
        FRAME_RATE_COLUMN,
    ]
    if level == "team":
        required_columns.append(_PLAYER_POSSESSION_COLUMN)
    else:
        required_columns.append(_TEAM_POSSESSION_COLUMN)
    schema = df.collect_schema()
    missing_columns = [
        column for column in required_columns if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing possession tempo columns: {missing}.")

    numeric_columns = (
        _FRAME_COLUMN,
        _DISPLACEMENT_COLUMN,
        _DELTA_FRAME_COLUMN,
        FRAME_RATE_COLUMN,
    )
    nonnumeric_columns = [
        column for column in numeric_columns if not schema[column].is_numeric()
    ]
    if nonnumeric_columns:
        invalid = ", ".join(nonnumeric_columns)
        raise TypeError(
            f"Possession tempo columns must be numeric: {invalid}."
        )

    return possession_column


def starting_pitch_third() -> pl.Expr:
    """Retain the pitch third at a possession's earliest frame."""
    return (
        pl.col(_PITCH_THIRD_COLUMN)
        .sort_by(pl.col(_FRAME_COLUMN))
        .first()
        .alias(_STARTING_PITCH_THIRD_COLUMN)
    )


def _level_metadata_expressions(
    level: PossessionLevel,
) -> tuple[pl.Expr, ...]:
    """Build metadata aggregations specific to a possession level."""
    if level == "team":
        return (
            pl.col(_PLAYER_POSSESSION_COLUMN)
            .drop_nulls()
            .n_unique()
            .cast(pl.UInt32)
            .alias(_UNIQUE_PLAYER_POSSESSION_COUNT),
        )

    return (
        pl.col(_TEAM_START_FRAME_COLUMN)
        .first()
        .alias(_TEAM_START_FRAME_COLUMN),
    )


def _add_possession_sequence(df: pl.LazyFrame) -> pl.LazyFrame:
    """Assign chronological numbers within a team possession.

    Sequence numbers are one-based. A team possession receives null
    sequence numbers if any player possessions share a start frame or
    have a null start frame. This exposes invalid ordering data rather
    than resolving ties arbitrarily.
    """
    team_group = (_GAME_COLUMN, _TEAM_POSSESSION_COLUMN)
    start_group = (*team_group, "start_frame")
    invalid_start = (pl.len().over(start_group) > 1) | pl.col(
        "start_frame"
    ).is_null()
    invalid_team_sequence = invalid_start.any().over(team_group)
    sequence_number = (
        pl.col("start_frame")
        .rank(method="ordinal")
        .over(team_group)
        .cast(pl.UInt32)
    )

    return df.with_columns(
        pl.when(~invalid_team_sequence)
        .then(sequence_number)
        .alias("player_possession_sequence_number")
    )


def _add_possession_time_elapsed(df: pl.LazyFrame) -> pl.LazyFrame:
    """Measure each player possession's start from the true team start."""
    elapsed_frames = pl.col("start_frame") - pl.col(_TEAM_START_FRAME_COLUMN)
    valid_elapsed_frames = elapsed_frames.is_not_null() & (elapsed_frames >= 0)

    return df.with_columns(
        pl.when(valid_elapsed_frames)
        .then(elapsed_frames)
        .alias("elapsed_frames_team_possession"),
        pl.when(valid_elapsed_frames)
        .then(elapsed_frames.cast(pl.Float64) / pl.col(FRAME_RATE_COLUMN))
        .alias("elapsed_seconds_team_possession"),
    )


def aggregate_possession_tempo(
    df: pl.LazyFrame,
    level: PossessionLevel,
) -> pl.LazyFrame:
    """Summarize observed ball movement for each possession.

    A valid movement segment has a non-null displacement and a positive
    frame interval. Tempo is total valid displacement divided by the
    observed time represented by those segments. This avoids counting
    leading rows, duplicate frames, or missing-coordinate intervals as
    though they were measured movement.

    Possessions are isolated by both game and possession ID. Rows with
    a null key are excluded instead of being combined into an artificial
    null possession. Output is sorted by those keys for deterministic
    downstream files. Invalid or conflicting frame-rate metadata makes
    tempo null rather than selecting an arbitrary rate.

    Args:
        df: Lazy frame-level rows with ball movement metrics.
        level: Team or player possession granularity.

    Returns:
        One row per possession with frame bounds, pitch-third movement,
        valid segment count, total displacement, elapsed frames, and ball
        tempo. Metrics are null when a possession has no valid movement
        segment. Team rows also count their distinct non-null player
        possessions. Player rows include their team ID, chronological
        sequence, and elapsed time from the true team-possession start.

    """
    possession_column = _validate_tempo_inputs(
        df,
        level,
    )
    group_columns = (
        (_GAME_COLUMN, possession_column)
        if level == "team"
        else (
            _GAME_COLUMN,
            _TEAM_POSSESSION_COLUMN,
            possession_column,
        )
    )
    metric_input = (
        df
        if level == "team"
        else df.with_columns(
            pl.col(_FRAME_COLUMN)
            .min()
            .over(_GAME_COLUMN, _TEAM_POSSESSION_COLUMN)
            .alias(_TEAM_START_FRAME_COLUMN)
        )
    )
    valid_key = pl.all_horizontal(
        *(pl.col(column).is_not_null() for column in group_columns)
    )
    valid_segment = pl.col(_DISPLACEMENT_COLUMN).is_not_null() & (
        pl.col(_DELTA_FRAME_COLUMN) > 0
    ).fill_null(False)
    has_valid_segment = pl.col("valid_segment_count") > 0
    valid_frame_rate = pl.col(FRAME_RATE_COLUMN).is_finite().fill_null(
        False
    ) & (pl.col(FRAME_RATE_COLUMN) > 0).fill_null(False)
    has_consistent_frame_rate = ~pl.col("__has_invalid_frame_rate") & (
        pl.col("__minimum_frame_rate") == pl.col("__maximum_frame_rate")
    )
    tempo_column = f"ball_speed_tempo_{level}"
    metadata_expressions = _level_metadata_expressions(level)
    metadata_columns = (
        [_UNIQUE_PLAYER_POSSESSION_COUNT]
        if level == "team"
        else [_TEAM_START_FRAME_COLUMN]
    )
    base_output_columns = [
        *group_columns,
        "start_frame",
        "end_frame",
        *metadata_columns,
        _STARTING_PITCH_THIRD_COLUMN,
        FRAME_RATE_COLUMN,
        "valid_segment_count",
        "total_ball_displacement",
        "elapsed_frames",
        "elapsed_duration",
        tempo_column,
    ]

    aggregated = (
        metric_input.filter(valid_key)
        .group_by(group_columns)
        .agg(
            pl.col(_FRAME_COLUMN).min().alias("start_frame"),
            pl.col(_FRAME_COLUMN).max().alias("end_frame"),
            *metadata_expressions,
            starting_pitch_third(),
            valid_segment.sum().cast(pl.UInt32).alias("valid_segment_count"),
            pl.when(valid_segment)
            .then(pl.col(_DISPLACEMENT_COLUMN))
            .sum()
            .cast(pl.Float64)
            .alias("__total_ball_displacement"),
            pl.when(valid_segment)
            .then(pl.col(_DELTA_FRAME_COLUMN))
            .sum()
            .cast(pl.Float64)
            .alias("__elapsed_frames"),
            (~valid_frame_rate).any().alias("__has_invalid_frame_rate"),
            pl.col(FRAME_RATE_COLUMN)
            .cast(pl.Float64)
            .min()
            .alias("__minimum_frame_rate"),
            pl.col(FRAME_RATE_COLUMN)
            .cast(pl.Float64)
            .max()
            .alias("__maximum_frame_rate"),
        )
        .with_columns(
            pl.when(has_valid_segment)
            .then(pl.col("__total_ball_displacement"))
            .alias("total_ball_displacement"),
            pl.when(has_valid_segment)
            .then(pl.col("__elapsed_frames"))
            .alias("elapsed_frames"),
            pl.when(has_consistent_frame_rate)
            .then(pl.col("__minimum_frame_rate"))
            .alias(FRAME_RATE_COLUMN),
        )
        .with_columns(
            (pl.col("elapsed_frames") / pl.col(FRAME_RATE_COLUMN)).alias(
                "elapsed_duration"
            ),
            (
                pl.col("total_ball_displacement")
                * pl.col(FRAME_RATE_COLUMN)
                / pl.col("elapsed_frames")
            ).alias(tempo_column),
        )
        .select(base_output_columns)
    )

    if level == "team":
        return aggregated.rename(
            {
                _TEAM_POSSESSION_COLUMN: _OUTPUT_TEAM_POSSESSION_COLUMN,
            }
        ).sort(_GAME_COLUMN, _OUTPUT_TEAM_POSSESSION_COLUMN)

    player_output_columns = [
        *group_columns,
        "player_possession_sequence_number",
        "start_frame",
        "end_frame",
        "elapsed_frames_team_possession",
        "elapsed_seconds_team_possession",
        _STARTING_PITCH_THIRD_COLUMN,
        FRAME_RATE_COLUMN,
        "valid_segment_count",
        "total_ball_displacement",
        "elapsed_frames",
        tempo_column,
    ]
    return (
        aggregated.pipe(_add_possession_sequence)
        .pipe(_add_possession_time_elapsed)
        .select(player_output_columns)
        .rename(
            {
                _TEAM_POSSESSION_COLUMN: _OUTPUT_TEAM_POSSESSION_COLUMN,
                _PLAYER_POSSESSION_COLUMN: _OUTPUT_PLAYER_POSSESSION_COLUMN,
            }
        )
        .sort(
            _GAME_COLUMN,
            _OUTPUT_TEAM_POSSESSION_COLUMN,
            "player_possession_sequence_number",
        )
    )
