"""Transform Gradient Sports possession data."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from tempoctrl.gradient_sports.ball_metrics import add_ball_metrics
from tempoctrl.gradient_sports.frame_rates import FrameRateSpec
from tempoctrl.gradient_sports.ingest import scan_processed_files
from tempoctrl.gradient_sports.interpolations import (
    interpolate_ball_coordinates,
)

POSSESSION_COLUMNS = (
    "match_team_possession_id",
    "match_team_player_possession_id",
)

TRACKING_COLUMNS = (
    "balls_smooth",
    "away_players_smooth",
    "home_players_smooth",
)

REQUIRED_COLUMNS = (
    "game_id",
    "framenum",
    *POSSESSION_COLUMNS,
    *TRACKING_COLUMNS,
    "successful_pass_or_cross",
)

# FIXME: should we add period?
SORT_COLUMNS = (
    "game_id",
    "framenum",
)

ATTACKING_DIRECTION_COLUMNS = (
    "match_team_possession_id",
    "attacking_team_direction",
)

_CURRENT_POSSESSION_COLUMN = "__current_player_possession_id"
_SUCCESSFUL_DELIVERY_COLUMN = "__successful_delivery_possession_id"
SYNTHETIC_PASS_END_COLUMN = "is_synthetic_pass_end"

PITCH_LENGTH_METERS = 105.0
PITCH_HALF_LENGTH_METERS = PITCH_LENGTH_METERS / 2
PITCH_THIRD_BOUNDARY_METERS = PITCH_LENGTH_METERS / 6
PITCH_THIRD_DTYPE = pl.Enum(["Defensive", "Middle", "Attacking"])


def validate_possession_columns(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Require the columns used by possession propagation.

    Schema validation reads only Parquet metadata and does not collect
    the tracking rows.
    """
    schema = df_in.collect_schema()
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing possession columns: {missing}")

    return df_in


def _bounded_possession_id(poss_col: str) -> pl.Expr:
    """Build an expression that fills bounded IDs within each game."""
    possession_id = pl.col(poss_col)
    previous_id = possession_id.forward_fill().over("game_id")
    next_id = possession_id.backward_fill().over("game_id")
    return (
        pl.when(previous_id == next_id)
        .then(previous_id)
        .otherwise(possession_id)
        .alias(f"dev_{poss_col}")
    )


def fill_bounded_possessions_id(
    df: pl.LazyFrame,
    poss_col: str | Sequence[str],
) -> pl.LazyFrame:
    """Fill bounded null IDs in one or more possession columns.

    Leading nulls, trailing nulls, and gaps bounded by different IDs are
    left unchanged. The source column is preserved and the filled values
    are written to columns prefixed with ``dev_``. Multiple columns are
    accepted so callers can sort the input only once. Duplicate rows for
    a frame share their single non-null possession annotation.
    """
    possession_columns = (poss_col,) if isinstance(poss_col, str) else poss_col
    frame_columns = {
        column: f"__frame_{column}" for column in possession_columns
    }
    return (
        df.sort(SORT_COLUMNS)
        .with_columns(
            *(
                pl.col(column)
                .drop_nulls()
                .first()
                .over(SORT_COLUMNS)
                .alias(frame_column)
                for column, frame_column in frame_columns.items()
            )
        )
        .with_columns(
            *(
                _bounded_possession_id(frame_column).alias(f"dev_{column}")
                for column, frame_column in frame_columns.items()
            )
        )
        .drop(*frame_columns.values())
    )


def _successful_delivery_id(poss_col: str) -> pl.Expr:
    """Track the latest successful delivery possession in each game."""
    possession_id = pl.col(poss_col)
    return (
        pl.when(
            possession_id.is_not_null()
            & pl.col("successful_pass_or_cross").fill_null(False)
        )
        .then(possession_id)
        .forward_fill()
        .over("game_id")
        .alias(_SUCCESSFUL_DELIVERY_COLUMN)
    )


def propagate_successful_delivery_possession(
    df_in: pl.LazyFrame,
) -> pl.LazyFrame:
    """Propagate a successful delivery until another possession starts.

    Existing bounded-gap fills are retained. An unbounded null receives
    the successful player's possession ID only while that ID remains the
    latest observed player possession in the game.
    """
    possession_column = "match_team_player_possession_id"
    development_column = f"dev_{possession_column}"
    possession_id = pl.col(possession_column)
    delivery_is_active = pl.col(_CURRENT_POSSESSION_COLUMN) == pl.col(
        _SUCCESSFUL_DELIVERY_COLUMN
    )
    propagated_id = pl.when(delivery_is_active).then(
        pl.col(_SUCCESSFUL_DELIVERY_COLUMN)
    )

    return (
        df_in.with_columns(
            possession_id.forward_fill()
            .over("game_id")
            .alias(_CURRENT_POSSESSION_COLUMN),
            _successful_delivery_id(possession_column),
        )
        .with_columns(
            pl.coalesce(
                pl.col(development_column),
                propagated_id,
            ).alias(development_column)
        )
        .drop(
            _CURRENT_POSSESSION_COLUMN,
            _SUCCESSFUL_DELIVERY_COLUMN,
        )
    )


def create_synthetic_final_pass_frame(
    df_in: pl.LazyFrame,
) -> pl.LazyFrame:
    """Add a passer row at each successful delivery's receiver frame.

    The synthetic row carries the passer's derived possession IDs and
    the receiver frame's tracking state. All other source fields are
    null. Exactly one row is created per delivery boundary, even when a
    frame has duplicate source rows.
    """
    schema = df_in.collect_schema()
    development_team = "dev_match_team_possession_id"
    development_player = "dev_match_team_player_possession_id"
    required_columns = (
        "game_id",
        "framenum",
        "match_team_player_possession_id",
        development_team,
        development_player,
        "successful_pass_or_cross",
        *TRACKING_COLUMNS,
    )
    missing_columns = [
        column for column in required_columns if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing synthetic pass columns: {missing}")

    receiver_player = "__receiver_player_possession_id"
    frame_player = "__frame_player_possession_id"
    frame_team = "__frame_team_possession_id"
    successful_player = "__successful_player_possession_id"
    previous_player = "__previous_player_possession_id"
    previous_team = "__previous_team_possession_id"
    previous_success = "__previous_successful_player_possession_id"

    frame_state = (
        df_in.group_by(SORT_COLUMNS)
        .agg(
            pl.col("match_team_player_possession_id")
            .drop_nulls()
            .first()
            .alias(receiver_player),
            pl.col(development_player)
            .drop_nulls()
            .first()
            .alias(frame_player),
            pl.col(development_team).drop_nulls().first().alias(frame_team),
            pl.col("match_team_player_possession_id")
            .filter(pl.col("successful_pass_or_cross").fill_null(False))
            .drop_nulls()
            .first()
            .alias(successful_player),
            *(
                pl.col(column).drop_nulls().first().alias(column)
                for column in TRACKING_COLUMNS
            ),
        )
        .with_columns(
            pl.col(frame_player)
            .shift(1)
            .over("game_id", order_by="framenum")
            .alias(previous_player),
            pl.col(frame_team)
            .shift(1)
            .over("game_id", order_by="framenum")
            .alias(previous_team),
            pl.col(successful_player)
            .forward_fill()
            .shift(1)
            .over("game_id", order_by="framenum")
            .alias(previous_success),
        )
    )
    delivery_boundaries = frame_state.filter(
        pl.col(receiver_player).is_not_null()
        & pl.col(previous_player).is_not_null()
        & (pl.col(receiver_player) != pl.col(previous_player))
        & (pl.col(previous_success) == pl.col(previous_player))
    )

    synthetic_values = {
        "game_id": pl.col("game_id"),
        "framenum": pl.col("framenum"),
        development_team: pl.col(previous_team),
        development_player: pl.col(previous_player),
        **{column: pl.col(column) for column in TRACKING_COLUMNS},
    }
    source_schema = {
        column: dtype
        for column, dtype in schema.items()
        if column != SYNTHETIC_PASS_END_COLUMN
    }
    synthetic_rows = delivery_boundaries.select(
        *(
            synthetic_values.get(
                column,
                pl.lit(None, dtype=dtype),
            )
            .cast(dtype)
            .alias(column)
            for column, dtype in source_schema.items()
        ),
        pl.lit(True).alias(SYNTHETIC_PASS_END_COLUMN),
    )
    original_rows = df_in.select(*source_schema).with_columns(
        pl.lit(False).alias(SYNTHETIC_PASS_END_COLUMN)
    )

    return pl.concat(
        [original_rows, synthetic_rows],
        how="vertical",
    ).sort(
        [*SORT_COLUMNS, SYNTHETIC_PASS_END_COLUMN],
        descending=[False, False, True],
    )


def append_attacking_direction(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Append each possession's attacking direction."""
    path_events = Path("data/processed/gradient_sports/events/")
    df_events = scan_processed_files(
        path_events, columns=ATTACKING_DIRECTION_COLUMNS
    )

    return df_in.join(
        df_events.drop_nulls().unique(),
        left_on="dev_match_team_possession_id",
        right_on="match_team_possession_id",
        how="left",
        maintain_order="left",
        suffix="_event",
        coalesce=True,
        validate="m:1",
    )


def _flip_if_attacking_left(struct_name: str, struct_column: str) -> pl.Expr:
    """Reflect coordinates when the team is attacking left.

    This makes all downstream coordinates run from left to right.
    """
    value = pl.col(struct_name).struct.field(struct_column)

    return (
        pl.when(pl.col("attacking_team_direction") == "L")
        .then(-value)
        .otherwise(value)
        .alias(struct_column)
    )


def normalize_ball_coordinates(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize ball coordinates to a common attacking direction."""
    return df_in.with_columns(
        pl.struct(
            [
                pl.col("balls_smooth").struct.field("visibility"),
                _flip_if_attacking_left("balls_smooth", "x"),
                _flip_if_attacking_left("balls_smooth", "y"),
                pl.col("balls_smooth").struct.field("z"),
            ]
        ).alias("balls_smooth")
    )


def label_pitch_thirds(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Label valid, normalized ball x coordinates by pitch third.

    Coordinates must use a centered 105-meter pitch and be normalized so
    the possessing team attacks from left to right. Boundary coordinates
    belong to the third immediately to their left: ``-17.5`` is
    defensive and ``17.5`` is middle. Missing, non-finite, and
    out-of-pitch values remain unlabeled.
    """
    coord_x = pl.col("balls_smooth").struct.field("x")

    valid_coordinate = coord_x.is_finite().fill_null(
        False
    ) & coord_x.is_between(
        -PITCH_HALF_LENGTH_METERS,
        PITCH_HALF_LENGTH_METERS,
        closed="both",
    ).fill_null(False)

    pitch_third = (
        pl.when(~valid_coordinate)
        .then(pl.lit(None, dtype=PITCH_THIRD_DTYPE))
        .when(coord_x <= -PITCH_THIRD_BOUNDARY_METERS)
        .then(pl.lit("Defensive", dtype=PITCH_THIRD_DTYPE))
        .when(coord_x <= PITCH_THIRD_BOUNDARY_METERS)
        .then(pl.lit("Middle", dtype=PITCH_THIRD_DTYPE))
        .otherwise(pl.lit("Attacking", dtype=PITCH_THIRD_DTYPE))
        .alias("pitch_third")
    )

    return df_in.with_columns(pitch_third)


def transform_possessions(
    df_in: pl.LazyFrame,
    *,
    frame_rate: FrameRateSpec,
) -> pl.LazyFrame:
    """Fill bounded gaps and extend successful player deliveries."""
    return (
        df_in.pipe(validate_possession_columns)
        .pipe(fill_bounded_possessions_id, POSSESSION_COLUMNS)
        .pipe(propagate_successful_delivery_possession)
        .pipe(create_synthetic_final_pass_frame)
        .pipe(append_attacking_direction)
        .pipe(normalize_ball_coordinates)
        .pipe(interpolate_ball_coordinates)
        .pipe(label_pitch_thirds)
        .pipe(
            add_ball_metrics,
            "dev_match_team_possession_id",
            frame_rate=frame_rate,
        )
    )
