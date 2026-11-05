"""Row-level ball movement metrics for possession tracking data."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from tempoctrl.gradient_sports.frame_rates import (
    FRAME_RATE_COLUMN,
    FrameRateSpec,
    add_frame_rate_column,
)

SYNTHETIC_PASS_END_COLUMN = "is_synthetic_pass_end"

_BALL_COLUMN = "balls_smooth"
_FRAME_COLUMN = "framenum"
_IS_METRIC_FRAME_COLUMN = "__is_ball_metric_frame"
_METRIC_INPUT_ORDER_COLUMN = "__ball_metric_input_order"


def _resolve_group_columns(
    df: pl.LazyFrame,
    possession_groups: str | Sequence[str],
) -> tuple[str, ...]:
    """Build game-aware partitions from one or more possession IDs.

    ``game_id`` is always included so identical possession IDs from
    different games cannot share movement history. Schema inspection
    reads metadata only and does not collect tracking rows.
    """
    possession_columns = (
        (possession_groups,)
        if isinstance(possession_groups, str)
        else tuple(possession_groups)
    )
    if not possession_columns:
        raise ValueError("At least one possession group is required.")
    if not all(isinstance(column, str) for column in possession_columns):
        raise TypeError("Possession group names must be strings.")
    if any(not column for column in possession_columns):
        raise ValueError("Possession group names cannot be empty.")
    if len(set(possession_columns)) != len(possession_columns):
        raise ValueError("Possession group names must be unique.")

    group_columns = ("game_id", *possession_columns)
    schema = df.collect_schema()
    missing_columns = [
        column for column in group_columns if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing ball metric group columns: {missing}.")

    return group_columns


def _validate_displacement_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """Validate frame and ball-coordinate inputs for displacement."""
    schema = df.collect_schema()
    required_columns = (_FRAME_COLUMN, _BALL_COLUMN)
    missing_columns = [
        column for column in required_columns if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing ball metric columns: {missing}.")

    if not schema[_FRAME_COLUMN].is_numeric():
        raise TypeError(f"{_FRAME_COLUMN} must be numeric.")

    ball_dtype = schema[_BALL_COLUMN]
    if not isinstance(ball_dtype, pl.Struct):
        raise TypeError(f"{_BALL_COLUMN} must be a struct column.")

    ball_fields = {field.name: field.dtype for field in ball_dtype.fields}
    missing_fields = [
        field for field in ("x", "y") if field not in ball_fields
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"{_BALL_COLUMN} is missing fields: {missing}.")

    nonnumeric_fields = [
        field for field in ("x", "y") if not ball_fields[field].is_numeric()
    ]
    if nonnumeric_fields:
        invalid = ", ".join(nonnumeric_fields)
        raise TypeError(f"{_BALL_COLUMN} fields must be numeric: {invalid}.")

    if (
        SYNTHETIC_PASS_END_COLUMN in schema
        and schema[SYNTHETIC_PASS_END_COLUMN] != pl.Boolean
    ):
        raise TypeError(f"{SYNTHETIC_PASS_END_COLUMN} must be Boolean.")

    return df


def _add_metric_frame_marker(
    df: pl.LazyFrame,
    group_columns: tuple[str, ...],
) -> pl.LazyFrame:
    """Select one row to own movement metrics at each frame.

    Synthetic pass endpoints take priority over source rows at the same
    frame. Otherwise, the first input row is selected. Rows with null
    game, possession, or frame keys are never selected.
    """
    schema = df.collect_schema()
    frame_columns = (*group_columns, _FRAME_COLUMN)
    input_order = pl.col(_METRIC_INPUT_ORDER_COLUMN)
    is_synthetic = (
        pl.col(SYNTHETIC_PASS_END_COLUMN).fill_null(False)
        if SYNTHETIC_PASS_END_COLUMN in schema
        else pl.lit(False)
    )

    with_input_order = df.with_columns(
        pl.int_range(pl.len()).alias(_METRIC_INPUT_ORDER_COLUMN)
    )
    first_input_order = input_order.min().over(frame_columns)
    first_synthetic_order = (
        pl.when(is_synthetic)
        .then(input_order)
        .otherwise(None)
        .min()
        .over(frame_columns)
    )
    valid_partition = pl.all_horizontal(
        *(pl.col(column).is_not_null() for column in group_columns),
        pl.col(_FRAME_COLUMN).is_not_null(),
    )
    selected_order = pl.coalesce(
        first_synthetic_order,
        first_input_order,
    )

    return with_input_order.with_columns(
        (valid_partition & (input_order == selected_order)).alias(
            _IS_METRIC_FRAME_COLUMN
        )
    )


def add_ball_displacement(
    df: pl.LazyFrame,
    possession_groups: str | Sequence[str],
) -> pl.LazyFrame:
    """Add consecutive-frame 2D ball movement within possessions.

    Each non-null displacement describes the segment ending at the
    current row. Ball x and y coordinates are assumed to be meters.
    Calculations use numerical frame order within each game and
    possession, while the returned row order remains unchanged.

    One row owns the metrics at each frame. A synthetic pass endpoint is
    preferred so the incoming pass segment belongs to the passer. Other
    duplicate rows receive null metrics. Missing coordinates also make
    every segment touching that observation null.

    Adds ``delta_x``, ``delta_y``, ``delta_frame``, and
    ``ball_displacement``. Displacement is the Euclidean distance in
    meters; speed is intentionally calculated by ``add_ball_speed``.

    Args:
        df: Lazy tracking rows containing frame and ball coordinates.
        possession_groups: Possession columns that define independent
            ball trajectories. ``game_id`` is included automatically.

    Returns:
        The lazy input with four row-level movement columns added.

    """
    group_columns = _resolve_group_columns(df, possession_groups)
    order_columns = (_FRAME_COLUMN, _METRIC_INPUT_ORDER_COLUMN)

    ball = pl.col(_BALL_COLUMN)
    x = ball.struct.field("x")
    y = ball.struct.field("y")
    is_metric_frame = pl.col(_IS_METRIC_FRAME_COLUMN)
    metric_state = pl.when(is_metric_frame).then(
        pl.struct(
            x.alias("x"),
            y.alias("y"),
            pl.col(_FRAME_COLUMN).alias("frame"),
        )
    )
    previous_state = (
        metric_state.forward_fill()
        .shift()
        .over(
            group_columns,
            order_by=order_columns,
        )
    )
    previous_x = previous_state.struct.field("x")
    previous_y = previous_state.struct.field("y")
    previous_frame = previous_state.struct.field("frame")
    delta_x = pl.when(is_metric_frame).then(x - previous_x)
    delta_y = pl.when(is_metric_frame).then(y - previous_y)
    delta_frame = pl.when(is_metric_frame).then(
        pl.col(_FRAME_COLUMN) - previous_frame
    )

    return (
        df.pipe(_validate_displacement_schema)
        .pipe(_add_metric_frame_marker, group_columns)
        .with_columns(
            delta_x.alias("delta_x"),
            delta_y.alias("delta_y"),
            delta_frame.alias("delta_frame"),
            (delta_x.pow(2) + delta_y.pow(2))
            .sqrt()
            .alias("ball_displacement"),
        )
        .drop(
            _IS_METRIC_FRAME_COLUMN,
            _METRIC_INPUT_ORDER_COLUMN,
        )
    )


def add_ball_speed(
    df: pl.LazyFrame,
    frame_rate: FrameRateSpec,
) -> pl.LazyFrame:
    """Convert ball displacement to meters per second.

    Speed is calculated as ``ball_displacement * frame_rate /
    delta_frame``. Rows with null, zero, or negative frame intervals
    receive null speed. The resolved rate is retained as metadata.

    Args:
        df: Lazy rows containing ``ball_displacement`` and
            ``delta_frame``.
        frame_rate: Shared rate or mapping from game ID to rate.

    Returns:
        The lazy input with Float64 ``frame_rate`` and ``ball_speed``
        columns added.

    Raises:
        ValueError: If the frame rate or required columns are invalid.
        TypeError: If either input metric column is nonnumeric.

    """
    schema = df.collect_schema()
    required_columns = ("ball_displacement", "delta_frame")
    missing_columns = [
        column for column in required_columns if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing ball speed columns: {missing}.")

    nonnumeric_columns = [
        column
        for column in required_columns
        if not schema[column].is_numeric()
    ]
    if nonnumeric_columns:
        invalid = ", ".join(nonnumeric_columns)
        raise TypeError(f"Ball speed columns must be numeric: {invalid}.")

    with_frame_rate = add_frame_rate_column(df, frame_rate)
    valid_frame_delta = pl.col("delta_frame") > 0
    ball_speed = pl.when(valid_frame_delta).then(
        pl.col("ball_displacement")
        * pl.col(FRAME_RATE_COLUMN)
        / pl.col("delta_frame")
    )

    return with_frame_rate.with_columns(ball_speed.alias("ball_speed"))


def add_ball_metrics(
    df: pl.LazyFrame,
    possession_groups: str | Sequence[str],
    *,
    frame_rate: FrameRateSpec,
) -> pl.LazyFrame:
    """Add row-level ball displacement and speed in one pipeline step.

    This convenience function composes ``add_ball_displacement`` and
    ``add_ball_speed``. The explicit frame rate keeps dataset metadata
    outside the metric implementation.

    Args:
        df: Lazy possession-level tracking rows.
        possession_groups: Columns defining independent trajectories.
        frame_rate: Shared rate or mapping from game ID to rate.

    Returns:
        The lazy input with displacement and speed columns added.

    """
    return df.pipe(add_ball_displacement, possession_groups).pipe(
        add_ball_speed, frame_rate=frame_rate
    )
