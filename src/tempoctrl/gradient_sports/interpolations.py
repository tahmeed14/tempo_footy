"""Interpolate values in Gradient Sports tracking data."""

import polars as pl

BALL_COLUMN = "balls_smooth"
FRAME_COLUMN = "framenum"
GAME_COLUMN = "game_id"

_CAN_IMPUTE_COLUMN = "__ball_can_impute"
_IS_FIRST_FRAME_COLUMN = "__ball_is_first_frame"
_NEXT_FRAME_COLUMN = "__ball_next_frame"
_NEXT_X_COLUMN = "__ball_next_x"
_NEXT_Y_COLUMN = "__ball_next_y"
_PREVIOUS_FRAME_COLUMN = "__ball_previous_frame"
_PREVIOUS_X_COLUMN = "__ball_previous_x"
_PREVIOUS_Y_COLUMN = "__ball_previous_y"
_ROW_ORDER_COLUMN = "__ball_input_order"
_SOURCE_X_COLUMN = "__ball_source_x"
_SOURCE_Y_COLUMN = "__ball_source_y"
_X_CANDIDATE_COLUMN = "__ball_x_candidate"
_Y_CANDIDATE_COLUMN = "__ball_y_candidate"

_TEMPORARY_COLUMNS = (
    _ROW_ORDER_COLUMN,
    _IS_FIRST_FRAME_COLUMN,
    _SOURCE_X_COLUMN,
    _SOURCE_Y_COLUMN,
    _PREVIOUS_FRAME_COLUMN,
    _NEXT_FRAME_COLUMN,
    _PREVIOUS_X_COLUMN,
    _NEXT_X_COLUMN,
    _PREVIOUS_Y_COLUMN,
    _NEXT_Y_COLUMN,
    _CAN_IMPUTE_COLUMN,
    _X_CANDIDATE_COLUMN,
    _Y_CANDIDATE_COLUMN,
)


def _validate_interpolation_input(
    df: pl.LazyFrame,
    *,
    possession_col: str,
    max_gap: int,
    overwrite: bool = False,
) -> pl.LazyFrame:
    """Validate shared interpolation columns and configuration."""
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1.")

    schema = df.collect_schema()
    required_columns = (
        GAME_COLUMN,
        FRAME_COLUMN,
        possession_col,
        BALL_COLUMN,
    )
    missing_columns = [
        column for column in required_columns if column not in schema
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}.")

    if not schema[FRAME_COLUMN].is_numeric():
        raise TypeError(f"{FRAME_COLUMN} must be numeric.")

    ball_dtype = schema[BALL_COLUMN]
    if not isinstance(ball_dtype, pl.Struct):
        raise TypeError(f"{BALL_COLUMN} must be a struct column.")

    ball_fields = {field.name: field.dtype for field in ball_dtype.fields}
    missing_fields = [
        field for field in ("x", "y") if field not in ball_fields
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"{BALL_COLUMN} is missing fields: {missing}.")

    nonnumeric_fields = [
        field for field in ("x", "y") if not ball_fields[field].is_numeric()
    ]
    if nonnumeric_fields:
        invalid = ", ".join(nonnumeric_fields)
        raise TypeError(f"{BALL_COLUMN} fields must be numeric: {invalid}.")

    imputation_dtype = ball_fields.get("is_imputed")
    if imputation_dtype is not None and imputation_dtype != pl.Boolean:
        raise TypeError(f"{BALL_COLUMN}.is_imputed must be Boolean.")
    if imputation_dtype is not None and not overwrite:
        raise ValueError(
            f"{BALL_COLUMN} has already been interpolated. "
            "Pass overwrite=True to recalculate imputed coordinates."
        )

    return df


def _add_source_coordinates(
    df: pl.LazyFrame,
    *,
    overwrite: bool,
) -> pl.LazyFrame:
    """Expose original coordinates, clearing prior imputations on rerun."""
    ball = pl.col(BALL_COLUMN)
    ball_dtype = df.collect_schema()[BALL_COLUMN]
    field_names = {field.name for field in ball_dtype.fields}
    was_imputed = pl.lit(False)

    if overwrite and "is_imputed" in field_names:
        was_imputed = ball.struct.field("is_imputed").fill_null(False)

    return df.with_columns(
        pl.when(was_imputed)
        .then(None)
        .otherwise(ball.struct.field("x"))
        .alias(_SOURCE_X_COLUMN),
        pl.when(was_imputed)
        .then(None)
        .otherwise(ball.struct.field("y"))
        .alias(_SOURCE_Y_COLUMN),
    )


def _add_ball_gap_context(
    df: pl.LazyFrame,
    *,
    partition_columns: tuple[str, ...],
) -> pl.LazyFrame:
    """Add numerical frame and coordinate bounds for each ball gap."""
    frame_partition = (*partition_columns, FRAME_COLUMN)
    order_columns = (FRAME_COLUMN, _ROW_ORDER_COLUMN)
    source_x = pl.col(_SOURCE_X_COLUMN)
    source_y = pl.col(_SOURCE_Y_COLUMN)

    with_row_order = df.with_columns(
        pl.int_range(pl.len()).alias(_ROW_ORDER_COLUMN)
    )
    with_first_frame = with_row_order.with_columns(
        (
            pl.col(_ROW_ORDER_COLUMN)
            == pl.col(_ROW_ORDER_COLUMN).min().over(frame_partition)
        ).alias(_IS_FIRST_FRAME_COLUMN)
    )
    ball_observed = (
        pl.col(_IS_FIRST_FRAME_COLUMN)
        & source_x.is_not_null()
        & source_y.is_not_null()
    )

    return with_first_frame.with_columns(
        pl.when(ball_observed)
        .then(pl.col(FRAME_COLUMN))
        .otherwise(None)
        .forward_fill()
        .over(partition_columns, order_by=order_columns)
        .alias(_PREVIOUS_FRAME_COLUMN),
        pl.when(ball_observed)
        .then(pl.col(FRAME_COLUMN))
        .otherwise(None)
        .backward_fill()
        .over(partition_columns, order_by=order_columns)
        .alias(_NEXT_FRAME_COLUMN),
        pl.when(ball_observed)
        .then(source_x)
        .otherwise(None)
        .forward_fill()
        .over(partition_columns, order_by=order_columns)
        .alias(_PREVIOUS_X_COLUMN),
        pl.when(ball_observed)
        .then(source_x)
        .otherwise(None)
        .backward_fill()
        .over(partition_columns, order_by=order_columns)
        .alias(_NEXT_X_COLUMN),
        pl.when(ball_observed)
        .then(source_y)
        .otherwise(None)
        .forward_fill()
        .over(partition_columns, order_by=order_columns)
        .alias(_PREVIOUS_Y_COLUMN),
        pl.when(ball_observed)
        .then(source_y)
        .otherwise(None)
        .backward_fill()
        .over(partition_columns, order_by=order_columns)
        .alias(_NEXT_Y_COLUMN),
    )


def _add_interpolation_eligibility(
    df: pl.LazyFrame,
    *,
    possession_col: str,
    max_gap: int,
) -> pl.LazyFrame:
    """Mark first rows in short, numerically bounded frame gaps."""
    previous_frame = pl.col(_PREVIOUS_FRAME_COLUMN)
    next_frame = pl.col(_NEXT_FRAME_COLUMN)
    frame_gap = next_frame - previous_frame - 1
    coordinates_missing = (
        pl.col(_SOURCE_X_COLUMN).is_null() & pl.col(_SOURCE_Y_COLUMN).is_null()
    )

    return df.with_columns(
        (
            pl.col(possession_col).is_not_null()
            & pl.col(FRAME_COLUMN).is_not_null()
            & pl.col(_IS_FIRST_FRAME_COLUMN)
            & coordinates_missing
            & previous_frame.is_not_null()
            & next_frame.is_not_null()
            & (frame_gap <= max_gap)
        ).alias(_CAN_IMPUTE_COLUMN)
    )


def _add_linear_interpolation_candidates(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Calculate linear candidates using numerical frame distances."""
    frame = pl.col(FRAME_COLUMN)
    previous_frame = pl.col(_PREVIOUS_FRAME_COLUMN)
    next_frame = pl.col(_NEXT_FRAME_COLUMN)
    frame_fraction = (frame - previous_frame) / (next_frame - previous_frame)
    x_candidate = (
        pl.col(_PREVIOUS_X_COLUMN)
        + (pl.col(_NEXT_X_COLUMN) - pl.col(_PREVIOUS_X_COLUMN))
        * frame_fraction
    )
    y_candidate = (
        pl.col(_PREVIOUS_Y_COLUMN)
        + (pl.col(_NEXT_Y_COLUMN) - pl.col(_PREVIOUS_Y_COLUMN))
        * frame_fraction
    )

    return df.with_columns(
        pl.when(pl.col(_SOURCE_X_COLUMN).is_not_null())
        .then(pl.col(_SOURCE_X_COLUMN))
        .otherwise(x_candidate)
        .alias(_X_CANDIDATE_COLUMN),
        pl.when(pl.col(_SOURCE_Y_COLUMN).is_not_null())
        .then(pl.col(_SOURCE_Y_COLUMN))
        .otherwise(y_candidate)
        .alias(_Y_CANDIDATE_COLUMN),
    )


def _apply_interpolation_candidates(df: pl.LazyFrame) -> pl.LazyFrame:
    """Apply eligible candidates and add the ball imputation marker."""
    ball = pl.col(BALL_COLUMN)
    can_impute = pl.col(_CAN_IMPUTE_COLUMN)
    ball_dtype = df.collect_schema()[BALL_COLUMN]
    field_expressions: list[pl.Expr] = []

    for field in ball_dtype.fields:
        if field.name == "x":
            expression = (
                pl.when(can_impute)
                .then(pl.col(_X_CANDIDATE_COLUMN))
                .otherwise(pl.col(_SOURCE_X_COLUMN))
                .alias("x")
            )
        elif field.name == "y":
            expression = (
                pl.when(can_impute)
                .then(pl.col(_Y_CANDIDATE_COLUMN))
                .otherwise(pl.col(_SOURCE_Y_COLUMN))
                .alias("y")
            )
        elif field.name == "is_imputed":
            expression = can_impute.alias("is_imputed")
        else:
            expression = ball.struct.field(field.name).alias(field.name)
        field_expressions.append(expression)

    if "is_imputed" not in {field.name for field in ball_dtype.fields}:
        field_expressions.append(can_impute.alias("is_imputed"))

    return df.with_columns(pl.struct(*field_expressions).alias(BALL_COLUMN))


def _drop_interpolation_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Remove implementation columns shared by interpolation methods."""
    return df.drop(*_TEMPORARY_COLUMNS)


def interpolate_ball_coordinates(
    df: pl.LazyFrame,
    *,
    possession_col: str = "dev_match_team_player_possession_id",
    max_gap: int = 5,
    overwrite: bool = False,
) -> pl.LazyFrame:
    """Linearly interpolate short, bounded ball-coordinate gaps.

    Interpolation uses numerical frame distance within each game and
    possession. Only the first row at a duplicate frame participates.
    Rows where both x and y are null require observed coordinates on
    both sides whose frame difference contains at most ``max_gap``
    frames. Existing results require ``overwrite=True`` to recalculate.
    The ball struct receives an ``is_imputed`` Boolean field.
    """
    partition_columns = (GAME_COLUMN, possession_col)

    return (
        df.pipe(
            _validate_interpolation_input,
            possession_col=possession_col,
            max_gap=max_gap,
            overwrite=overwrite,
        )
        .pipe(_add_source_coordinates, overwrite=overwrite)
        .pipe(
            _add_ball_gap_context,
            partition_columns=partition_columns,
        )
        .pipe(
            _add_interpolation_eligibility,
            possession_col=possession_col,
            max_gap=max_gap,
        )
        .pipe(_add_linear_interpolation_candidates)
        .pipe(_apply_interpolation_candidates)
        .pipe(_drop_interpolation_columns)
    )
