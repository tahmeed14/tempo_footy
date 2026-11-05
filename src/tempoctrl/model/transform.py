"""Transform model data and categorical reference levels."""

import pandas as pd
import polars as pl

PITCH_THIRD_ORDER = pl.Enum(
    [
        "Middle",
        "Defensive",
        "Attacking",
    ]
)

SETPIECE_TYPE_ORDER = pl.Enum(
    [
        "Open Play",
        "Dead Ball",
    ]
)

BALL_HEIGHT_ORDER = pl.Enum(["Ground", "Air"])

PRESSURE_ORDER = pl.Enum(
    ["No Pressure", "Pressured (inc. Attempted Pressure)"]
)

DEFENDER_CHALLENGES_ORDER = pl.Enum(
    [
        "None",
        "1",
        "2+",
    ]
)

POSITION_GROUP_ORDER = pl.Enum(
    [
        "Midfielder",
        "Goalkeeper",
        "Centre Back",
        "Full Back",
        "Winger",
        "Forward",
    ]
)

PERIOD_ORDER = pl.Enum(["1", "2", "AET"])

MODEL_ENUM_COLUMNS = [
    "setpiecetype",
    "starting_pitch_third",
    "player_position_group",
    "first_touch_ballheight",
    "first_touch_defender_pressure_type",
    "defender_num_challenges",
    "game_period",
]


# FIXME:
def filter_model_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """Filter observations that are unsuitable for model fitting."""
    return df.filter(
        pl.col("ball_speed_tempo_player").is_not_null(),
        pl.col("starting_pitch_third").is_not_null(),
        pl.col("ball_speed_tempo_player") > 0,
        pl.col("ball_speed_tempo_player") < 25,
        pl.col("setpiecetype") != "Drop Kick",
        pl.col("first_touch_ballheight") != "N/A",
        pl.col("setpiecetype") == "Open Play",
    )


def reclassify_setpiece_type(
    df: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Reclassify setpiece types into broader categories.

    Args:
        df (pl.LazyFrame | pl.DataFrame): Input DataFrame containing the
        'setpiecetype' column.

    Returns:
        pl.LazyFrame | pl.DataFrame: DataFrame with the 'setpiecetype'
        column reclassified.

    """
    return df.with_columns(
        pl.when(pl.col("setpiecetype") == "Open Play")
        .then(pl.lit("Open Play"))
        .otherwise(pl.lit("Dead Ball"))
        .alias("setpiecetype")
    )


def categorize_defender_num_challenges(
    df: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Categorize the number of defender challenges."""
    return df.with_columns(
        pl.when(pl.col("defender_num_challenges") == 0)
        .then(pl.lit("None"))
        .when(pl.col("defender_num_challenges") == 1)
        .then(pl.lit("1"))
        .when(pl.col("defender_num_challenges") >= 2)
        .then(pl.lit("2+"))
        .otherwise(None)
        .cast(DEFENDER_CHALLENGES_ORDER)
        .alias("defender_num_challenges")
    )


def lock_period_order(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast match periods to their modeled order."""
    return df.with_columns(pl.col("game_period").cast(PERIOD_ORDER))


def lock_setpiece_type_levels(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast set-piece types to their modeled order."""
    return df.with_columns(pl.col("setpiecetype").cast(SETPIECE_TYPE_ORDER))


def lock_ball_height_levels(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast ball-height values to their modeled order."""
    return df.with_columns(
        pl.col("first_touch_ballheight").cast(BALL_HEIGHT_ORDER)
    )


# FIXME: need to rename upstream
def lock_position_group_levels(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast position groups to their modeled order."""
    fixme = df.rename({"player_possession_group": "player_position_group"})

    return fixme.with_columns(
        pl.col("player_position_group").cast(POSITION_GROUP_ORDER)
    )


def lock_pitch_third_levels(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast pitch thirds to their modeled order."""
    return df.with_columns(
        pl.col("starting_pitch_third").cast(PITCH_THIRD_ORDER)
    )


def lock_pressure_levels(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast pressure values to their modeled order."""
    return df.with_columns(
        pl.col("first_touch_defender_pressure_type").cast(PRESSURE_ORDER)
    )


def lock_reference_levels(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast all categorical predictors to their modeled orders."""
    return (
        df.pipe(lock_setpiece_type_levels)
        .pipe(lock_pitch_third_levels)
        .pipe(lock_position_group_levels)
        .pipe(lock_ball_height_levels)
        .pipe(lock_pressure_levels)
        .pipe(lock_period_order)
    )


def add_log_sequence(
    df: pl.DataFrame,
    source_col: str = "player_possession_sequence_number",
) -> pl.DataFrame:
    """Add transformed and standardized possession-sequence variables.

    Creates:
        player_possession_sequence_log
        player_possession_sequence_log_z
    """
    log_col = "player_possession_sequence_log"
    z_col = "player_possession_sequence_log_z"

    df = df.with_columns(
        pl.col(source_col).cast(pl.Float64).log1p().alias(log_col)
    )

    df = df.with_columns(
        (
            (pl.col(log_col) - pl.col(log_col).mean()) / pl.col(log_col).std()
        ).alias(z_col)
    )

    return df


def polars_enum_to_ordered_pandas(
    pl_df: pl.DataFrame,
    columns: list[str] = MODEL_ENUM_COLUMNS,
) -> pd.DataFrame:
    """Convert a Polars DataFrame to an ordered pandas DataFrame.

    Preserve the category ordering of specified Polars Enum columns.
    """
    category_orders = {
        col: pl_df.schema[col].categories.to_list() for col in columns
    }

    pd_df = pl_df.to_pandas()

    for col, categories in category_orders.items():
        pd_df[col] = pd.Categorical(
            pd_df[col],
            categories=categories,
            ordered=True,
        )

    return pd_df


def transform_model_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """Apply the complete model-data transformation."""
    return (
        df.pipe(filter_model_data)  # FIXME:
        .pipe(reclassify_setpiece_type)
        .pipe(categorize_defender_num_challenges)
        .pipe(lock_reference_levels)
        .pipe(add_log_sequence)
    )
