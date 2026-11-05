"""Provide data-investigation helpers."""

from __future__ import annotations

from typing import Any

import polars as pl


def _normalize_filter_values(filter_values: Any) -> list[Any]:
    """Return the filter input as a list of values."""
    if isinstance(filter_values, (list, tuple, set, frozenset)):
        return list(filter_values)
    return [filter_values]


def get_rows_with_context(
    df: pl.DataFrame,
    column_name: str,
    filter_values: Any,
    n: int = 5,
) -> pl.DataFrame:
    """Return matched rows plus `n` rows above and below each match.

    Parameters
    ----------
    df
        The Polars dataframe to inspect.
    column_name
        Column to filter on.
    filter_values
        A single value or a list/tuple/set of values to match against.
    n
        Number of rows to include above and below each matched row.

    Returns
    -------
    pl.DataFrame
        A dataframe containing the matched rows and their surrounding context,
        in the same order as the original dataframe.

    """
    if n < 0:
        raise ValueError("n must be greater than or equal to 0")
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in dataframe")

    values = _normalize_filter_values(filter_values)
    if not values:
        return df.head(0)

    indexed_df = df.with_row_index("__row_idx")

    condition = None
    for value in values:
        current = (
            pl.col(column_name).is_null()
            if value is None
            else pl.col(column_name) == value
        )
        condition = current if condition is None else (condition | current)

    matched_indices = (
        indexed_df.filter(condition).get_column("__row_idx").to_list()
    )

    if not matched_indices:
        return df.head(0)

    context_indices: set[int] = set()
    for row_idx in matched_indices:
        start = max(0, row_idx - n)
        stop = min(df.height, row_idx + n + 1)
        context_indices.update(range(start, stop))

    return (
        indexed_df.filter(pl.col("__row_idx").is_in(sorted(context_indices)))
        .sort("__row_idx")
        .drop("__row_idx")
    )


def column_missingness(df: pl.DataFrame) -> pl.DataFrame:
    """Return the null count and percentage for every column.

    Percentages are reported on a 0-to-100 scale. An empty DataFrame has
    an undefined percentage, represented by null for each column.
    """
    null_counts = df.null_count().row(0)
    row_count = df.height
    missing_percent = [
        count / row_count * 100 if row_count else None for count in null_counts
    ]

    return pl.DataFrame(
        {
            "column_name": pl.Series(df.columns, dtype=pl.String),
            "null_count": pl.Series(null_counts, dtype=pl.UInt32),
            "missing_percent": pl.Series(
                missing_percent,
                dtype=pl.Float64,
            ),
        }
    )


def summarize_variables(
    df: pl.DataFrame,
    categorical_columns: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Summarize continuous and categorical DataFrame columns.

    Numeric columns are continuous unless included in
    ``categorical_columns``. All other columns are categorical. Continuous
    statistics ignore nulls and use sample standard deviation and variance.
    Category counts and percentages exclude nulls.

    Args:
        df: DataFrame containing the variables to summarize.
        categorical_columns: Numeric or other columns that should be
            summarized as categories instead of continuous variables.

    Returns:
        A dictionary containing ``continuous`` and ``categorical``
        DataFrames. Percentages are on a 0-to-100 scale.

    """
    categorical_overrides = set(categorical_columns or [])
    missing_columns = categorical_overrides.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    continuous_columns = [
        name
        for name, dtype in df.schema.items()
        if dtype.is_numeric() and name not in categorical_overrides
    ]
    category_columns = [
        name for name in df.columns if name not in continuous_columns
    ]

    continuous_rows: list[dict[str, Any]] = []
    for column_name in continuous_columns:
        values = df.get_column(column_name).cast(pl.Float64)
        mean = values.mean()
        standard_deviation = values.std()
        if mean is None or standard_deviation is None:
            outside_standard_deviation = 0
        else:
            lower_bound = mean - standard_deviation
            upper_bound = mean + standard_deviation
            outside_standard_deviation = values.filter(
                (values < lower_bound) | (values > upper_bound)
            ).len()

        continuous_rows.append(
            {
                "column_name": column_name,
                "min": values.min(),
                "median": values.median(),
                "mean": mean,
                "max": values.max(),
                "standard_deviation": standard_deviation,
                "variance": values.var(),
                "outside_mean_plus_minus_std_count": (
                    outside_standard_deviation
                ),
            }
        )

    continuous_summary = pl.DataFrame(
        continuous_rows,
        schema={
            "column_name": pl.String,
            "min": pl.Float64,
            "median": pl.Float64,
            "mean": pl.Float64,
            "max": pl.Float64,
            "standard_deviation": pl.Float64,
            "variance": pl.Float64,
            "outside_mean_plus_minus_std_count": pl.UInt32,
        },
    )

    categorical_rows: list[dict[str, Any]] = []
    for column_name in category_columns:
        values = df.get_column(column_name).drop_nulls().cast(pl.String)
        non_null_count = values.len()
        unique_count = values.n_unique()
        value_counts = values.value_counts(sort=True, name="count")

        if value_counts.is_empty():
            categorical_rows.append(
                {
                    "column_name": column_name,
                    "category": None,
                    "count": 0,
                    "percentage": None,
                    "unique_category_count": 0,
                }
            )
            continue

        for category, count in value_counts.iter_rows():
            categorical_rows.append(
                {
                    "column_name": column_name,
                    "category": category,
                    "count": count,
                    "percentage": count / non_null_count * 100,
                    "unique_category_count": unique_count,
                }
            )

    categorical_summary = pl.DataFrame(
        categorical_rows,
        schema={
            "column_name": pl.String,
            "category": pl.String,
            "count": pl.UInt32,
            "percentage": pl.Float64,
            "unique_category_count": pl.UInt32,
        },
    )

    return {
        "continuous": continuous_summary,
        "categorical": categorical_summary,
    }


def _flip_if_attacking_left(field: str) -> pl.Expr:
    """Flip a ball-coordinate field when attacking left."""
    value = pl.col("balls_smooth").struct.field(field)

    return (
        pl.when(pl.col("attacking_team_direction") == "L")
        .then(-value)
        .otherwise(value)
        .alias(field)
    )


def normalize_ball_coordinates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize ball coordinates to the attacking direction."""
    return lf.with_columns(
        pl.struct(
            [
                pl.col("balls_smooth").struct.field("type").alias("type"),
                _flip_if_attacking_left("x"),
                _flip_if_attacking_left("y"),
                pl.col("balls_smooth").struct.field("z").alias("z"),
            ]
        ).alias("balls_smooth")
    )
