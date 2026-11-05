"""Generate descriptive statistics for the fitted analytic sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

# Change this path to summarize a different final model dataset when this
# module is run as a script.
MODEL_DATA_PATH = Path("data/analysis/model_data_vFINAL.parquet")
MODEL_SUMMARY_OUTPUT_DIRECTORY = Path("paper/tables")
DEFAULT_MODEL_SUMMARY_FILENAME = "model_summary_statistics.csv"

OUTCOME_VARIABLE = "ball_speed_tempo_player"
CONTINUOUS_PREDICTORS = (
    "game_state_goal_diff",
    "player_possession_sequence_log_z",
)
CATEGORICAL_PREDICTORS = (
    "first_touch_ballheight",
    "first_touch_defender_pressure_type",
    "defender_num_challenges",
    "starting_pitch_third",
    "player_position_group",
    "game_period",
)
GROUPING_VARIABLES = ("player_id", "opponent_id")
REQUIRED_MODEL_VARIABLES = (
    OUTCOME_VARIABLE,
    *GROUPING_VARIABLES,
    *CONTINUOUS_PREDICTORS,
    *CATEGORICAL_PREDICTORS,
)

VARIABLE_LABELS = {
    "__analytic_sample__": "Analytic sample",
    "ball_speed_tempo_player": "Ball-speed tempo",
    "game_state_goal_diff": "Goal difference",
    "first_touch_ballheight": "Ball height at first touch",
    "first_touch_defender_pressure_type": ("Defender pressure at first touch"),
    "defender_num_challenges": "Number of defender challenges",
    "starting_pitch_third": "Starting pitch third",
    "player_position_group": "Player position group",
    "player_possession_sequence_log_z": (
        "Player possession sequence (log-standardized)"
    ),
    "game_period": "Match period",
    "player_id": "Player",
    "opponent_id": "Opposition team",
    "game_id": "Match",
    "team_id": "Team",
}

USED_IN_MU = frozenset(REQUIRED_MODEL_VARIABLES)
USED_IN_ALPHA = frozenset(
    {
        "player_id",
        "first_touch_defender_pressure_type",
        "defender_num_challenges",
        "starting_pitch_third",
        "player_position_group",
    }
)

MODEL_SUMMARY_COLUMNS = (
    "section",
    "variable",
    "variable_label",
    "variable_type",
    "level",
    "is_reference_level",
    "statistic",
    "value",
    "count",
    "percentage",
    "missing_count",
    "missing_percentage",
    "used_in_mu",
    "used_in_alpha",
)

MODEL_SUMMARY_SCHEMA = {
    "section": pl.String,
    "variable": pl.String,
    "variable_label": pl.String,
    "variable_type": pl.String,
    "level": pl.String,
    "is_reference_level": pl.Boolean,
    "statistic": pl.String,
    "value": pl.Float64,
    "count": pl.Int64,
    "percentage": pl.Float64,
    "missing_count": pl.Int64,
    "missing_percentage": pl.Float64,
    "used_in_mu": pl.Boolean,
    "used_in_alpha": pl.Boolean,
}

_CONTINUOUS_STATISTICS = (
    "count",
    "mean",
    "sd",
    "median",
    "q25",
    "q75",
    "p05",
    "p95",
    "min",
    "max",
)
_GROUP_SIZE_STATISTICS = (
    "min_group_n",
    "q25_group_n",
    "median_group_n",
    "mean_group_n",
    "q75_group_n",
    "max_group_n",
)


def generate_model_summary_statistics(
    model_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    """Summarize the exact dataframe used to fit the Gamma model.

    Parameters
    ----------
    model_df
        Final analytic dataframe supplied to the model. Polars dataframes,
        lazy frames, and pandas dataframes are accepted.
    output_path
        Optional CSV destination. A directory receives the filename
        ``model_summary_statistics.csv``. Pass ``None`` to skip writing.

    Returns
    -------
    polars.DataFrame
        Long-form sample, variable, frequency, and group-size summaries.

    """
    frame, category_orders = _normalize_model_dataframe(model_df)
    _validate_model_dataframe(frame)

    rows: list[dict[str, Any]] = []
    _append_sample_rows(rows, frame)
    _append_continuous_rows(
        rows,
        frame,
        OUTCOME_VARIABLE,
        section="outcome",
    )
    for variable in CONTINUOUS_PREDICTORS:
        _append_continuous_rows(
            rows,
            frame,
            variable,
            section="continuous_predictor",
        )
    for variable in CATEGORICAL_PREDICTORS:
        _append_categorical_rows(
            rows,
            frame,
            variable,
            category_orders.get(variable),
        )
    for variable in GROUPING_VARIABLES:
        _append_grouping_rows(rows, frame, variable)

    table = pl.DataFrame(rows, schema=MODEL_SUMMARY_SCHEMA)
    if table.columns != list(MODEL_SUMMARY_COLUMNS):
        raise RuntimeError("The summary table does not match its schema.")

    if output_path is not None:
        destination = _resolve_output_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(destination)

    return table


def _normalize_model_dataframe(
    model_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> tuple[pl.DataFrame, dict[str, list[Any]]]:
    """Return a Polars frame and any explicitly declared category orders."""
    category_orders: dict[str, list[Any]] = {}

    if isinstance(model_df, pl.LazyFrame):
        model_df = model_df.collect()

    if isinstance(model_df, pd.DataFrame):
        for variable in CATEGORICAL_PREDICTORS:
            if variable not in model_df.columns:
                continue
            dtype = model_df[variable].dtype
            if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
                category_orders[variable] = dtype.categories.to_list()
        frame = pl.from_pandas(model_df)
    elif isinstance(model_df, pl.DataFrame):
        frame = model_df
        for variable in CATEGORICAL_PREDICTORS:
            if variable not in frame.columns:
                continue
            dtype = frame.schema[variable]
            if isinstance(dtype, pl.Enum):
                category_orders[variable] = dtype.categories.to_list()
    else:
        raise TypeError(
            "model_df must be a Polars DataFrame, Polars LazyFrame, or "
            "pandas DataFrame."
        )

    float_columns = [
        name
        for name, dtype in frame.schema.items()
        if dtype in {pl.Float32, pl.Float64}
    ]
    if float_columns:
        frame = frame.with_columns(pl.col(float_columns).fill_nan(None))
    return frame, category_orders


def _validate_model_dataframe(frame: pl.DataFrame) -> None:
    """Validate required columns and the Gamma response support."""
    missing_variables = sorted(
        set(REQUIRED_MODEL_VARIABLES).difference(frame.columns)
    )
    if missing_variables:
        raise ValueError(
            "The model dataframe is missing required variables: "
            f"{missing_variables}."
        )
    if frame.is_empty():
        raise ValueError("The model dataframe must contain observations.")

    response = _numeric_series(frame, OUTCOME_VARIABLE).drop_nulls()
    invalid_count = response.filter(response <= 0).len()
    if invalid_count:
        raise ValueError(
            f"{OUTCOME_VARIABLE!r} contains {invalid_count} zero or "
            "negative value(s), which are outside the Gamma response "
            "support."
        )


def _append_sample_rows(
    rows: list[dict[str, Any]],
    frame: pl.DataFrame,
) -> None:
    """Append sample-size and reliable identifier summaries."""
    rows.append(
        _make_row(
            frame,
            section="sample",
            variable="__analytic_sample__",
            variable_type="sample",
            statistic="n_observations",
            value=float(frame.height),
        )
    )
    identifier_variables = [*GROUPING_VARIABLES]
    identifier_variables.extend(
        variable
        for variable in ("game_id", "team_id")
        if variable in frame.columns
    )
    for variable in identifier_variables:
        rows.append(
            _make_row(
                frame,
                section="sample",
                variable=variable,
                variable_type="sample",
                statistic="n_unique",
                value=float(frame[variable].drop_nulls().n_unique()),
            )
        )


def _append_continuous_rows(
    rows: list[dict[str, Any]],
    frame: pl.DataFrame,
    variable: str,
    *,
    section: str,
) -> None:
    """Append distribution summaries for one numeric variable."""
    values = _numeric_series(frame, variable).drop_nulls()
    statistics = {
        "count": float(values.len()),
        "mean": values.mean(),
        "sd": values.std(),
        "median": values.median(),
        "q25": values.quantile(0.25, interpolation="linear"),
        "q75": values.quantile(0.75, interpolation="linear"),
        "p05": values.quantile(0.05, interpolation="linear"),
        "p95": values.quantile(0.95, interpolation="linear"),
        "min": values.min(),
        "max": values.max(),
    }
    for statistic in _CONTINUOUS_STATISTICS:
        rows.append(
            _make_row(
                frame,
                section=section,
                variable=variable,
                variable_type="continuous",
                statistic=statistic,
                value=_optional_float(statistics[statistic]),
            )
        )


def _append_categorical_rows(
    rows: list[dict[str, Any]],
    frame: pl.DataFrame,
    variable: str,
    category_order: list[Any] | None,
) -> None:
    """Append ordered category frequencies for one variable."""
    frequencies = (
        frame.select(variable)
        .drop_nulls()
        .group_by(variable)
        .len(name="count")
    )
    counts = dict(frequencies.iter_rows())
    non_missing_count = sum(counts.values())

    if category_order is not None:
        levels = [level for level in category_order if level in counts]
        reference_level = category_order[0] if category_order else None
    else:
        levels = [
            level
            for level, _ in sorted(
                counts.items(),
                key=lambda item: (-item[1], str(item[0])),
            )
        ]
        reference_level = None

    for level in levels:
        count = counts[level]
        percentage = 100 * count / non_missing_count
        is_reference = (
            level == reference_level if reference_level is not None else None
        )
        rows.append(
            _make_row(
                frame,
                section="categorical_predictor",
                variable=variable,
                variable_type="categorical",
                statistic="frequency",
                level=str(level),
                is_reference_level=is_reference,
                count=count,
                percentage=percentage,
            )
        )


def _append_grouping_rows(
    rows: list[dict[str, Any]],
    frame: pl.DataFrame,
    variable: str,
) -> None:
    """Append unique-level and group-size summaries."""
    group_sizes = (
        frame.select(variable)
        .drop_nulls()
        .group_by(variable)
        .len(name="group_n")["group_n"]
        .cast(pl.Float64)
    )
    rows.append(
        _make_row(
            frame,
            section="grouping_variable",
            variable=variable,
            variable_type="grouping",
            statistic="n_unique",
            value=float(group_sizes.len()),
        )
    )
    statistics = {
        "min_group_n": group_sizes.min(),
        "q25_group_n": group_sizes.quantile(
            0.25,
            interpolation="linear",
        ),
        "median_group_n": group_sizes.median(),
        "mean_group_n": group_sizes.mean(),
        "q75_group_n": group_sizes.quantile(
            0.75,
            interpolation="linear",
        ),
        "max_group_n": group_sizes.max(),
    }
    for statistic in _GROUP_SIZE_STATISTICS:
        rows.append(
            _make_row(
                frame,
                section="grouping_variable",
                variable=variable,
                variable_type="grouping",
                statistic=statistic,
                value=_optional_float(statistics[statistic]),
            )
        )


def _make_row(
    frame: pl.DataFrame,
    *,
    section: str,
    variable: str,
    variable_type: str,
    statistic: str,
    level: str | None = None,
    is_reference_level: bool | None = None,
    value: float | None = None,
    count: int | None = None,
    percentage: float | None = None,
) -> dict[str, Any]:
    """Build one schema-complete summary row."""
    if variable in frame.columns:
        missing_count = frame[variable].null_count()
        missing_percentage = 100 * missing_count / frame.height
    else:
        missing_count = 0
        missing_percentage = 0.0

    return {
        "section": section,
        "variable": variable,
        "variable_label": VARIABLE_LABELS[variable],
        "variable_type": variable_type,
        "level": level,
        "is_reference_level": is_reference_level,
        "statistic": statistic,
        "value": value,
        "count": count,
        "percentage": percentage,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "used_in_mu": variable in USED_IN_MU,
        "used_in_alpha": variable in USED_IN_ALPHA,
    }


def _numeric_series(frame: pl.DataFrame, variable: str) -> pl.Series:
    """Return a numeric variable as Float64 or raise a clear error."""
    try:
        return frame[variable].cast(pl.Float64, strict=True)
    except (
        TypeError,
        ValueError,
        pl.exceptions.InvalidOperationError,
    ) as error:
        raise TypeError(f"{variable!r} must be numeric.") from error


def _optional_float(value: Any) -> float | None:
    """Convert a scalar statistic to a native float when present."""
    return None if value is None else float(value)


def _resolve_output_path(output_path: str | Path) -> Path:
    """Resolve a file or directory destination for the summary CSV."""
    destination = Path(output_path)
    if destination.is_dir() or not destination.suffix:
        return destination / DEFAULT_MODEL_SUMMARY_FILENAME
    return destination


def main() -> None:
    """Generate the model summary from the configurable global path."""
    model_df = pl.read_parquet(MODEL_DATA_PATH)
    generate_model_summary_statistics(
        model_df,
        output_path=MODEL_SUMMARY_OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()
