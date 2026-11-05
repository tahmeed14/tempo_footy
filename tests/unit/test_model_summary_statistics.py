"""Test model summary-statistic helpers."""

from pathlib import Path

import polars as pl
import pytest

from tempoctrl.reproduce_tables import (
    MODEL_SUMMARY_COLUMNS,
    generate_model_summary_statistics,
)


def _model_frame() -> pl.DataFrame:
    """Build explicit model rows for summary-statistics assertions."""
    frame = pl.DataFrame(
        {
            "ball_speed_tempo_player": [2.0, 4.0, 6.0, 8.0],
            "player_id": [10, 10, 20, 20],
            "opponent_id": [100, 100, 100, 200],
            "game_state_goal_diff": [-1, 0, 1, None],
            "player_possession_sequence_log_z": [
                -1.0,
                float("nan"),
                0.5,
                1.5,
            ],
            "first_touch_ballheight": [
                "Ground",
                "Ground",
                "Air",
                None,
            ],
            "first_touch_defender_pressure_type": [
                "No Pressure",
                "Pressured",
                "No Pressure",
                "No Pressure",
            ],
            "defender_num_challenges": ["None", "1", "2+", "None"],
            "starting_pitch_third": [
                "Middle",
                "Defensive",
                "Attacking",
                "Middle",
            ],
            "player_position_group": [
                "Midfielder",
                "Forward",
                "Midfielder",
                "Forward",
            ],
            "game_period": ["1", "1", "2", "AET"],
            "game_id": [1, 1, 2, 2],
            "team_id": [1000, 1000, 2000, 2000],
        }
    )
    orders = {
        "first_touch_ballheight": ["Ground", "Air"],
        "first_touch_defender_pressure_type": [
            "No Pressure",
            "Pressured",
        ],
        "defender_num_challenges": ["None", "1", "2+"],
        "starting_pitch_third": ["Middle", "Defensive", "Attacking"],
        "player_position_group": ["Midfielder", "Forward"],
        "game_period": ["1", "2", "AET"],
    }
    return frame.with_columns(
        pl.col(variable).cast(pl.Enum(levels))
        for variable, levels in orders.items()
    )


def _row(
    table: pl.DataFrame,
    variable: str,
    statistic: str,
    level: str | None = None,
    section: str | None = None,
) -> dict[str, object]:
    """Select a named summary row for explicit value assertions."""
    result = table.filter(
        (pl.col("variable") == variable) & (pl.col("statistic") == statistic)
    )
    if level is not None:
        result = result.filter(pl.col("level") == level)
    if section is not None:
        result = result.filter(pl.col("section") == section)
    assert result.height == 1
    return result.row(0, named=True)


def test_generate_model_summary_statistics_has_required_values(
    tmp_path: Path,
) -> None:
    """Recover known counts, moments, and ranges from explicit model
    rows.
    """
    table = generate_model_summary_statistics(
        _model_frame(),
        output_path=tmp_path,
    )

    assert table.columns == list(MODEL_SUMMARY_COLUMNS)
    assert (
        _row(
            table,
            "__analytic_sample__",
            "n_observations",
        )["value"]
        == 4.0
    )
    assert (
        _row(
            table,
            "player_id",
            "n_unique",
            section="sample",
        )["value"]
        == 2.0
    )
    assert (
        _row(
            table,
            "opponent_id",
            "n_unique",
            section="sample",
        )["value"]
        == 2.0
    )
    assert _row(table, "game_id", "n_unique")["value"] == 2.0
    assert _row(table, "team_id", "n_unique")["value"] == 2.0

    outcome_mean = _row(table, "ball_speed_tempo_player", "mean")
    assert outcome_mean["value"] == 5.0
    assert outcome_mean["used_in_mu"] is True
    assert outcome_mean["used_in_alpha"] is False

    sequence_count = _row(
        table,
        "player_possession_sequence_log_z",
        "count",
    )
    assert sequence_count["value"] == 3.0
    assert sequence_count["missing_count"] == 1
    assert sequence_count["missing_percentage"] == 25.0

    ground = _row(
        table,
        "first_touch_ballheight",
        "frequency",
        "Ground",
    )
    assert ground["count"] == 2
    assert ground["percentage"] == pytest.approx(200 / 3)
    assert ground["missing_count"] == 1
    assert ground["is_reference_level"] is True

    ball_height_rows = table.filter(
        pl.col("variable") == "first_touch_ballheight",
        pl.col("statistic") == "frequency",
    )
    assert ball_height_rows["level"].to_list() == ["Ground", "Air"]
    assert ball_height_rows["is_reference_level"].to_list() == [True, False]
    assert ball_height_rows["percentage"].sum() == pytest.approx(100.0)

    assert _row(table, "player_id", "min_group_n")["value"] == 2.0
    assert _row(table, "player_id", "mean_group_n")["value"] == 2.0
    assert _row(table, "opponent_id", "max_group_n")["value"] == 3.0
    assert (tmp_path / "model_summary_statistics.csv").is_file()


def test_generate_model_summary_statistics_can_skip_writing() -> None:
    """Return model summaries without requiring an output artifact."""
    table = generate_model_summary_statistics(
        _model_frame(),
        output_path=None,
    )

    assert table.height > 0


def test_summary_statistics_rejects_nonpositive_response() -> None:
    """Reject zero or negative responses before logarithmic summaries.
    """
    frame = _model_frame().with_columns(
        pl.Series("ball_speed_tempo_player", [0.0, 4.0, 6.0, 8.0])
    )

    with pytest.raises(ValueError, match="outside the Gamma response support"):
        generate_model_summary_statistics(frame)


def test_generate_model_summary_statistics_requires_model_variables() -> None:
    """Reject model rows missing variables required for the summary."""
    frame = _model_frame().drop("game_period")

    with pytest.raises(ValueError, match="game_period"):
        generate_model_summary_statistics(frame)
