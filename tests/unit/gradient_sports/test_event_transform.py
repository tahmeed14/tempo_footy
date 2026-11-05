"""Test Gradient Sports event transformations."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from tempoctrl.gradient_sports import event_transform as event_transform_module
from tempoctrl.gradient_sports.event_transform import (
    add_possession_identifiers,
    create_team_possession_flag,
    drop_foul_events,
    transform_events,
)


def test_event_after_out_starts_new_team_possession() -> None:
    """Restart possession after OUT while retaining all four event rows.
    """
    events = pl.DataFrame(
        {
            "gameid": [101, 101, 101, 101],
            "ge_teamid": [1, 1, 1, 1],
            "ge_setpiecetype": [None, None, None, None],
            "ge_gameeventtype": ["FIRSTKICKOFF", "OUT", "PASS", "PASS"],
        },
        schema_overrides={"ge_setpiecetype": pl.String},
    )

    result = create_team_possession_flag(events)

    assert_series_equal(
        result["ge_gameeventtype"],
        pl.Series(
            "ge_gameeventtype",
            ["FIRSTKICKOFF", "OUT", "PASS", "PASS"],
        ),
    )
    assert_series_equal(
        result["team_possession_start"],
        pl.Series("team_possession_start", [True, False, True, False]),
    )


def _possession_events(
    *,
    event_number: list[int],
    event_type: list[str],
    team_id: list[int | None],
    team_name: list[str | None],
    player_id: list[int | None],
) -> pl.DataFrame:
    """Build typed possession events from explicit identifiers and
    flags.
    """
    row_count = len(event_number)
    return pl.DataFrame(
        {
            "event_number": event_number,
            "gameid": [10517] * row_count,
            "ge_gameeventtype": event_type,
            "ge_setpiecetype": [None] * row_count,
            "ge_teamid": team_id,
            "ge_teamname": team_name,
            "ge_playerid": player_id,
        },
        schema_overrides={"ge_setpiecetype": pl.String},
    )


def test_pause_rows_do_not_create_gaps_in_possession_ids() -> None:
    """Keep pause IDs null and resume known teams without counter gaps.
    """
    events = _possession_events(
        event_number=[2291, 2292, 2293, 2294, 2295, 2296],
        event_type=["FIRSTKICKOFF", "OTB", "OUT", "SUB", "OTB", "OTB"],
        team_id=[363, 363, None, None, 363, 363],
        team_name=["France", "France", None, None, "France", "France"],
        player_id=[1789, 1789, None, None, 1789, 4012],
    )

    result = add_possession_identifiers(events)

    assert result["event_number"].to_list() == [
        2291,
        2292,
        2293,
        2294,
        2295,
        2296,
    ]
    assert result["team_possession_start"].to_list() == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert result["match_team_possession_id"].to_list() == [
        "10517_France_1",
        "10517_France_1",
        None,
        None,
        "10517_France_2",
        "10517_France_2",
    ]
    assert result["match_team_player_possession_id"].to_list() == [
        "10517_France_1789_1",
        "10517_France_1789_1",
        None,
        None,
        "10517_France_1789_2",
        "10517_France_4012_3",
    ]


def test_normal_team_and_player_changes_keep_existing_behavior() -> None:
    """Assign explicit IDs for repeated players and a change of team."""
    events = _possession_events(
        event_number=[1, 2, 3, 4],
        event_type=["FIRSTKICKOFF", "PASS", "PASS", "PASS"],
        team_id=[1, 1, 1, 2],
        team_name=["Lions", "Lions", "Lions", "Tigers"],
        player_id=[11, 11, 12, 21],
    )

    result = add_possession_identifiers(events)

    assert result["match_team_possession_id"].to_list() == [
        "10517_Lions_1",
        "10517_Lions_1",
        "10517_Lions_1",
        "10517_Tigers_2",
    ]
    assert result["match_team_player_possession_id"].to_list() == [
        "10517_Lions_11_1",
        "10517_Lions_11_1",
        "10517_Lions_12_2",
        "10517_Tigers_21_3",
    ]


def test_pause_metadata_is_transparent_to_possession_counters() -> None:
    """Ignore a substitute annotation when continuing player possession.
    """
    events = _possession_events(
        event_number=[1, 2, 3],
        event_type=["FIRSTKICKOFF", "SUB", "PASS"],
        team_id=[1, 1, 1],
        team_name=["Lions", "Lions", "Lions"],
        player_id=[11, 99, 11],
    )

    result = add_possession_identifiers(events)

    assert result["team_possession_start"].to_list() == [True, False, False]
    assert result["match_team_possession_id"].to_list() == [
        "10517_Lions_1",
        None,
        "10517_Lions_1",
    ]
    assert result["match_team_player_possession_id"].to_list() == [
        "10517_Lions_11_1",
        None,
        "10517_Lions_11_1",
    ]


def test_drop_foul_events_removes_only_supplemental_foul_rows() -> None:
    """Drop uppercase FOUL rows while preserving nulls and other types.
    """
    events = pl.DataFrame(
        {
            "event_number": [1, 2, 3, 4, 5],
            "ge_gameeventtype": ["PASS", "FOUL", None, "foul", "OUT"],
            "source_value": [10, 20, 30, 40, 50],
        },
        schema_overrides={"ge_gameeventtype": pl.String},
    )

    result = drop_foul_events(events)

    assert result.to_dict(as_series=False) == {
        "event_number": [1, 3, 4, 5],
        "ge_gameeventtype": ["PASS", None, "foul", "OUT"],
        "source_value": [10, 30, 40, 50],
    }
    assert result.schema == events.schema


def test_drop_foul_events_preserves_empty_input_schema() -> None:
    """Retain typed empty event columns when filtering foul rows."""
    events = pl.DataFrame(
        {
            "event_number": pl.Series([], dtype=pl.UInt32),
            "ge_gameeventtype": pl.Series([], dtype=pl.String),
        }
    )

    result = drop_foul_events(events)

    assert result.equals(events)
    assert result.schema == events.schema


def test_transform_events_runs_foul_filter_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filter FOUL rows through the production transform orchestration.
    """
    events = pl.DataFrame(
        {
            "event_number": [1, 2, 3],
            "ge_gameeventtype": ["PASS", "FOUL", "SHOT"],
        }
    )
    identity_steps = (
        "select_events_columns",
        "drop_pen_shootouts",
        "reclassify_period",
        "reclassify_ballheight",
        "reclassify_pressuretype",
        "reclassify_linesbrokentype",
        "reclassify_firsttouch",
        "rename_classes_setpieces",
        "add_possession_identifiers",
    )
    for step_name in identity_steps:
        monkeypatch.setattr(
            event_transform_module,
            step_name,
            lambda frame: frame,
        )

    result = transform_events(events)

    assert result["event_number"].to_list() == [1, 3]
