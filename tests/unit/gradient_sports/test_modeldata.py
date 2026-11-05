"""Tests for assembling model data from player possessions."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.modeldata import (
    POSSESSION_COLUMNS,
    build_modeldata,
    join_event_features,
    join_possession_lookup,
)


def test_modeldata_selects_tempo_outcome_and_context() -> None:
    """Retain tempo and its movement evidence in the selection contract.

    The explicit field set guards against dropping the model outcome,
    its distance and duration, or chronological possession context.
    """
    assert {
        "ball_speed_tempo_player",
        "total_ball_displacement",
        "elapsed_frames",
        "player_possession_sequence_number",
        "elapsed_seconds_team_possession",
        "starting_pitch_third",
    }.issubset(POSSESSION_COLUMNS)


def test_build_modeldata_preserves_unmatched_player_possessions() -> None:
    """Preserve the base rows through both model-data joins.

    Behavior:
        Both left joins retain every player possession.

    Input:
        Three player possessions, event features for the first and
        third, and lookup metadata for the first and second.

    Expected:
        Three rows with null event or lookup values where a match is
        absent, and stable integer and string output dtypes.

    Guards against:
        Inner joins, dropped possessions, or a changed join order.
    """
    df_player_possessions = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "match_team_possession_id": [10, 10, 20],
            "match_team_player_possession_id": ["p1", "p2", "p3"],
            "elapsed_frames": [12, 8, 5],
        }
    ).lazy()
    df_events = pl.DataFrame(
        {
            "game_id": [1, 1],
            "match_team_possession_id": [10, 20],
            "match_team_player_possession_id": ["p1", "p3"],
            "setpiecetype": ["OPEN_PLAY", "CORNER"],
        }
    ).lazy()
    df_lookup = pl.DataFrame(
        {
            "match_team_player_possession_id": ["p1", "p2"],
            "game_id": [1, 1],
            "opposition_teamname": ["United", "United"],
        }
    ).lazy()

    result = build_modeldata(
        df_player_possessions,
        df_events,
        df_lookup,
    ).collect()

    expected = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "match_team_possession_id": [10, 10, 20],
            "match_team_player_possession_id": ["p1", "p2", "p3"],
            "elapsed_frames": [12, 8, 5],
            "setpiecetype": ["OPEN_PLAY", None, "CORNER"],
            "opposition_teamname": ["United", "United", None],
        }
    )
    assert_frame_equal(result, expected)


def test_join_event_features_rejects_duplicate_join_keys() -> None:
    """Reject event rows that would multiply player possessions.

    Behavior:
        The event join enforces one-to-one join-key cardinality.

    Input:
        One player possession and two events with its composite key.

    Expected:
        Collecting the lazy result raises a Polars compute error.

    Guards against:
        Accidental many-to-many joins and duplicated model rows.
    """
    df_player_possessions = pl.DataFrame(
        {
            "game_id": [1],
            "match_team_possession_id": [10],
            "match_team_player_possession_id": ["p1"],
        }
    ).lazy()
    df_events = pl.DataFrame(
        {
            "game_id": [1, 1],
            "match_team_possession_id": [10, 10],
            "match_team_player_possession_id": ["p1", "p1"],
            "event_number": [3, 4],
        }
    ).lazy()

    with pytest.raises(pl.exceptions.ComputeError, match="1:1 validation"):
        join_event_features(df_player_possessions, df_events).collect()


def test_join_possession_lookup_rejects_duplicate_keys() -> None:
    """Reject lookup rows that would multiply model rows.

    Behavior:
        The lookup join enforces one-to-one key cardinality.

    Input:
        One model row and two lookup rows for its possession ID.

    Expected:
        Collecting the lazy result raises a Polars compute error.

    Guards against:
        Duplicate lookup records silently expanding model data.
    """
    df_model = pl.DataFrame(
        {
            "match_team_player_possession_id": ["p1"],
            "game_id": [1],
            "elapsed_frames": [12],
        }
    ).lazy()
    df_lookup = pl.DataFrame(
        {
            "match_team_player_possession_id": ["p1", "p1"],
            "game_id": [1, 1],
            "opposition_teamname": ["United", "City"],
        }
    ).lazy()

    with pytest.raises(pl.exceptions.ComputeError, match="1:1 validation"):
        join_possession_lookup(df_model, df_lookup).collect()
