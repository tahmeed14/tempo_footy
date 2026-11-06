"""Test provider-specific possession features."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from tempoctrl.gradient_sports.event_transform import (
    create_player_possession_id,
    create_team_possession_flag,
    create_team_possession_id,
)


@pytest.mark.parametrize(
    ("setpiece_label", "starts_possession"),
    [
        ("Corner", True),
        ("Drop Kick", True),
        ("Free Kick", True),
        ("Goal Kick", True),
        ("Kickoff", True),
        ("Penalty", True),
        ("Throw In", True),
        ("Open Play", False),
        ("N/A", False),
        ("UNKNOWN", False),
        (None, False),
    ],
)
def test_team_possession_flags_recognize_reclassified_setpieces(
    setpiece_label: str | None,
    starts_possession: bool,
) -> None:
    """Recognize dead-ball labels after set-piece reclassification.

    Behavior:
        Start a possession for each dead-ball label without relying on
        team changes, kickoff event types, or a preceding OUT.

    Input:
        Three PASS rows for one team. Open Play surrounds a candidate
        set-piece label, including unknown, N/A, and null candidates.

    Expected:
        The first row starts a possession. Only the seven dead-ball
        candidates start another; the final Open Play row never does.
        All source columns, rows, and order remain unchanged, and the
        added flag has Boolean dtype with no nulls.

    Guards against:
        Comparing readable labels against raw provider codes or treating
        Open Play, missing, and unknown labels as dead-ball restarts.
    """
    events = pl.DataFrame(
        {
            "gameid": [101, 101, 101],
            "event_number": [1, 2, 3],
            "ge_teamid": [1, 1, 1],
            "ge_gameeventtype": ["PASS", "PASS", "PASS"],
            "ge_setpiecetype": ["Open Play", setpiece_label, "Open Play"],
        }
    )
    expected = events.with_columns(
        pl.Series(
            "team_possession_start",
            [True, starts_possession, False],
            dtype=pl.Boolean,
        )
    )

    assert_frame_equal(create_team_possession_flag(events), expected)


def test_team_possession_flags_are_scoped_to_each_match() -> None:
    """Compare each team only with the previous team in the same match.
    """
    events = pl.DataFrame(
        {
            "event_number": [1, 1, 2, 2, 3, 3],
            "gameid": [101, 202, 101, 202, 101, 202],
            "ge_teamid": [1, 9, 1, 9, 2, 8],
            "ge_setpiecetype": [None] * 6,
            "ge_gameeventtype": ["PASS"] * 6,
        },
        schema_overrides={"ge_setpiecetype": pl.String},
    )

    result = create_team_possession_flag(events)

    assert_series_equal(
        result["team_possession_start"],
        pl.Series(
            "team_possession_start",
            [True, True, False, False, True, True],
        ),
    )


def _possession_events(data: dict[str, list[object]]) -> pl.DataFrame:
    """Build typed possession-event input for the unit tests.

    Behavior:
        Construct an event DataFrame using the lowercase column names
        emitted by the Gradient Sports transform.

    Input:
        Explicit values for each required input column and any
        passthrough columns used by a test.

    Expected:
        Boolean, integer, and string columns retain stable dtypes even
        when their values include nulls or the DataFrame is empty.

    Guards against:
        Polars inference changing a test's meaning instead of
        exercising the possession-ID behavior under test.
    """
    row_count = len(data["gameid"])
    data = {
        **data,
        "ge_teamid": [1] * row_count,
        "ge_gameeventtype": ["PASS"] * row_count,
    }
    return pl.DataFrame(
        data,
        schema_overrides={
            "event_number": pl.Int64,
            "gameid": pl.Int64,
            "ge_teamname": pl.String,
            "team_possession_start": pl.Boolean,
            "ge_teamid": pl.Int64,
            "ge_gameeventtype": pl.String,
        },
    )


def test_creates_ids_for_each_possession_in_one_match() -> None:
    """Create stable match and team possession IDs for a normal match.

    Behavior:
        Increment the match possession number only when the start flag
        is true and combine the game, team, and possession values into
        a string ID.

    Input:
        Five ordered events from one match containing three possessions
        split between two teams.

    Expected:
        Possession numbers are ``[1, 1, 2, 2, 3]``; string IDs match
        those possessions; input row order and the event-number column
        are retained; and the start flag is retained.

    Guards against:
        Incrementing every row, building an incorrect composite ID,
        reordering or dropping events, or dropping the start flag.
    """
    events = _possession_events(
        {
            "event_number": [1, 2, 3, 4, 5],
            "gameid": [101, 101, 101, 101, 101],
            "ge_teamname": ["Lions", "Lions", "Tigers", "Tigers", "Lions"],
            "team_possession_start": [True, False, True, False, True],
        }
    )
    expected = pl.DataFrame(
        {
            "event_number": [1, 2, 3, 4, 5],
            "gameid": [101, 101, 101, 101, 101],
            "ge_teamname": ["Lions", "Lions", "Tigers", "Tigers", "Lions"],
            "match_possession_id": pl.Series(
                [1, 1, 2, 2, 3],
                dtype=pl.UInt32,
            ),
            "match_team_possession_id": [
                "101_Lions_1",
                "101_Lions_1",
                "101_Tigers_2",
                "101_Tigers_2",
                "101_Lions_3",
            ],
        }
    )

    result = create_team_possession_id(events)

    assert result["team_possession_start"].to_list() == [
        True, False, True, False, True,
    ]
    assert_frame_equal(
        result.drop("team_possession_start", "ge_teamid", "ge_gameeventtype"),
        expected,
    )


def test_restarts_possession_numbers_for_each_match() -> None:
    """Restart match possession numbering when a new match begins.

    Behavior:
        Calculate possession numbers independently within each game.

    Input:
        Two ordered matches, each containing two possessions.

    Expected:
        Each match has possession numbers ``[1, 1, 2]`` rather than
        the second match continuing the first match's cumulative count.

    Guards against:
        A global cumulative sum that makes IDs depend on which other
        matches happen to be present in the same DataFrame.
    """
    events = _possession_events(
        {
            "event_number": [1, 2, 3, 1, 2, 3],
            "gameid": [101, 101, 101, 202, 202, 202],
            "ge_teamname": [
                "Lions",
                "Lions",
                "Tigers",
                "Owls",
                "Owls",
                "Bears",
            ],
            "team_possession_start": [True, False, True, True, False, True],
        }
    )

    result = create_team_possession_id(events)

    expected_ids = pl.Series(
        "match_possession_id",
        [1, 1, 2, 1, 1, 2],
        dtype=pl.UInt32,
    )
    assert_series_equal(result["match_possession_id"], expected_ids)


# def test_null_start_flag_does_not_create_or_erase_a_possession_id() ->
# None:
#     """Treat a null possession-start flag as no new possession.

#     Behavior:
#         Continue the current possession when the Boolean start flag is
#         null.

#     Input:
#         One match with a true start, a null flag, a false flag, and
#         another true start.

#     Expected:
# Possession numbers are ``[1, 1, 1, 2]`` with no null output ID.

#     Guards against:
#         Propagating a null cumulative value into an otherwise valid
#         possession.
#     """
#     events = _possession_events(
#         {
#             "event_number": [1, 2, 3, 4],
#             "gameid": [101, 101, 101, 101],
#             "ge_teamname": ["Lions", "Lions", "Lions", "Tigers"],
#             "team_possession_start": [True, None, False, True],
#         }
#     )

#     result = create_team_possession_id(events)

#     expected_ids = pl.Series(
#         "match_possession_id",
#         [1, 1, 1, 2],
#         dtype=pl.UInt32,
#     )
#     assert_series_equal(result["match_possession_id"], expected_ids)


def test_null_identifier_component_produces_null_composite_id() -> None:
    """Avoid inventing a composite ID when a required component is null.

    Behavior:
        Preserve possession numbering while returning a null
        team-possession ID when either the game ID or team name is
        unknown.

    Input:
        Three possession starts: one complete row, one with a null game
        ID, and one with a null team name.

    Expected:
        Numeric possession IDs remain populated, while the latter two
        string identifiers are null.

    Guards against:
        Producing misleading strings such as ``null_Lions_2`` that
        appear to be valid identifiers.
    """
    events = _possession_events(
        {
            "event_number": [1, 2, 3],
            "gameid": [101, None, 101],
            "ge_teamname": ["Lions", "Lions", None],
            "team_possession_start": [True, True, True],
        }
    )

    result = create_team_possession_id(events)

    expected_ids = pl.Series(
        "match_team_possession_id",
        ["101_Lions_1", None, None],
        dtype=pl.String,
    )
    assert_series_equal(result["match_team_possession_id"], expected_ids)


def test_empty_input_returns_expected_output_schema() -> None:
    """Return a typed, empty result for a typed, empty input.

    Behavior:
        Support an empty event batch without raising an exception.

    Input:
        Zero rows with the normal input schema.

    Expected:
        Zero rows; the source flag is retained; the possession number
        is ``UInt32``; and the composite identifier is ``String``.

    Guards against:
        Empty-batch failures and unstable output schemas that break
        downstream concatenation or persistence.
    """
    events = _possession_events(
        {
            "event_number": [],
            "gameid": [],
            "ge_teamname": [],
            "team_possession_start": [],
        }
    )

    result = create_team_possession_id(events)

    assert result.shape == (0, 8)
    assert result.schema == {
        "event_number": pl.Int64,
        "gameid": pl.Int64,
        "ge_teamname": pl.String,
        "team_possession_start": pl.Boolean,
        "ge_teamid": pl.Int64,
        "ge_gameeventtype": pl.String,
        "match_possession_id": pl.UInt32,
        "match_team_possession_id": pl.String,
    }


def _player_possession_events(
    data: dict[str, list[object]],
) -> pl.DataFrame:
    """Build typed player-possession input for the unit tests.

    Behavior:
        Construct the schema produced by the team-possession step and
        consumed by the player-possession step.

    Input:
        Explicit event, match, team, player, and team-possession values.

    Expected:
        Identifier columns retain stable dtypes for normal, null, and
        empty test cases.

    Guards against:
        DataFrame inference masking player-possession behavior with an
        unintended test dtype.
    """
    return pl.DataFrame(
        data,
        schema_overrides={
            "event_number": pl.Int64,
            "gameid": pl.Int64,
            "ge_teamname": pl.String,
            "ge_playerid": pl.Int64,
            "match_possession_id": pl.UInt32,
            "match_team_possession_id": pl.String,
        },
    )


def test_creates_player_ids_for_normal_possession_sequence() -> None:
    """Create a new ID when the player or team possession changes.

    Behavior:
        Continue an ID for consecutive events by the same player and
        increment it when the player or team possession changes.

    Input:
        Six events containing repeated players, a player returning after
        another player, and a change of team possession.

    Expected:
        Explicit player-possession IDs use indices ``1, 1, 2, 3, 4, 4``;
        input columns and row order remain intact.

    Guards against:
        Incrementing every event, merging nonconsecutive player touches,
        ignoring team-possession changes, or dropping upstream columns.
    """
    events = _player_possession_events(
        {
            "event_number": [1, 2, 3, 4, 5, 6],
            "gameid": [101, 101, 101, 101, 101, 101],
            "ge_teamname": [
                "Lions",
                "Lions",
                "Lions",
                "Lions",
                "Tigers",
                "Tigers",
            ],
            "ge_playerid": [11, 11, 12, 11, 21, 21],
            "match_team_possession_id": [
                "101_Lions_1",
                "101_Lions_1",
                "101_Lions_1",
                "101_Lions_1",
                "101_Tigers_2",
                "101_Tigers_2",
            ],
        }
    )
    expected = events.with_columns(
        pl.Series(
            "match_team_player_possession_id",
            [
                "101_Lions_11_1",
                "101_Lions_11_1",
                "101_Lions_12_2",
                "101_Lions_11_3",
                "101_Tigers_21_4",
                "101_Tigers_21_4",
            ],
        )
    )

    result = create_player_possession_id(events)

    assert_frame_equal(result, expected)


def test_team_possession_change_starts_new_player_possession() -> None:
    """Start a new player possession at a team-possession boundary.

    Behavior:
        Increment the player-possession index when the team possession
        changes, even if the player ID remains unchanged.

    Input:
        Three events for one player spanning two team possessions.

    Expected:
        The first two events share index 1 and the final event uses
        index 2.

    Guards against:
        Merging one player's touches across a new team possession.
    """
    events = _player_possession_events(
        {
            "event_number": [1, 2, 3],
            "gameid": [101, 101, 101],
            "ge_teamname": ["Lions", "Lions", "Lions"],
            "ge_playerid": [11, 11, 11],
            "match_possession_id": [1, 1, 2],
            "match_team_possession_id": [
                "101_Lions_1",
                "101_Lions_1",
                "101_Lions_2",
            ],
        }
    )

    result = create_player_possession_id(events)

    expected_ids = pl.Series(
        "match_team_player_possession_id",
        ["101_Lions_11_1", "101_Lions_11_1", "101_Lions_11_2"],
    )
    assert_series_equal(
        result["match_team_player_possession_id"],
        expected_ids,
    )


def test_player_possession_numbers_restart_for_each_match() -> None:
    """Restart player-possession numbering when the match changes.

    Behavior:
        Calculate player-possession indices independently for each game.

    Input:
        Two matches containing two consecutive player possessions each.

    Expected:
        Both matches use indices 1 and 2 rather than the second match
        continuing at indices 3 and 4.

    Guards against:
        Global numbering whose values depend on other matches in the
        same DataFrame.
    """
    events = _player_possession_events(
        {
            "event_number": [1, 2, 1, 2],
            "gameid": [101, 101, 202, 202],
            "ge_teamname": ["Lions", "Lions", "Owls", "Owls"],
            "ge_playerid": [11, 12, 31, 32],
            "match_possession_id": [1, 1, 1, 1],
            "match_team_possession_id": [
                "101_Lions_1",
                "101_Lions_1",
                "202_Owls_1",
                "202_Owls_1",
            ],
        }
    )

    result = create_player_possession_id(events)

    expected_ids = pl.Series(
        "match_team_player_possession_id",
        [
            "101_Lions_11_1",
            "101_Lions_12_2",
            "202_Owls_31_1",
            "202_Owls_32_2",
        ],
    )
    assert_series_equal(
        result["match_team_player_possession_id"],
        expected_ids,
    )


def test_player_changes_are_scoped_for_interleaved_matches() -> None:
    """Ignore rows from other matches when finding the previous player.
    """
    events = _player_possession_events(
        {
            "event_number": [1, 1, 2, 2],
            "gameid": [101, 202, 101, 202],
            "ge_teamname": ["Lions", "Owls", "Lions", "Owls"],
            "ge_playerid": [11, 31, 11, 31],
            "match_possession_id": [1, 1, 1, 1],
            "match_team_possession_id": [
                "101_Lions_1",
                "202_Owls_1",
                "101_Lions_1",
                "202_Owls_1",
            ],
        }
    )

    result = create_player_possession_id(events)

    assert_series_equal(
        result["match_team_player_possession_id"],
        pl.Series(
            "match_team_player_possession_id",
            [
                "101_Lions_11_1",
                "202_Owls_31_1",
                "101_Lions_11_1",
                "202_Owls_31_1",
            ],
        ),
    )


def test_null_player_does_not_advance_player_possession() -> None:
    """Do not create a player possession for an unknown player.

    Behavior:
        Leave the composite ID null for an unknown player and continue
        the known player's possession after the null event.

    Input:
        One team possession with player 11, a null player, then player
        11 again.

    Expected:
        The known-player events share index 1 and the null-player event
        has a null composite identifier.

    Guards against:
        Treating missing player data as two player changes and advancing
        the counter for an event that cannot identify a player.
    """
    events = _player_possession_events(
        {
            "event_number": [1, 2, 3],
            "gameid": [101, 101, 101],
            "ge_teamname": ["Lions", "Lions", "Lions"],
            "ge_playerid": [11, None, 11],
            "match_possession_id": [1, 1, 1],
            "match_team_possession_id": [
                "101_Lions_1",
                "101_Lions_1",
                "101_Lions_1",
            ],
        }
    )

    result = create_player_possession_id(events)

    expected_ids = pl.Series(
        "match_team_player_possession_id",
        ["101_Lions_11_1", None, "101_Lions_11_1"],
        dtype=pl.String,
    )
    assert_series_equal(
        result["match_team_player_possession_id"],
        expected_ids,
    )


def test_empty_player_input_returns_expected_schema() -> None:
    """Return a typed, empty result for an empty player-event batch.

    Behavior:
        Support an empty DataFrame without raising an exception.

    Input:
        Zero rows with the normal player-possession input schema.

    Expected:
        Zero rows, all upstream columns, and one String
        player-possession identifier column.

    Guards against:
        Empty-batch failures, dropped upstream identifiers, and unstable
        output schemas.
    """
    events = _player_possession_events(
        {
            "event_number": [],
            "gameid": [],
            "ge_teamname": [],
            "ge_playerid": [],
            "match_possession_id": [],
            "match_team_possession_id": [],
        }
    )

    result = create_player_possession_id(events)

    assert result.shape == (0, 6)
    assert result.schema == {
        "event_number": pl.Int64,
        "gameid": pl.Int64,
        "ge_teamname": pl.String,
        "ge_playerid": pl.Int64,
        "match_team_possession_id": pl.String,
        "match_team_player_possession_id": pl.String,
    }
