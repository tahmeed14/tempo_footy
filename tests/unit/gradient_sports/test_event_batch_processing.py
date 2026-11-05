"""Verify combined and separate processing produce identical events."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.event_features import features_events
from tempoctrl.gradient_sports.event_transform import (
    cleanup_events,
    transform_events,
)
from tempoctrl.gradient_sports.ingest import scan_events


def _raw_match_events(
    game_id: int,
    missing_challenge_player: bool,
) -> pl.DataFrame:
    """Build raw events with explicit match boundaries and edge cases.

    Behavior:
        Supply the complete nested schema used by the event transforms.

    Input:
        Match 101 or 202 and whether its challenge lacks a player ID.
        Both matches reuse team, player, and game-event IDs.

    Expected:
        Eight raw rows with a kickoff, pass, challenge, supplemental
        foul, OUT pause, cross, and end marker. Match 101 has an earlier
        end marker and a trailing penalty. Match 202 has an additional
        event of an unknown type. Only match 101's pass succeeds.

    Guards against:
        Fixtures hiding cross-match key collisions or relying on private
        data, inferred null dtypes, or production mapping constants.
    """
    event_types = [
        "FIRSTKICKOFF", "PASS", "OTB", "FOUL", "OUT", "PASS",
        "END" if game_id == 101 else "UNKNOWN_EVENT",
        "PENALTY" if game_id == 101 else "END",
    ]
    possession_types = ["IT", "PA", "CH", None, None, "CR", "IT", None]
    game_event_ids = [1, 2, 2, 4, 5, 6, 7, 8]
    rows = []
    for index, event_type in enumerate(event_types):
        is_pause = event_type in ("OUT", "END")
        is_opponent = index == 5
        team_name = "Opp" if is_opponent else "Shared"
        player_id = 21 if is_opponent else 11
        if is_pause or (index == 2 and missing_challenge_player):
            player_id = None
        goal_code = None
        if index == 5:
            goal_code = "A" if game_id == 101 else "H"
        elif index == 3 or (index == 7 and game_id == 101):
            goal_code = "H"
        rows.append(
            {
                "gameId": game_id,
                "gameEventId": game_event_ids[index],
                "possessionEventId": index + 1,
                "duration": 1.0,
                "initialTouch": {
                    "initialPressureType": "UNKNOWN" if index == 2 else "L",
                    "initialHeightType": "UNKNOWN" if index == 2 else "G",
                    "initialBodyType": "UNKNOWN" if index == 2 else "RF",
                },
                "gameEvents": {
                    "gameEventType": event_type,
                    "playerId": player_id,
                    "playerName": None if player_id is None else "Player",
                    "teamId": None if is_pause else (2 if is_opponent else 1),
                    "teamName": None if is_pause else team_name,
                    "homeTeam": None if is_pause else not is_opponent,
                    "setpieceType": "K" if index == 0 else "O",
                    "outType": goal_code,
                    "endType": "G" if event_type == "END" else None,
                    "period": 1,
                    "touches": 1,
                },
                "possessionEvents": {
                    "possessionEventType": possession_types[index],
                    "formattedGameClock": f"00:0{index}",
                    "linesBrokenType": "UNKNOWN" if index == 2 else "A",
                    "pressureType": "UNKNOWN" if index == 2 else "L",
                    "crossOutcomeType": "C" if index == 5 else None,
                    "passOutcomeType": (
                        ("C" if game_id == 101 else "F")
                        if index == 1 else None
                    ),
                },
                "stadiumMetadata": {"teamAttackingDirection": "R"},
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "gameId": pl.Int64,
            "gameEventId": pl.Int64,
            "possessionEventId": pl.Int64,
            "duration": pl.Float64,
            "initialTouch": pl.Struct(
                {
                    "initialPressureType": pl.String,
                    "initialHeightType": pl.String,
                    "initialBodyType": pl.String,
                }
            ),
            "gameEvents": pl.Struct(
                {
                    "gameEventType": pl.String,
                    "playerId": pl.Int64,
                    "playerName": pl.String,
                    "teamId": pl.Int64,
                    "teamName": pl.String,
                    "homeTeam": pl.Boolean,
                    "setpieceType": pl.String,
                    "outType": pl.String,
                    "endType": pl.String,
                    "period": pl.Int64,
                    "touches": pl.Int64,
                }
            ),
            "possessionEvents": pl.Struct(
                {
                    "possessionEventType": pl.String,
                    "formattedGameClock": pl.String,
                    "linesBrokenType": pl.String,
                    "pressureType": pl.String,
                    "crossOutcomeType": pl.String,
                    "passOutcomeType": pl.String,
                }
            ),
            "stadiumMetadata": pl.Struct(
                {"teamAttackingDirection": pl.String}
            ),
        },
    )


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("reverse_matches", [False, True])
@pytest.mark.parametrize("missing_challenge_player", [False, True])
def test_combined_event_processing_matches_separate_matches(
    tmp_path: Path,
    interleaved: bool,
    reverse_matches: bool,
    missing_challenge_player: bool,
) -> None:
    """Preserve full event outputs when selected matches are combined.

    Starting datasets:
        Two eight-row match frames share team, player, and event IDs.
        Each contains a removable foul, pause rows, and a challenge. One
        includes a removable shootout row. Challenge player IDs may be
        null. Files contain either complete matches or interleaved rows
        split across two files, in normal or reversed match order.

    Expected:
        No joins occur and no unmatched rows are removed. The scanner,
        transform_events, features_events, and cleanup_events retain six
        rows for match 101 and seven for match 202: 13 rows, 31 columns.
        Values, column order, dtypes, and row order match separate
        processing. Event numbers retain filtering gaps. Score,
        possession IDs, challenge counts, null states, and pass outcomes
        also match explicit expected values below.

    Guards against:
        Cross-match window leakage, accidental row multiplication or
        loss, global event numbers, differing shootout cutoffs, schema
        changes, or output reordering hidden by sorted comparisons.
    """
    game_ids = [202, 101] if reverse_matches else [101, 202]
    frames = [
        _raw_match_events(game_id, missing_challenge_player)
        for game_id in game_ids
    ]
    separate_outputs = []
    for game_id, frame in zip(game_ids, frames, strict=True):
        source_path = tmp_path / f"{game_id}.parquet"
        frame.write_parquet(source_path)
        separate_outputs.append(
            scan_events(source_path).collect()
            .pipe(transform_events)
            .pipe(features_events)
            .pipe(cleanup_events)
        )
    expected = pl.concat(separate_outputs)

    if interleaved:
        interleaved_frame = pl.concat(
            [frame.slice(index, 1) for index in range(8) for frame in frames]
        )
        selected_frames = [
            interleaved_frame.head(8), interleaved_frame.tail(8)
        ]
        expected = expected.sort(
            ["event_number", "game_id"], descending=[False, reverse_matches]
        )
    else:
        selected_frames = frames
    paths = []
    for index, frame in enumerate(selected_frames):
        path = tmp_path / f"selected_{index}.parquet"
        frame.write_parquet(path)
        paths.append(path)

    actual = (
        scan_events(paths).collect()
        .pipe(transform_events)
        .pipe(features_events)
        .pipe(cleanup_events)
    )

    assert_frame_equal(actual, expected)
    assert actual.shape == (13, 31)
    assert actual.schema["event_number"] == pl.UInt32
    assert actual.schema["home_score"] == pl.Int32
    assert actual.schema["game_state_goal_diff"] == pl.Int32
    assert actual.schema["defender_num_challenges"] == pl.UInt32
    for game_id in (101, 202):
        result = actual.filter(pl.col("game_id") == game_id)
        if game_id == 101:
            assert result["event_number"].to_list() == [1, 2, 3, 5, 6, 7]
            assert result["home_score"].to_list() == [0, 0, 0, 0, 0, 0]
            assert result["away_score"].to_list() == [0, 0, 0, 0, 1, 1]
            assert result["game_state"].to_list() == [
                "Drawing", "Drawing", "Drawing", None, "Winning", None,
            ]
            assert result["successful_pass_or_cross"].to_list() == [
                None, True, None, None, True, None,
            ]
            expected_players = [
                "101_Shared_11_1", "101_Shared_11_1",
                None if missing_challenge_player else "101_Shared_11_1",
                None, "101_Opp_21_2", None,
            ]
        else:
            assert result["event_number"].to_list() == [1, 2, 3, 5, 6, 7, 8]
            assert result["home_score"].to_list() == [0, 0, 0, 0, 1, 1, 1]
            assert result["away_score"].to_list() == [0, 0, 0, 0, 0, 0, 0]
            assert result["game_state"].to_list() == [
                "Drawing", "Drawing", "Drawing", None,
                "Losing", "Winning", None,
            ]
            assert result["successful_pass_or_cross"].to_list() == [
                None, False, None, None, True, None, None,
            ]
            expected_players = [
                "202_Shared_11_1", "202_Shared_11_1",
                None if missing_challenge_player else "202_Shared_11_1",
                None, "202_Opp_21_2", "202_Shared_11_3", None,
            ]
        assert result["match_team_player_possession_id"].to_list() == (
            expected_players
        )
        assert result["defender_num_challenges"].to_list() == (
            [0, 0 if missing_challenge_player else 1, 1, 0, 0, 0]
            + ([0] if game_id == 202 else [])
        )
        challenge = result.filter(pl.col("event_number") == 3)
        assert challenge["first_touch_ballheight"].item() == "N/A"
        assert challenge["first_touch_bodypart"].item() == "N/A"
        assert challenge["defender_pressure_type"].item() == "No Pressure"
        assert challenge["lines_broken_type"].item() == "None"
