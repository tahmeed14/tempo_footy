"""Test match isolation and ordering in event score features."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.event_features import (
    add_game_state_features,
    create_match_scores,
    create_num_challenges,
    create_team_scores,
)


def test_challenge_counts_exclude_unidentified_possessions() -> None:
    """Count whole identified possessions without pooling unknown rows.

    Behavior:
        Count challenges across game events within each match and player
        possession, leaving unidentified rows' counts null.

    Input:
        Nine interleaved rows from two matches reuse possession A across
        multiple game-event IDs. Possession B has no challenges. Both
        matches also have unidentified challenges and a null-type event.

    Expected:
        UInt32 counts 2, 1, 2, 1, 0, null, null, null, 2 in input order.
        Every source column, row, value, and dtype remains unchanged.

    Guards against:
        Counting by individual game event, leaking counts between
        matches, pooling unidentified rows, or treating unknown as zero.
    """
    events = pl.DataFrame(
        {
            "gameid": [101, 202, 101, 202, 101, 101, 202, 101, 101],
            "gameeventid": [1, 1, 2, 2, 3, 4, 4, 5, 6],
            "match_team_player_possession_id": [
                "A", "A", "A", "A", "B", None, None, None, "A"
            ],
            "pe_possessioneventtype": [
                "CH", "PA", "CH", "CH", "PA", "CH", "CH", None, None
            ],
        }
    )
    expected = events.with_columns(
        pl.Series(
            "defender_num_challenges",
            [2, 1, 2, 1, 0, None, None, None, 2],
            dtype=pl.UInt32,
        )
    )

    assert_frame_equal(create_num_challenges(events), expected)


@pytest.mark.parametrize("possession_dtype", [pl.Null, pl.String])
@pytest.mark.parametrize("row_count", [0, 2])
def test_challenge_counts_preserve_unknown_possessions_and_empty_schema(
    possession_dtype: pl.DataType,
    row_count: int,
) -> None:
    """Keep unidentified-only counts null with stable empty schemas.

    Behavior:
        Support Null and String possession-ID columns with no known IDs.

    Input:
        Zero or two events from match 101 with null possession IDs and
        entirely Null-typed event types.

    Expected:
        Preserve all source data and add a UInt32 count column with
        only nulls, or zero rows for empty input.

    Guards against:
        Inventing zero counts for unknown possessions, Null-dtype
        errors, fabricated empty rows, or count dtype drift.
    """
    events = pl.DataFrame(
        {
            "gameid": pl.Series([101] * row_count, dtype=pl.Int64),
            "match_team_player_possession_id": pl.Series(
                [None] * row_count, dtype=possession_dtype
            ),
            "pe_possessioneventtype": pl.Series(
                [None] * row_count, dtype=pl.Null
            ),
        }
    )
    expected = events.with_columns(
        pl.Series(
            "defender_num_challenges", [None] * row_count, dtype=pl.UInt32
        )
    )

    assert_frame_equal(create_num_challenges(events), expected)


@pytest.mark.parametrize("indicator_dtype", [pl.Null, pl.Boolean])
@pytest.mark.parametrize("row_count", [0, 2])
def test_create_match_scores_accepts_null_home_team_columns(
    indicator_dtype: pl.DataType,
    row_count: int,
) -> None:
    """Leave game state unknown for missing home-team indicators.

    Behavior:
        Accept Null dtype and Boolean-typed nulls without errors.

    Input:
        Zero or two rows with Int32 home/away scores, null home-team
        indicators, and UInt32 event numbers in unsorted order.

    Expected:
        Preserve source rows, order, values, and dtypes. Add entirely
        null Int32 goal differences and String game states, including
        those output types for empty input.

    Guards against:
        Negating a Null-typed column, treating unknown teams as home or
        away, fabricating empty rows, or unstable output schemas.
    """
    events = pl.DataFrame(
        {
            "event_number": pl.Series([2, 1][:row_count], dtype=pl.UInt32),
            "home_score": pl.Series([2, 0][:row_count], dtype=pl.Int32),
            "away_score": pl.Series([1, 1][:row_count], dtype=pl.Int32),
            "ge_hometeam": pl.Series(
                [None] * row_count, dtype=indicator_dtype
            ),
        }
    )
    expected = events.with_columns(
        pl.Series("game_state_goal_diff", [None] * row_count, dtype=pl.Int32),
        pl.Series("game_state", [None] * row_count, dtype=pl.String),
    )

    assert_frame_equal(create_match_scores(events), expected)


def test_create_match_scores_preserves_home_and_away_perspectives() -> None:
    """Keep score perspectives unchanged when accepting Null dtype.

    Behavior:
        Label home and away teams as winning, losing, or drawing while
        leaving a missing team indicator unknown.

    Input:
        Seven rows cover both teams leading, trailing, drawing, and a
        null indicator in a Boolean column, using Int32 scores.

    Expected:
        Goal differences 1, -1, -1, 1, 0, 0, null and corresponding
        String state labels, with every source column unchanged.

    Guards against:
        Reversing the away perspective or mapping unknown teams to a
        valid state while correcting the Null-dtype error.
    """
    events = pl.DataFrame(
        {
            "home_score": pl.Series([2, 2, 1, 1, 2, 1, 2], dtype=pl.Int32),
            "away_score": pl.Series([1, 1, 2, 2, 2, 1, 1], dtype=pl.Int32),
            "ge_hometeam": [True, False, True, False, True, False, None],
        }
    )
    expected = events.with_columns(
        pl.Series(
            "game_state_goal_diff", [1, -1, -1, 1, 0, 0, None],
            dtype=pl.Int32,
        ),
        pl.Series(
            "game_state",
            [
                "Winning", "Losing", "Losing", "Winning", "Drawing",
                "Drawing", None,
            ],
            dtype=pl.String,
        ),
    )

    assert_frame_equal(create_match_scores(events), expected)


@pytest.mark.parametrize(
    ("out_types", "home_scores", "away_scores"),
    [
        ([None, "H", "A", "H"], [0, 1, 1, 2], [0, 0, 1, 1]),
        ([None, None, None, None], [0, 0, 0, 0], [0, 0, 0, 0]),
        (["UNKNOWN", "h", "A", "H"], [0, 0, 0, 1], [0, 0, 1, 1]),
    ],
)
def test_create_team_scores_preserves_single_match_behavior(
    out_types: list[str | None],
    home_scores: list[int],
    away_scores: list[int],
) -> None:
    """Retain cumulative scores for one chronologically ordered match.

    Behavior:
        Increment home or away scores only for their recognized codes.

    Input:
        Four ordered events for one match with home/away codes, nulls,
        or unexpected codes, including lowercase h.

    Expected:
        Explicit cumulative Int32 scores, unchanged source columns,
        and all four rows in their original order.

    Guards against:
        Changing existing single-match scores, counting missing or
        unknown codes, or changing score dtypes.
    """
    events = pl.DataFrame(
        {
            "gameid": [101] * 4,
            "event_number": pl.Series([1, 2, 3, 4], dtype=pl.UInt32),
            "ge_outtype": pl.Series(out_types, dtype=pl.String),
        }
    )
    expected = events.with_columns(
        pl.Series("home_score", home_scores, dtype=pl.Int32),
        pl.Series("away_score", away_scores, dtype=pl.Int32),
    )

    assert_frame_equal(create_team_scores(events), expected)


def test_create_team_scores_isolates_unsorted_interleaved_matches() -> None:
    """Reset scores per match and accumulate in event-number order.

    Behavior:
        Use each match's event sequence while retaining input row order.

    Input:
        Six interleaved, unsorted events from two matches. Match 101
        scores home then away; match 202 scores only home. Both start
        with a null scoring code at event number 1.

    Expected:
        Each match starts at 0-0. Scores follow its event numbers, with
        no rows reordered, dropped, or transferred between matches.

    Guards against:
        Score leakage across matches or accumulation in physical order.
    """
    events = pl.DataFrame(
        {
            "gameid": [101, 202, 101, 202, 101, 202],
            "event_number": pl.Series([3, 2, 1, 3, 2, 1], dtype=pl.UInt32),
            "ge_outtype": ["A", "H", None, None, "H", None],
        }
    )
    expected = events.with_columns(
        pl.Series("home_score", [1, 1, 0, 1, 1, 0], dtype=pl.Int32),
        pl.Series("away_score", [1, 0, 0, 0, 0, 0], dtype=pl.Int32),
    )

    assert_frame_equal(create_team_scores(events), expected)


def test_create_team_scores_preserves_empty_schema() -> None:
    """Return typed score columns for a typed empty event batch.

    Behavior:
        Support zero events without generating rows or losing schema.

    Input:
        An empty frame with Int64 game IDs, UInt32 event numbers,
        and String scoring codes.

    Expected:
        Zero rows, all original columns, and two Int32 score columns.

    Guards against:
        Empty-batch failures or unstable output schemas.
    """
    events = pl.DataFrame(
        schema={
            "gameid": pl.Int64,
            "event_number": pl.UInt32,
            "ge_outtype": pl.String,
        }
    )
    expected = pl.DataFrame(
        schema={
            "gameid": pl.Int64,
            "event_number": pl.UInt32,
            "ge_outtype": pl.String,
            "home_score": pl.Int32,
            "away_score": pl.Int32,
        }
    )

    assert_frame_equal(create_team_scores(events), expected)


def test_game_state_matches_separate_match_processing() -> None:
    """Match separate score and game-state results in a combined batch.

    Behavior:
        Exercise score accumulation followed by possession-team state.

    Input:
        Four events from two matches: match 101 scores home then away,
        and match 202 opens at 0-0 before an away goal. One event has no
        home-team indicator.

    Expected:
        Four rows with unchanged event columns plus scores, goal
        differences, and state labels. No joins occur and no unmatched
        rows are removed. Match 202 begins Drawing, then its home team
        is Losing at -1; its opening row inherits no earlier goals.
        A null home-team indicator produces null state fields.

    Guards against:
        Cross-match score leakage propagating into model features or
        a combined batch differing from separate processing.
    """
    events = pl.DataFrame(
        {
            "gameid": [101, 101, 202, 202],
            "event_number": pl.Series([1, 2, 1, 2], dtype=pl.UInt32),
            "ge_outtype": ["H", "A", None, "A"],
            "ge_hometeam": [True, None, False, True],
        }
    )
    combined = add_game_state_features(events)
    separate = pl.concat(
        [
            add_game_state_features(events.filter(pl.col("gameid") == 101)),
            add_game_state_features(events.filter(pl.col("gameid") == 202)),
        ]
    )
    expected = events.with_columns(
        pl.Series("home_score", [1, 1, 0, 0], dtype=pl.Int32),
        pl.Series("away_score", [0, 1, 0, 1], dtype=pl.Int32),
        pl.Series("game_state_goal_diff", [1, None, 0, -1], dtype=pl.Int32),
        pl.Series("game_state", ["Winning", None, "Drawing", "Losing"]),
    )

    assert_frame_equal(combined, expected)
    assert_frame_equal(combined, separate)
