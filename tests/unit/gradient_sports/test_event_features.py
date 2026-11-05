"""Test match isolation and ordering in event score features."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.event_features import (
    add_game_state_features,
    create_team_scores,
)


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
