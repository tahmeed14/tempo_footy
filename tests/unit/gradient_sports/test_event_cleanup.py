"""Test provider-specific event transformations."""

import polars as pl
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.event_transform import (
    drop_pen_shootouts,
    remove_event_prefixes,
)


def _event_data(data: dict[str, list[object]]) -> pl.DataFrame:
    """Build typed event data for normal-time filtering tests.

    Behavior:
        Construct the minimal transformed-event schema required by the
        shootout filter.

    Input:
        Explicit match, event-number, event-type, and end-type values.

    Expected:
        Empty and null-containing inputs retain stable dtypes.

    Guards against:
        Polars inference changing the behavior under test.
    """
    return pl.DataFrame(
        data,
        schema_overrides={
            "gameid": pl.Int64,
            "event_number": pl.Int64,
            "ge_gameeventtype": pl.String,
            "ge_endtype": pl.String,
        },
    )


def test_drops_events_after_normal_time_boundary() -> None:
    """Keep the qualifying end event and remove later events.

    Behavior:
        Filter one match at the row containing both required end codes.

    Input:
        Events before the boundary, partial code matches, the qualifying
        boundary, and two penalty-shootout rows.

    Expected:
        Events through and including event 4 are retained.

    Guards against:
        Matching only one code, dropping the boundary, or retaining
        shootout events.
    """
    events = _event_data(
        {
            "gameid": [101] * 6,
            "event_number": [1, 2, 3, 4, 5, 6],
            "ge_gameeventtype": [
                "PASS",
                "END",
                "PASS",
                "END",
                "PENALTY",
                "PENALTY",
            ],
            "ge_endtype": [None, "SECOND", "G", "G", None, None],
        }
    )
    expected = events.head(4)

    result = drop_pen_shootouts(events)

    assert_frame_equal(result, expected)


def test_filters_each_match_independently() -> None:
    """Apply a separate normal-time boundary to every match.

    Behavior:
        Calculate the cutoff within each game rather than globally.

    Input:
        Two interleaved matches whose boundary event numbers differ.

    Expected:
        Each boundary is retained, later rows from its match are
        removed,
        and the original order of the remaining rows is preserved.

    Guards against:
        A cutoff from one match truncating or extending another match.
    """
    events = _event_data(
        {
            "gameid": [101, 202, 101, 202, 101, 202, 202],
            "event_number": [1, 1, 2, 2, 3, 3, 4],
            "ge_gameeventtype": [
                "PASS",
                "PASS",
                "END",
                "PASS",
                "PENALTY",
                "END",
                "PENALTY",
            ],
            "ge_endtype": [None, None, "G", None, None, "G", None],
        }
    )
    expected = events[[0, 1, 2, 3, 5]]

    result = drop_pen_shootouts(events)

    assert_frame_equal(result, expected)


def test_keeps_match_without_qualifying_end_event() -> None:
    """Retain all data when the normal-time marker is unavailable.

    Behavior:
        Treat nulls, unexpected codes, and partial matches as no cutoff.

    Input:
        One match containing null values, an ``END`` with another end
        type, and a ``G`` end type on a non-``END`` event.

    Expected:
        The complete input DataFrame and its schema are unchanged.

    Guards against:
        Silently deleting incomplete matches or treating nulls and
        partial code matches as the boundary.
    """
    events = _event_data(
        {
            "gameid": [101, 101, 101, 101],
            "event_number": [1, 2, 3, 4],
            "ge_gameeventtype": [None, "END", "PASS", "UNKNOWN"],
            "ge_endtype": [None, "SECOND", "G", "UNKNOWN"],
        }
    )

    result = drop_pen_shootouts(events)

    assert_frame_equal(result, events)


def test_empty_input_preserves_schema() -> None:
    """Return a typed empty DataFrame for typed empty input.

    Behavior:
        Support an empty event batch without raising an exception.

    Input:
        Zero rows with the required transformed-event schema.

    Expected:
        Zero rows with exactly the original columns and dtypes.

    Guards against:
        Empty-batch failures or temporary filtering columns leaking into
        the result schema.
    """
    events = _event_data(
        {
            "gameid": [],
            "event_number": [],
            "ge_gameeventtype": [],
            "ge_endtype": [],
        }
    )

    result = drop_pen_shootouts(events)

    assert_frame_equal(result, events)


def test_remove_event_prefixes_preserves_rows_and_values() -> None:
    """Remove source prefixes without discarding rows or changing
    values.

    Input includes a null possession ID and an unrelated embedded
    prefix.
    Both rows and the embedded prefix must remain unchanged, catching
    accidental filtering or replacement outside the start of a name.
    """
    events = pl.DataFrame(
        {
            "match_team_possession_id": ["101_Lions_1", None],
            "ge_playername": ["Alex", None],
            "pe_possessioneventtype": ["PA", None],
            "source_ge_value": [1, 2],
        }
    )
    expected = pl.DataFrame(
        {
            "match_team_possession_id": ["101_Lions_1", None],
            "playername": ["Alex", None],
            "possessioneventtype": ["PA", None],
            "source_ge_value": [1, 2],
        }
    )

    assert_frame_equal(remove_event_prefixes(events), expected)
