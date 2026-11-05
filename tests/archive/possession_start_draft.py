"""Test provider-specific event features."""

# import polars as pl
# import pytest
# from polars.testing import assert_series_equal

# from tempoctrl.providers.gradient_sports.event_features import (
#     team_possession_start,
# )


# def _events(number_of_rows: int = 2,
#             **overrides: list[object]) -> pl.DataFrame:
# """Build a minimal sequence of events with no possession changes."""
#     event_data = {
#         "gameid": [1] * number_of_rows,
#         "ge_teamid": [10] * number_of_rows,
#         "ge_setpiecetype": ["O"] * number_of_rows,
#         "ge_gameeventtype": ["OTB"] * number_of_rows,
#     }
#     event_data.update(overrides)

#     return pl.DataFrame(
#         event_data,
#         schema_overrides={
#             "gameid": pl.Int64,
#             "ge_teamid": pl.Int64,
#             "ge_setpiecetype": pl.String,
#             "ge_gameeventtype": pl.String,
#         },
#     )


# def _assert_possession_flags(
#     events: pl.DataFrame,
#     expected: list[bool],
# ) -> None:
# """Assert that the function adds the expected possession-start
# flags."""
#     result = team_possession_start(events)
#     expected_series = pl.Series("team_possession_start", expected)

#     assert_series_equal(
#         result["team_possession_start"],
#         expected_series,
#     )


# def test_first_event_starts_possession_and_same_team_continues() ->
# None:
# """Flag the first event, but not a following event with no trigger."""
#     _assert_possession_flags(_events(), [True, False])


# def test_first_event_starts_possession_even_when_values_are_null() ->
# None:
# """Treat the first row as a start independently of its event
# values."""
#     events = _events(
#         1,
#         gameid=[None],
#         ge_teamid=[None],
#         ge_setpiecetype=[None],
#         ge_gameeventtype=[None],
#     )

#     _assert_possession_flags(events, [True])


# def test_team_change_starts_possession() -> None:
#     """Flag events whose team differs from the preceding event."""
#     events = _events(ge_teamid=[10, 20])

#     _assert_possession_flags(events, [True, True])


# def test_game_change_starts_possession() -> None:
#     """Flag events whose game differs from the preceding event."""
#     events = _events(gameid=[1, 2])

#     _assert_possession_flags(events, [True, True])


# @pytest.mark.parametrize(
#     "set_piece_type",
#     ["C", "D", "F", "G", "K", "P", "T"],
# )
# def test_dead_ball_set_piece_starts_possession(set_piece_type: str) ->
# None:
# """Flag each supported dead-ball set-piece type as a possession
# start."""
#     events = _events(ge_setpiecetype=["O", set_piece_type])

#     _assert_possession_flags(events, [True, True])


# @pytest.mark.parametrize("pause_event", ["SUB", "ON", "OFF", "OUT",
# "END"])
# def test_pause_event_starts_possession(pause_event: str) -> None:
#     """Flag each supported game-pause event as a possession start."""
#     events = _events(ge_gameeventtype=["OTB", pause_event])

#     _assert_possession_flags(events, [True, True])


# @pytest.mark.parametrize(
#     ("column", "values"),
#     [
#         ("ge_teamid", [10, None, 10]),
#         ("gameid", [1, None, 1]),
#         ("ge_setpiecetype", ["O", None, "O"]),
#         ("ge_gameeventtype", ["OTB", None, "OTB"]),
#     ],
# )
# def test_null_values_do_not_start_possession(
#     column: str,
#     values: list[object],
# ) -> None:
# """Do not treat a null trigger value, or its shifted comparison, as a
#     start.

# The third event is included because a null team or game ID also makes
# the
# following row's comparison null. Neither row should start a
# possession.
#     """
#     events = _events(3, **{column: values})

#     _assert_possession_flags(events, [True, False, False])
