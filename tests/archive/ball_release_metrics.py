"""Historical release-time tests for APIs absent from this version."""

import polars as pl

from tempoctrl.gradient_sports.tempo_metrics import (
    add_player_ball_release_metrics,
    aggregate_possession_tempo,
)

def test_player_ball_release_metrics_apply_one_frame_minimum() -> None:
    """Calculate release seconds while preserving observed zero frames.

    Behavior:
        Convert release-frame differences to seconds and replace an
        observed zero difference with one frame for the model outcome.

    Input:
        Positive, zero, and missing release durations at 20 frames per
        second.

    Expected:
        The positive value remains five frames, the zero value retains
        a raw zero but models one frame, and the missing release remains
        null throughout.

    Guards against:
        Including no-attempt possessions, losing the imputation audit,
        or applying a fixed frame rate across matches.
    """
    possessions = pl.DataFrame(
        {
            "start_frame": [10, 20, 30],
            "ball_release_frame": [15, 20, None],
            "frame_rate": [20.0, 20.0, 20.0],
        },
        schema_overrides={"ball_release_frame": pl.Int64},
    ).lazy()

    result = add_player_ball_release_metrics(possessions).collect()

    assert result.select(
        "ball_release_frames_observed",
        "ball_release_frames",
        "ball_release_time",
        "ball_release_one_frame_imputed",
    ).to_dicts() == [
        {
            "ball_release_frames_observed": 5,
            "ball_release_frames": 5,
            "ball_release_time": 0.25,
            "ball_release_one_frame_imputed": False,
        },
        {
            "ball_release_frames_observed": 0,
            "ball_release_frames": 1,
            "ball_release_time": 0.05,
            "ball_release_one_frame_imputed": True,
        },
        {
            "ball_release_frames_observed": None,
            "ball_release_frames": None,
            "ball_release_time": None,
            "ball_release_one_frame_imputed": None,
        },
    ]


def test_player_aggregation_uses_earliest_release_attempt() -> None:
    """Use the earliest pass-or-cross frame within each possession.

    Behavior:
        Aggregate release markers with player frames before calculating
        the model outcome.

    Input:
        One player possession beginning at frame 100 with marked release
        attempts at frames 105 and 106, supplied out of frame order.

    Expected:
        Frame 105 is selected and produces five frames or 0.2 seconds
        at 25 frames per second.

    Guards against:
        Using physical row order, selecting the final release marker, or
        including successful-delivery flight time.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "dev_match_team_possession_id": ["team_a"] * 3,
            "dev_match_team_player_possession_id": ["player_a"] * 3,
            "framenum": [106, 100, 105],
            "pitch_third": ["Middle"] * 3,
            "ball_displacement": [1.0, None, 2.0],
            "delta_frame": [1, None, 5],
            "frame_rate": [25.0] * 3,
            "is_ball_release_attempt": [True, False, True],
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    result = aggregate_possession_tempo(frames, "player").collect()

    assert result.select(
        "ball_release_frame",
        "ball_release_frames_observed",
        "ball_release_frames",
        "ball_release_time",
        "ball_release_one_frame_imputed",
    ).to_dicts() == [
        {
            "ball_release_frame": 105,
            "ball_release_frames_observed": 5,
            "ball_release_frames": 5,
            "ball_release_time": 0.2,
            "ball_release_one_frame_imputed": False,
        }
    ]
