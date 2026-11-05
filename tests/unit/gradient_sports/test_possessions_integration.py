"""Test Gradient Sports possession transformations."""

import polars as pl
import pytest

from tempoctrl.gradient_sports import possessions_transform
from tempoctrl.gradient_sports.possessions_transform import (
    transform_possessions,
)

BALL_DTYPE = pl.Struct(
    {
        "visibility": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
    }
)


@pytest.fixture(autouse=True)
def stub_attacking_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply local direction metadata without reading private datasets.
    """
    directions = pl.DataFrame(
        {
            "match_team_possession_id": ["team_1", "team_2"],
            "attacking_team_direction": ["R", "R"],
        }
    ).lazy()
    monkeypatch.setattr(
        possessions_transform,
        "scan_processed_files",
        lambda *args, **kwargs: directions,
    )


def _tracking_frames(data: dict[str, list[object]]) -> pl.LazyFrame:
    """Build typed tracking frames for possession transform tests.

    Behavior:
        Construct the required integrated tracking schema.

    Input:
        Explicit frame, event, possession, and outcome values.

    Expected:
        Missing optional test values receive stable typed defaults.

    Guards against:
        Schema inference changing the behavior being tested.
    """
    frame_count = len(data["game_id"])
    defaults: dict[str, list[object]] = {
        "balls_smooth": [
            {
                "visibility": "visible",
                "x": float(index),
                "y": float(index + 10),
                "z": 0.0,
            }
            for index in range(frame_count)
        ],
        "away_players_smooth": [[]] * frame_count,
        "home_players_smooth": [[]] * frame_count,
        "player_id": [None] * frame_count,
        "playername": [None] * frame_count,
        "team_id": [None] * frame_count,
        "teamname": [None] * frame_count,
        "hometeam": [None] * frame_count,
        "game_event_type": [None] * frame_count,
    }
    frame_data = {**defaults, **data}

    return pl.DataFrame(
        frame_data,
        schema_overrides={
            "game_id": pl.Int32,
            "period": pl.Int64,
            "framenum": pl.Int64,
            "event_number": pl.UInt32,
            "possession_event_id": pl.Int32,
            "match_team_possession_id": pl.String,
            "match_team_player_possession_id": pl.String,
            "possession_event_type": pl.String,
            "successful_pass_or_cross": pl.Boolean,
            "balls_smooth": BALL_DTYPE,
            "away_players_smooth": pl.List(pl.Int64),
            "home_players_smooth": pl.List(pl.Int64),
            "player_id": pl.Int64,
            "playername": pl.String,
            "team_id": pl.Int64,
            "teamname": pl.String,
            "hometeam": pl.Boolean,
            "game_event_type": pl.String,
        },
    ).lazy()


def test_fills_internal_team_and_player_gaps_independently() -> None:
    """Fill only gaps enclosed by matching IDs.

    Behavior:
        Evaluate team- and player-possession gaps independently.

    Input:
        One gap enclosed by the same team and player, followed by a gap
        enclosed by one team but two different players.

    Expected:
        Both IDs fill in the first gap; only the team ID fills in the
        second gap. Raw source identifiers remain unchanged.

    Guards against:
        Unconditional forward filling or coupling the two ID columns.
    """
    frames = _tracking_frames(
        {
            "game_id": [1] * 7,
            "period": [1] * 7,
            "framenum": [1, 2, 3, 4, 5, 6, 7],
            "event_number": [1, None, 2, 3, None, 4, None],
            "possession_event_id": [10, None, 11, 12, None, 13, None],
            "match_team_possession_id": [
                "team_1",
                None,
                "team_1",
                "team_1",
                None,
                "team_1",
                None,
            ],
            "match_team_player_possession_id": [
                "player_a",
                None,
                "player_a",
                "player_b",
                None,
                "player_c",
                None,
            ],
            "possession_event_type": [
                "BC",
                None,
                "CH",
                "BC",
                None,
                "CH",
                None,
            ],
            "successful_pass_or_cross": [None] * 7,
        }
    )

    result = transform_possessions(
        frames,
        frame_rate=29.97,
    ).collect()

    assert result.height == 7
    assert result["match_team_player_possession_id"].to_list() == [
        "player_a",
        None,
        "player_a",
        "player_b",
        None,
        "player_c",
        None,
    ]
    assert result["dev_match_team_player_possession_id"].to_list() == [
        "player_a",
        "player_a",
        "player_a",
        "player_b",
        None,
        "player_c",
        None,
    ]
    assert result["dev_match_team_possession_id"].to_list() == [
        "team_1",
        "team_1",
        "team_1",
        "team_1",
        "team_1",
        "team_1",
        None,
    ]


def test_successful_pass_adds_passer_at_receiver_frame() -> None:
    """Add a shared-frame endpoint for a successful pass.

    Behavior:
        Extend the passer over intervening null frames and insert one
        passer-attributed row at the receiver's first frame.

    Input:
        A successful pass, one null frame, and the receiver's
        possession.

    Expected:
        The synthetic row copies only the receiver-frame ball
        coordinates
        and frame identity while retaining derived passer possession
        IDs. Other source metadata remains null. The original receiver
        row remains unchanged.

    Guards against:
        Losing delivery distance, overwriting the receiver, or copying
        receiver event metadata into the passer's synthetic row.
    """
    receiver_ball = {
        "visibility": "visible",
        "x": 42.5,
        "y": 11.0,
        "z": 0.25,
    }
    frames = _tracking_frames(
        {
            "game_id": [1, 1, 1],
            "period": [1, 1, 1],
            "framenum": [10, 11, 12],
            "event_number": [1, None, 2],
            "possession_event_id": [10, None, 11],
            "match_team_possession_id": ["team_1", None, "team_1"],
            "match_team_player_possession_id": [
                "player_a",
                None,
                "player_b",
            ],
            "possession_event_type": ["PA", None, "IT"],
            "successful_pass_or_cross": [True, None, None],
            "balls_smooth": [
                {
                    "visibility": "visible",
                    "x": 10.0,
                    "y": 5.0,
                    "z": 0.0,
                },
                {
                    "visibility": "visible",
                    "x": 25.0,
                    "y": 8.0,
                    "z": 0.1,
                },
                receiver_ball,
            ],
            "player_id": [101, None, 202],
            "playername": ["Passer", None, "Receiver"],
            "team_id": [7, None, 7],
            "teamname": ["Home", None, "Home"],
            "hometeam": [True, None, True],
            "game_event_type": ["PASS", None, "RECEIVE"],
        }
    )

    result = transform_possessions(
        frames,
        frame_rate=29.97,
    ).collect()
    synthetic = result.filter("is_synthetic_pass_end")
    receiver = result.filter(
        (pl.col("framenum") == 12) & ~pl.col("is_synthetic_pass_end")
    )

    assert result["framenum"].to_list() == [10, 11, 12, 12]
    assert result["is_synthetic_pass_end"].to_list() == [
        False,
        False,
        True,
        False,
    ]
    assert (
        result.filter(pl.col("framenum") == 11)[
            "dev_match_team_player_possession_id"
        ].item()
        == "player_a"
    )
    assert synthetic["balls_smooth"].item() == {
        **receiver_ball, "is_imputed": False,
    }
    assert synthetic["player_id"].item() is None
    assert synthetic["playername"].item() is None
    assert synthetic["team_id"].item() is None
    assert synthetic["teamname"].item() is None
    assert synthetic["hometeam"].item() is None
    assert synthetic["match_team_possession_id"].item() is None
    assert synthetic["match_team_player_possession_id"].item() is None
    assert synthetic["possession_event_id"].item() is None
    assert synthetic["possession_event_type"].item() is None
    assert synthetic["game_event_type"].item() is None
    assert synthetic["dev_match_team_possession_id"].item() == "team_1"
    assert (
        synthetic["dev_match_team_player_possession_id"].item()
        == "player_a"
    )
    assert receiver["player_id"].item() == 202
    assert receiver["playername"].item() == "Receiver"
    assert (
        receiver["dev_match_team_player_possession_id"].item()
        == "player_b"
    )


def test_repeated_passer_annotation_does_not_stop_extension() -> None:
    """Keep extending after repeated annotations of the same possession.

    Behavior:
        Start a new propagation segment only when the observed player-
        possession identifier changes.

    Input:
        A successful-pass event, a repeated raw passer ID, a null frame,
        and then a receiver possession.

    Expected:
        The null frame retains the passer's effective possession ID.

    Guards against:
        Treating every non-null annotation as a new possession boundary.
    """
    frames = _tracking_frames(
        {
            "game_id": [1, 1, 1, 1],
            "period": [1, 1, 1, 1],
            "framenum": [1, 2, 3, 4],
            "event_number": [1, None, None, 2],
            "possession_event_id": [10, None, None, 11],
            "match_team_possession_id": [
                "team_1",
                "team_1",
                None,
                "team_1",
            ],
            "match_team_player_possession_id": [
                "player_a",
                "player_a",
                None,
                "player_b",
            ],
            "possession_event_type": ["PA", None, None, "IT"],
            "successful_pass_or_cross": [True, None, None, None],
            "player_id": [101, None, None, 102],
            "playername": ["A", None, None, "B"],
            "team_id": [7, None, None, 7],
            "teamname": ["Home", None, None, "Home"],
            "hometeam": [True, None, None, True],
        }
    )

    result = transform_possessions(
        frames,
        frame_rate=29.97,
    ).collect()

    assert (
        result.filter(pl.col("framenum") == 3)[
            "dev_match_team_player_possession_id"
        ].item()
        == "player_a"
    )
    assert result["is_synthetic_pass_end"].sum() == 1


def test_includes_crosses_and_excludes_failed_passes() -> None:
    """Extend successful crosses but not unsuccessful passes.

    Behavior:
        Treat ``CR`` as eligible alongside ``PA`` and require true
        success.

    Input:
        Separate games containing a successful cross and a failed pass.

    Expected:
        Only the cross receives an extension and synthetic endpoint.

    Guards against:
        Excluding crosses or treating false outcomes as successful.
    """
    frames = _tracking_frames(
        {
            "game_id": [1, 1, 1, 2, 2, 2],
            "period": [1] * 6,
            "framenum": [1, 2, 3, 1, 2, 3],
            "event_number": [1, None, 2, 1, None, 2],
            "possession_event_id": [10, None, 11, 20, None, 21],
            "match_team_possession_id": [
                "team_1",
                None,
                "team_1",
                "team_2",
                None,
                "team_2",
            ],
            "match_team_player_possession_id": [
                "crosser",
                None,
                "receiver_1",
                "passer",
                None,
                "receiver_2",
            ],
            "possession_event_type": [
                "CR",
                None,
                "IT",
                "PA",
                None,
                "IT",
            ],
            "successful_pass_or_cross": [
                True,
                None,
                None,
                False,
                None,
                None,
            ],
            "player_id": [101, None, 102, 201, None, 202],
            "playername": ["A", None, "B", "C", None, "D"],
            "team_id": [7, None, 7, 8, None, 8],
            "teamname": ["Home", None, "Home", "Away", None, "Away"],
            "hometeam": [True, None, True, False, None, False],
        }
    )

    result = transform_possessions(
        frames,
        frame_rate=29.97,
    ).collect()
    game_one = result.filter(pl.col("game_id") == 1)
    game_two = result.filter(pl.col("game_id") == 2)

    assert game_one["is_synthetic_pass_end"].sum() == 1
    assert (
        game_one.filter(pl.col("framenum") == 2)[
            "dev_match_team_player_possession_id"
        ].item()
        == "crosser"
    )
    assert game_two["is_synthetic_pass_end"].sum() == 0
    assert (
        game_two.filter(pl.col("framenum") == 2)[
            "dev_match_team_player_possession_id"
        ].item()
        is None
    )


def test_preserves_duplicates_and_adds_one_synthetic_row() -> None:
    """Preserve source duplicates and create one delivery endpoint.

    Behavior:
        Calculate state once per frame and restore duplicate
        annotations.

    Input:
        Two identical successful-pass rows and one receiver row.

    Expected:
        All source rows remain and exactly one synthetic row is added.

    Guards against:
        Accidental deduplication or a many-to-many synthetic-row join.
    """
    frames = _tracking_frames(
        {
            "game_id": [1, 1, 1],
            "period": [1, 1, 1],
            "framenum": [1, 1, 2],
            "event_number": [1, 1, 2],
            "possession_event_id": [10, 10, 11],
            "match_team_possession_id": ["team_1", None, "team_1"],
            "match_team_player_possession_id": [
                "player_a",
                None,
                "player_b",
            ],
            "possession_event_type": ["PA", "PA", "IT"],
            "successful_pass_or_cross": [True, True, None],
            "player_id": [101, 101, 102],
            "playername": ["A", "A", "B"],
            "team_id": [7, 7, 7],
            "teamname": ["Home"] * 3,
            "hometeam": [True] * 3,
        }
    )

    result = transform_possessions(
        frames,
        frame_rate=29.97,
    ).collect()

    assert result.height == 4
    assert result["is_synthetic_pass_end"].sum() == 1
    assert result.filter(~pl.col("is_synthetic_pass_end")).height == 3
    duplicate = result.filter(
        (pl.col("framenum") == 1)
        & pl.col("match_team_player_possession_id").is_null()
    )
    assert (
        duplicate["dev_match_team_player_possession_id"].item()
        == "player_a"
    )
    assert duplicate["is_synthetic_pass_end"].item() is False
    assert duplicate["player_id"].item() == 101


def test_rejects_missing_required_columns() -> None:
    """Reject input that cannot support possession transformation.

    Behavior:
        Validate required columns before constructing the lazy pipeline.

    Input:
        A LazyFrame containing only a game identifier.

    Expected:
        A ``ValueError`` names the missing possession columns.

    Guards against:
        Deferred, unclear column-not-found errors during collection.
    """
    frames = pl.DataFrame({"game_id": [1]}).lazy()

    with pytest.raises(ValueError, match="Missing possession columns"):
        transform_possessions(frames, frame_rate=29.97)
