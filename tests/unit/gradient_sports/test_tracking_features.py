"""Test Gradient Sports tracking features."""

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.tracking_features import (
    add_tracking_player_features,
    build_tracking_team_lookup,
    tracking_to_player_rows,
)

PLAYER_LIST_DTYPE = pl.List(
    pl.Struct(
        {
            "jerseyNum": pl.String,
            "confidence": pl.String,
            "visibility": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
        }
    )
)


def _write_metadata(
    metadata_path: Path,
    *,
    game_id: str,
    home_team_id: str | None,
    away_team_id: str | None,
) -> None:
    """Write one explicit raw metadata game for a lookup test."""
    metadata = [
        {
            "id": game_id,
            "homeTeam": {"id": home_team_id, "name": "Home FC"},
            "awayTeam": {"id": away_team_id, "name": "Away FC"},
        }
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _tracked_player(
    shirt_number: str,
    x_coordinate: float,
) -> dict[str, object]:
    """Create one explicit nested tracking-player observation."""
    return {
        "jerseyNum": shirt_number,
        "confidence": "HIGH",
        "visibility": "VISIBLE",
        "x": x_coordinate,
        "y": 1.0,
    }


def _tracking_frames() -> pl.LazyFrame:
    """Create two processed tracking frames for enrichment tests."""
    return pl.DataFrame(
        {
            "game_id": pl.Series([101, 101], dtype=pl.Int32),
            "framenum": [1, 2],
            "home_players_smooth": [
                [
                    _tracked_player("4", 10.0),
                    _tracked_player("8", 20.0),
                ],
                [],
            ],
            "away_players_smooth": [
                [_tracked_player("9", -10.0)],
                [_tracked_player("10", -20.0)],
            ],
        },
        schema_overrides={
            "home_players_smooth": PLAYER_LIST_DTYPE,
            "away_players_smooth": PLAYER_LIST_DTYPE,
        },
    ).lazy()


def _team_lookup() -> pl.LazyFrame:
    """Create explicit home-away team IDs for tracking tests."""
    return pl.DataFrame(
        {
            "game_id": pl.Series([101, 101], dtype=pl.Int64),
            "team_side": ["home", "away"],
            "team_id": pl.Series([10, 20], dtype=pl.Int64),
        }
    ).lazy()


def _player_lookup() -> pl.LazyFrame:
    """Create explicit shirt-number player features for tests."""
    return pl.DataFrame(
        {
            "game_id": pl.Series([101, 101, 101], dtype=pl.Int64),
            "team_id": pl.Series([10, 10, 20], dtype=pl.Int64),
            "shirt_number": pl.Series([4, 8, 9], dtype=pl.Int32),
            "player_id": pl.Series([1, 2, 3], dtype=pl.Int64),
            "opponent_id": pl.Series([20, 20, 10], dtype=pl.Int64),
            "player_position_group": [
                "Centre Back",
                "Midfielder",
                "Forward",
            ],
        }
    ).lazy()


def test_build_tracking_team_lookup_maps_both_sides(
    tmp_path: Path,
) -> None:
    """Test deterministic metadata conversion to game-side team rows.

    Behavior:
        Each metadata game produces one home and one away team key.

    Input:
        Two metadata files created in reverse numeric game order.

    Expected:
        Four typed rows sorted by game ID and home-away side key.

    Guards against:
        Swapped teams, omitted games, filesystem ordering, and string ID
        leakage into tracking joins.
    """
    _write_metadata(
        tmp_path / "202.json",
        game_id="202",
        home_team_id="30",
        away_team_id="40",
    )
    _write_metadata(
        tmp_path / "101.json",
        game_id="101",
        home_team_id="10",
        away_team_id="20",
    )

    result = build_tracking_team_lookup(tmp_path)

    expected = pl.DataFrame(
        {
            "game_id": pl.Series([101, 101, 202, 202], dtype=pl.Int64),
            "team_side": ["away", "home", "away", "home"],
            "team_id": pl.Series([20, 10, 40, 30], dtype=pl.Int64),
        }
    )
    assert_frame_equal(result, expected)


def test_build_tracking_team_lookup_rejects_null_team(
    tmp_path: Path,
) -> None:
    """Test rejection of incomplete home-away metadata.

    Behavior:
        Both tracking sides require a non-null team identifier.

    Input:
        One metadata game with a valid home team and null away team ID.

    Expected:
        A `ValueError` reporting null team IDs.

    Guards against:
        Unmatchable away-player rows caused by incomplete metadata.
    """
    _write_metadata(
        tmp_path / "101.json",
        game_id="101",
        home_team_id="10",
        away_team_id=None,
    )

    with pytest.raises(
        ValueError,
        match="Tracking team lookup contains null team IDs",
    ):
        build_tracking_team_lookup(tmp_path)


def test_tracking_to_player_rows_drops_empty_lists() -> None:
    """Test frame-list normalization without phantom player rows.

    Behavior:
        Home and away lists explode to one row per actual observation;
        empty lists add no rows.

    Input:
        Two frames containing two home observations, two away
        observations, and one empty home list.

    Expected:
        Four rows with side labels and Int32 shirt numbers, with the two
        nested list columns removed.

    Guards against:
        Null placeholder players, retained nested lists, and string
        jersey IDs that cannot join to the lookup.
    """
    result = tracking_to_player_rows(_tracking_frames()).collect()

    assert result.height == 4
    assert result["team_side"].to_list() == [
        "home",
        "home",
        "away",
        "away",
    ]
    assert result["shirt_number"].to_list() == [4, 8, 9, 10]
    assert result["shirt_number"].dtype == pl.Int32
    assert "home_players_smooth" not in result.columns
    assert "away_players_smooth" not in result.columns


def test_add_tracking_player_features_preserves_observations() -> None:
    """Test complete tracking-player enrichment and unmatched behavior.

    Behavior:
        Long tracking rows receive side-derived teams and shirt-derived
        player, opponent, and position features.

    Starting datasets:
        Two tracking frames with four player observations, a two-row
        game-side lookup, and three matching player lookup rows.

    Cardinality:
        Two consecutive many-to-one left joins: side to team, then shirt
        number to player within game and team.

    Expected columns:
        Frame fields, player observation fields, `team_id`, `player_id`,
        `opponent_id`, and `player_position_group`.

    Expected rows:
        Four rows, exactly matching the number of nested observations.

    Unmatched behavior:
        Away shirt 10 remains present with null player lookup features.

    Known values:
        Home shirt 4 maps to player 1 and `Centre Back`; away shirt 9
        maps to player 3, opponent 10, and `Forward`.

    Components:
        List normalization, metadata team join, key dtype alignment, and
        validated tracking player join.

    Guards against:
        Row multiplication, side inversion, cross-team shirt matches,
        dropped observations, and inner-join behavior.
    """
    result = add_tracking_player_features(
        _tracking_frames(),
        _team_lookup(),
        _player_lookup(),
    ).collect()

    expected = pl.DataFrame(
        {
            "game_id": pl.Series([101, 101, 101, 101], dtype=pl.Int32),
            "framenum": [1, 1, 1, 2],
            "team_side": ["home", "home", "away", "away"],
            "shirt_number": pl.Series([4, 8, 9, 10], dtype=pl.Int32),
            "confidence": ["HIGH", "HIGH", "HIGH", "HIGH"],
            "visibility": ["VISIBLE", "VISIBLE", "VISIBLE", "VISIBLE"],
            "x": [10.0, 20.0, -10.0, -20.0],
            "y": [1.0, 1.0, 1.0, 1.0],
            "team_id": pl.Series([10, 10, 20, 20], dtype=pl.Int64),
            "player_id": pl.Series([1, 2, 3, None], dtype=pl.Int64),
            "opponent_id": pl.Series([20, 20, 10, None], dtype=pl.Int64),
            "player_position_group": [
                "Centre Back",
                "Midfielder",
                "Forward",
                None,
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_add_tracking_player_features_rejects_duplicate_lookup() -> None:
    """Test tracking enrichment's many-to-one player guarantee.

    Behavior:
        A duplicated game-team-shirt key fails during lazy collection.

    Input:
        Four tracking observations and a player lookup with one matching
        row duplicated under different player IDs.

    Expected:
        A Polars `ComputeError` identifying failed m:1 validation.

    Guards against:
        Duplicate player observations caused by ambiguous shirt lookup
        records.
    """
    duplicate_lookup = pl.concat(
        [
            _player_lookup(),
            pl.DataFrame(
                {
                    "game_id": pl.Series([101], dtype=pl.Int64),
                    "team_id": pl.Series([10], dtype=pl.Int64),
                    "shirt_number": pl.Series([4], dtype=pl.Int32),
                    "player_id": pl.Series([99], dtype=pl.Int64),
                    "opponent_id": pl.Series([20], dtype=pl.Int64),
                    "player_position_group": ["Full Back"],
                }
            ).lazy(),
        ]
    )

    with pytest.raises(pl.exceptions.ComputeError, match="m:1"):
        add_tracking_player_features(
            _tracking_frames(),
            _team_lookup(),
            duplicate_lookup,
        ).collect()
