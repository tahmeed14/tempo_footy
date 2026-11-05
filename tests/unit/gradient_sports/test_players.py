"""Test Gradient Sports player-data helpers."""

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports import players as players_module
from tempoctrl.gradient_sports.players import (
    add_opponent_ids,
    add_player_lookup_features,
    build_player_game_lookup,
    read_roster_file,
    scan_player_game_lookup,
    write_player_game_lookup,
)


def _write_roster(
    roster_path: Path,
    roster_rows: list[dict[str, object]],
) -> None:
    """Write explicit roster rows for a player lookup unit test."""
    roster_path.write_text(json.dumps(roster_rows), encoding="utf-8")


def _roster_row(
    *,
    player_id: str | None,
    team_id: str | None,
    shirt_number: str | None,
    player_name: str,
    team_name: str,
    position: str = "CM",
    started: bool = True,
) -> dict[str, object]:
    """Create one explicit raw roster row for lookup tests."""
    return {
        "player": {"id": player_id, "nickname": player_name},
        "positionGroupType": position,
        "shirtNumber": shirt_number,
        "started": started,
        "team": {"id": team_id, "name": team_name},
    }


def _valid_player_lookup() -> pl.DataFrame:
    """Create an explicit valid player lookup for persistence tests."""
    return pl.DataFrame(
        {
            "game_id": pl.Series([101, 101], dtype=pl.Int64),
            "team_id": pl.Series([10, 20], dtype=pl.Int64),
            "opponent_id": pl.Series([20, 10], dtype=pl.Int64),
            "player_id": pl.Series([1, 2], dtype=pl.Int64),
            "shirt_number": pl.Series([4, 9], dtype=pl.Int32),
            "position_group_type": ["CB", "ST"],
            "player_position_group": ["Centre Back", "Forward"],
            "player_name": ["Defender", "Forward"],
            "team_name": ["Home FC", "Away FC"],
            "started": [True, True],
        }
    )


def test_read_roster_file_normalizes_and_maps_players(
    tmp_path: Path,
) -> None:
    """Test normal roster flattening and position mapping.

    Behavior:
        A roster file becomes one typed row per player, including its
        filename-derived game ID and mapped position group.

    Input:
        A game 10517 JSON roster containing an MCB and an RW from the
        two participating teams.

    Expected:
        Two rows in the documented column order with explicit IDs,
        names, starter flags, and mapped position groups.

    Guards against:
        Incorrect nested-field extraction, filename parsing, MCB
        handling, column order, and inferred identifier dtypes.
    """
    roster_path = tmp_path / "10517.json"
    _write_roster(
        roster_path,
        [
            {
                "player": {"id": "11", "nickname": "Centre Player"},
                "positionGroupType": "MCB",
                "shirtNumber": "19",
                "started": True,
                "team": {"id": "364", "name": "Argentina"},
            },
            {
                "player": {"id": "4622", "nickname": "Wide Player"},
                "positionGroupType": "RW",
                "shirtNumber": "20",
                "started": False,
                "team": {"id": "363", "name": "France"},
            },
        ],
    )

    result = read_roster_file(roster_path)

    expected = pl.DataFrame(
        {
            "game_id": pl.Series([10517, 10517], dtype=pl.Int64),
            "team_id": pl.Series([364, 363], dtype=pl.Int64),
            "player_id": pl.Series([11, 4622], dtype=pl.Int64),
            "shirt_number": pl.Series([19, 20], dtype=pl.Int32),
            "position_group_type": ["MCB", "RW"],
            "player_position_group": ["Centre Back", "Winger"],
            "player_name": ["Centre Player", "Wide Player"],
            "team_name": ["Argentina", "France"],
            "started": [True, False],
        }
    )
    assert_frame_equal(result, expected)


def test_read_roster_file_preserves_null_position(
    tmp_path: Path,
) -> None:
    """Test null position handling without inventing a category.

    Behavior:
        A null source position remains null after position mapping.

    Input:
        One otherwise valid roster row with a null
        `positionGroupType` value.

    Expected:
        Both position columns are null strings and the row is retained.

    Guards against:
        Dropped players or fabricated categories for missing source
        data.
    """
    roster_path = tmp_path / "101.json"
    _write_roster(
        roster_path,
        [
            {
                "player": {"id": "7", "nickname": "Unknown Role"},
                "positionGroupType": None,
                "shirtNumber": "12",
                "started": False,
                "team": {"id": "20", "name": "Example FC"},
            }
        ],
    )

    result = read_roster_file(roster_path)

    assert result.height == 1
    assert result["position_group_type"].to_list() == [None]
    assert result["position_group_type"].dtype == pl.String
    assert result["player_position_group"].to_list() == [None]
    assert result["player_position_group"].dtype == pl.String


def test_read_roster_file_rejects_unmapped_position(
    tmp_path: Path,
) -> None:
    """Test explicit failure for an unexpected position code.

    Behavior:
        Non-null position codes absent from `POSITION_MAPPER` fail with
        the offending value in the error message.

    Input:
        One valid player row whose position code is `SW`.

    Expected:
        A `ValueError` naming `SW` instead of an output table.

    Guards against:
        Silent null feature values when the provider adds or changes a
        position category.
    """
    roster_path = tmp_path / "101.json"
    _write_roster(
        roster_path,
        [
            {
                "player": {"id": "7", "nickname": "Sweeper"},
                "positionGroupType": "SW",
                "shirtNumber": "5",
                "started": True,
                "team": {"id": "20", "name": "Example FC"},
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="Unmapped positionGroupType values: SW",
    ):
        read_roster_file(roster_path)


def test_read_roster_file_rejects_non_numeric_filename(
    tmp_path: Path,
) -> None:
    """Test rejection of a roster without a numeric game filename.

    Behavior:
        The reader requires the filename stem to be the game ID.

    Input:
        A path named `final.json`; file contents are unnecessary because
        filename validation happens first.

    Expected:
        A `ValueError` naming the invalid roster filename.

    Guards against:
        Creating lookup rows with a missing or incorrectly inferred
        game identifier.
    """
    roster_path = tmp_path / "final.json"

    with pytest.raises(
        ValueError,
        match="Roster filename must be a numeric game ID: final.json",
    ):
        read_roster_file(roster_path)


def test_build_player_game_lookup_combines_and_sorts_games(
    tmp_path: Path,
) -> None:
    """Test deterministic construction of a multi-game lookup.

    Behavior:
        All roster files are combined and sorted by the documented
        player lookup key.

    Input:
        Two roster files created out of order, each containing one
        player from each of two teams.

    Expected:
        Four explicitly ordered, typed player-game rows with mapped
        position groups.

    Guards against:
        Processing only one game, filesystem-order output, changed
        schemas, or lost roster values during concatenation.
    """
    _write_roster(
        tmp_path / "202.json",
        [
            _roster_row(
                player_id="22",
                team_id="4",
                shirt_number="9",
                player_name="Second Forward",
                team_name="Fourth FC",
                position="ST",
            ),
            _roster_row(
                player_id="21",
                team_id="3",
                shirt_number="1",
                player_name="Second Keeper",
                team_name="Third FC",
                position="GK",
            ),
        ],
    )
    _write_roster(
        tmp_path / "101.json",
        [
            _roster_row(
                player_id="12",
                team_id="2",
                shirt_number="11",
                player_name="First Winger",
                team_name="Second FC",
                position="LW",
                started=False,
            ),
            _roster_row(
                player_id="11",
                team_id="1",
                shirt_number="4",
                player_name="First Defender",
                team_name="First FC",
                position="CB",
            ),
        ],
    )

    result = build_player_game_lookup(tmp_path)

    expected = pl.DataFrame(
        {
            "game_id": pl.Series(
                [101, 101, 202, 202],
                dtype=pl.Int64,
            ),
            "team_id": pl.Series([1, 2, 3, 4], dtype=pl.Int64),
            "opponent_id": pl.Series([2, 1, 4, 3], dtype=pl.Int64),
            "player_id": pl.Series([11, 12, 21, 22], dtype=pl.Int64),
            "shirt_number": pl.Series([4, 11, 1, 9], dtype=pl.Int32),
            "position_group_type": ["CB", "LW", "GK", "ST"],
            "player_position_group": [
                "Centre Back",
                "Winger",
                "Goalkeeper",
                "Forward",
            ],
            "player_name": [
                "First Defender",
                "First Winger",
                "Second Keeper",
                "Second Forward",
            ],
            "team_name": [
                "First FC",
                "Second FC",
                "Third FC",
                "Fourth FC",
            ],
            "started": [True, False, True, True],
        }
    )
    assert_frame_equal(result, expected)


def test_add_opponent_ids_preserves_rows_and_assigns_opponents() -> None:
    """Test opponent enrichment for teams with uneven roster sizes.

    Behavior:
        Every player receives the other team in its game as
        `opponent_id` without changing input order or row count.

    Input:
        One valid two-team game with two team 10 players and one team
        20 player.

    Expected:
        Three output rows; team 10 maps to 20 twice, team 20 maps to 10,
        and `opponent_id` has the Int64 dtype.

    Guards against:
        Accidental many-to-many joins, dropped players, self-opponents,
        and inconsistent opponent identifier types.
    """
    df_lookup = pl.DataFrame(
        {
            "game_id": pl.Series([101, 101, 101], dtype=pl.Int64),
            "team_id": pl.Series([10, 10, 20], dtype=pl.Int64),
            "player_id": pl.Series([1, 2, 3], dtype=pl.Int64),
            "shirt_number": pl.Series([4, 8, 9], dtype=pl.Int32),
            "position_group_type": ["CB", "CM", "ST"],
            "player_position_group": [
                "Centre Back",
                "Midfielder",
                "Forward",
            ],
            "player_name": ["Defender", "Midfielder", "Forward"],
            "team_name": ["Home FC", "Home FC", "Away FC"],
            "started": [True, True, True],
        }
    )

    result = add_opponent_ids(df_lookup)

    assert result.height == 3
    assert result["player_id"].to_list() == [1, 2, 3]
    assert result["opponent_id"].to_list() == [20, 20, 10]
    assert result["opponent_id"].dtype == pl.Int64


def test_build_player_game_lookup_rejects_null_identifier(
    tmp_path: Path,
) -> None:
    """Test rejection of null identifiers in a combined lookup.

    Behavior:
        Required game, team, player, and shirt identifiers must be
        non-null before the lookup can be used for joins.

    Input:
        A two-team roster whose first player has a null player ID.

    Expected:
        A `ValueError` reporting one null `player_id`.

    Guards against:
        Ambiguous lookup rows that can never match downstream player
        data reliably.
    """
    _write_roster(
        tmp_path / "101.json",
        [
            _roster_row(
                player_id=None,
                team_id="1",
                shirt_number="4",
                player_name="Missing ID",
                team_name="First FC",
            ),
            _roster_row(
                player_id="12",
                team_id="2",
                shirt_number="8",
                player_name="Valid Player",
                team_name="Second FC",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Player lookup contains null identifiers: player_id=1",
    ):
        build_player_game_lookup(tmp_path)


def test_build_player_game_lookup_requires_two_teams(
    tmp_path: Path,
) -> None:
    """Test the two-team invariant for every game.

    Behavior:
        A player lookup game must contain exactly two unique teams.

    Input:
        One roster file containing two players from only team 1.

    Expected:
        A `ValueError` reporting that game 101 has one team.

    Guards against:
        Opponent derivation from incomplete or malformed roster data.
    """
    _write_roster(
        tmp_path / "101.json",
        [
            _roster_row(
                player_id="11",
                team_id="1",
                shirt_number="4",
                player_name="First Player",
                team_name="First FC",
            ),
            _roster_row(
                player_id="12",
                team_id="1",
                shirt_number="8",
                player_name="Second Player",
                team_name="First FC",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=("Each game must contain exactly two teams; found: 101=1"),
    ):
        build_player_game_lookup(tmp_path)


def test_build_player_game_lookup_rejects_duplicate_player_key(
    tmp_path: Path,
) -> None:
    """Test uniqueness of the canonical player lookup key.

    Behavior:
        `(game_id, team_id, player_id)` must identify one roster row.

    Input:
        A two-team game where team 1 repeats player 11 under two shirt
        numbers.

    Expected:
        A `ValueError` reporting the duplicated `(101, 1, 11)` key.

    Guards against:
        Many-to-many joins and duplicated downstream player records.
    """
    _write_roster(
        tmp_path / "101.json",
        [
            _roster_row(
                player_id="11",
                team_id="1",
                shirt_number="4",
                player_name="Repeated Player",
                team_name="First FC",
            ),
            _roster_row(
                player_id="11",
                team_id="1",
                shirt_number="5",
                player_name="Repeated Player",
                team_name="First FC",
            ),
            _roster_row(
                player_id="21",
                team_id="2",
                shirt_number="8",
                player_name="Other Player",
                team_name="Second FC",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Duplicate player lookup key \(game_id, team_id, "
            r"player_id\): \(101, 1, 11\)"
        ),
    ):
        build_player_game_lookup(tmp_path)


def test_build_player_game_lookup_rejects_duplicate_tracking_key(
    tmp_path: Path,
) -> None:
    """Test uniqueness of the tracking-specific player key.

    Behavior:
        `(game_id, team_id, shirt_number)` must identify one player.

    Input:
        A two-team game where two team 1 players share shirt number 4.

    Expected:
        A `ValueError` reporting the duplicated `(101, 1, 4)` key.

    Guards against:
        Ambiguous tracking enrichment and accidental many-to-many
        joins by jersey number.
    """
    _write_roster(
        tmp_path / "101.json",
        [
            _roster_row(
                player_id="11",
                team_id="1",
                shirt_number="4",
                player_name="First Player",
                team_name="First FC",
            ),
            _roster_row(
                player_id="12",
                team_id="1",
                shirt_number="4",
                player_name="Second Player",
                team_name="First FC",
            ),
            _roster_row(
                player_id="21",
                team_id="2",
                shirt_number="8",
                player_name="Other Player",
                team_name="Second FC",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Duplicate player lookup key \(game_id, team_id, "
            r"shirt_number\): \(101, 1, 4\)"
        ),
    ):
        build_player_game_lookup(tmp_path)


def test_write_and_scan_player_game_lookup_round_trip(
    tmp_path: Path,
) -> None:
    """Test a player lookup Parquet write and lazy scan round trip.

    Behavior:
        The writer creates missing parent directories and the scanner
        lazily reads the resulting Parquet file.

    Input:
        A two-row valid lookup and a nested temporary output path.

    Expected:
        The returned path exists, scanning stays lazy, and collection
        reproduces every explicit value, column, and dtype.

    Guards against:
        Incorrect artifact paths, eager loading, schema drift, and
        lossy Parquet serialization.
    """
    expected = _valid_player_lookup()
    output_path = tmp_path / "processed" / "player_lookup.parquet"

    result_path = write_player_game_lookup(expected, output_path)
    result_lazy = scan_player_game_lookup(result_path)

    assert result_path == output_path
    assert result_path.is_file()
    assert isinstance(result_lazy, pl.LazyFrame)
    assert_frame_equal(result_lazy.collect(), expected)


def test_write_player_game_lookup_controls_overwrite(
    tmp_path: Path,
) -> None:
    """Test preservation and replacement of an existing lookup file.

    Behavior:
        An existing artifact is preserved by default and replaced only
        when `overwrite=True` is supplied.

    Input:
        An original lookup and a replacement with changed player names,
        both written to the same temporary path.

    Expected:
        The default second write retains the original values; an
        explicit overwrite stores the replacement values.

    Guards against:
        Accidental destruction of processed lookup data or an ignored
        overwrite request.
    """
    original = _valid_player_lookup()
    replacement = original.with_columns(
        pl.lit("Changed Player").alias("player_name")
    )
    output_path = tmp_path / "player_lookup.parquet"

    write_player_game_lookup(original, output_path)
    write_player_game_lookup(replacement, output_path)

    assert_frame_equal(pl.read_parquet(output_path), original)

    write_player_game_lookup(
        replacement,
        output_path,
        overwrite=True,
    )
    assert_frame_equal(pl.read_parquet(output_path), replacement)


def test_scan_player_game_lookup_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Test a descriptive failure when the lookup artifact is absent.

    Behavior:
        The scanner checks for the requested Parquet file before
        constructing a lazy query.

    Input:
        A temporary path where no player lookup has been written.

    Expected:
        A `FileNotFoundError` containing the complete requested path.

    Guards against:
        Deferred, unclear Polars errors when a pipeline is configured
        with a missing lookup artifact.
    """
    lookup_path = tmp_path / "missing_lookup.parquet"

    with pytest.raises(
        FileNotFoundError,
        match=f"Player lookup file does not exist: {lookup_path}",
    ):
        scan_player_game_lookup(lookup_path)


def test_add_player_lookup_features_preserves_downstream_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test downstream enrichment with matches and unmatched rows.

    Behavior:
        The helper lazily adds opponent and position features while
        retaining every downstream record in its original order.

    Starting datasets:
        Four event-like rows with Int32 game and team IDs, plus a
        two-row Int64-keyed player lookup supplied by the default scan.

    Cardinality:
        A many-to-one left join from downstream players to lookup rows.

    Expected columns:
        The four input columns followed by `opponent_id` and
        `player_position_group`.

    Expected rows:
        Four output rows, equal to the downstream input row count.

    Unmatched behavior:
        An unknown player and a null team retain their rows with null
        lookup features.

    Known values:
        Player 1 maps to opponent 20 and `Centre Back`; player 2 maps to
        opponent 10 and `Forward`.

    Components:
        Default lookup scanning, key dtype alignment, feature selection,
        and the validated lazy join.

    Guards against:
        Inner joins, row multiplication, key dtype failures, and
        incorrect unmatched-record handling.
    """
    df_in = pl.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "game_id": pl.Series([101, 101, 101, 101], dtype=pl.Int32),
            "team_id": pl.Series([10, 20, 10, None], dtype=pl.Int32),
            "player_id": pl.Series([1, 2, 99, 1], dtype=pl.Int64),
        }
    ).lazy()
    df_lookup = _valid_player_lookup().lazy()
    monkeypatch.setattr(
        players_module,
        "scan_player_game_lookup",
        lambda: df_lookup,
    )

    result_lazy = add_player_lookup_features(df_in)
    result = result_lazy.collect()

    expected = pl.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "game_id": pl.Series([101, 101, 101, 101], dtype=pl.Int32),
            "team_id": pl.Series([10, 20, 10, None], dtype=pl.Int32),
            "player_id": pl.Series([1, 2, 99, 1], dtype=pl.Int64),
            "opponent_id": pl.Series([20, 10, None, None], dtype=pl.Int64),
            "player_position_group": [
                "Centre Back",
                "Forward",
                None,
                None,
            ],
        }
    )
    assert isinstance(result_lazy, pl.LazyFrame)
    assert_frame_equal(result, expected)


def test_add_player_lookup_features_rejects_duplicate_lookup_key() -> None:
    """Test enforcement of many-to-one lookup cardinality.

    Behavior:
        Collection fails when one player key maps to multiple feature
        rows.

    Input:
        One downstream player and a lookup containing that key twice.

    Expected:
        A Polars `ComputeError` identifying the failed m:1 validation.

    Guards against:
        Accidental many-to-many joins that duplicate downstream rows.
    """
    df_in = pl.DataFrame(
        {
            "game_id": [101],
            "team_id": [10],
            "player_id": [1],
        }
    ).lazy()
    df_lookup = pl.DataFrame(
        {
            "game_id": [101, 101],
            "team_id": [10, 10],
            "player_id": [1, 1],
            "opponent_id": [20, 20],
            "player_position_group": ["Centre Back", "Full Back"],
        }
    ).lazy()

    with pytest.raises(pl.exceptions.ComputeError, match="m:1"):
        add_player_lookup_features(df_in, df_lookup).collect()


def test_add_player_lookup_features_requires_input_keys() -> None:
    """Test descriptive validation of downstream join keys.

    Behavior:
        Enrichment rejects downstream data missing a composite key.

    Input:
        A frame containing game and team IDs but no `player_id`.

    Expected:
        A `ValueError` naming the missing player key.

    Guards against:
        Deferred Polars errors that obscure an invalid input schema.
    """
    df_in = pl.DataFrame(
        {
            "game_id": [101],
            "team_id": [10],
        }
    ).lazy()

    with pytest.raises(
        ValueError,
        match="Downstream data is missing player lookup keys: player_id",
    ):
        add_player_lookup_features(df_in, _valid_player_lookup().lazy())


def test_add_player_lookup_features_rejects_existing_features() -> None:
    """Test protection against ambiguous feature replacement.

    Behavior:
        Enrichment rejects input that already has a lookup-owned
        feature.

    Input:
        A valid player key and an existing `opponent_id` column.

    Expected:
        A `ValueError` naming the conflicting feature column.

    Guards against:
        Silent suffix columns or accidental replacement of prior
        feature calculations.
    """
    df_in = pl.DataFrame(
        {
            "game_id": [101],
            "team_id": [10],
            "player_id": [1],
            "opponent_id": [999],
        }
    ).lazy()

    with pytest.raises(
        ValueError,
        match="Downstream data already contains player features: opponent_id",
    ):
        add_player_lookup_features(df_in, _valid_player_lookup().lazy())
