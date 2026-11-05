"""Test possession-null investigation helpers."""

from pathlib import Path

import polars as pl
import pytest

from tempoctrl.gradient_sports.dev_investigate_poss_nulls import (
    find_verified_possession_gaps,
    select_shortest_gap_per_type,
    summarize_possession_gaps_by_match,
    sync_possession_gap_examples,
)


def _gap_frames() -> pl.DataFrame:
    """Build event types with valid, invalid, and tied gap candidates.
    """
    return pl.DataFrame(
        {
            "game_id": [1] * 16,
            "period": [1] * 7 + [2] * 3 + [3] * 3 + [4] * 3,
            "framenum": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
                3,
            ],
            "event_number": [
                1,
                None,
                None,
                2,
                3,
                None,
                4,
                5,
                None,
                6,
                7,
                None,
                8,
                9,
                None,
                10,
            ],
            "match_team_possession_id": [
                "team_a",
                None,
                None,
                "team_a",
                "team_b",
                None,
                "team_b",
                "team_c",
                None,
                "team_c",
                "team_d",
                "team_d",
                "team_d",
                "team_e",
                None,
                "team_e",
            ],
            "match_team_player_possession_id": [
                "player_a",
                None,
                None,
                "player_a",
                "player_b",
                None,
                "player_b",
                "player_c",
                None,
                "player_c",
                "player_d",
                None,
                "player_d",
                "player_e",
                None,
                "player_e",
            ],
            "possession_event_type": [
                "BC",
                None,
                None,
                "PA",
                "BC",
                None,
                "PA",
                "PA",
                None,
                "CR",
                "CH",
                None,
                "CH",
                None,
                None,
                "IT",
            ],
        },
        schema_overrides={
            "game_id": pl.Int32,
            "period": pl.Int64,
            "framenum": pl.Int64,
            "event_number": pl.UInt32,
            "match_team_possession_id": pl.String,
            "match_team_player_possession_id": pl.String,
            "possession_event_type": pl.String,
        },
    )


def _other_match_frames() -> pl.DataFrame:
    """Build one match with a BC gap and one match without gaps."""
    return pl.DataFrame(
        {
            "game_id": [2, 2, 2, 3, 3],
            "period": [1, 1, 1, 1, 1],
            "framenum": [1, 2, 3, 1, 2],
            "event_number": [1, None, 2, 1, 2],
            "match_team_possession_id": [
                "team_f",
                None,
                "team_f",
                "team_g",
                "team_h",
            ],
            "match_team_player_possession_id": [
                "player_f",
                None,
                "player_f",
                "player_g",
                "player_h",
            ],
            "possession_event_type": ["BC", None, "PA", "IT", "PA"],
        },
        schema_overrides={
            "game_id": pl.Int32,
            "period": pl.Int64,
            "framenum": pl.Int64,
            "event_number": pl.UInt32,
            "match_team_possession_id": pl.String,
            "match_team_player_possession_id": pl.String,
            "possession_event_type": pl.String,
        },
    )


def test_selects_shortest_verified_gap_for_each_event_type() -> None:
    """Select the shortest fully null gap for every leading type.

    Behavior:
        Verify complete endpoint IDs and select one shortest gap per
        type.

    Input:
        Two valid BC gaps, valid PA and null-type gaps, and one CH
        interval
        where only the player ID is null.

    Expected:
        The one-frame BC gap wins, PA and null are retained, and CH is
        excluded because both IDs were not null in its interior.

    Guards against:
        Selecting partial-null gaps, omitting null event types, or
        choosing
        a longer example for an event category.
    """
    candidates = find_verified_possession_gaps(_gap_frames().lazy()).collect()

    result = select_shortest_gap_per_type(candidates)

    assert result["possession_type_key"].to_list() == [
        "BC",
        "PA",
        "null",
    ]
    assert result["leading_frame"].to_list() == [5, 1, 1]
    assert result["trailing_frame"].to_list() == [7, 3, 3]
    assert result["null_gap_frames"].to_list() == [1, 1, 1]


def test_summarizes_possession_counts_and_gaps_for_every_match() -> None:
    """Report total possession IDs and gap involvement by match.

    Behavior:
        Build summary rows for matches with and without verified gaps.

    Input:
        Match 1 has four verified gaps, match 2 has one, and match 3 has
        no gap but two distinct team and player possession identifiers.

    Expected:
        All three matches appear with total unique-ID counts and
        separate
        unique-ID counts for the verified gaps.

    Guards against:
        Omitting clean matches or confusing all possessions with the
        subset of possessions involved in null gaps.
    """
    source = pl.concat(
        [_gap_frames(), _other_match_frames()],
        how="vertical",
    ).lazy()
    gaps = find_verified_possession_gaps(source).collect()

    result = summarize_possession_gaps_by_match(source, gaps)

    assert result["game_id"].to_list() == [1, 2, 3]
    assert result["unique_team_possession_ids"].to_list() == [5, 1, 2]
    assert result["unique_player_possession_ids"].to_list() == [5, 1, 2]
    assert result["verified_gap_count"].to_list() == [4, 1, 0]
    assert result["gap_event_types"].to_list() == [
        "BC, PA, null",
        "BC",
        "",
    ]
    assert result["gap_team_possession_ids"].to_list() == [4, 1, 0]
    assert result["gap_player_possession_ids"].to_list() == [4, 1, 0]


def test_sync_creates_preserves_replaces_and_removes_examples(
    tmp_path: Path,
) -> None:
    """Synchronize managed examples without needless rewrites.

    Behavior:
        Create missing files, retain exact matches, replace changed
        files,
        and delete stale managed categories.

    Input:
        A temporary integrated match, then a deliberately corrupted BC
        example and an unrelated stale managed example.

    Expected:
        First run creates three categories, second marks them unchanged,
        and the final run repairs BC and removes the stale file.

    Guards against:
        Rewriting identical examples, retaining outdated diagnostics, or
        overwriting unrelated files in the investigation directory.
    """
    input_dir = tmp_path / "integrated"
    output_dir = tmp_path / "investigate"
    input_dir.mkdir()
    _gap_frames().write_parquet(input_dir / "1.parquet")
    _other_match_frames().write_parquet(input_dir / "2.parquet")

    first_statuses, summary = sync_possession_gap_examples(
        1,
        input_dir,
        output_dir,
    )

    assert first_statuses == {
        "BC": "created",
        "PA": "created",
        "null": "created",
    }
    assert summary["game_id"].to_list() == [1, 2, 3]
    assert (output_dir / "possession_gap_summary_by_match.parquet").exists()
    bc_path = output_dir / "bc_shortest_gap.parquet"
    original_bc = pl.read_parquet(bc_path)
    assert original_bc["game_id"].unique().to_list() == [1]
    assert original_bc["framenum"].to_list() == [5, 6, 7]
    assert original_bc["gap_frame_role"].to_list() == [
        "leading_id_observed",
        "internal_null_gap",
        "trailing_id_observed",
    ]

    second_statuses, _ = sync_possession_gap_examples(
        1,
        input_dir,
        output_dir,
    )

    assert set(second_statuses.values()) == {"unchanged"}

    original_bc.with_columns(
        pl.lit("corrupt").alias("developer_note")
    ).write_parquet(bc_path)
    stale_path = output_dir / "stale_shortest_gap.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(stale_path)
    unrelated_path = output_dir / "developer_notes.parquet"
    pl.DataFrame({"note": ["keep"]}).write_parquet(unrelated_path)

    final_statuses, _ = sync_possession_gap_examples(
        1,
        input_dir,
        output_dir,
    )

    assert final_statuses["BC"] == "replaced"
    assert final_statuses["stale"] == "removed"
    assert pl.read_parquet(bc_path).equals(original_bc)
    assert not stale_path.exists()
    assert unrelated_path.exists()


def test_rejects_unknown_example_match(tmp_path: Path) -> None:
    """Reject a requested example match absent from integrated data.

    Behavior:
        Validate ``--m`` against all analyzed integrated matches.

    Input:
        Integrated data containing match 1 and a request for match 999.

    Expected:
        A ``ValueError`` reports the missing match and available match.

    Guards against:
        Silently deleting examples because of a mistyped match ID.
    """
    input_dir = tmp_path / "integrated"
    input_dir.mkdir()
    _gap_frames().write_parquet(input_dir / "1.parquet")

    with pytest.raises(ValueError, match="Available matches: 1"):
        sync_possession_gap_examples(
            999,
            input_dir,
            tmp_path / "investigate",
        )


def test_rejects_missing_gap_columns() -> None:
    """Reject data that cannot verify possession-ID boundaries.

    Behavior:
        Validate the investigation schema before building the query.

    Input:
        A LazyFrame containing only a game identifier.

    Expected:
        A clear ``ValueError`` identifies missing gap columns.

    Guards against:
        Deferred Polars errors that obscure malformed integrated data.
    """
    incomplete = pl.DataFrame({"game_id": [1]}).lazy()

    with pytest.raises(ValueError, match="Missing possession-gap columns"):
        find_verified_possession_gaps(incomplete)
