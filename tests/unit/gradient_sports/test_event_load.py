"""Test per-match event writes and batch possession-lookup updates."""

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports import event_load


@pytest.fixture
def events() -> pl.DataFrame:
    """Provide interleaved events with repeated and missing possessions.

    Behavior:
        Represent a processed batch with rows that must all be written.

    Input:
        Five events from matches 101 and 202, including two annotations
        of possession 101_p1 and one null-possession row per match.

    Expected:
        Three rows belong to match 101 and two to 202. Their lookup has
        two distinct identified possessions and excludes the null IDs.

    Guards against:
        Fixtures masking deduplication, reordering, or dropped pauses.
    """
    return pl.DataFrame(
        {
            "game_id": [202, 101, 101, 202, 101],
            "match_team_player_possession_id": [
                "202_p2", "101_p1", "101_p1", None, None,
            ],
            "player_id": [22, 11, 11, None, None],
            "playername": ["Two", "One", "One", None, None],
            "team_id": [2, 1, 1, None, None],
            "teamname": ["Beta", "Alpha", "Alpha", None, None],
            "event_number": pl.Series([2, 1, 3, 4, 5], dtype=pl.UInt32),
            "marker": ["first", "second", "third", "fourth", "fifth"],
        }
    )


@pytest.fixture
def existing_lookup() -> pl.DataFrame:
    """Provide old possessions and metadata for an unrelated match.

    Behavior:
        Represent a shared lookup before selected matches are replaced.

    Input:
        One stale possession each for matches 101 and 202, plus an
        enriched possession for match 303 absent from current metadata.

    Expected:
        A three-row lookup with all ten existing columns and stable
        integer, string, and Boolean dtypes.

    Guards against:
        Tests overlooking stale selected rows or lost retained metadata.
    """
    return pl.DataFrame(
        {
            "game_id": [101, 202, 303],
            "match_team_player_possession_id": [
                "101_old", "202_old", "303_p3",
            ],
            "player_id": [11, 22, 33],
            "playername": ["Old One", "Old Two", "Three"],
            "team_id": [1, 2, 3],
            "teamname": ["Alpha", "Beta", "Keep"],
            "opponent_id": [9, 8, 7],
            "player_possession_group": ["Old", "Old", "Winger"],
            "started": [False, True, True],
            "opposition_teamname": ["Old Gamma", "Old Delta", "Kept"],
        }
    )


@pytest.fixture
def player_lookup(tmp_path: Path) -> Path:
    """Write metadata for both selected matches and their opponents.

    Behavior:
        Supply player-game and opposing-team metadata for enrichment.

    Input:
        Players 11 and 22 and their opposing teams in matches 101 and
        202, with different position groups and starter flags.

    Expected:
        One Parquet file; player 11 maps to Forward, starter, Gamma,
        while player 22 maps to Midfielder, substitute, Delta.

    Guards against:
        Dependence on private metadata or missing opponent-name rows.
    """
    path = tmp_path / "player_game_lookup.parquet"
    pl.DataFrame(
        {
            "game_id": [101, 101, 202, 202],
            "team_id": [1, 9, 2, 8],
            "opponent_id": [9, 1, 8, 2],
            "player_id": [11, 99, 22, 88],
            "team_name": ["Alpha", "Gamma", "Beta", "Delta"],
            "player_position_group": [
                "Forward", "Goalkeeper", "Midfielder", "Goalkeeper",
            ],
            "started": [True, True, False, True],
        }
    ).write_parquet(path)
    return path


def test_load_events_partitions_batch_and_updates_lookup_once(
    tmp_path: Path,
    events: pl.DataFrame,
    existing_lookup: pl.DataFrame,
    player_lookup: Path,
) -> None:
    """Write complete match files and upsert the shared lookup once.

    Starting datasets:
        Five interleaved events from two selected matches, an existing
        three-match lookup, player metadata for the selected matches,
        and an unrelated event file for match 303.

    Expected:
        Event files contain three rows for 101 and two for 202. All
        source columns, dtypes, and order remain, including null IDs.
        The metadata join is many-to-one and keeps unmatched rows. The
        lookup is updated and enriched once, has three rows and ten
        columns, and retains match 303's original metadata. Duplicate
        event annotations yield one possession row; null IDs yield none.
        Paths are sorted event paths followed by the shared lookup.

    Guards against:
        Per-match lookup rewrites, row loss or multiplication, stale
        possessions, incorrect enrichment, or unrelated output changes.
    """
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    unrelated_event_path = output_dir / "303.parquet"
    pl.DataFrame({"marker": ["keep"]}).write_parquet(unrelated_event_path)
    unrelated_bytes = unrelated_event_path.read_bytes()
    lookup_path = tmp_path / "lookup.parquet"
    existing_lookup.write_parquet(lookup_path)
    pl.DataFrame({"marker": ["stale"]}).write_parquet(
        output_dir / "101.parquet"
    )

    with (
        patch.object(
            event_load,
            "write_match_possession_lookup",
            wraps=event_load.write_match_possession_lookup,
        ) as writer,
        patch.object(
            event_load,
            "add_player_metadata",
            wraps=event_load.add_player_metadata,
        ) as metadata,
    ):
        paths = event_load.load_events(
            events,
            output_dir=str(output_dir),
            possession_lookup_path=str(lookup_path),
            player_lookup_path=str(player_lookup),
        )

    assert paths == (
        output_dir / "101.parquet", output_dir / "202.parquet", lookup_path,
    )
    assert writer.call_count == 1
    assert writer.call_args.args[0] is events
    assert metadata.call_count == 1
    assert metadata.call_args.args[0].height == 2
    for game_id in (101, 202):
        assert_frame_equal(
            pl.read_parquet(output_dir / f"{game_id}.parquet"),
            events.filter(pl.col("game_id") == game_id),
        )
    assert unrelated_event_path.read_bytes() == unrelated_bytes
    expected = pl.DataFrame(
        {
            "game_id": [101, 202, 303],
            "match_team_player_possession_id": ["101_p1", "202_p2", "303_p3"],
            "player_id": [11, 22, 33],
            "playername": ["One", "Two", "Three"],
            "team_id": [1, 2, 3],
            "teamname": ["Alpha", "Beta", "Keep"],
            "opponent_id": [9, 8, 7],
            "player_possession_group": ["Forward", "Midfielder", "Winger"],
            "started": [True, False, True],
            "opposition_teamname": ["Gamma", "Delta", "Kept"],
        }
    )
    assert_frame_equal(pl.read_parquet(lookup_path), expected)


def test_load_events_batch_matches_separate_writes_and_is_idempotent(
    tmp_path: Path,
    events: pl.DataFrame,
    existing_lookup: pl.DataFrame,
    player_lookup: Path,
) -> None:
    """Produce the same files for batch, separate, and repeated writes.

    Starting datasets:
        The same five events and seeded three-match lookup in two
        temporary output trees, with identical player metadata.

    Expected:
        Batched writing and two separate single-match writes produce
        equal event files and a three-row, ten-column lookup. Metadata
        joins remain many-to-one; null event IDs are retained in event
        files and excluded from lookups. Repeating the batch does not
        add possessions. Single-match calls return two paths.

    Guards against:
        Batch-specific output changes, duplicate upserts, lost unrelated
        matches, or breaking the existing single-match return contract.
    """
    batch_dir = tmp_path / "batch"
    separate_dir = tmp_path / "separate"
    for root in (batch_dir, separate_dir):
        root.mkdir()
        existing_lookup.write_parquet(root / "lookup.parquet")
    batch_paths = event_load.load_events(
        events,
        output_dir=batch_dir / "events",
        possession_lookup_path=batch_dir / "lookup.parquet",
        player_lookup_path=player_lookup,
    )
    for game_id in (101, 202):
        paths = event_load.load_events(
            events.filter(pl.col("game_id") == game_id),
            output_dir=separate_dir / "events",
            possession_lookup_path=separate_dir / "lookup.parquet",
            player_lookup_path=player_lookup,
        )
        assert paths == (
            separate_dir / "events" / f"{game_id}.parquet",
            separate_dir / "lookup.parquet",
        )
        assert_frame_equal(
            pl.read_parquet(batch_dir / "events" / f"{game_id}.parquet"),
            pl.read_parquet(paths[0]),
        )
    expected_lookup = pl.read_parquet(separate_dir / "lookup.parquet")
    assert_frame_equal(pl.read_parquet(batch_paths[-1]), expected_lookup)

    assert event_load.load_events(
        events,
        output_dir=batch_dir / "events",
        possession_lookup_path=batch_dir / "lookup.parquet",
        player_lookup_path=player_lookup,
    ) == batch_paths
    assert_frame_equal(pl.read_parquet(batch_paths[-1]), expected_lookup)


def test_load_events_preserves_default_single_match_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: pl.DataFrame,
) -> None:
    """Keep the existing default destinations for single-match callers.

    Behavior:
        Accept a processed DataFrame without new keyword arguments.

    Input:
        Three match-101 events in an isolated working directory, without
        optional player metadata or an existing lookup.

    Expected:
        The original two relative paths, three unchanged event rows,
        and one possession row with the six identity columns.

    Guards against:
        Requiring caller changes or introducing new output locations.
    """
    monkeypatch.chdir(tmp_path)
    selected = events.filter(pl.col("game_id") == 101)

    paths = event_load.load_events(selected)

    assert paths == (
        Path("data/processed/gradient_sports/events/101.parquet"),
        Path(
            "data/curated/gradient_sports/possession_lookup/"
            "match_possession_lookup.parquet"
        ),
    )
    assert_frame_equal(pl.read_parquet(paths[0]), selected)
    lookup = pl.read_parquet(paths[1])
    assert lookup.shape == (1, 6)
    assert lookup["match_team_player_possession_id"].to_list() == ["101_p1"]


@pytest.mark.parametrize("metadata_exists", [False, True])
def test_load_events_preserves_unmatched_and_unrelated_metadata(
    tmp_path: Path,
    events: pl.DataFrame,
    existing_lookup: pl.DataFrame,
    player_lookup: Path,
    metadata_exists: bool,
) -> None:
    """Retain unmatched incoming rows and unrelated existing metadata.

    Starting datasets:
        Two match-202 events, an enriched lookup row for match 303, and
        metadata containing only match 101 or no metadata file at all.

    Expected:
        Two unchanged event rows and a two-row, ten-column lookup. The
        optional many-to-one join retains possession 202_p2 with null
        enrichment. Match 303 retains its group, starter flag, and
        opponent name despite being absent from the metadata source.

    Guards against:
        Inner joins, dropped possessions, or stripping metadata from
        matches outside the selected batch.
    """
    lookup_path = tmp_path / "lookup.parquet"
    retained = existing_lookup.filter(pl.col("game_id") == 303)
    retained.write_parquet(lookup_path)
    if metadata_exists:
        pl.read_parquet(player_lookup).filter(
            pl.col("game_id") == 101
        ).write_parquet(player_lookup)
    else:
        player_lookup.unlink()
    selected = events.filter(pl.col("game_id") == 202)

    paths = event_load.load_events(
        selected,
        output_dir=tmp_path / "events",
        possession_lookup_path=lookup_path,
        player_lookup_path=player_lookup,
    )

    assert_frame_equal(pl.read_parquet(paths[0]), selected)
    lookup = pl.read_parquet(lookup_path)
    assert lookup.shape == (2, 10)
    assert_frame_equal(lookup.filter(pl.col("game_id") == 303), retained)
    unmatched = lookup.filter(pl.col("game_id") == 202)
    assert unmatched["match_team_player_possession_id"].item() == "202_p2"
    assert unmatched.select(
        "opponent_id", "player_possession_group", "started",
        "opposition_teamname",
    ).row(0) == (None, None, None, None)


def test_load_events_removes_stale_possessions_for_pause_only_matches(
    tmp_path: Path,
    events: pl.DataFrame,
    existing_lookup: pl.DataFrame,
) -> None:
    """Remove selected stale possessions even when all new IDs are null.

    Behavior:
        Determine replacement matches from all events before filtering
        null player-possession IDs out of the lookup.

    Input:
        One null-possession event each from matches 101 and 202, and
        existing lookup rows for both matches and unrelated match 303.

    Expected:
        Both events are written to their match files. The shared lookup
        retains only match 303 with all its metadata unchanged.

    Guards against:
        Keeping stale possessions or dropping pause-only match outputs.
    """
    lookup_path = tmp_path / "lookup.parquet"
    existing_lookup.write_parquet(lookup_path)
    selected = events.filter(
        pl.col("match_team_player_possession_id").is_null()
    )

    paths = event_load.load_events(
        selected,
        output_dir=tmp_path / "events",
        possession_lookup_path=lookup_path,
        player_lookup_path=tmp_path / "absent.parquet",
    )

    assert len(paths) == 3
    for path in paths[:-1]:
        assert pl.read_parquet(path).height == 1
    assert_frame_equal(
        pl.read_parquet(lookup_path),
        existing_lookup.filter(pl.col("game_id") == 303),
    )


def test_load_events_rejects_metadata_that_would_multiply_possessions(
    tmp_path: Path,
    events: pl.DataFrame,
    existing_lookup: pl.DataFrame,
    player_lookup: Path,
) -> None:
    """Reject ambiguous metadata before replacing lookup or event files.

    Starting datasets:
        A five-event batch, prior event and lookup outputs, and player
        metadata containing duplicate keys for player 11 in match 101.

    Expected:
        The many-to-one metadata join raises a Polars ComputeError.
        Previous output bytes remain unchanged and no match-202 event
        file is created. No multiplied possession rows are published.

    Guards against:
        Many-to-many joins silently expanding lookup rows or replacing
        valid outputs before metadata validation succeeds.
    """
    metadata = pl.read_parquet(player_lookup)
    pl.concat([metadata, metadata.head(1)]).write_parquet(player_lookup)
    output_dir = tmp_path / "events"
    output_dir.mkdir()
    previous_event_path = output_dir / "101.parquet"
    pl.DataFrame({"marker": ["old"]}).write_parquet(previous_event_path)
    previous_event_bytes = previous_event_path.read_bytes()
    lookup_path = tmp_path / "lookup.parquet"
    existing_lookup.write_parquet(lookup_path)
    previous_lookup_bytes = lookup_path.read_bytes()

    with pytest.raises(pl.exceptions.ComputeError, match="m:1 validation"):
        event_load.load_events(
            events,
            output_dir=output_dir,
            possession_lookup_path=lookup_path,
            player_lookup_path=player_lookup,
        )

    assert previous_event_path.read_bytes() == previous_event_bytes
    assert lookup_path.read_bytes() == previous_lookup_bytes
    assert not (output_dir / "202.parquet").exists()


@pytest.mark.parametrize("invalid_input", ["empty", "all_null", "mixed_null"])
def test_load_events_rejects_empty_or_unidentified_batches(
    tmp_path: Path,
    events: pl.DataFrame,
    invalid_input: str,
) -> None:
    """Reject batches whose rows cannot all be assigned to match files.

    Behavior:
        Validate nonempty input and non-null game IDs before writing.

    Input:
        An empty typed event frame, a frame with all-null game IDs, or
        a frame with one null game ID and otherwise identified rows.

    Expected:
        ValueError occurs before event or lookup destinations are made.

    Guards against:
        Row loss during partitioning or publishing partial inputs.
    """
    if invalid_input == "empty":
        selected = events.head(0)
    elif invalid_input == "all_null":
        selected = events.with_columns(
            pl.lit(None).cast(pl.Int64).alias("game_id")
        )
    else:
        selected = events.with_columns(
            pl.when(pl.col("event_number") == 1)
            .then(None)
            .otherwise(pl.col("game_id"))
            .alias("game_id")
        )
    output_dir = tmp_path / "events"
    lookup_path = tmp_path / "lookup.parquet"

    with pytest.raises(ValueError, match="non-null game_id"):
        event_load.load_events(
            selected,
            output_dir=output_dir,
            possession_lookup_path=lookup_path,
        )

    assert not output_dir.exists()
    assert not lookup_path.exists()
