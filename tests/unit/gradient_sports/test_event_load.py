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
            "_build_match_possession_lookup",
            wraps=event_load._build_match_possession_lookup,
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


@pytest.fixture
def replacement_events(events: pl.DataFrame) -> pl.DataFrame:
    """Provide distinguishable replacements for two selected matches.

    Behavior:
        Change possession IDs and event markers without row loss.

    Input:
        Five interleaved events from matches 101 and 202, including
        repeated possessions and null possession rows.

    Expected:
        Non-null IDs and all markers gain a _new suffix. Nulls and all
        other values remain unchanged.

    Guards against:
        Failure tests passing because replacement bytes equal old data.
    """
    return events.with_columns(
        (pl.col("match_team_player_possession_id") + "_new"),
        (pl.col("marker") + "_new"),
    )


@pytest.fixture
def publication_state(
    tmp_path: Path,
    events: pl.DataFrame,
    existing_lookup: pl.DataFrame,
    existing_outputs: bool,
) -> tuple[dict[str, Path], dict[Path, bytes]]:
    """Seed publication paths and capture their original bytes.

    Behavior:
        Isolate all writes and retain an unrelated match in every case.

    Input:
        Five selected event rows, an existing three-match lookup, and
        whether selected files and the shared lookup already exist.

    Expected:
        A keyword-argument mapping for load_events and byte snapshots
        of all original files. Match 303's event always exists; its
        enriched lookup row exists only with preexisting outputs.

    Guards against:
        Tests missing rollback of new files or unrelated-data loss.
    """
    kwargs = {
        "output_dir": tmp_path / "events",
        "possession_lookup_path": tmp_path / "lookup.parquet",
        "player_lookup_path": tmp_path / "absent.parquet",
    }
    kwargs["output_dir"].mkdir()
    pl.DataFrame({"marker": ["keep"]}).write_parquet(
        kwargs["output_dir"] / "303.parquet"
    )
    if existing_outputs:
        existing_lookup.write_parquet(kwargs["possession_lookup_path"])
        event_load.load_events(events, **kwargs)
    snapshots = {
        path: path.read_bytes() for path in tmp_path.rglob("*.parquet")
    }
    return kwargs, snapshots


@pytest.mark.parametrize("existing_outputs", [False, True])
@pytest.mark.parametrize(
    "failed_name", ["101.parquet", "202.parquet", "lookup.parquet"]
)
def test_event_write_failure_preserves_all_outputs_and_allows_retry(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
    failed_name: str,
) -> None:
    """Leave every public output unchanged after an interrupted write.

    Starting datasets:
        Five replacement events from two matches, an unrelated match,
        and either existing selected files and lookup or no prior batch.
        The first event, second event, or lookup writer writes partial
        bytes and raises an OSError.

    Expected:
        No public files change or appear, and staging directories are
        removed. A retry writes complete three-row and two-row event
        files, including null IDs; the lookup has two new possessions
        plus the unrelated possession when present. No metadata joins
        occur because the optional metadata file is absent.

    Guards against:
        Publishing a lookup or earlier matches before all writes finish,
        corrupting existing files, blocking retries, or leaking copies.
    """
    kwargs, snapshots = publication_state
    original_write = pl.DataFrame.write_parquet

    def interrupted_write(
        frame: pl.DataFrame,
        file: str | Path,
        **write_kwargs: object,
    ) -> None:
        """Simulate partial bytes followed by a disk-write failure.

        Input is a pending frame and path. Only the selected basename
        fails; other writes use Polars. The expected partial file and
        OSError detect public overwrites or early publication.
        """
        path = Path(file)
        if path.name == failed_name:
            path.write_bytes(b"partial parquet")
            raise OSError("simulated write failure")
        original_write(frame, file, **write_kwargs)

    with patch.object(pl.DataFrame, "write_parquet", interrupted_write):
        with pytest.raises(OSError, match="simulated write failure"):
            event_load.load_events(replacement_events, **kwargs)

    current = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    assert current == snapshots
    assert not list(tmp_path.rglob(".event-output-*"))
    paths = event_load.load_events(replacement_events, **kwargs)
    for path in paths[:-1]:
        assert_frame_equal(
            pl.read_parquet(path),
            replacement_events.filter(pl.col("game_id") == int(path.stem)),
        )
    lookup = pl.read_parquet(paths[-1])
    assert lookup.filter(pl.col("game_id").is_in([101, 202]))[
        "match_team_player_possession_id"
    ].to_list() == ["101_p1_new", "202_p2_new"]
    assert not list(tmp_path.rglob(".event-output-*"))


@pytest.mark.parametrize("existing_outputs", [False, True])
@pytest.mark.parametrize("corruption", ["truncated", "schema", "values"])
def test_invalid_pending_parquet_prevents_all_publication(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
    corruption: str,
) -> None:
    """Validate pending contents before replacing any public outputs.

    Behavior:
        Reject unreadable files and readable files with wrong data.

    Input:
        A two-match replacement batch. Its second pending match file
        is truncated, loses marker, or replaces every marker with wrong.

    Expected:
        Raise a validation error while retaining every original file
        byte for byte, creating no new public files or staging residue.

    Guards against:
        Trusting writer success without validating Parquet schema and
        values, or publishing a match before validation finishes.
    """
    kwargs, snapshots = publication_state
    original_write = pl.DataFrame.write_parquet

    def corrupt_write(
        frame: pl.DataFrame,
        file: str | Path,
        **write_kwargs: object,
    ) -> None:
        """Write corrupt data while reporting a successful write.

        Input is a pending frame and path; only match 202 is altered.
        Expected unreadable bytes or mismatching schema/values expose
        missing validation without touching unrelated writes.
        """
        path = Path(file)
        if path.name == "202.parquet":
            if corruption == "truncated":
                path.write_bytes(b"truncated parquet")
                return
            if corruption == "schema":
                frame = frame.drop("marker")
            else:
                frame = frame.with_columns(pl.lit("wrong").alias("marker"))
        original_write(frame, file, **write_kwargs)

    with patch.object(pl.DataFrame, "write_parquet", corrupt_write):
        with pytest.raises((ValueError, pl.exceptions.PolarsError)):
            event_load.load_events(replacement_events, **kwargs)

    current = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    assert current == snapshots
    assert not list(tmp_path.rglob(".event-output-*"))


@pytest.mark.parametrize("existing_outputs", [False, True])
@pytest.mark.parametrize(
    "failed_name", ["101.parquet", "202.parquet", "lookup.parquet"]
)
def test_publication_failure_rolls_back_the_whole_event_batch(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
    failed_name: str,
) -> None:
    """Restore prior files after failure during ordered publication.

    Starting datasets:
        Two selected matches, an unrelated match, and optional prior
        event files and lookup. All three pending Parquets are valid,
        but atomic replacement fails for one selected destination.

    Expected:
        Propagate OSError and restore all original bytes. Remove any
        newly published files that had no predecessor. Leave unrelated
        files untouched and no staging directories behind. Ordinary
        retry succeeds, retaining every event and null possession row.
        No joins occur because player metadata is absent.

    Guards against:
        Mixed old/new batches after a replacement failure, deleting
        prior data, leaving new files behind, or cleanup errors.
    """
    kwargs, snapshots = publication_state
    original_replace = Path.replace

    def failed_replace(source: Path, target: str | Path) -> Path:
        """Fail one pending publication, allowing backup restoration.

        Input is a source and destination. A pending file with the
        selected destination name raises OSError; backups replace
        normally. This isolates publication failure from write failure.
        """
        if (
            source.parent.name.startswith(".event-output-")
            and Path(target).name == failed_name
        ):
            raise OSError("simulated publication failure")
        return original_replace(source, target)

    with patch.object(Path, "replace", failed_replace):
        with pytest.raises(OSError, match="simulated publication failure"):
            event_load.load_events(replacement_events, **kwargs)

    current = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    assert current == snapshots
    assert not list(tmp_path.rglob(".event-output-*"))
    event_load.load_events(replacement_events, **kwargs)
    assert not list(tmp_path.rglob(".event-output-*"))


@pytest.mark.parametrize("existing_outputs", [False, True])
def test_lookup_publishes_last_after_every_output_is_prepared(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
) -> None:
    """Publish the lookup only after complete match files are prepared.

    Behavior:
        Complete every pending write before publishing any output.

    Input:
        A two-match replacement batch with optional previous outputs
        and an unrelated match 303.

    Expected:
        All three writes finish before the first atomic replacement.
        Publication order is match 101, match 202, then the lookup.
        Returned paths match that order; no temporary copies remain.

    Guards against:
        Early lookup publication, per-match prepare/publish loops, or
        history and backup files accumulating after successful runs.
    """
    kwargs, _ = publication_state
    original_replace = Path.replace
    published: list[Path] = []

    def inspect_replace(source: Path, target: str | Path) -> Path:
        """Check preparation is complete at each public replacement.

        Input is a pending file and public path. Expect three completed
        writes and record destination order, catching early publication.
        """
        assert writer.call_count == 3
        published.append(Path(target))
        return original_replace(source, target)

    with (
        patch.object(
            pl.DataFrame, "write_parquet", autospec=True,
            side_effect=pl.DataFrame.write_parquet,
        ) as writer,
        patch.object(Path, "replace", inspect_replace),
    ):
        paths = event_load.load_events(replacement_events, **kwargs)

    assert published == [
        kwargs["output_dir"] / "101.parquet",
        kwargs["output_dir"] / "202.parquet",
        kwargs["possession_lookup_path"],
    ]
    assert paths == tuple(published)
    assert not list(tmp_path.rglob(".event-output-*"))


@pytest.mark.parametrize("existing_outputs", [False, True])
def test_standalone_lookup_write_is_safe_and_retryable(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
) -> None:
    """Preserve a standalone lookup after failure and allow retry.

    Behavior:
        Apply safe temporary writes to the public standalone writer too.

    Input:
        Two selected matches, optional existing lookup metadata for an
        unrelated match, and a writer that fails after partial bytes.

    Expected:
        Retain all original files and remove partial staging data. Retry
        without the injected failure writes two new possessions, keeping
        match 303 when present. Event files stay unchanged throughout.

    Guards against:
        Fixing only the pipeline path, corrupting the standalone lookup,
        losing unrelated metadata, or blocking the next ordinary retry.
    """
    kwargs, snapshots = publication_state

    def interrupted_write(
        frame: pl.DataFrame,
        file: str | Path,
        **write_kwargs: object,
    ) -> None:
        """Write partial bytes and fail the standalone pending lookup.

        Input is a pending lookup frame and path. Expect OSError after
        partial bytes, catching direct overwrites of the public lookup.
        """
        Path(file).write_bytes(b"partial lookup")
        raise OSError("simulated lookup failure")

    with patch.object(pl.DataFrame, "write_parquet", interrupted_write):
        with pytest.raises(OSError, match="simulated lookup failure"):
            event_load.write_match_possession_lookup(
                replacement_events,
                output_path=kwargs["possession_lookup_path"],
                player_lookup_path=kwargs["player_lookup_path"],
            )

    current = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    assert current == snapshots
    assert not list(tmp_path.rglob(".event-output-*"))
    path = event_load.write_match_possession_lookup(
        replacement_events,
        output_path=kwargs["possession_lookup_path"],
        player_lookup_path=kwargs["player_lookup_path"],
    )
    assert path == kwargs["possession_lookup_path"]
    lookup = pl.read_parquet(path)
    assert lookup.filter(pl.col("game_id").is_in([101, 202]))[
        "match_team_player_possession_id"
    ].to_list() == ["101_p1_new", "202_p2_new"]
    for original, data in snapshots.items():
        if original != path:
            assert original.read_bytes() == data
    if path in snapshots:
        assert lookup.filter(pl.col("game_id") == 303)[
            "player_possession_group"
        ].item() == "Winger"
    assert not list(tmp_path.rglob(".event-output-*"))


@pytest.mark.parametrize("existing_outputs", [True])
def test_failed_rollback_retains_recovery_copies_and_reports_location(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
) -> None:
    """Retain old bytes for recovery when rollback cannot restore data.

    Behavior:
        Keep backup directories instead of deleting the last old copy.

    Input:
        Existing event and lookup files. Publishing the new lookup
        fails, then restoring match 101's previous file also fails.

    Expected:
        Re-raise the publication error with a note identifying the
        failed restoration and backup path. The old match-101 bytes
        survive in that backup. Match 202, the shared lookup, and the
        unrelated match retain their original bytes.

    Guards against:
        Cleanup destroying recovery copies after a rollback failure,
        masking the original error, or stopping other restorations.
    """
    kwargs, snapshots = publication_state
    original_replace = Path.replace

    def fail_publication_and_restore(
        source: Path,
        target: str | Path,
    ) -> Path:
        """Fail lookup publication and only match 101's restoration.

        Input is a staged or backup path and destination. Expected
        OSErrors expose backup deletion and incomplete rollback.
        Other replacements use their normal filesystem operation.
        """
        if (
            source.parent.name.startswith(".event-output-")
            and Path(target).name == "lookup.parquet"
        ):
            raise OSError("publication failed")
        if source.parent.name == "backups" and source.name == "101.parquet":
            raise OSError("restoration failed")
        return original_replace(source, target)

    with patch.object(Path, "replace", fail_publication_and_restore):
        with pytest.raises(OSError, match="publication failed") as caught:
            event_load.load_events(replacement_events, **kwargs)

    backups = [
        path for path in tmp_path.rglob("101.parquet")
        if path.parent.name == "backups"
    ]
    assert len(backups) == 1
    assert backups[0].read_bytes() == snapshots[
        kwargs["output_dir"] / "101.parquet"
    ]
    assert any(str(backups[0]) in note for note in caught.value.__notes__)
    for path, original in snapshots.items():
        if path.name != "101.parquet":
            assert path.read_bytes() == original


@pytest.mark.parametrize("existing_outputs", [False, True])
def test_colliding_event_and_lookup_paths_are_rejected_before_writing(
    tmp_path: Path,
    replacement_events: pl.DataFrame,
    publication_state: tuple[dict[str, Path], dict[Path, bytes]],
) -> None:
    """Reject configuring the lookup to overwrite a selected event file.

    Behavior:
        Require distinct resolved destination paths before publication.

    Input:
        A two-match batch whose custom lookup path equals match 101's
        output path, with or without existing outputs.

    Expected:
        ValueError and no new public files, changed bytes, or temporary
        directories. All original files remain available.

    Guards against:
        Publishing a lookup over a match file or ambiguous backups.
    """
    kwargs, snapshots = publication_state
    kwargs = {
        **kwargs,
        "possession_lookup_path": kwargs["output_dir"] / "101.parquet",
    }

    with pytest.raises(ValueError, match="paths must be distinct"):
        event_load.load_events(replacement_events, **kwargs)

    current = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    assert current == snapshots
    assert not list(tmp_path.rglob(".event-output-*"))
