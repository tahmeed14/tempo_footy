"""Test event staging, cache reuse, and staged event numbering."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.ingest import (
    event_stage_is_current,
    resolve_event_paths,
    scan_events,
    stage_events,
)


@pytest.fixture
def raw_event_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Provide nested raw events in an isolated working directory.

    Behavior:
        Keep every staged artifact inside the temporary test directory.

    Input:
        Three JSON rows with unsorted markers, nulls, a list, and a
        nested field first populated on the final row.

    Expected:
        A source path in the default raw-event directory for match 101.

    Guards against:
        Tests reading private data or writing repository artifacts.
    """
    monkeypatch.chdir(tmp_path)
    raw_path = Path("data/raw/gradient_sports/events/101.json")
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            [
                {
                    "marker": 30,
                    "event": {"type": "PASS"},
                    "tags": ["open"],
                    "gameId": 101,
                },
                {"marker": 10, "event": None, "tags": None, "gameId": 101},
                {
                    "marker": 20,
                    "event": {"type": "UNKNOWN", "value": 7},
                    "tags": [],
                    "gameId": 101,
                },
            ]
        ),
        encoding="utf-8",
    )
    return raw_path


@pytest.mark.parametrize("match_id", [101, "101"])
def test_resolve_event_paths_supports_defaults_and_override(
    match_id: int | str,
    tmp_path: Path,
) -> None:
    """Resolve one staged destination for default and custom sources.

    Behavior:
        Accept numeric or string IDs and optional source paths.

    Input:
        Match 101 and a custom JSON file outside the default directory.

    Expected:
        The source changes with the override; the staged path does not.

    Guards against:
        Ignoring explicit sources or staging into the raw directory.
    """
    expected_staged = Path("data/staged/gradient_sports/events/101.parquet")
    assert resolve_event_paths(match_id) == (
        Path("data/raw/gradient_sports/events/101.json"),
        expected_staged,
    )
    custom_path = tmp_path / "custom.json"
    assert resolve_event_paths(match_id, str(custom_path)) == (
        custom_path,
        expected_staged,
    )


def test_stage_and_scan_preserve_nested_rows_and_source_order(
    raw_event_path: Path,
) -> None:
    """Stage source values and assign row numbers only during scanning.

    Behavior:
        Round-trip raw JSON through Parquet without business transforms.

    Input:
        Three nested events with nulls, an unknown category, list data,
        and a nested numeric field absent from the first row.

    Expected:
        Three rows in source order, unchanged columns and dtypes, and
        UInt32 event numbers 1, 2, 3 only on the lazy scan. Filtering
        retains original numbers 1 and 3. No joins are involved, so no
        unmatched rows are dropped.

    Guards against:
        Sorting, schema loss, dropping nulls, adding indices to staged
        files, or numbering events after downstream filters.
    """
    expected = pl.DataFrame(
        {
            "marker": [30, 10, 20],
            "event": [
                {"type": "PASS", "value": None},
                None,
                {"type": "UNKNOWN", "value": 7},
            ],
            "tags": [["open"], None, []],
            "gameId": [101, 101, 101],
        },
        schema={
            "marker": pl.Int64,
            "event": pl.Struct({"type": pl.String, "value": pl.Int64}),
            "tags": pl.List(pl.String),
            "gameId": pl.Int64,
        },
    )

    staged_path = stage_events(101)
    assert staged_path == Path(
        "data/staged/gradient_sports/events/101.parquet"
    )
    assert_frame_equal(pl.read_parquet(staged_path), expected)

    scanned = scan_events(str(staged_path))
    assert isinstance(scanned, pl.LazyFrame)
    assert_frame_equal(
        scanned.collect(),
        expected.insert_column(
            0, pl.Series("event_number", [1, 2, 3], dtype=pl.UInt32)
        ),
    )
    assert scanned.filter(pl.col("marker") >= 20).select(
        "event_number"
    ).collect().to_series().to_list() == [1, 3]


def test_stage_events_reuses_current_file_without_reading_json(
    raw_event_path: Path,
) -> None:
    """Reuse staged events without parsing JSON or rewriting Parquet.

    Behavior:
        Preserve cached bytes and modification time when overwrite is
        false and the stage is at least as new as its source.

    Input:
        A valid source and existing stage with controlled timestamps.

    Expected:
        The same path, bytes, and timestamp; no JSON read occurs.

    Guards against:
        Repeating JSON parsing or rewriting a reusable stage every run.
    """
    os.utime(raw_event_path, (1000, 1000))
    staged_path = stage_events(101)
    os.utime(staged_path, (2000, 2000))
    staged_bytes = staged_path.read_bytes()
    staged_mtime = staged_path.stat().st_mtime_ns

    with patch(
        "tempoctrl.gradient_sports.ingest.pl.read_json",
        side_effect=AssertionError("Cached events must not reparse JSON"),
    ):
        assert event_stage_is_current(101)
        assert stage_events(101) == staged_path

    assert staged_path.read_bytes() == staged_bytes
    assert staged_path.stat().st_mtime_ns == staged_mtime


@pytest.mark.parametrize(
    ("overwrite", "recorded_mtime", "raw_mtime", "expected_current"),
    [
        (False, 1000, 1000, True),
        (False, 2000, 2000, True),
        (False, 1000, 3000, False),
        (False, 1000, 500, False),
        (False, 1000, 2000, False),
        (True, 1000, 1000, False),
    ],
)
def test_event_stage_freshness_respects_source_time_and_overwrite(
    raw_event_path: Path,
    overwrite: bool,
    recorded_mtime: int,
    raw_mtime: int,
    expected_current: bool,
) -> None:
    """Resolve freshness from recorded source times and overwrite.

    Behavior:
        Reuse stages only when source metadata agrees and overwrite is
        false, even when changed source timestamps move backward.

    Input:
        A stage timestamp of 2000 and sources with unchanged, older,
        equal, or newer times relative to recorded source metadata.

    Expected:
        Only stages with matching provenance and no forced overwrite
        are current. Equal source/stage times are valid when the
        recorded source time also agrees.

    Guards against:
        Reusing stale data or ignoring the overwrite flag.
    """
    os.utime(raw_event_path, (recorded_mtime, recorded_mtime))
    staged_path = stage_events(101)
    os.utime(staged_path, (2000, 2000))
    os.utime(raw_event_path, (raw_mtime, raw_mtime))

    assert event_stage_is_current(101, overwrite) is expected_current


@pytest.mark.parametrize("overwrite", [False, True])
def test_stage_events_refreshes_stale_or_forced_stage(
    raw_event_path: Path,
    overwrite: bool,
) -> None:
    """Replace stages when the source changes or overwrite is requested.

    Behavior:
        Restage newer sources and refresh current sources on demand.

    Input:
        An existing stage and replacement JSON containing marker 99.
        The replacement is newer only when overwrite is false.

    Expected:
        The same staged path contains the replacement row and no index.

    Guards against:
        Keeping old rows or applying overwrite to processed outputs.
    """
    staged_path = stage_events(101)
    raw_event_path.write_text(
        '[{"marker": 99, "gameId": 101}]', encoding="utf-8"
    )
    os.utime(staged_path, (2000, 2000))
    raw_mtime = 1000 if overwrite else 3000
    os.utime(raw_event_path, (raw_mtime, raw_mtime))

    assert stage_events(101, overwrite) == staged_path
    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"marker": [99], "gameId": [101]}),
    )


def test_stage_events_honors_custom_raw_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage a custom event source without requiring a default source.

    Behavior:
        Use the supplied JSON path for reading and freshness checks.

    Input:
        A single-row JSON file outside the default raw directory.

    Expected:
        Match 101's staged file contains marker 42 and is current when
        checked with the same custom source.

    Guards against:
        Reverting to a default path or misplacing custom-source outputs.
    """
    monkeypatch.chdir(tmp_path)
    raw_path = tmp_path / "custom.json"
    raw_path.write_text(
        '[{"marker": 42, "gameId": "101"}]', encoding="utf-8"
    )

    staged_path = stage_events(101, raw_path=str(raw_path))

    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"marker": [42], "gameId": ["101"]}),
    )
    assert event_stage_is_current(101, raw_path=raw_path)


def test_event_stage_reports_missing_stage(raw_event_path: Path) -> None:
    """Mark an unstaged source as requiring staging.

    Behavior:
        Report no reusable cache before the first staging operation.

    Input:
        Valid raw JSON without any corresponding staged file.

    Expected:
        The freshness check returns false.

    Guards against:
        Skipping first-time staging because the source alone exists.
    """
    assert not event_stage_is_current(101)


@pytest.mark.parametrize("overwrite", [False, True])
def test_event_stage_rejects_missing_raw_source(
    raw_event_path: Path,
    overwrite: bool,
) -> None:
    """Require the raw source even when a cached stage exists.

    Behavior:
        Match tracking's source validation for staging and reuse.

    Input:
        An existing stage whose raw JSON file has been removed.

    Expected:
        Both freshness checks and staging raise FileNotFoundError.

    Guards against:
        Silently treating a missing source as a current cache.
    """
    stage_events(101)
    raw_event_path.unlink()

    with pytest.raises(FileNotFoundError, match="Raw event file not found"):
        event_stage_is_current(101, overwrite)
    with pytest.raises(FileNotFoundError, match="Raw event file not found"):
        stage_events(101, overwrite)


def test_scan_events_preserves_empty_schema(tmp_path: Path) -> None:
    """Scan empty staged events with a typed one-based index column.

    Behavior:
        Return a lazy, typed result even when no events are present.

    Input:
        Empty staged Parquet with an Int64 marker column.

    Expected:
        Zero rows and columns event_number UInt32 and marker Int64.

    Guards against:
        Empty-input failures, fabricated rows, or index dtype drift.
    """
    staged_path = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"marker": pl.Int64}).write_parquet(staged_path)

    assert_frame_equal(
        scan_events(staged_path).collect(),
        pl.DataFrame(schema={"event_number": pl.UInt32, "marker": pl.Int64}),
    )
