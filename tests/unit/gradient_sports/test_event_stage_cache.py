"""Test event-stage provenance, atomic writes, and cache recovery."""

import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.ingest import (
    event_stage_is_current,
    stage_events,
)


@pytest.fixture
def raw_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Provide one source with a stable identity and timestamp.

    Behavior:
        Keep all staging operations in an isolated temporary directory.

    Input:
        JSON containing gameId 101 and marker A, timestamped at 1000.

    Expected:
        A custom source path and no preexisting staged artifacts.

    Guards against:
        Tests relying on production data or uncontrolled source times.
    """
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source.json"
    path.write_text('[{"gameId":101,"marker":"A"}]', encoding="utf-8")
    os.utime(path, (1000, 1000))
    return path


@pytest.mark.parametrize("second_timestamp", [1000, 500])
def test_stage_events_restages_when_source_path_changes(
    raw_source: Path,
    second_timestamp: int,
) -> None:
    """Decline an existing cache when a different source is requested.

    Behavior:
        Include the resolved source path in cache identity.

    Input:
        Sources A and B for match 101 have equal byte sizes. B has the
        same or older timestamp than A, which has already been staged.

    Expected:
        B is not current against A's stage. Restaging returns the same
        destination containing marker B and recording B's source path.

    Guards against:
        The audited bug where a custom source silently reuses data from
        another source solely because its timestamp is older.
    """
    staged_path = stage_events(101, raw_path=raw_source)
    second_source = raw_source.with_name("second.json")
    second_source.write_text(
        '[{"gameId":101,"marker":"B"}]', encoding="utf-8"
    )
    os.utime(second_source, (second_timestamp, second_timestamp))
    assert second_source.stat().st_size == raw_source.stat().st_size

    assert not event_stage_is_current(101, raw_path=second_source)
    assert stage_events(101, raw_path=second_source) == staged_path

    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"gameId": [101], "marker": ["B"]}),
    )
    assert pl.read_parquet_metadata(staged_path)[
        "tempoctrl.event_stage.source_path"
    ] == str(second_source.resolve())
    assert event_stage_is_current(101, raw_path=second_source)


@pytest.mark.parametrize("change", ["older_time", "size"])
def test_stage_events_detects_source_changes_without_newer_timestamps(
    raw_source: Path,
    change: str,
) -> None:
    """Detect source changes without newer timestamps.

    Behavior:
        Compare recorded source time and size instead of relying only on
        whether the cache is newer than the source.

    Input:
        A staged source is replaced with an equal-size row and older
        timestamp, or a longer row with its original timestamp restored.

    Expected:
        Both changes require restaging and produce the replacement row.

    Guards against:
        Missing source replacements whose timestamps move backward or
        whose file size changes while their timestamp remains unchanged.
    """
    staged_path = stage_events(101, raw_path=raw_source)
    marker = "B" if change == "older_time" else "longer"
    raw_source.write_text(
        '[{"gameId":101,"marker":"' + marker + '"}]', encoding="utf-8"
    )
    timestamp = 500 if change == "older_time" else 1000
    os.utime(raw_source, (timestamp, timestamp))

    assert not event_stage_is_current(101, raw_path=raw_source)
    assert stage_events(101, raw_path=raw_source) == staged_path

    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"gameId": [101], "marker": [marker]}),
    )


def test_stage_events_records_provenance_without_extra_artifacts(
    raw_source: Path,
) -> None:
    """Store provenance without extra columns or old versions.

    Behavior:
        Keep metadata in the Parquet footer and replace one match file.

    Input:
        Match-101 JSON and three forced restaging operations.

    Expected:
        Raw columns and values are unchanged. The footer records version
        1, the resolved path, timestamp 1000000000000 nanoseconds,
        and the source byte size. Exactly one staged file remains, with
        no sidecars, temporary directories, or previous versions.

    Guards against:
        Leaking provenance into downstream data or accumulating copies
        and temporary files on repeated pipeline runs.
    """
    for _ in range(3):
        staged_path = stage_events(101, overwrite=True, raw_path=raw_source)

    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"gameId": [101], "marker": ["A"]}),
    )
    metadata = pl.read_parquet_metadata(staged_path)
    assert metadata["tempoctrl.event_stage.version"] == "1"
    assert metadata["tempoctrl.event_stage.source_path"] == str(
        raw_source.resolve()
    )
    assert metadata["tempoctrl.event_stage.source_mtime_ns"] == "1000000000000"
    assert metadata["tempoctrl.event_stage.source_size"] == str(
        raw_source.stat().st_size
    )
    assert list(staged_path.parent.iterdir()) == [staged_path]


def test_stage_events_migrates_legacy_metadata_once(raw_source: Path) -> None:
    """Rebuild legacy stages once, then reuse the migrated artifact.

    Behavior:
        Decline stages without source provenance even if IDs are valid.

    Input:
        Source marker A and a newer stage with marker legacy, without
        provenance metadata.

    Expected:
        One JSON read rebuilds the stage with marker A. A second call
        reuses it without another source read, and no old copies remain.

    Guards against:
        Trusting unverified legacy sources or reparsing them every run.
    """
    staged_path = Path("data/staged/gradient_sports/events/101.parquet")
    staged_path.parent.mkdir(parents=True)
    pl.DataFrame({"gameId": [101], "marker": ["legacy"]}).write_parquet(
        staged_path
    )
    os.utime(staged_path, (2000, 2000))

    assert not event_stage_is_current(101, raw_path=raw_source)
    with patch(
        "tempoctrl.gradient_sports.ingest.pl.read_json", wraps=pl.read_json
    ) as reader:
        assert stage_events(101, raw_path=raw_source) == staged_path
        assert stage_events(101, raw_path=raw_source) == staged_path

    assert reader.call_count == 1
    assert pl.read_parquet(staged_path)["marker"].item() == "A"
    assert list(staged_path.parent.iterdir()) == [staged_path]


@pytest.mark.parametrize(
    "cached_bytes", [b"", b"not parquet", b"PAR1brokenPAR1"]
)
def test_stage_events_recovers_unreadable_cached_parquet(
    raw_source: Path,
    cached_bytes: bytes,
) -> None:
    """Recover unreadable caches through ordinary staging calls.

    Behavior:
        Treat unreadable Parquet as stale so staging can rebuild it.

    Input:
        Valid raw JSON and a newer empty, invalid, or truncated stage.

    Expected:
        The freshness check returns false. Staging without overwrite
        rebuilds the valid source and leaves one reusable Parquet file.

    Guards against:
        The audited retry failure where corrupt cached data prevents
        processing even though a valid raw source is available.
    """
    staged_path = stage_events(101, raw_path=raw_source)
    staged_path.write_bytes(cached_bytes)
    os.utime(staged_path, (2000, 2000))

    assert not event_stage_is_current(101, raw_path=raw_source)
    assert stage_events(101, raw_path=raw_source) == staged_path

    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"gameId": [101], "marker": ["A"]}),
    )
    assert event_stage_is_current(101, raw_path=raw_source)
    assert list(staged_path.parent.iterdir()) == [staged_path]


@pytest.mark.parametrize("existing_stage", [False, True])
def test_failed_stage_write_preserves_previous_data_and_cleans_temporary_files(
    raw_source: Path,
    existing_stage: bool,
) -> None:
    """Preserve prior data and remove partial files after write errors.

    Behavior:
        Write temporary data before replacing the public stage.

    Input:
        Source marker A, optionally staged, followed by valid marker B.
        The Parquet writer writes partial bytes and raises OSError.

    Expected:
        The error propagates. An existing stage retains its original
        bytes; a missing stage stays absent. No temporary files remain.
        A subsequent ordinary call successfully stages marker B.

    Guards against:
        Losing a valid cache, publishing partial data, or leaving failed
        write artifacts that consume space and break future runs.
    """
    staged_path = Path("data/staged/gradient_sports/events/101.parquet")
    previous_bytes = None
    if existing_stage:
        stage_events(101, raw_path=raw_source)
        previous_bytes = staged_path.read_bytes()
    raw_source.write_text('[{"gameId":101,"marker":"B"}]', encoding="utf-8")
    os.utime(raw_source, (2000, 2000))
    write_targets: list[Path] = []

    def interrupted_write(
        frame: pl.DataFrame,
        target: str | Path,
        **kwargs: object,
    ) -> None:
        """Simulate an interrupted Parquet write.

        Behavior:
            Write partial bytes before raising a disk-write error.

        Input:
            The staged frame and the writer's destination path.

        Expected:
            Record the destination, write partial bytes, and raise.

        Guards against:
            Direct overwrites of the public staged file.
        """
        path = Path(target)
        write_targets.append(path)
        path.write_bytes(b"interrupted write")
        raise OSError("simulated write failure")

    with patch.object(pl.DataFrame, "write_parquet", interrupted_write):
        with pytest.raises(OSError, match="simulated write failure"):
            stage_events(101, overwrite=True, raw_path=raw_source)

    assert len(write_targets) == 1
    assert write_targets[0] != staged_path
    if existing_stage:
        assert staged_path.read_bytes() == previous_bytes
        assert list(staged_path.parent.iterdir()) == [staged_path]
    else:
        assert not staged_path.exists()
        assert list(staged_path.parent.iterdir()) == []

    assert stage_events(101, raw_path=raw_source) == staged_path
    assert pl.read_parquet(staged_path)["marker"].item() == "B"
    assert list(staged_path.parent.iterdir()) == [staged_path]


def test_failed_stage_replacement_preserves_cache_and_cleans_temporary_files(
    raw_source: Path,
) -> None:
    """Preserve the old stage when publication fails after a good write.

    Behavior:
        Replace the public path only after temporary output validation.

    Input:
        A valid cache and a valid replacement source. Path.replace is
        forced to raise OSError during publication.

    Expected:
        The original bytes remain unchanged and no temporary files
        survive. Retrying without the injected error stages marker B.

    Guards against:
        Removing the original before publication or leaking temporary
        artifacts when replacement itself fails.
    """
    staged_path = stage_events(101, raw_path=raw_source)
    previous_bytes = staged_path.read_bytes()
    raw_source.write_text('[{"gameId":101,"marker":"B"}]', encoding="utf-8")
    os.utime(raw_source, (2000, 2000))

    with patch.object(
        Path, "replace", side_effect=OSError("simulated replacement failure")
    ):
        with pytest.raises(OSError, match="simulated replacement failure"):
            stage_events(101, overwrite=True, raw_path=raw_source)

    assert staged_path.read_bytes() == previous_bytes
    assert list(staged_path.parent.iterdir()) == [staged_path]
    assert stage_events(101, raw_path=raw_source) == staged_path
    assert pl.read_parquet(staged_path)["marker"].item() == "B"
    assert list(staged_path.parent.iterdir()) == [staged_path]


def test_stage_events_reuses_equivalent_source_paths(raw_source: Path) -> None:
    """Reuse equivalent relative and absolute source paths.

    Behavior:
        Resolve path spellings before comparing provenance.

    Input:
        A source staged using its absolute path, then requested using
        its relative filename in the same working directory.

    Expected:
        The stage is current and reused without reparsing JSON.

    Guards against:
        Unnecessary restaging for equivalent source paths.
    """
    staged_path = stage_events(101, raw_path=raw_source)

    with patch(
        "tempoctrl.gradient_sports.ingest.pl.read_json",
        side_effect=AssertionError("Equivalent sources must reuse the cache"),
    ):
        assert event_stage_is_current(101, raw_path=Path(raw_source.name))
        assert stage_events(101, raw_path=raw_source.name) == staged_path
