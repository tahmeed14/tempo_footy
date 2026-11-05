"""Test event match identity before staging and cache reuse."""

import json
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


@pytest.mark.parametrize(
    "game_ids",
    [
        [202],
        [101, 202],
        [101, None],
        [None],
        [],
        ["unknown"],
        ["0101"],
        [True],
        [101.5],
        None,
    ],
    ids=[
        "wrong_match", "mixed_matches", "partial_null", "all_null", "empty",
        "nonnumeric", "noncanonical_string", "boolean", "fractional",
        "missing_column",
    ],
)
def test_stage_events_rejects_invalid_source_identity_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    game_ids: list[object] | None,
) -> None:
    """Reject source data that cannot identify only the requested match.

    Behavior:
        Validate source gameId values before creating staged outputs.

    Input:
        Custom-named JSON with wrong, mixed, null, missing, empty, or
        malformed IDs. Request match 101 regardless of source contents.

    Expected:
        ValueError names the source path and gameId or match ID problem.
        No staged file or staged directory is created.

    Guards against:
        Mislabeling stages, hiding rows from other matches, or allowing
        invalid IDs to determine downstream filenames.
    """
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "snapshot.json"
    rows = (
        [{"marker": 1}] if game_ids is None
        else [{"gameId": game_id, "marker": 1} for game_id in game_ids]
    )
    source_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="gameId|match ID") as error_info:
        stage_events(101, raw_path=source_path)

    assert str(source_path) in str(error_info.value)
    assert not Path("data/staged").exists()


@pytest.mark.parametrize("match_id", [101, "101"])
@pytest.mark.parametrize("source_id", [101, "101"])
def test_stage_events_accepts_matching_integer_and_string_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    match_id: int | str,
    source_id: int | str,
) -> None:
    """Accept matching integer and canonical string IDs without casting.

    Behavior:
        Support valid ID representations and arbitrary source names.

    Input:
        Custom JSON containing two match-101 rows, with integer or
        string IDs and markers deliberately in descending order.

    Expected:
        One staged 101.parquet preserves source values, dtypes, and row
        order. Its validated cache is reusable without reparsing JSON.

    Guards against:
        Rejecting valid custom sources, mutating raw IDs, or changing
        normal cached-stage behavior while adding identity validation.
    """
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "snapshot.json"
    rows = [
        {"gameId": source_id, "marker": 20},
        {"gameId": source_id, "marker": 10},
    ]
    source_path.write_text(json.dumps(rows), encoding="utf-8")

    staged_path = stage_events(match_id, raw_path=source_path)

    assert staged_path == Path(
        "data/staged/gradient_sports/events/101.parquet"
    )
    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"gameId": [source_id, source_id], "marker": [20, 10]}),
    )
    with patch(
        "tempoctrl.gradient_sports.ingest.pl.read_json",
        side_effect=AssertionError("Do not reparse a current source"),
    ):
        assert event_stage_is_current(match_id, raw_path=source_path)
        assert stage_events(match_id, raw_path=source_path) == staged_path


@pytest.mark.parametrize("overwrite", [False, True])
def test_stage_events_rejects_wrong_numeric_filename_even_with_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    """Reject conflicting numeric filenames before accepting a cache.

    Behavior:
        Require numeric source filenames to identify the requested ID.

    Input:
        A valid match-101 cache and a custom source named 202.json whose
        timestamp is older than the cache. Request match 101 with that
        source, both with and without forced overwrite.

    Expected:
        Freshness checks and staging raise ValueError naming expected
        101, found 202, and the source path. Cache bytes stay unchanged.
        No JSON parser runs.

    Guards against:
        A current cache masking an explicitly mismatched source path.
    """
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "101.json"
    source_path.write_text('[{"gameId": 101}]', encoding="utf-8")
    staged_path = stage_events(101, raw_path=source_path)
    cached_bytes = staged_path.read_bytes()
    wrong_source = tmp_path / "202.json"
    wrong_source.write_text('[{"gameId": 202}]', encoding="utf-8")
    os.utime(wrong_source, (1000, 1000))
    os.utime(staged_path, (2000, 2000))

    with patch(
        "tempoctrl.gradient_sports.ingest.pl.read_json",
        side_effect=AssertionError("Reject filename before reading JSON"),
    ):
        with pytest.raises(
            ValueError, match="expected 101, found 202"
        ) as error_info:
            stage_events(101, overwrite, raw_path=wrong_source)
        with pytest.raises(ValueError, match="filename match ID mismatch"):
            event_stage_is_current(101, overwrite, raw_path=wrong_source)

    assert str(wrong_source) in str(error_info.value)
    assert staged_path.read_bytes() == cached_bytes


@pytest.mark.parametrize("overwrite", [False, True])
def test_stage_events_keeps_previous_stage_when_new_source_id_is_wrong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    """Preserve a good stage when a replacement source has the wrong ID.

    Behavior:
        Validate changed source contents before overwriting staged data.

    Input:
        A valid stage for 101, then custom-named JSON changed to match
        202. The source is newer for automatic restaging, or older when
        forced overwrite is enabled.

    Expected:
        ValueError reports expected 101 and found 202. Prior stage bytes
        and modification time remain unchanged.

    Guards against:
        Replacing good cached data with a mislabeled source during
        normal refreshes or forced restaging.
    """
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "snapshot.json"
    source_path.write_text('[{"gameId": 101}]', encoding="utf-8")
    staged_path = stage_events(101, raw_path=source_path)
    source_path.write_text('[{"gameId": 202}]', encoding="utf-8")
    os.utime(staged_path, (2000, 2000))
    source_time = 1000 if overwrite else 3000
    os.utime(source_path, (source_time, source_time))
    cached_bytes = staged_path.read_bytes()
    cached_time = staged_path.stat().st_mtime_ns

    with pytest.raises(ValueError, match="expected 101, found \\[202\\]"):
        stage_events(101, overwrite, raw_path=source_path)

    assert staged_path.read_bytes() == cached_bytes
    assert staged_path.stat().st_mtime_ns == cached_time


@pytest.mark.parametrize("cached_ids", [[202], [101, 202], [None], None])
def test_stage_events_repairs_invalid_cached_match_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_ids: list[int | None] | None,
) -> None:
    """Restage invalid cached identities from a valid raw source.

    Behavior:
        Treat a cache with wrong or missing game IDs as not current.

    Input:
        Valid raw JSON for match 101 and a newer cached file containing
        match 202, mixed IDs, null IDs, or no gameId column.

    Expected:
        The freshness check returns false. One source read replaces the
        invalid stage with the original match-101 row without requiring
        the caller to force overwrite.

    Guards against:
        Reusing stages polluted before identity validation was added,
        or permanently blocking a valid source behind a bad cache.
    """
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "snapshot.json"
    source_path.write_text('[{"gameId": 101, "marker": 7}]', encoding="utf-8")
    os.utime(source_path, (1000, 1000))
    staged_path = stage_events(101, raw_path=source_path)
    cached_metadata = pl.read_parquet_metadata(staged_path)
    bad_cache = pl.DataFrame(
        {"marker": [0]} if cached_ids is None else {"gameId": cached_ids}
    )
    bad_cache.write_parquet(staged_path, metadata=cached_metadata)
    os.utime(staged_path, (2000, 2000))

    assert not event_stage_is_current(101, raw_path=source_path)
    with patch(
        "tempoctrl.gradient_sports.ingest.pl.read_json", wraps=pl.read_json
    ) as reader:
        assert stage_events(101, raw_path=source_path) == staged_path

    assert reader.call_count == 1
    assert_frame_equal(
        pl.read_parquet(staged_path),
        pl.DataFrame({"gameId": [101], "marker": [7]}),
    )


def test_stage_events_rejects_invalid_cache_and_invalid_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject mismatched data even when both cache and source agree.

    Behavior:
        Validate source identity after declining an invalid cache.

    Input:
        Match-202 JSON and a newer match-202 Parquet file incorrectly
        stored under the requested match-101 stage filename.

    Expected:
        ValueError reports the raw source mismatch. The invalid cached
        file is neither returned nor rewritten.

    Guards against:
        Accepting matching wrong IDs merely because both copies agree
        with each other rather than with the requested match.
    """
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "snapshot.json"
    source_path.write_text('[{"gameId": 202}]', encoding="utf-8")
    staged_path = Path("data/staged/gradient_sports/events/101.parquet")
    staged_path.parent.mkdir(parents=True)
    pl.DataFrame({"gameId": [202]}).write_parquet(staged_path)
    os.utime(source_path, (1000, 1000))
    os.utime(staged_path, (2000, 2000))
    cached_bytes = staged_path.read_bytes()

    with pytest.raises(ValueError, match="expected 101, found \\[202\\]"):
        stage_events(101, raw_path=source_path)

    assert staged_path.read_bytes() == cached_bytes
