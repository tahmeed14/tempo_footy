"""Test joins between tracking and event data."""

from pathlib import Path

import polars as pl
import pytest

from tempoctrl.gradient_sports import join as join_module


def _tracking_rows() -> pl.LazyFrame:
    """Provide two matching frames and one unmatched tracking frame."""
    return pl.DataFrame(
        {
            "game_id": [10514, 10514, 10514],
            "game_event_id": [10, 20, 30],
            "possession_event_id": [100, 200, 300],
            "frame_number": [1, 2, 3],
        }
    ).lazy()


def _event_rows(duplicate: bool = False) -> pl.LazyFrame:
    """Provide current event fields, optionally duplicating a join key.
    """
    game_events = [10, 10, 20] if duplicate else [10, 20]
    possession_events = [100, 100, 200] if duplicate else [100, 200]
    event_types = ["PASS", "OUT", "SHOT"] if duplicate else ["PASS", "SHOT"]
    count = len(game_events)
    return pl.DataFrame(
        {
            "game_id": [10514] * count,
            "game_event_id": game_events,
            "possession_event_id": possession_events,
            "formattedgameclock": ["00:10"] * count,
            "match_team_possession_id": ["team_1"] * count,
            "match_team_player_possession_id": ["player_1"] * count,
            "team_possession_start": [False] * count,
            "player_id": [11] * count,
            "team_id": [1] * count,
            "setpiecetype": ["O"] * count,
            "event_number": list(range(1, count + 1)),
            "game_event_type": event_types,
            "possession_event_type": ["PA"] * count,
            "successful_pass_or_cross": [True] * count,
            "unneeded_source_field": ["discard"] * count,
        }
    ).lazy()


def test_valid_many_to_one_join_preserves_existing_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve all tracking rows through event enrichment.

    Three tracking frames and two unique events must yield three rows,
    including null event fields on the unmatched frame. This catches
    accidental inner joins, dropped rows, and leaked source columns.
    """
    events = _event_rows()
    monkeypatch.setattr(join_module, "scan_events", lambda _: events)
    monkeypatch.setattr(
        join_module, "scan_tracking", lambda _: _tracking_rows()
    )
    monkeypatch.setattr(join_module, "JOIN_ISSUES_DIR", tmp_path)

    result = join_module.possession_join(10514).collect()

    assert result.height == 3
    assert set(result.columns) == {
        *join_module.KEEP_EVENT_COLUMNS,
        "frame_number",
    }
    assert result.sort("frame_number").select(
        "frame_number", "game_event_type", "player_id"
    ).rows() == [(1, "PASS", 11), (2, "SHOT", 11), (3, None, None)]
    assert list(tmp_path.iterdir()) == []


def test_failed_many_to_one_join_saves_duplicate_event_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject row multiplication and save the offending event keys.

    Two events share one tracking key. The join must raise and write
    exactly those two events with a UInt32 key count of two, catching
    silent many-to-many expansion or incomplete diagnostics.
    """
    events = _event_rows(duplicate=True)
    monkeypatch.setattr(join_module, "scan_events", lambda _: events)
    monkeypatch.setattr(
        join_module, "scan_tracking", lambda _: _tracking_rows()
    )
    monkeypatch.setattr(join_module, "JOIN_ISSUES_DIR", tmp_path)

    with pytest.raises(pl.exceptions.ComputeError, match="m:1 validation"):
        join_module.possession_join(10514)

    diagnostic_path = tmp_path / "10514_duplicate_event_rows.parquet"
    assert diagnostic_path.is_file()
    diagnostics = pl.read_parquet(diagnostic_path)
    assert diagnostics.height == 2
    assert diagnostics.select(
        "game_id", "game_event_id", "possession_event_id",
        "game_event_type", "join_key_count",
    ).rows() == [
        (10514, 10, 100, "PASS", 2),
        (10514, 10, 100, "OUT", 2),
    ]
    assert diagnostics.schema["join_key_count"] == pl.UInt32
