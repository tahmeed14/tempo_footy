"""Tests for match-level Gradient Sports frame-rate metadata."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tempoctrl.gradient_sports.frame_rates import (
    GRADIENT_SPORTS_DEFAULT_FPS,
    GameFrameRate,
    resolve_gradient_sports_frame_rates,
)


def test_resolver_uses_metadata_and_default_by_match(
    tmp_path: Path,
) -> None:
    """Resolve each game's FPS without reading tracking rows.

    Behavior:
        Match integrated parquet filenames to optional metadata JSON and
        use the global default only when a matching JSON is absent.

    Input:
        Empty match files 10516 and 10517, with a 50-fps metadata file
        supplied only for match 10517.

    Expected:
        Sorted resolutions assign the global default to 10516 and the
        metadata rate to 10517, with their sources recorded.

    Guards against:
        Applying one match's FPS globally, requiring every metadata
        file,
        or depending on the contents of large parquet datasets.
    """
    match_dir = tmp_path / "integrated"
    metadata_dir = tmp_path / "metadata"
    match_dir.mkdir()
    metadata_dir.mkdir()
    (match_dir / "10517.parquet").touch()
    (match_dir / "10516.parquet").touch()
    pl.DataFrame({"fps": [50.0]}).write_json(metadata_dir / "10517.json")

    result = resolve_gradient_sports_frame_rates(
        match_dir,
        metadata_dir,
    )

    assert result == (
        GameFrameRate(
            game_id=10516,
            frame_rate=GRADIENT_SPORTS_DEFAULT_FPS,
            source="default",
        ),
        GameFrameRate(
            game_id=10517,
            frame_rate=50.0,
            source="metadata",
        ),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"other_field": [29.97]},
        {"fps": [0.0]},
        {"fps": [float("nan")]},
    ],
)
def test_resolver_rejects_invalid_existing_metadata(
    tmp_path: Path,
    metadata: dict[str, list[float]],
) -> None:
    """Reject an existing metadata file with unusable FPS data.

    Behavior:
        Treat present metadata as authoritative and validate it rather
        than silently replacing malformed values with the default.

    Input:
        One integrated match and metadata with a missing, zero, or NaN
        FPS value.

    Expected:
        Resolution raises a clear ``ValueError`` mentioning metadata.

    Guards against:
        Hiding corrupt provider metadata behind a plausible default
        rate.
    """
    match_dir = tmp_path / "integrated"
    metadata_dir = tmp_path / "metadata"
    match_dir.mkdir()
    metadata_dir.mkdir()
    (match_dir / "10517.parquet").touch()
    pl.DataFrame(metadata).write_json(metadata_dir / "10517.json")

    with pytest.raises(ValueError, match="Metadata"):
        resolve_gradient_sports_frame_rates(
            match_dir,
            metadata_dir,
        )
