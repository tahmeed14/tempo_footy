"""Frame-rate metadata helpers for multi-game tracking datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

FrameRateSpec = float | Mapping[int, float]
FrameRateSource = Literal["metadata", "default"]

FRAME_RATE_COLUMN = "frame_rate"
GAME_COLUMN = "game_id"
GRADIENT_SPORTS_DEFAULT_FPS = 29.97


@dataclass(frozen=True, slots=True)
class GameFrameRate:
    """Record the resolved sampling rate and its source for one game."""

    game_id: int
    frame_rate: float
    source: FrameRateSource


def _validate_frame_rate_spec(frame_rate: FrameRateSpec) -> None:
    """Validate a shared rate or a complete per-game rate mapping."""
    rates = (
        tuple(frame_rate.values())
        if isinstance(frame_rate, Mapping)
        else (frame_rate,)
    )
    if not rates:
        raise ValueError("frame_rate mapping cannot be empty.")
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError(
            "frame_rate values must be finite and greater than 0."
        )


def add_frame_rate_column(
    df: pl.LazyFrame,
    frame_rate: FrameRateSpec,
) -> pl.LazyFrame:
    """Attach the rate used for every game's tracking rows.

    A scalar applies to the entire dataset. A mapping is resolved from
    ``game_id`` and uses strict replacement, so collection fails if any
    non-null game is missing from the supplied metadata.

    Args:
        df: Lazy tracking rows.
        frame_rate: Shared rate or mapping from game ID to rate.

    Returns:
        The lazy input with a Float64 ``frame_rate`` column.

    """
    _validate_frame_rate_spec(frame_rate)

    if isinstance(frame_rate, Mapping):
        if GAME_COLUMN not in df.collect_schema():
            raise ValueError("game_id is required for per-game frame rates.")
        frame_rate_expression = pl.col(GAME_COLUMN).replace_strict(
            dict(frame_rate),
            return_dtype=pl.Float64,
        )
    else:
        frame_rate_expression = pl.lit(
            float(frame_rate),
            dtype=pl.Float64,
        )

    return df.with_columns(frame_rate_expression.alias(FRAME_RATE_COLUMN))


def _read_metadata_frame_rate(
    metadata_path: Path,
    game_id: int,
) -> float:
    """Read and validate one match's FPS metadata."""
    metadata = pl.read_json(metadata_path)
    if metadata.height != 1:
        raise ValueError(
            f"Expected one metadata row for game {game_id}, "
            f"found {metadata.height}."
        )
    if "fps" not in metadata.columns:
        raise ValueError(f"Metadata for game {game_id} is missing fps.")

    frame_rate = metadata.item(0, "fps")
    if (
        not isinstance(frame_rate, (int, float))
        or isinstance(frame_rate, bool)
        or not math.isfinite(frame_rate)
        or frame_rate <= 0
    ):
        raise ValueError(
            f"Metadata fps for game {game_id} must be finite and "
            "greater than 0."
        )

    return float(frame_rate)


def resolve_gradient_sports_frame_rates(
    match_dir: str | Path | Sequence[str | Path],
    metadata_dir: str | Path,
    *,
    default_frame_rate: float = GRADIENT_SPORTS_DEFAULT_FPS,
) -> tuple[GameFrameRate, ...]:
    """Resolve FPS metadata for selected files or a match directory.

    Match IDs come from integrated parquet filenames. This avoids
    scanning tracking rows merely to discover which games are present.
    A matching JSON file supplies ``fps``; a missing file uses the
    configured global default.

    Args:
        match_dir: One or more ``<game_id>.parquet`` files, or a
            directory containing such match files.
        metadata_dir: Directory of optional ``<game_id>.json`` files.
        default_frame_rate: Fallback for missing metadata files.

    Returns:
        Sorted frame-rate resolutions, one per integrated match.

    Raises:
        FileNotFoundError: If the match path or parquet files are missing.
        ValueError: If filenames or existing metadata are invalid.

    """
    _validate_frame_rate_spec(default_frame_rate)
    if isinstance(match_dir, (str, Path)):
        integrated_path = Path(match_dir)
        if integrated_path.is_file():
            if integrated_path.suffix != ".parquet":
                raise ValueError(
                    f"Integrated match file must be Parquet: {integrated_path}"
                )
            match_files = [integrated_path]
        elif integrated_path.is_dir():
            match_files = sorted(integrated_path.glob("*.parquet"))
        else:
            raise FileNotFoundError(
                f"Integrated match path does not exist: {integrated_path}"
            )
    else:
        match_files = sorted(Path(path) for path in match_dir)
        if not match_files:
            raise ValueError(
                "At least one integrated match Parquet file is required."
            )
        for match_file in match_files:
            if not match_file.is_file():
                raise FileNotFoundError(
                    f"Integrated match file does not exist: {match_file}"
                )
            if match_file.suffix != ".parquet":
                raise ValueError(
                    f"Integrated match file must be Parquet: {match_file}"
                )

    if not match_files:
        raise FileNotFoundError(
            f"No integrated match parquet files found in: {match_dir}"
        )

    metadata_directory = Path(metadata_dir)
    resolutions: list[GameFrameRate] = []
    for match_path in match_files:
        try:
            game_id = int(match_path.stem)
        except ValueError as error:
            raise ValueError(
                "Integrated match filenames must be numeric game IDs: "
                f"{match_path.name}."
            ) from error

        metadata_path = metadata_directory / f"{game_id}.json"
        if metadata_path.is_file():
            frame_rate = _read_metadata_frame_rate(
                metadata_path,
                game_id,
            )
            source: FrameRateSource = "metadata"
        else:
            frame_rate = float(default_frame_rate)
            source = "default"

        resolutions.append(
            GameFrameRate(
                game_id=game_id,
                frame_rate=frame_rate,
                source=source,
            )
        )

    return tuple(resolutions)
