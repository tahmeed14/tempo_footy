"""Ingest processed Gradient Sports data."""

import bz2
from collections.abc import Sequence
from pathlib import Path

import polars as pl


def read_events(local_path: str | Path) -> pl.DataFrame:
    """Read and unnest events JSON for a single match.

    Reads the raw events JSON for `local_path`, assigns a 1-based
    `event_number` index, and returns the DataFrame with selected
    columns unnested via `selectRelevantEventsColumns`.

    Args:
        local_path: Path to the events JSON file.

    Returns:
    -------
    - pl.DataFrame: unnested events ready for downstream processing.

    """
    # TODO: Issue #3
    df = pl.read_json(local_path, infer_schema_length=None)
    return df.with_row_index("event_number", offset=1)


def resolve_tracking_paths(
    match_id: int | str,
    raw_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return the raw and staged paths for one tracking match."""
    resolved_raw_path = (
        Path(raw_path)
        if raw_path is not None
        else Path(f"data/raw/gradient_sports/tracking/{match_id}.jsonl.bz2")
    )
    staged_path = Path(
        f"data/staged/gradient_sports/tracking/{match_id}.parquet"
    )
    return resolved_raw_path, staged_path


def tracking_stage_is_current(
    match_id: int | str,
    overwrite: bool = False,
    *,
    raw_path: str | Path | None = None,
) -> bool:
    """Return whether an existing staged file can be reused."""
    resolved_raw_path, staged_path = resolve_tracking_paths(
        match_id,
        raw_path,
    )
    if not resolved_raw_path.is_file():
        raise FileNotFoundError(
            f"Raw tracking file not found: {resolved_raw_path}"
        )

    return (
        staged_path.is_file()
        and staged_path.stat().st_mtime >= resolved_raw_path.stat().st_mtime
        and not overwrite
    )


def stage_tracking(
    match_id: int | str,
    overwrite: bool = False,
    *,
    raw_path: str | Path | None = None,
) -> Path:
    """Stage one raw tracking file and return its parquet path."""
    resolved_raw_path, staged_path = resolve_tracking_paths(
        match_id,
        raw_path,
    )

    if tracking_stage_is_current(
        match_id,
        overwrite,
        raw_path=resolved_raw_path,
    ):
        return staged_path

    staged_path.parent.mkdir(parents=True, exist_ok=True)

    with bz2.open(resolved_raw_path, "rb") as file:
        (
            pl.scan_ndjson(
                file,
                infer_schema_length=10_000,
            ).sink_parquet(staged_path)
        )

    return staged_path


def scan_tracking(df_path: str | Path) -> pl.LazyFrame:
    """Lazily scan a staged tracking Parquet file.

    Args:
        df_path: Path returned by ``stage_tracking``.

    Returns:
        A lazy tracking-data query for downstream transformations.

    """
    return pl.scan_parquet(df_path)


def scan_processed_files(
    df_path: str | Path | Sequence[str | Path],
    columns: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    """Lazily scan one Parquet file or every file in a directory.

    Args:
        df_path: One or more integrated match-level Parquet files, or a
            directory containing such files.
        columns: Optional columns to read from each file.

    Returns:
        One lazy query spanning the selected match files.

    Raises:
        FileNotFoundError: If the path or Parquet files do not exist.

    """
    if isinstance(df_path, (str, Path)):
        input_path = Path(df_path)
        if input_path.is_file():
            if input_path.suffix != ".parquet":
                raise ValueError(
                    f"Integrated input file must be Parquet: {input_path}"
                )
            parquet_files = [input_path]
        elif input_path.is_dir():
            parquet_files = sorted(input_path.glob("*.parquet"))
        else:
            raise FileNotFoundError(
                f"Integrated data path does not exist: {input_path}"
            )
    else:
        parquet_files = sorted(Path(path) for path in df_path)
        if not parquet_files:
            raise ValueError(
                "At least one integrated Parquet file is required."
            )
        for parquet_path in parquet_files:
            if not parquet_path.is_file():
                raise FileNotFoundError(
                    f"Integrated data file does not exist: {parquet_path}"
                )
            if parquet_path.suffix != ".parquet":
                raise ValueError(
                    f"Integrated input file must be Parquet: {parquet_path}"
                )

    if not parquet_files:
        raise FileNotFoundError(
            f"No integrated Parquet files found in: {input_path}"
        )

    lf_out = pl.scan_parquet(parquet_files)

    if columns:
        return lf_out.select(columns)

    return lf_out
