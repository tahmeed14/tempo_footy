"""Ingest processed Gradient Sports data."""

import bz2
from collections.abc import Sequence
from pathlib import Path

import polars as pl

RAW_EVENTS_DIRECTORY = Path("data/raw/gradient_sports/events")


def discover_event_files(
    input_dir: str | Path = RAW_EVENTS_DIRECTORY,
) -> tuple[tuple[int, Path], ...]:
    """Discover raw event files with distinct numeric match IDs.

    Raise ValueError when multiple filenames resolve to the same ID.
    """
    input_directory = Path(input_dir)
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Raw event directory does not exist: {input_directory}"
        )

    event_files = sorted(input_directory.glob("*.json"))
    if not event_files:
        raise FileNotFoundError(
            f"No raw event JSON files found in: {input_directory}"
        )

    discovered_files: list[tuple[int, Path]] = []
    paths_by_match_id: dict[int, Path] = {}
    for event_path in event_files:
        try:
            match_id = int(event_path.stem)
        except ValueError as error:
            raise ValueError(
                "Raw event filenames must be numeric match IDs: "
                f"{event_path.name}"
            ) from error
        if match_id in paths_by_match_id:
            raise ValueError(
                f"Duplicate raw event match ID {match_id}: "
                f"{paths_by_match_id[match_id]} and {event_path}"
            )
        paths_by_match_id[match_id] = event_path
        discovered_files.append((match_id, event_path))

    return tuple(sorted(discovered_files))


def resolve_event_paths(
    match_id: int | str,
    raw_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve event paths and reject conflicting numeric source names.

    Custom nonnumeric filenames are allowed; their content is validated
    when staging. Numeric filenames must identify the requested match.
    """
    resolved_raw_path = (
        Path(raw_path)
        if raw_path is not None
        else RAW_EVENTS_DIRECTORY / f"{match_id}.json"
    )
    staged_path = Path(
        f"data/staged/gradient_sports/events/{match_id}.parquet"
    )
    try:
        filename_id = int(resolved_raw_path.stem)
    except ValueError:
        pass
    else:
        if filename_id != int(match_id):
            raise ValueError(
                f"Raw event filename match ID mismatch: expected {match_id}, "
                f"found {filename_id} in {resolved_raw_path}"
            )
    return resolved_raw_path, staged_path


def _validate_event_match_id(
    df_events: pl.LazyFrame,
    match_id: int | str,
    source_path: Path,
) -> None:
    """Require every source row to identify the requested numeric match.

    Read only gameId from cached Parquet.
    Accept integer IDs or canonical numeric strings.
    Reject missing, null, empty, or mixed IDs.
    """
    expected_id = int(match_id)
    schema = df_events.collect_schema()
    if "gameId" not in schema:
        raise ValueError(f"Event data is missing gameId: {source_path}")
    if not (schema["gameId"].is_integer() or schema["gameId"] == pl.String):
        raise ValueError(f"Event data has invalid gameId dtype: {source_path}")
    game_ids = (
        df_events.select("gameId").unique().collect().get_column("gameId")
    )
    if (
        game_ids.len() != 1
        or game_ids.null_count() > 0
        or str(game_ids.item()) != str(expected_id)
    ):
        raise ValueError(
            f"Event match ID mismatch in {source_path}: "
            f"expected {expected_id}, found {game_ids.to_list()}"
        )


def event_stage_is_current(
    match_id: int | str,
    overwrite: bool = False,
    *,
    raw_path: str | Path | None = None,
) -> bool:
    """Reuse a current stage only when all rows match the requested ID.

    A stage with invalid match IDs is not current and must be restaged.
    A conflicting numeric source filename raises ValueError.
    """
    resolved_raw_path, staged_path = resolve_event_paths(match_id, raw_path)
    if not resolved_raw_path.is_file():
        raise FileNotFoundError(
            f"Raw event file not found: {resolved_raw_path}"
        )

    is_current = (
        staged_path.is_file()
        and staged_path.stat().st_mtime >= resolved_raw_path.stat().st_mtime
        and not overwrite
    )
    if not is_current:
        return False
    try:
        _validate_event_match_id(
            pl.scan_parquet(staged_path), match_id, staged_path
        )
    except ValueError:
        return False
    return True


def stage_events(
    match_id: int | str,
    overwrite: bool = False,
    *,
    raw_path: str | Path | None = None,
) -> Path:
    """Stage raw event JSON as Parquet without adding derived columns.

    With overwrite=False, create a missing stage, replace a stage older
    than the raw JSON, and reuse a current stage. With overwrite=True,
    always restage and replace any existing staged file.

    Read the full JSON schema and preserve nested values and row order.
    Validate gameId against the requested match before writing.
    Invalid cached match IDs trigger restaging. Numeric source filenames
    must also agree with the requested ID.

    Args:
        match_id: Match identifier used for the staged filename.
        overwrite: Force restaging even when the staged file is current.
        raw_path: Optional source path instead of the default raw path.

    Returns:
        The path of the reused or newly written staged Parquet file.

    Raises:
        ValueError: If filenames or gameId values disagree with the
            requested match, or source IDs are missing or invalid.

    """
    resolved_raw_path, staged_path = resolve_event_paths(match_id, raw_path)
    if event_stage_is_current(
        match_id,
        overwrite,
        raw_path=resolved_raw_path,
    ):
        return staged_path

    df = pl.read_json(resolved_raw_path, infer_schema_length=None)
    _validate_event_match_id(df.lazy(), match_id, resolved_raw_path)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(staged_path, compression="zstd")
    return staged_path


def scan_events(
    df_path: str | Path | Sequence[str | Path],
) -> pl.LazyFrame:
    """Scan selected staged events and add one-based event numbers.

    A single file retains its source row numbering. For multiple files,
    number rows within each ``gameId`` across the combined input.
    Preserve supplied file order and source row order within each file.
    Combine differing source schemas by name and fill absent fields with
    nulls, promoting compatible dtypes when needed.

    Args:
        df_path: A staged Parquet path or sequence of selected paths.

    Returns:
        A lazy query with a leading UInt32 ``event_number`` column.

    Raises:
        ValueError: If no paths are supplied or a multi-file input lacks
            the ``gameId`` column required for per-match numbering.

    """
    paths = [df_path] if isinstance(df_path, (str, Path)) else list(df_path)
    if not paths:
        raise ValueError("At least one staged event Parquet path is required.")

    scans = [pl.scan_parquet(path) for path in paths]
    if len(scans) == 1:
        return scans[0].with_row_index("event_number", offset=1)

    for path, scan in zip(paths, scans, strict=True):
        if "gameId" not in scan.collect_schema():
            raise ValueError(f"Staged event file is missing gameId: {path}")

    return pl.concat(scans, how="diagonal_relaxed").select(
        pl.int_range(1, pl.len() + 1, dtype=pl.UInt32)
        .over("gameId")
        .alias("event_number"),
        pl.all(),
    )


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
