"""Load Gradient Sports event data."""

from collections.abc import Sequence
from pathlib import Path
from shutil import copy2, rmtree
from tempfile import mkdtemp

import polars as pl

from tempoctrl.gradient_sports.players import (
    PLAYER_LOOKUP_KEYS,
    PLAYER_LOOKUP_SCHEMA,
)

EVENTS_OUTPUT_DIRECTORY = Path("data/processed/gradient_sports/events")
MATCH_POSSESSION_LOOKUP_PATH = Path(
    "data/curated/gradient_sports/possession_lookup/"
    "match_possession_lookup.parquet"
)
PLAYER_GAME_LOOKUP_PATH = Path(
    "data/curated/gradient_sports/metadata_lookup/player_game_lookup.parquet"
)

MATCH_POSSESSION_LOOKUP_COLUMNS = (
    "game_id",
    "match_team_player_possession_id",
    "player_id",
    "playername",
    "team_id",
    "teamname",
)

ADDITIONAL_POSSESSION_COLUMNS = (
    "opponent_id",
    "player_possession_group",
    "started",
)


def add_opposition_teamname(
    df_player_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Add each opponent ID's team name to player-game metadata."""
    df_opponent_names = (
        df_player_lookup.select("game_id", "team_id", "team_name")
        .unique()
        .rename(
            {
                "team_id": "opponent_id",
                "team_name": "opposition_teamname",
            }
        )
    )
    return df_player_lookup.join(
        df_opponent_names,
        on=("game_id", "opponent_id"),
        how="left",
        validate="m:1",
    )


def add_player_metadata(
    df_possession_lookup: pl.DataFrame,
    player_lookup_path: str | Path = PLAYER_GAME_LOOKUP_PATH,
) -> pl.DataFrame:
    """Normalize lookup IDs and add available player-game metadata.

    Normalize incoming game, team, and player join keys to the player
    lookup's Int64 convention, including empty and Null-typed columns.
    Preserve null values. Invalid string IDs raise instead of silently
    becoming null. Normalize even when optional metadata is absent.
    """
    df_possession_lookup = _normalize_lookup_ids(df_possession_lookup)
    player_lookup_path = Path(player_lookup_path)
    if not player_lookup_path.is_file():
        return df_possession_lookup

    df_player_metadata = (
        pl.read_parquet(player_lookup_path)
        .pipe(add_opposition_teamname)
        .select(
            "game_id",
            "team_id",
            "player_id",
            "opponent_id",
            pl.col("player_position_group").alias("player_possession_group"),
            "started",
            "opposition_teamname",
        )
    )
    return df_possession_lookup.join(
        df_player_metadata,
        on=PLAYER_LOOKUP_KEYS,
        how="left",
        validate="m:1",
    )


def _normalize_lookup_ids(df_in: pl.DataFrame) -> pl.DataFrame:
    """Normalize integer, String, and Null lookup keys to canonical IDs.

    Reject other dtypes to avoid truncating fractional IDs or treating
    Boolean values as numeric identifiers.
    """
    for column in PLAYER_LOOKUP_KEYS:
        dtype = df_in.schema[column]
        if dtype not in (pl.String, pl.Null) and not dtype.is_integer():
            raise ValueError(
                f"Unsupported lookup ID dtype for {column}: {dtype}"
            )
    return df_in.cast(
        {
            column: PLAYER_LOOKUP_SCHEMA[column]
            for column in PLAYER_LOOKUP_KEYS
        },
        strict=True,
    )


def _build_match_possession_lookup(
    df_in: pl.DataFrame,
    output_path: Path,
    player_lookup_path: str | Path,
) -> pl.DataFrame:
    """Prepare an enriched lookup without changing existing files."""
    df_in = _normalize_lookup_ids(df_in)
    game_ids = df_in.get_column("game_id").drop_nulls().unique().to_list()
    df_lookup = (
        df_in.select(MATCH_POSSESSION_LOOKUP_COLUMNS)
        .filter(pl.col("match_team_player_possession_id").is_not_null())
        .unique()
        .pipe(add_player_metadata, player_lookup_path)
    )

    if output_path.is_file():
        df_existing = (
            pl.read_parquet(output_path).pipe(_normalize_lookup_ids)
            .filter(~pl.col("game_id").is_in(game_ids))
        )
        df_lookup = pl.concat(
            (df_existing, df_lookup),
            how="diagonal_relaxed",
        )

    return df_lookup.sort(
        ("game_id", "match_team_player_possession_id")
    )


def _write_event_outputs(
    outputs: Sequence[tuple[Path, pl.DataFrame]],
) -> None:
    """Prepare Parquets before publishing them in the supplied order.

    Validate each temporary file against its source frame. Keep backup
    copies beside staged files and atomically replace each destination.
    Restore published files if replacement raises, removing new files
    that had no predecessor. Retain backups if restoration itself fails.

    Each staging directory shares its destination's filesystem. This
    supports rollback after handled errors, not a transaction across
    files during a process crash or concurrent reads and writes.
    """
    targets = [path.resolve() for path, _ in outputs]
    if len(set(targets)) != len(targets):
        raise ValueError("Event output paths must be distinct")

    temporary_dirs: list[Path] = []
    prepared: list[tuple[Path, Path, Path | None]] = []
    published: list[tuple[Path, Path | None]] = []
    keep_backups = False
    try:
        for target, frame in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir = Path(
                mkdtemp(dir=target.parent, prefix=".event-output-")
            )
            temporary_dirs.append(temporary_dir)
            staged = temporary_dir / target.name
            frame.write_parquet(staged, compression="zstd")
            restored = pl.read_parquet(staged)
            if restored.schema != frame.schema or not restored.equals(frame):
                raise ValueError(f"Event output validation failed: {target}")
            backup = None
            if target.exists():
                backup = temporary_dir / "backups" / target.name
                backup.parent.mkdir()
                copy2(target, backup)
            prepared.append((target, staged, backup))

        try:
            for target, staged, backup in prepared:
                staged.replace(target)
                published.append((target, backup))
        except BaseException as error:
            for target, backup in reversed(published):
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        backup.replace(target)
                except OSError as recovery_error:
                    keep_backups = True
                    error.add_note(
                        f"Could not restore {target}: {recovery_error}. "
                        f"Recovery copy: {backup}"
                    )
            raise
    finally:
        if not keep_backups:
            for temporary_dir in temporary_dirs:
                rmtree(temporary_dir)


def write_match_possession_lookup(
    df_in: pl.DataFrame,
    output_path: str | Path = MATCH_POSSESSION_LOOKUP_PATH,
    player_lookup_path: str | Path = PLAYER_GAME_LOOKUP_PATH,
) -> Path:
    """Replace selected matches while preserving other lookup rows.

    Enrich incoming possessions once and retain all columns on existing
    rows from other matches. A selected match without any identified
    possessions removes its old lookup rows. Write and validate a
    temporary Parquet before atomically replacing the existing lookup.
    """
    output_path = Path(output_path)
    df_lookup = _build_match_possession_lookup(
        df_in, output_path, player_lookup_path
    )
    _write_event_outputs([(output_path, df_lookup)])
    return output_path


def load_events(
    df_in: pl.DataFrame,
    *,
    output_dir: str | Path = EVENTS_OUTPUT_DIRECTORY,
    possession_lookup_path: str | Path = MATCH_POSSESSION_LOOKUP_PATH,
    player_lookup_path: str | Path = PLAYER_GAME_LOOKUP_PATH,
) -> tuple[Path, ...]:
    """Write event files per match and update the shared lookup once.

    Preserve each match's source row order, columns, and dtypes. Replace
    event files only for selected matches and retain unrelated files.
    Prepare and validate all Parquets before publishing match files,
    then the shared lookup. Handled replacement failures restore prior
    outputs. Publication is not crash-atomic across multiple files.

    Args:
        df_in: Processed events for one or more matches.
        output_dir: Directory receiving per-match event Parquet files.
        possession_lookup_path: Shared possession lookup destination.
        player_lookup_path: Optional player metadata source.

    Returns:
        Event paths ordered by match ID, followed by the lookup path.
        Single-match calls retain the original two-path return value.

    Raises:
        ValueError: If the input is empty or any game_id is null.

    """
    if df_in.is_empty() or df_in.get_column("game_id").null_count() > 0:
        raise ValueError(
            "Event data must contain rows with non-null game_id values"
        )

    match_frames = df_in.partition_by(
        "game_id", as_dict=True, maintain_order=True
    )
    output_directory = Path(output_dir)
    lookup_path = Path(possession_lookup_path)
    df_lookup = _build_match_possession_lookup(
        df_in, lookup_path, player_lookup_path
    )
    outputs = [
        (output_directory / f"{match_id}.parquet", df_match)
        for (match_id,), df_match in sorted(match_frames.items())
    ]
    _write_event_outputs([*outputs, (lookup_path, df_lookup)])

    return (*(path for path, _ in outputs), lookup_path)
