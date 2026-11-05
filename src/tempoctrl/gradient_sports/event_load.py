"""Load Gradient Sports event data."""

from pathlib import Path

import polars as pl

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
    """Add available player-game metadata to a possession lookup."""
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
        on=("game_id", "team_id", "player_id"),
        how="left",
        validate="m:1",
    )


def write_match_possession_lookup(
    df_in: pl.DataFrame,
    output_path: str | Path = MATCH_POSSESSION_LOOKUP_PATH,
    player_lookup_path: str | Path = PLAYER_GAME_LOOKUP_PATH,
) -> Path:
    """Replace selected matches while preserving other lookup rows.

    Enrich incoming possessions once and retain all columns on existing
    rows from other matches. A selected match without any identified
    possessions removes its old lookup rows.
    """
    output_path = Path(output_path)
    game_ids = df_in.get_column("game_id").drop_nulls().unique().to_list()
    df_lookup = (
        df_in.select(MATCH_POSSESSION_LOOKUP_COLUMNS)
        .filter(pl.col("match_team_player_possession_id").is_not_null())
        .unique()
        .pipe(add_player_metadata, player_lookup_path)
    )

    if output_path.is_file():
        df_existing = (
            pl.read_parquet(output_path)
            .filter(~pl.col("game_id").is_in(game_ids))
        )
        df_lookup = pl.concat(
            (df_existing, df_lookup),
            how="diagonal_relaxed",
        )

    df_lookup = df_lookup.sort(
        ("game_id", "match_team_player_possession_id")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_lookup.write_parquet(output_path, compression="zstd")
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
    output_directory.mkdir(parents=True, exist_ok=True)
    lookup_path = write_match_possession_lookup(
        df_in,
        output_path=possession_lookup_path,
        player_lookup_path=player_lookup_path,
    )
    output_paths = []
    for (match_id,), df_match in sorted(match_frames.items()):
        output_path = output_directory / f"{match_id}.parquet"
        df_match.write_parquet(output_path, compression="zstd")
        output_paths.append(output_path)

    return (*output_paths, lookup_path)
