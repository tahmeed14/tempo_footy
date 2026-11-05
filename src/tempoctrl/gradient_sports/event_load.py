"""Load Gradient Sports event data."""

from pathlib import Path

import polars as pl

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
    """Upsert one match and overwrite the possession lookup file."""
    output_path = Path(output_path)
    game_ids = df_in.get_column("game_id").drop_nulls().unique()
    df_lookup = (
        df_in.select(MATCH_POSSESSION_LOOKUP_COLUMNS)
        .filter(pl.col("match_team_player_possession_id").is_not_null())
        .unique()
    )

    if output_path.is_file():
        df_existing = (
            pl.read_parquet(output_path)
            .select(MATCH_POSSESSION_LOOKUP_COLUMNS)
            .filter(~pl.col("game_id").is_in(game_ids))
        )
        df_lookup = pl.concat(
            (df_existing, df_lookup),
            how="vertical_relaxed",
        )

    df_lookup = df_lookup.pipe(add_player_metadata, player_lookup_path).sort(
        ("game_id", "match_team_player_possession_id")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_lookup.write_parquet(output_path, compression="zstd")
    return output_path


def load_events(df_in: pl.DataFrame) -> tuple[Path, Path]:
    """Write one match of event data and return both output paths.

    Returns:
        The event parquet path followed by the possession lookup path.

    """
    match_ids = df_in.get_column("game_id").drop_nulls().unique().to_list()
    if len(match_ids) != 1:
        raise ValueError(
            "Event data must contain exactly one non-null game_id"
        )

    output_path = Path(
        f"data/processed/gradient_sports/events/{match_ids[0]}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    possession_lookup_path = write_match_possession_lookup(df_in)
    df_in.write_parquet(output_path, compression="zstd")
    return output_path, possession_lookup_path
