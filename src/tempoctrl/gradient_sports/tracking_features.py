"""Create player-level features from processed tracking data."""

from pathlib import Path

import polars as pl

from tempoctrl.gradient_sports.players import (
    TRACKING_LOOKUP_KEYS,
    scan_player_game_lookup,
)

METADATA_DIRECTORY = Path("data/raw/gradient_sports/metadata")

TRACKING_PLAYER_COLUMNS = (
    "home_players_smooth",
    "away_players_smooth",
)

TEAM_LOOKUP_KEYS = (
    "game_id",
    "team_side",
)

TRACKING_PLAYER_FEATURES = (
    "player_id",
    "opponent_id",
    "player_position_group",
)


def read_tracking_team_file(metadata_path: str | Path) -> pl.DataFrame:
    """Read home and away team IDs from one game metadata file.

    Args:
        metadata_path: Game metadata JSON path named by game ID.

    Returns:
        Two rows mapping the game's home and away sides to team IDs.

    Raises:
        ValueError: If the file does not contain exactly one game or its
            filename does not match the metadata game ID.

    """
    metadata_path = Path(metadata_path)
    df_metadata = pl.read_json(metadata_path, infer_schema_length=None)
    if df_metadata.height != 1:
        raise ValueError(
            "Tracking metadata file must contain exactly one game: "
            f"{metadata_path}"
        )

    game_id = int(metadata_path.stem)
    metadata_game_id = int(df_metadata.get_column("id").item())
    if metadata_game_id != game_id:
        raise ValueError(
            "Metadata game ID does not match its filename: "
            f"{metadata_game_id} != {game_id}"
        )

    df_home = df_metadata.select(
        pl.col("id").cast(pl.Int64).alias("game_id"),
        pl.lit("home").alias("team_side"),
        pl.col("homeTeam").struct.field("id").cast(pl.Int64).alias("team_id"),
    )
    df_away = df_metadata.select(
        pl.col("id").cast(pl.Int64).alias("game_id"),
        pl.lit("away").alias("team_side"),
        pl.col("awayTeam").struct.field("id").cast(pl.Int64).alias("team_id"),
    )
    return pl.concat((df_home, df_away), how="vertical")


def build_tracking_team_lookup(
    metadata_dir: str | Path = METADATA_DIRECTORY,
) -> pl.DataFrame:
    """Build a validated home-away team lookup for tracking games.

    Args:
        metadata_dir: Directory containing game metadata JSON files.

    Returns:
        A sorted table keyed by `game_id` and `team_side`.

    Raises:
        FileNotFoundError: If the directory or metadata files are
            absent.
        ValueError: If team IDs are null or side keys are duplicated.

    """
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"Metadata directory does not exist: {metadata_dir}"
        )

    metadata_paths = sorted(metadata_dir.glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No metadata JSON files found in: {metadata_dir}"
        )

    df_lookup = pl.concat(
        [read_tracking_team_file(path) for path in metadata_paths],
        how="vertical",
    )
    if df_lookup.get_column("team_id").null_count() > 0:
        raise ValueError("Tracking team lookup contains null team IDs")

    duplicate_keys = (
        df_lookup.group_by(TEAM_LOOKUP_KEYS)
        .len(name="key_count")
        .filter(pl.col("key_count") > 1)
    )
    if not duplicate_keys.is_empty():
        raise ValueError(
            "Tracking team lookup contains duplicate game-side keys"
        )

    return df_lookup.sort(TEAM_LOOKUP_KEYS)


def _tracking_side_rows(
    df_tracking: pl.LazyFrame,
    frame_columns: tuple[str, ...],
    player_column: str,
    team_side: str,
) -> pl.LazyFrame:
    """Normalize one home or away tracking-player list."""
    return (
        df_tracking.filter(
            pl.col(player_column).is_not_null()
            & (pl.col(player_column).list.len() > 0)
        )
        .select(
            *frame_columns,
            pl.lit(team_side).alias("team_side"),
            pl.col(player_column).alias("tracked_player"),
        )
        .explode("tracked_player")
        .unnest("tracked_player")
        .rename({"jerseyNum": "shirt_number"})
        .with_columns(pl.col("shirt_number").cast(pl.Int32))
    )


def tracking_to_player_rows(
    df_tracking: pl.LazyFrame,
) -> pl.LazyFrame:
    """Convert frame-level tracking lists to long player rows.

    Args:
        df_tracking: Processed tracking frames with home and away lists.

    Returns:
        Lazy tracking data with one row per observed tracked player.

    Raises:
        ValueError: If required tracking columns are missing.

    """
    tracking_schema = df_tracking.collect_schema()
    required_columns = ("game_id", *TRACKING_PLAYER_COLUMNS)
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in tracking_schema
    ]
    if missing_columns:
        formatted_columns = ", ".join(missing_columns)
        raise ValueError(
            f"Tracking data is missing required columns: {formatted_columns}"
        )

    frame_columns = tuple(
        column_name
        for column_name in tracking_schema
        if column_name not in TRACKING_PLAYER_COLUMNS
    )
    df_home = _tracking_side_rows(
        df_tracking,
        frame_columns,
        "home_players_smooth",
        "home",
    )
    df_away = _tracking_side_rows(
        df_tracking,
        frame_columns,
        "away_players_smooth",
        "away",
    )
    return pl.concat((df_home, df_away), how="vertical")


def add_tracking_player_features(
    df_tracking: pl.LazyFrame,
    df_team_lookup: pl.LazyFrame | None = None,
    df_player_lookup: pl.LazyFrame | None = None,
    *,
    metadata_dir: str | Path = METADATA_DIRECTORY,
) -> pl.LazyFrame:
    """Add team and player lookup features to long-form tracking.

    Args:
        df_tracking: Processed frame-level tracking data.
        df_team_lookup: Optional home-away team lookup for testing or
            reuse; defaults to lookup data built from game metadata.
        df_player_lookup: Optional player-game lookup; defaults to the
            processed player lookup Parquet file.
        metadata_dir: Raw metadata source used for the default team
            lookup.

    Returns:
        Lazy player-level tracking with team, player, opponent, and
        position identifiers.

    """
    df_players = tracking_to_player_rows(df_tracking)
    if df_team_lookup is None:
        df_team_lookup = build_tracking_team_lookup(metadata_dir).lazy()
    if df_player_lookup is None:
        df_player_lookup = scan_player_game_lookup()

    player_schema = df_players.collect_schema()
    team_schema = df_team_lookup.collect_schema()
    missing_team_columns = [
        column_name
        for column_name in (*TEAM_LOOKUP_KEYS, "team_id")
        if column_name not in team_schema
    ]
    if missing_team_columns:
        formatted_columns = ", ".join(missing_team_columns)
        raise ValueError(
            f"Tracking team lookup is missing required columns: "
            f"{formatted_columns}"
        )

    df_team_features = df_team_lookup.select(
        pl.col("game_id").cast(player_schema["game_id"]),
        "team_side",
        pl.col("team_id").cast(pl.Int64),
    )
    df_players = df_players.join(
        df_team_features,
        on=TEAM_LOOKUP_KEYS,
        how="left",
        coalesce=True,
        validate="m:1",
    )

    enriched_schema = df_players.collect_schema()
    lookup_schema = df_player_lookup.collect_schema()
    required_player_columns = (
        *TRACKING_LOOKUP_KEYS,
        *TRACKING_PLAYER_FEATURES,
    )
    missing_player_columns = [
        column_name
        for column_name in required_player_columns
        if column_name not in lookup_schema
    ]
    if missing_player_columns:
        formatted_columns = ", ".join(missing_player_columns)
        raise ValueError(
            f"Player lookup is missing tracking columns: {formatted_columns}"
        )

    player_key_expressions = [
        pl.col(column_name)
        .cast(enriched_schema[column_name])
        .alias(column_name)
        for column_name in TRACKING_LOOKUP_KEYS
    ]
    df_player_features = df_player_lookup.select(
        *player_key_expressions,
        *TRACKING_PLAYER_FEATURES,
    )
    return df_players.join(
        df_player_features,
        on=TRACKING_LOOKUP_KEYS,
        how="left",
        coalesce=True,
        validate="m:1",
    )
