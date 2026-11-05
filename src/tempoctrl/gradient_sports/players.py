"""Build player lookup data from Gradient Sports roster files."""

from pathlib import Path

import polars as pl

PLAYER_LOOKUP_PATH = Path(
    "data/curated/gradient_sports/metadata_lookup/player_game_lookup.parquet"
)

POSITION_MAPPER = {
    # Goalkeepers
    "GK": "Goalkeeper",
    # Defenders
    "CB": "Centre Back",
    "RCB": "Centre Back",
    "LCB": "Centre Back",
    "MCB": "Centre Back",
    "LB": "Full Back",
    "RB": "Full Back",
    "LWB": "Full Back",
    "RWB": "Full Back",
    # Midfielders
    "DM": "Midfielder",
    "LDM": "Midfielder",
    "RDM": "Midfielder",
    "CM": "Midfielder",
    "LCM": "Midfielder",
    "RCM": "Midfielder",
    "AM": "Midfielder",
    "LAM": "Midfielder",
    "RAM": "Midfielder",
    # Attackers
    "LW": "Winger",
    "RW": "Winger",
    "LM": "Winger",
    "RM": "Winger",
    "ST": "Forward",
    "CF": "Forward",
    "SS": "Forward",
    "LS": "Forward",
    "RS": "Forward",
}

ROSTER_PLAYER_COLUMNS = (
    "game_id",
    "team_id",
    "player_id",
    "shirt_number",
    "position_group_type",
    "player_position_group",
    "player_name",
    "team_name",
    "started",
)

ROSTER_PLAYER_SCHEMA = {
    "game_id": pl.Int64,
    "team_id": pl.Int64,
    "player_id": pl.Int64,
    "shirt_number": pl.Int32,
    "position_group_type": pl.String,
    "player_position_group": pl.String,
    "player_name": pl.String,
    "team_name": pl.String,
    "started": pl.Boolean,
}

PLAYER_LOOKUP_COLUMNS = (
    "game_id",
    "team_id",
    "opponent_id",
    "player_id",
    "shirt_number",
    "position_group_type",
    "player_position_group",
    "player_name",
    "team_name",
    "started",
)

PLAYER_LOOKUP_SCHEMA = {
    **ROSTER_PLAYER_SCHEMA,
    "opponent_id": pl.Int64,
}

PLAYER_LOOKUP_KEYS = (
    "game_id",
    "team_id",
    "player_id",
)

PLAYER_FEATURE_COLUMNS = (
    "opponent_id",
    "player_position_group",
)

TRACKING_LOOKUP_KEYS = (
    "game_id",
    "team_id",
    "shirt_number",
)

REQUIRED_ID_COLUMNS = (
    "game_id",
    "team_id",
    "player_id",
    "shirt_number",
)


def _parse_game_id(roster_path: Path) -> int:
    """Parse a numeric game identifier from a roster filename."""
    try:
        return int(roster_path.stem)
    except ValueError as error:
        raise ValueError(
            f"Roster filename must be a numeric game ID: {roster_path.name}"
        ) from error


def _validate_position_groups(df_roster: pl.DataFrame) -> None:
    """Raise an error when a roster has an unmapped position code."""
    position_codes = set(
        df_roster.get_column("position_group_type").drop_nulls().to_list()
    )
    unmapped_codes = sorted(position_codes - POSITION_MAPPER.keys())
    if unmapped_codes:
        formatted_codes = ", ".join(unmapped_codes)
        raise ValueError(
            f"Unmapped positionGroupType values: {formatted_codes}"
        )


def _validate_required_ids(df_lookup: pl.DataFrame) -> None:
    """Raise an error when a required lookup identifier is null."""
    null_counts = {
        column_name: df_lookup.get_column(column_name).null_count()
        for column_name in REQUIRED_ID_COLUMNS
    }
    invalid_counts = {
        column_name: null_count
        for column_name, null_count in null_counts.items()
        if null_count > 0
    }
    if invalid_counts:
        formatted_counts = ", ".join(
            f"{column_name}={null_count}"
            for column_name, null_count in invalid_counts.items()
        )
        raise ValueError(
            f"Player lookup contains null identifiers: {formatted_counts}"
        )


def _validate_two_teams_per_game(df_lookup: pl.DataFrame) -> None:
    """Raise an error unless every game contains exactly two teams."""
    invalid_games = (
        df_lookup.group_by("game_id")
        .agg(pl.col("team_id").n_unique().alias("team_count"))
        .filter(pl.col("team_count") != 2)
        .sort("game_id")
    )
    if not invalid_games.is_empty():
        game_counts = ", ".join(
            f"{game_id}={team_count}"
            for game_id, team_count in invalid_games.iter_rows()
        )
        raise ValueError(
            f"Each game must contain exactly two teams; found: {game_counts}"
        )


def _validate_unique_key(
    df_lookup: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> None:
    """Raise an error when a composite lookup key is duplicated."""
    duplicate_keys = (
        df_lookup.group_by(key_columns)
        .len(name="key_count")
        .filter(pl.col("key_count") > 1)
        .sort(key_columns)
    )
    if not duplicate_keys.is_empty():
        key_name = ", ".join(key_columns)
        duplicate_values = ", ".join(
            str(tuple(row[:-1])) for row in duplicate_keys.iter_rows()
        )
        raise ValueError(
            f"Duplicate player lookup key ({key_name}): {duplicate_values}"
        )


def read_roster_file(roster_path: str | Path) -> pl.DataFrame:
    """Read and normalize one Gradient Sports roster file.

    Args:
        roster_path: JSON roster path named with its numeric game ID.

    Returns:
        A typed player-game table with mapped position groups.

    Raises:
        ValueError: If the filename or a position code is invalid.

    """
    roster_path = Path(roster_path)
    game_id = _parse_game_id(roster_path)
    df_raw = pl.read_json(roster_path, infer_schema_length=None)

    df_roster = df_raw.select(
        pl.lit(game_id).cast(pl.Int64).alias("game_id"),
        pl.col("team").struct.field("id").cast(pl.Int64).alias("team_id"),
        pl.col("player").struct.field("id").cast(pl.Int64).alias("player_id"),
        pl.col("shirtNumber").cast(pl.Int32).alias("shirt_number"),
        pl.col("positionGroupType")
        .cast(pl.String)
        .alias("position_group_type"),
        pl.col("player")
        .struct.field("nickname")
        .cast(pl.String)
        .alias("player_name"),
        pl.col("team").struct.field("name").cast(pl.String).alias("team_name"),
        pl.col("started").cast(pl.Boolean),
    )
    _validate_position_groups(df_roster)

    return (
        df_roster.with_columns(
            pl.col("position_group_type")
            .replace_strict(
                POSITION_MAPPER,
                default=None,
                return_dtype=pl.String,
            )
            .alias("player_position_group")
        )
        .cast(ROSTER_PLAYER_SCHEMA)
        .select(ROSTER_PLAYER_COLUMNS)
    )


def add_opponent_ids(df_lookup: pl.DataFrame) -> pl.DataFrame:
    """Add the opposing team identifier to every player-game row.

    Args:
        df_lookup: Validated player rows with two teams per game.

    Returns:
        Player rows with a non-null `opponent_id` column.

    Raises:
        ValueError: If required IDs or team cardinality are invalid.

    """
    _validate_required_ids(df_lookup)
    _validate_two_teams_per_game(df_lookup)

    df_teams = df_lookup.select("game_id", "team_id").unique()
    df_opponents = (
        df_teams.join(
            df_teams.rename({"team_id": "opponent_id"}),
            on="game_id",
            how="inner",
        )
        .filter(pl.col("team_id") != pl.col("opponent_id"))
        .sort(("game_id", "team_id"))
    )
    df_out = df_lookup.join(
        df_opponents,
        on=("game_id", "team_id"),
        how="left",
        validate="m:1",
    )
    if df_out.height != df_lookup.height:
        raise RuntimeError("Opponent join changed the player lookup row count")
    if df_out.get_column("opponent_id").null_count() > 0:
        raise RuntimeError("Opponent join produced null opponent IDs")

    return df_out.cast(PLAYER_LOOKUP_SCHEMA).select(PLAYER_LOOKUP_COLUMNS)


def build_player_game_lookup(roster_dir: str | Path) -> pl.DataFrame:
    """Build and validate a player-game lookup from roster files.

    Args:
        roster_dir: Directory containing game-ID-named roster files.

    Returns:
        A sorted player-game table spanning every roster file.

    Raises:
        FileNotFoundError: If the directory or roster files are absent.
        ValueError: If an input roster violates lookup invariants.

    """
    roster_dir = Path(roster_dir)
    if not roster_dir.is_dir():
        raise FileNotFoundError(
            f"Roster directory does not exist: {roster_dir}"
        )

    roster_paths = sorted(roster_dir.glob("*.json"))
    if not roster_paths:
        raise FileNotFoundError(f"No roster JSON files found in: {roster_dir}")

    roster_frames = []
    for roster_path in roster_paths:
        df_roster = read_roster_file(roster_path)
        if df_roster.is_empty():
            raise ValueError(f"Roster file contains no players: {roster_path}")
        roster_frames.append(df_roster)

    df_lookup = pl.concat(roster_frames, how="vertical")
    _validate_required_ids(df_lookup)
    _validate_two_teams_per_game(df_lookup)
    _validate_unique_key(df_lookup, PLAYER_LOOKUP_KEYS)
    _validate_unique_key(df_lookup, TRACKING_LOOKUP_KEYS)

    return add_opponent_ids(df_lookup).sort(PLAYER_LOOKUP_KEYS)


def write_player_game_lookup(
    df_lookup: pl.DataFrame,
    output_path: str | Path = PLAYER_LOOKUP_PATH,
    overwrite: bool = False,
) -> Path:
    """Write a player-game lookup to compressed Parquet.

    Args:
        df_lookup: Validated lookup returned by the lookup builder.
        output_path: Destination for the lookup Parquet file.
        overwrite: Whether to replace an existing lookup file.

    Returns:
        The resolved output path used by the writer.

    """
    output_path = Path(output_path)
    if output_path.is_file() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_lookup.write_parquet(output_path, compression="zstd")
    return output_path


def scan_player_game_lookup(
    lookup_path: str | Path = PLAYER_LOOKUP_PATH,
) -> pl.LazyFrame:
    """Lazily scan a processed player-game lookup.

    Args:
        lookup_path: Path to the processed player lookup Parquet file.

    Returns:
        A lazy query preserving the stored lookup schema.

    Raises:
        FileNotFoundError: If the lookup file does not exist.

    """
    lookup_path = Path(lookup_path)
    if not lookup_path.is_file():
        raise FileNotFoundError(
            f"Player lookup file does not exist: {lookup_path}"
        )

    return pl.scan_parquet(lookup_path)


def add_player_lookup_features(
    df_in: pl.LazyFrame,
    df_lookup: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Add player position and opponent features to downstream rows.

    Args:
        df_in: Lazy downstream data containing the player lookup keys.
        df_lookup: Optional lazy lookup; defaults to the processed file.

    Returns:
        A lazy left join preserving every downstream input row.

    Raises:
        ValueError: If required columns are missing or already exist.

    """
    if df_lookup is None:
        df_lookup = scan_player_game_lookup()

    input_schema = df_in.collect_schema()
    lookup_schema = df_lookup.collect_schema()
    missing_input_keys = [
        column_name
        for column_name in PLAYER_LOOKUP_KEYS
        if column_name not in input_schema
    ]
    if missing_input_keys:
        missing_columns = ", ".join(missing_input_keys)
        raise ValueError(
            f"Downstream data is missing player lookup keys: {missing_columns}"
        )

    required_lookup_columns = (
        *PLAYER_LOOKUP_KEYS,
        *PLAYER_FEATURE_COLUMNS,
    )
    missing_lookup_columns = [
        column_name
        for column_name in required_lookup_columns
        if column_name not in lookup_schema
    ]
    if missing_lookup_columns:
        missing_columns = ", ".join(missing_lookup_columns)
        raise ValueError(
            f"Player lookup is missing required columns: {missing_columns}"
        )

    existing_features = [
        column_name
        for column_name in PLAYER_FEATURE_COLUMNS
        if column_name in input_schema
    ]
    if existing_features:
        existing_columns = ", ".join(existing_features)
        raise ValueError(
            f"Downstream data already contains player features: "
            f"{existing_columns}"
        )

    lookup_key_expressions = [
        pl.col(column_name).cast(input_schema[column_name]).alias(column_name)
        for column_name in PLAYER_LOOKUP_KEYS
    ]
    df_features = df_lookup.select(
        *lookup_key_expressions,
        *PLAYER_FEATURE_COLUMNS,
    )

    return df_in.join(
        df_features,
        on=PLAYER_LOOKUP_KEYS,
        how="left",
        coalesce=True,
        validate="m:1",
    )
