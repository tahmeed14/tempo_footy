"""Transform Gradient Sports tracking data."""

import polars as pl

RENAME_MAPPER = {
    "gameRefId": "game_id",
    "homePlayersSmoothed": "home_players_smooth",
    "awayPlayersSmoothed": "away_players_smooth",
    "ballsSmoothed": "balls_smooth",
}

RECAST_MAPPER = {
    "game_id": pl.Int32,
    "possession_event_id": pl.Int32,
    "game_event_id": pl.Int32,
}

COLUMNS = (
    "game_id",
    "game_event_id",
    "possession_event_id",
    "framenum",
    "period",
    "game_event",
    "possession_event",
    "home_players_smooth",
    "away_players_smooth",
    "balls_smooth",
)


def rename_columns(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Rename raw tracking columns."""
    return df_in.rename(lambda col: RENAME_MAPPER.get(col, col.lower()))


def recast_columns(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Cast tracking columns to their expected data types."""
    return df_in.cast(RECAST_MAPPER)


def fill_game_id(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing game identifiers."""
    return df_in.with_columns(pl.col("game_id").forward_fill())


def select_columns(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Select the tracking columns used downstream."""
    return df_in.select(*COLUMNS)


def drop_dupes(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Remove duplicate tracking rows."""
    return df_in.unique(keep="first", maintain_order=True)


def transform_tracking(df_in: pl.LazyFrame) -> pl.LazyFrame:
    """Apply the complete tracking-data transformation."""
    return (
        df_in.pipe(rename_columns)
        .pipe(recast_columns)
        .pipe(fill_game_id)
        .pipe(select_columns)
        .pipe(drop_dupes)
    )
