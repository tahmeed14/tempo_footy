"""Build the dataset used for Bayesian modeling."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

EVENTS_PATH = Path("data/processed/gradient_sports/events")
PLAYER_POSSESSIONS_PATH = Path("data/analysis/player_possessions.parquet")
POSSESSION_LOOKUP_PATH = Path(
    "data/curated/gradient_sports/possession_lookup/"
    "match_possession_lookup.parquet"
)
MODELDATA_PATH = Path("data/analysis/modeldata_v0.parquet")

EVENT_FEATURES = (
    "event_number",
    "formattedgameclock",
    "player_id",
    "playername",
    "team_id",
    "teamname",
    "game_state_goal_diff",
    "setpiecetype",
    "first_touch_ballheight",
    "first_touch_bodypart",
    "first_touch_defender_pressure_type",
    "defender_num_challenges",
    "game_id",
    "game_period",
    "match_team_possession_id",
    "match_team_player_possession_id",
)

POSSESSION_COLUMNS = (
    "game_id",
    "match_team_possession_id",
    "match_team_player_possession_id",
    "player_possession_sequence_number",
    "elapsed_seconds_team_possession",
    "ball_speed_tempo_player",
    "total_ball_displacement",
    "elapsed_frames",
    "starting_pitch_third",
)

EVENT_JOIN_KEYS = (
    "game_id",
    "match_team_possession_id",
    "match_team_player_possession_id",
)

POSSESSION_LOOKUP_JOIN_KEY = "match_team_player_possession_id"

LOOKUP_IDENTITY_COLUMNS = (
    "player_id",
    "playername",
    "teamname",
    "team_id",
)


def prepare_events(df_events: pl.LazyFrame) -> pl.LazyFrame:
    """Select one representative event for each player possession."""
    return (
        df_events.select(EVENT_FEATURES)
        .sort(("game_id", "event_number"), nulls_last=True)
        .unique(
            subset=[POSSESSION_LOOKUP_JOIN_KEY],
            keep="first",
            maintain_order=True,
        )
    )


def join_event_features(
    df_player_possessions: pl.LazyFrame,
    df_events: pl.LazyFrame,
) -> pl.LazyFrame:
    """Add event features while preserving every player possession."""
    return df_player_possessions.join(
        df_events,
        on=EVENT_JOIN_KEYS,
        how="left",
        validate="1:1",
    )


def join_possession_lookup(
    df_model: pl.LazyFrame,
    df_possession_lookup: pl.LazyFrame,
) -> pl.LazyFrame:
    """Add possession attributes while preserving every model row."""
    return df_model.join(
        df_possession_lookup,
        on=(POSSESSION_LOOKUP_JOIN_KEY, "game_id"),
        how="left",
        validate="1:1",
    )


def build_modeldata(
    df_player_possessions: pl.LazyFrame,
    df_events: pl.LazyFrame,
    df_possession_lookup: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build model data from player possessions, events, and metadata."""
    return df_player_possessions.pipe(join_event_features, df_events).pipe(
        join_possession_lookup, df_possession_lookup
    )


def load_modeldata(
    events_path: str | Path | Sequence[str | Path] = EVENTS_PATH,
    player_possessions_path: str | Path = PLAYER_POSSESSIONS_PATH,
    possession_lookup_path: str | Path = POSSESSION_LOOKUP_PATH,
    output_path: str | Path = MODELDATA_PATH,
) -> Path:
    """Build model data from selected inputs and return its output path."""
    output_path = Path(output_path)
    df_events = pl.scan_parquet(events_path).pipe(prepare_events)
    df_player_possessions = pl.scan_parquet(player_possessions_path).select(
        POSSESSION_COLUMNS
    )
    df_possession_lookup = pl.scan_parquet(possession_lookup_path).drop(
        LOOKUP_IDENTITY_COLUMNS
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_modeldata(
        df_player_possessions,
        df_events,
        df_possession_lookup,
    ).sink_parquet(output_path, compression="zstd")
    return output_path
