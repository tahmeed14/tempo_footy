"""Create features from Gradient Sports event data."""

import polars as pl


# GAME STATE FEATURES
def create_team_scores(df_in: pl.DataFrame) -> pl.DataFrame:
    """Compute cumulative home and away scores for each event.

    Calculate the running score for both teams.

    Args:
        df_in: Event data containing ``event_number``, ``ge_endtype``, and
            ``ge_outtype``.

    Returns:
        Events through the end of play with cumulative score columns for
        the home and away teams.

    Raises:
        ValueError: If the data does not contain an end-of-play event.

    """
    score_expressions = [
        pl.when(pl.col("ge_outtype") == team_code)
        .then(1)
        .otherwise(0)
        .cum_sum()
        .alias(score_column)
        for team_code, score_column in (
            ("H", "home_score"),
            ("A", "away_score"),
        )
    ]

    return df_in.with_columns(score_expressions)


def create_match_scores(df_in: pl.DataFrame) -> pl.DataFrame:
    """Calculate game state from the in-possession team's perspective.

    Calculate the goal difference for the in-possession team, then label
    that team as winning, drawing, or losing. Rows without a home-team
    indicator retain null values for both game-state columns.

    Gradient Sports uses homeTeam to specify which team is associated
    with the possession event. The game-state columns are computed from
    the perspective of the in-possession team, so the home-team
    indicator is used to determine whether the home or away score should
    be used to calculate the goal difference.

    Args:
        df_in: Event data containing ``home_score``, ``away_score``, and
            ``ge_hometeam``.

    Returns:
        Event data with ``game_state_goal_diff`` and ``game_state``
        columns.

    """
    home_goal_difference = pl.col("home_score") - pl.col("away_score")
    goal_difference = (
        pl.when(pl.col("ge_hometeam"))
        .then(home_goal_difference)
        .when(~pl.col("ge_hometeam"))
        .then(-home_goal_difference)
        .otherwise(None)
    )

    return df_in.with_columns(
        goal_difference.alias("game_state_goal_diff")
    ).with_columns(
        pl.when(pl.col("game_state_goal_diff") > 0)
        .then(pl.lit("Winning"))
        .when(pl.col("game_state_goal_diff") == 0)
        .then(pl.lit("Drawing"))
        .when(pl.col("game_state_goal_diff") < 0)
        .then(pl.lit("Losing"))
        .otherwise(None)
        .alias("game_state")
    )


def add_game_state_features(df_in: pl.DataFrame) -> pl.DataFrame:
    """Add team score and game-state context to an events DataFrame.

    Calls `create_team_scores` and then `create_match_scores` to attach
    running scores and game-state labels to each event.

    Args:
        df_in: Event data game state variables.

    Returns:
        Event data with game state variables added.

    """
    return create_match_scores(create_team_scores(df_in))


# Possession Player Features
def create_num_challenges(df_in: pl.DataFrame) -> pl.DataFrame:
    """Count defender challenges for each player possession."""
    return df_in.with_columns(
        pl.col("pe_possessioneventtype")
        .eq("CH")
        .sum()
        .over(["gameeventid", "match_team_player_possession_id"])
        .alias("defender_num_challenges"),
    )


def create_pass_outcome(df_in: pl.DataFrame) -> pl.DataFrame:
    """Identify successful passes and crosses in each event."""
    return df_in.with_columns(
        pl.when(pl.col("pe_possessioneventtype").is_in(["PA", "CR"]))
        .then(
            pl.coalesce(["pe_passoutcometype", "pe_crossoutcometype"]) == "C"
        )
        .otherwise(None)
        .alias("successful_pass_or_cross"),
    ).select(pl.exclude("pe_passoutcometype", "pe_crossoutcometype"))


def add_possession_player_features(df_in: pl.DataFrame) -> pl.DataFrame:
    """Add player-possession context features."""
    return df_in.pipe(create_num_challenges).pipe(create_pass_outcome)


def features_events(df_in: pl.DataFrame) -> pl.DataFrame:
    """Add all derived event features."""
    df_out = add_game_state_features(df_in)
    df_out = add_possession_player_features(df_out)

    return df_in.pipe(add_game_state_features).pipe(
        add_possession_player_features
    )
