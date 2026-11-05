"""Transform Gradient Sports event data."""

import polars as pl

EVENT_MAIN_COLUMNS = (
    "event_number",
    "gameId",
    "gameEventId",
    "possessionEventId",
    "duration",
)

INITIAL_TOUCH_COLUMNS = (
    "initialPressureType",
    "initialHeightType",
    "initialBodyType",
)

GAME_EVENTS_COLUMNS = (
    "gameEventType",
    "playerId",
    "playerName",
    "teamId",
    "teamName",
    "homeTeam",
    "setpieceType",
    "outType",
    "endType",
    "period",
    "touches",
)

POSSESSION_EVENTS_COLUMNS = (
    "possessionEventType",
    "formattedGameClock",
    "linesBrokenType",
    "pressureType",
    "crossOutcomeType",
    "passOutcomeType",
)

# For ge_setpiecetype
DEAD_BALL_SET_PIECE_TYPES = (
    "C",  # corner
    "D",  # drop ball
    "F",  # free kick
    "G",  # goal kick
    "K",  # kickoff
    "P",  # open play
    "T",
)  # throw in

# For ge_gameeventtype
PAUSE_EVENTS = ("SUB", "ON", "OFF", "OUT", "END")

TRANSITION_EVENTS = (
    # WARNING: changing this will impact
    # create_team_possession_flag()
    "FIRSTKICKOFF",
    "SECONDKICKOFF",
    "THIRDKICKOFF",
    "FOURTHKICKOFF",
    "G",
    "OTB",
)  # goals and out-of-bounds events

BODYPART_MAPPER = {
    # Hands/Arms
    "2H": "Hands/Arms",
    "CA": "Hands/Arms",
    "PA": "Hands/Arms",
    "PU": "Hands/Arms",
    "LH": "Hands/Arms",
    "RH": "Hands/Arms",
    "LA": "Hands/Arms",
    "RA": "Hands/Arms",
    "TWOHANDS": "Hands/Arms",
    # Header
    "HE": "Head",
    # Upper Body
    "CH": "Torso/Other",
    "BA": "Torso/Other",
    "BO": "Torso/Other",  # BO refers to Bottom which is butt :O
    "LC": "Torso/Other",
    "RC": "Torso/Other",
    # Leg (not Foot)
    "LK": "Leg (not Foot)",
    "RK": "Leg (not Foot)",
    "LS": "Leg (not Foot)",
    "RS": "Leg (not Foot)",
    "LT": "Leg (not Foot)",
    "RT": "Leg (not Foot)",
    # Foot
    "LF": "Foot",
    "RF": "Foot",
    "LB": "Foot",
    "RB": "Foot",
    "R": "Foot",
    "L": "Foot",
    # Not available
    "VM": "N/A",
}

PRESSURE_MAPPER = {
    "L": "No Pressure",
    "P": "Pressured (inc. Attempted Pressure)",
    "A": "Pressured (inc. Attempted Pressure)",
    "N": "No Pressure",
}

BALL_HEIGHT_MAPPER = {
    "A": "Air",
    "H": "Air",
    "L": "Air",
    "V": "Air",
    "G": "Ground",
    "M": "N/A",
}

SETPIECE_MAPPER = {
    "O": "Open Play",
    "D": "Drop Kick",
    "F": "Free Kick",
    "G": "Goal Kick",
    "K": "Kickoff",
    "P": "Penalty",
    "T": "Throw In",
    "C": "Corner",
}

LINESBROKEN_MAPPER = {
    "A": "Attack",
    "AD": "Attack & Defense (Midfield Bypassed)",
    "AM": "Attack & Midfield",
    "AMD": "Attack, Midfield, & Defense",
    "D": "Defense",
    "M": "Midfield",
    "MD": "Midfield & Defense",
}

PERIOD_MAPPER = {1: "1", 2: "2", 3: "AET", 4: "AET"}

RENAME_MAPPER = {
    "gameid": "game_id",
    "gameeventid": "game_event_id",
    "possessioneventid": "possession_event_id",
    "ge_gameeventtype": "game_event_type",
    "pe_possessioneventtype": "possession_event_type",
    "ge_playerid": "player_id",
    "ge_teamid": "team_id",
    "teamattackingdirection": "attacking_team_direction",
}

FINALIZE_ORDER = (
    "game_id",
    "game_period",
    "game_event_id",
    "possession_event_id",
    "formattedgameclock",
    "game_state",
    "player_id",
    "team_id",
    "match_team_possession_id",
    "match_team_player_possession_id",
    "team_possession_start",
    "event_number",
    "game_event_type",
)

FINALIZE_EXCLUDE = (
    *FINALIZE_ORDER,
    "outtype",
    "endtype",
    "duration",
    # "period"
)


def select_events_columns(df_in: pl.DataFrame) -> pl.DataFrame:
    """Unnest selected event structs and return a flat DataFrame.

    Select the main event columns and configured fields from the nested
    event structs. Prefix extracted fields according to their source and
    include the attacking direction from ``stadiumMetadata``.

    Args:
        df_in: Event data containing the required identifier and nested
            struct columns.

    Returns:
        A flat Polars DataFrame containing the selected event fields.

    """
    struct_selections = (
        ("initialTouch", "it_", INITIAL_TOUCH_COLUMNS),
        ("gameEvents", "ge_", GAME_EVENTS_COLUMNS),
        ("possessionEvents", "pe_", POSSESSION_EVENTS_COLUMNS),
    )

    # loop through the struct selections then through the fields for
    # each struct, creating a list of fields to select from df
    nested_fields = [
        pl.col(struct_column).struct.field(field).alias(f"{prefix}{field}")
        for struct_column, prefix, fields in struct_selections
        for field in fields
    ]

    return df_in.select(
        *EVENT_MAIN_COLUMNS,
        pl.col("stadiumMetadata").struct.field("teamAttackingDirection"),
        *nested_fields,
    ).rename(str.lower)


def drop_pen_shootouts(df_in: pl.DataFrame) -> pl.DataFrame:
    """Remove shootout events occurring after the end of normal play.

    Find the last event in each match where ``ge_gameeventtype`` is
    ``"END"`` and ``ge_endtype`` is ``"G"``. Keep that event and every
    earlier event in the match. If a match has no qualifying end event,
    retain all of its rows.

    Args:
        df_in: Event data containing ``gameid``, ``event_number``,
            ``ge_gameeventtype``, and ``ge_endtype``.

    Returns:
        Event data through the end of normal play for each match.

    """
    normal_time_ended = (pl.col("ge_gameeventtype") == "END") & (
        pl.col("ge_endtype") == "G"
    )
    normal_time_end_event = (
        pl.when(normal_time_ended)
        .then(pl.col("event_number"))
        .max()
        .over("gameid")
    )

    return df_in.filter(
        normal_time_end_event.is_null()
        | (pl.col("event_number") <= normal_time_end_event)
    )


def reclassify_ballheight(df_in: pl.DataFrame) -> pl.DataFrame:
    """Reclassify ball-height values."""
    return df_in.with_columns(
        pl.col("it_initialheighttype")
        .replace_strict(BALL_HEIGHT_MAPPER, default="N/A")
        .alias("first_touch_ballheight")
    ).drop("it_initialheighttype")


def reclassify_pressuretype(df_in: pl.DataFrame) -> pl.DataFrame:
    """Reclassify pressure-type values."""
    return df_in.with_columns(
        pl.col("pe_pressuretype")
        .replace_strict(PRESSURE_MAPPER, default="No Pressure")
        .alias("defender_pressure_type"),
        pl.col("it_initialpressuretype")
        .replace_strict(PRESSURE_MAPPER, default="No Pressure")
        .alias("first_touch_defender_pressure_type"),
    ).drop(["it_initialpressuretype", "pe_pressuretype"])


def reclassify_firsttouch(df_in: pl.DataFrame) -> pl.DataFrame:
    """Reclassify first-touch values."""
    return df_in.with_columns(
        pl.col("it_initialbodytype")
        .replace_strict(BODYPART_MAPPER, default="N/A")
        .fill_null("Not available")
        .alias("first_touch_bodypart")
    ).drop("it_initialbodytype")


def reclassify_linesbrokentype(df_in: pl.DataFrame) -> pl.DataFrame:
    """Reclassify lines-broken values."""
    return df_in.with_columns(
        pl.col("pe_linesbrokentype")
        .replace_strict(LINESBROKEN_MAPPER, default="None")
        .fill_null("None")
        .alias("lines_broken_type")
    ).drop("pe_linesbrokentype")


def reclassify_period(df_in: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Reclassify match-period values."""
    return df_in.with_columns(
        pl.col("ge_period")
        .replace_strict(PERIOD_MAPPER, default="None")
        .fill_null("None")
        .alias("game_period")
    )


def rename_classes_setpieces(df_in: pl.DataFrame) -> pl.DataFrame:
    """Rename set-piece classes."""
    return df_in.with_columns(
        pl.col("ge_setpiecetype")
        .replace_strict(SETPIECE_MAPPER, default="N/A")
        .fill_null("Not available")
        # .alias("set_piece_type")
    )  # .drop("ge_setpiecetype")


def drop_foul_events(df_in: pl.DataFrame) -> pl.DataFrame:
    """Drop supplemental ``FOUL`` game-event rows.

    Gradient Sports describes this as follows: When multiple
    infringements occur in an event, they are added separately.
    For example, an on-the-ball foul followed by a yellow card
    for dissent are added as two separate fouls. The main foul
    is added on the possession event where it happened, and
    additional fouls are added as a separate row after the next
    OUT events. Additional fouls have gameEventId "FOUL" and
    possessionEventId "FO" even though technically they are not
    separate events.

    Null and all other game-event types are retained.
    """
    return df_in.filter(pl.col("ge_gameeventtype").ne_missing("FOUL"))


# Identify possessions for teams & players
def create_team_possession_flag(df_in: pl.DataFrame) -> pl.DataFrame:
    """Flag rows where a new team possession begins.

    Identify possession changes independently within each game. A team
    change, dead-ball set piece, kickoff, or first non-pause event after
    an ``OUT`` starts a possession. Pause rows remain in the result but
    are never flagged because they have no possession of their own.

    Args:
        df_in: Event data containing ``ge_teamid``, ``gameid``,
            ``ge_setpiecetype``, and ``ge_gameeventtype``.

    Returns:
        Event data with a Boolean ``team_possession_start`` column.
        Never a null

    """
    pause_event = (
        pl.col("ge_gameeventtype").is_in(PAUSE_EVENTS).fill_null(False)
    )
    team_has_possession = pl.col("ge_teamid").is_not_null() & ~pause_event
    previous_possession_team = (
        pl.when(team_has_possession)
        .then(pl.col("ge_teamid"))
        .otherwise(None)
        .forward_fill()
        .shift()
        .over("gameid")
    )
    team_changed = team_has_possession & pl.col("ge_teamid").ne_missing(
        previous_possession_team
    )
    dead_ball_started = pl.col("ge_setpiecetype").is_in(
        DEAD_BALL_SET_PIECE_TYPES
    )
    possession_resumed_after_out = (
        pl.when(pl.col("ge_gameeventtype") == "OUT")
        .then(True)
        .when(~pause_event)
        .then(False)
        .otherwise(None)
        .forward_fill()
        .shift()
        .over("gameid")
        .fill_null(False)
    )
    first_event = pl.col("ge_gameeventtype").is_in(TRANSITION_EVENTS[0:4])

    possession_started = (
        ~pause_event
        & (
            first_event
            | team_changed
            | dead_ball_started
            | possession_resumed_after_out
        )
    ).fill_null(False)

    return df_in.with_columns(
        possession_started.alias("team_possession_start")
    )


def create_team_possession_id(df_in: pl.DataFrame) -> pl.DataFrame:
    """Create match-level and team-level possession identifiers.

    Convert possession-start flags on possession-bearing rows into a
    cumulative possession number, then combine the game, team, and
    possession values into a unique identifier. Retained pause or
    null-team rows do not consume a possession number and receive a null
    composite ID.

    Args:
        df_in: Event data containing ``team_possession_start``,
            ``gameid``, and ``ge_teamname``.

    Returns:
        Event data with ``match_possession_id`` and
        ``match_team_possession_id`` columns.

    """
    team_has_possession = (
        pl.col("ge_teamid").is_not_null()
        & pl.col("ge_teamname").is_not_null()
        & ~pl.col("ge_gameeventtype").is_in(PAUSE_EVENTS).fill_null(False)
    )
    counted_possession_start = (
        pl.col("team_possession_start").fill_null(False) & team_has_possession
    )
    possession_data = df_in.with_columns(
        counted_possession_start.cum_sum()
        .over("gameid")
        .alias("match_possession_id")
    )

    # FIXME: Exlcude the team_possession_start flag
    # return possession_data.select(
    #     pl.exclude("team_possession_start"),
    #     pl.concat_str(
    #         ["gameid", "ge_teamname", "match_possession_id"],
    #         separator="_",
    #     ).alias("match_team_possession_id"),
    # )

    return possession_data.with_columns(
        pl.when(team_has_possession)
        .then(
            pl.concat_str(
                ["gameid", "ge_teamname", "match_possession_id"],
                separator="_",
            )
        )
        .otherwise(None)
        .alias("match_team_possession_id")
    )


def create_player_possession_id(df_in: pl.DataFrame) -> pl.DataFrame:
    """Create individual player-possession identifiers.

    Flag a new player possession when the player or team possession
    changes. Rows without a player or team possession remain transparent
    to the counter. Cumulatively number valid player possessions and combine
    the game, team, player, and possession values into a unique identifier.

    Args:
        df_in: Event data containing player and team-possession fields.

    Returns:
        Event data with an ``individual_possession_id`` column, without
        temporary player-possession columns.

    """
    player_has_team_possession = (
        pl.col("ge_playerid").is_not_null()
        & pl.col("match_team_possession_id").is_not_null()
    )
    previous_known_player = (
        pl.when(player_has_team_possession)
        .then(pl.col("ge_playerid"))
        .otherwise(None)
        .forward_fill()
        .shift()
        .over("gameid")
    )
    player_changed = player_has_team_possession & pl.col(
        "ge_playerid"
    ).ne_missing(previous_known_player)
    previous_player_team_possession = (
        pl.when(player_has_team_possession)
        .then(pl.col("match_team_possession_id"))
        .otherwise(None)
        .forward_fill()
        .shift()
        .over("gameid")
    )
    team_possession_changed = pl.col("match_team_possession_id").ne_missing(
        previous_player_team_possession
    )

    player_possession_index = (
        (
            player_has_team_possession
            & (player_changed | team_possession_changed)
        )
        .cum_sum()
        .over("gameid")
    )

    possession_data = df_in.with_columns(
        player_possession_index.alias("player_possession_index")
    )

    return possession_data.select(
        pl.exclude(["player_possession_index", "match_possession_id"]),
        pl.when(player_has_team_possession)
        .then(
            pl.concat_str(
                [
                    "gameid",
                    "ge_teamname",
                    "ge_playerid",
                    "player_possession_index",
                ],
                separator="_",
            )
        )
        .otherwise(None)
        .alias("match_team_player_possession_id"),
    )


def add_possession_identifiers(df_in: pl.DataFrame) -> pl.DataFrame:
    """Compute possession IDs while retaining every event row.

    Flag possession starts and assign team and player possession identifiers.
    Pause events remain in the output with null composite identifiers and do
    not consume sequence numbers.

    Args:
        df_in: pl.DataFrame containing event rows with
        `ge_gameeventtype`.

    Returns:
        Event data with team and player possession identifiers added.

    """
    return create_player_possession_id(
        create_team_possession_id(create_team_possession_flag(df_in=df_in))
    )


def transform_events(df_in: pl.DataFrame) -> pl.DataFrame:
    """Transform raw event data into a flat DataFrame with game state.

    Args:
        df_in: Event data containing the required identifier and nested
            struct columns.

    Returns:
        A flat Polars DataFrame containing the selected event fields and
        game state variables.

    """
    return (
        df_in.pipe(select_events_columns)
        .pipe(drop_pen_shootouts)
        .pipe(drop_foul_events)
        .pipe(reclassify_period)
        .pipe(reclassify_ballheight)
        .pipe(reclassify_pressuretype)
        .pipe(reclassify_linesbrokentype)
        .pipe(reclassify_firsttouch)
        .pipe(rename_classes_setpieces)
        .pipe(add_possession_identifiers)
    )


def rename_columns(df_in: pl.DataFrame) -> pl.DataFrame:
    """Rename event identifiers and event-type columns."""
    return df_in.rename(RENAME_MAPPER)


def remove_event_prefixes(df_in: pl.DataFrame) -> pl.DataFrame:
    """Remove leading game-event and possession-event prefixes."""
    return df_in.rename(
        lambda column_name: (
            column_name[3:]
            if column_name.startswith(("ge_", "pe_"))
            else column_name
        )
    )


def organize_event_columns(df_in: pl.DataFrame) -> pl.DataFrame:
    """Filter valid possessions and arrange final event columns."""
    return (
        df_in
        # .filter(pl.col("match_team_possession_id").is_not_null())
        .select(
            *FINALIZE_ORDER,
            pl.exclude(*FINALIZE_ORDER, *FINALIZE_EXCLUDE),
        )
    )


def cleanup_events(df_in: pl.DataFrame) -> pl.DataFrame:
    """Rename, remove prefixes, and organize processed event data."""
    return (
        df_in.pipe(rename_columns)
        .pipe(remove_event_prefixes)
        .pipe(organize_event_columns)
    )
