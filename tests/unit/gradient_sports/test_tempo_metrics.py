"""Test possession-tempo metrics."""

from pathlib import Path

import polars as pl
import pytest

from tempoctrl.gradient_sports.ball_metrics import (
    add_ball_displacement,
    add_ball_metrics,
    add_ball_speed,
)
from tempoctrl.gradient_sports.possessions_load import (
    write_possession_outputs,
)
from tempoctrl.gradient_sports.tempo_metrics import (
    _add_possession_sequence,
    _add_possession_time_elapsed,
    aggregate_possession_tempo,
)

BALL_DTYPE = pl.Struct(
    {
        "x": pl.Float64,
        "y": pl.Float64,
    }
)


def test_displacement_uses_frame_order_within_each_game() -> None:
    """Calculate displacement in frame order without sorting output.

    Behavior:
        Order each game's possession window by frame while preserving
        the input row order returned to downstream pipeline stages.

    Input:
        Two games with the same possession ID and independently
        unsorted frames containing horizontal ball movement.

    Expected:
        Each game's first chronological frame has null deltas. Later
        frames have explicit within-game displacement values and dtypes.

    Guards against:
        Using physical row order, leaking movement between games,
        sorting the output, or changing metric schemas unexpectedly.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "framenum": [3, 1, 2, 2, 1],
            "team_possession_id": ["shared"] * 5,
            "balls_smooth": [
                {"x": 3.0, "y": 0.0},
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 0.0},
                {"x": 14.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_displacement(
        frames,
        "team_possession_id",
    ).collect()

    assert result.height == 5
    assert result["framenum"].to_list() == [3, 1, 2, 2, 1]
    assert result["delta_x"].to_list() == [2.0, None, 1.0, 4.0, None]
    assert result["delta_y"].to_list() == [0.0, None, 0.0, 0.0, None]
    assert result["delta_frame"].to_list() == [1, None, 1, 1, None]
    assert result["ball_displacement"].to_list() == [
        2.0,
        None,
        1.0,
        4.0,
        None,
    ]
    assert result.schema["delta_x"] == pl.Float64
    assert result.schema["delta_y"] == pl.Float64
    assert result.schema["delta_frame"] == pl.Int64
    assert result.schema["ball_displacement"] == pl.Float64
    assert "ball_speed" not in result.columns


def test_displacement_accepts_multiple_possession_groups() -> None:
    """Partition movement by every configured possession column.

    Behavior:
        Normalize a sequence of group names and use all of them with the
        automatically included game ID.

    Input:
        Interleaved rows for two players in one team possession, with
        each player's second frame appearing before their first.

    Expected:
        Each second frame is compared only with the same player's first
        frame, producing distances two and three.

    Guards against:
        Treating a tuple as one grouping expression or shifting between
        different player possessions in the same team possession.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "framenum": [2, 2, 1, 1],
            "team_possession_id": ["team_a"] * 4,
            "player_possession_id": ["player_a", "player_b"] * 2,
            "balls_smooth": [
                {"x": 2.0, "y": 0.0},
                {"x": 13.0, "y": 0.0},
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_displacement(
        frames,
        ("team_possession_id", "player_possession_id"),
    ).collect()

    assert result["delta_x"].to_list() == [2.0, 3.0, None, None]
    assert result["ball_displacement"].to_list() == [
        2.0,
        3.0,
        None,
        None,
    ]


def test_synthetic_row_owns_duplicate_frame_displacement() -> None:
    """Assign a pass segment to the synthetic passer row.

    Behavior:
        Prefer a synthetic row as the sole metric row when a receiver
        and synthetic passer share a team-possession frame.

    Input:
        A passer frame, a receiver row followed physically by a
        synthetic passer row at frame two, and a receiver frame three.

    Expected:
        The synthetic row receives the four-meter pass. The duplicate
        receiver row is null, and its next row receives one meter.

    Guards against:
        Selecting duplicates by input order, crediting the pass to the
        receiver, or breaking receiver movement after the shared frame.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "framenum": [1, 2, 2, 3],
            "team_possession_id": ["team_a"] * 4,
            "player_possession_id": [
                "passer",
                "receiver",
                "passer",
                "receiver",
            ],
            "is_synthetic_pass_end": [False, False, True, False],
            "balls_smooth": [
                {"x": 0.0, "y": 0.0},
                {"x": 4.0, "y": 0.0},
                {"x": 4.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_displacement(
        frames,
        "team_possession_id",
    ).collect()

    assert result.height == 4
    assert result["ball_displacement"].to_list() == [
        None,
        None,
        4.0,
        1.0,
    ]
    assert result["delta_frame"].to_list() == [None, None, 1, 1]
    assert (
        result.filter(pl.col("player_possession_id") == "passer")[
            "ball_displacement"
        ].sum()
        == 4.0
    )
    assert (
        result.filter(pl.col("player_possession_id") == "receiver")[
            "ball_displacement"
        ].sum()
        == 1.0
    )
    assert "__is_ball_metric_frame" not in result.columns
    assert "__ball_metric_input_order" not in result.columns


def test_first_input_row_owns_nonsynthetic_duplicate_frame() -> None:
    """Use the first input row when duplicate frames are not synthetic.

    Behavior:
        Select the first physical row as the canonical metric row when
        no duplicate at that frame is synthetic.

    Input:
        Two frame-one rows at x zero and one hundred followed by frame
        two at x two, all within one team possession.

    Expected:
        Both frame-one rows have null metrics. Frame two measures two
        meters from the first frame-one row rather than negative 98.

    Guards against:
        Allowing a later ordinary duplicate to replace the first row as
        the coordinate anchor for the following frame.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "framenum": [1, 1, 2],
            "team_possession_id": ["team_a"] * 3,
            "is_synthetic_pass_end": [False, False, False],
            "balls_smooth": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 2.0, "y": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_displacement(
        frames,
        "team_possession_id",
    ).collect()

    assert result["delta_x"].to_list() == [None, None, 2.0]
    assert result["delta_y"].to_list() == [None, None, 0.0]
    assert result["delta_frame"].to_list() == [None, None, 1]
    assert result["ball_displacement"].to_list() == [None, None, 2.0]


def test_null_partition_keys_never_receive_movement_metrics() -> None:
    """Leave metrics null when a required partition key is null.

    Behavior:
        Exclude rows with null game, possession, or frame keys from the
        canonical movement sequence.

    Input:
        Null-possession and null-game sequences, a null-frame row, and a
        valid two-frame possession in the same LazyFrame.

    Expected:
        Only the second row of the valid possession receives movement;
        every row containing a null key has null metric columns.

    Guards against:
        Treating all null IDs as one possession or connecting unrelated
        rows through Polars' null grouping behavior.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, None, None, 1],
            "framenum": [1, 2, 1, 2, 1, 2, None],
            "team_possession_id": [
                None,
                None,
                "valid",
                "valid",
                "orphan",
                "orphan",
                "valid",
            ],
            "balls_smooth": [
                {"x": 0.0, "y": 0.0},
                {"x": 50.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 11.0, "y": 0.0},
                {"x": 20.0, "y": 0.0},
                {"x": 30.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_displacement(
        frames,
        "team_possession_id",
    ).collect()

    assert result["delta_x"].to_list() == [
        None,
        None,
        None,
        1.0,
        None,
        None,
        None,
    ]
    assert result["ball_displacement"].to_list() == [
        None,
        None,
        None,
        1.0,
        None,
        None,
        None,
    ]
    assert "ball_speed" not in result.columns


def test_null_coordinates_break_consecutive_movement_segments() -> None:
    """Propagate coordinate nulls through adjacent movement segments.

    Behavior:
        Compare consecutive canonical frames without skipping frames
        whose complete or partial ball coordinates are null.

    Input:
        Seven frames containing an outer-null ball struct followed later
        by a partially null coordinate pair and valid recovery frames.

    Expected:
        Segments touching either null coordinate are null. Movement
        resumes only after two consecutive frames have valid x and y.

    Guards against:
        Forward-filling through missing observations and reporting a
        misleading multi-frame straight-line displacement.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1] * 7,
            "framenum": list(range(1, 8)),
            "team_possession_id": ["team_a"] * 7,
            "balls_smooth": [
                {"x": 0.0, "y": 0.0},
                None,
                {"x": 3.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": None, "y": 1.0},
                {"x": 7.0, "y": 2.0},
                {"x": 8.0, "y": 2.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_displacement(
        frames,
        "team_possession_id",
    ).collect()

    assert result["delta_x"].to_list() == [
        None,
        None,
        None,
        2.0,
        None,
        None,
        1.0,
    ]
    assert result["delta_y"].to_list() == [
        None,
        None,
        None,
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    assert result["ball_displacement"].to_list() == [
        None,
        None,
        None,
        2.0,
        None,
        None,
        1.0,
    ]


def test_ball_speed_uses_explicit_rate_and_frame_delta() -> None:
    """Calculate speed separately from displacement.

    Behavior:
        Convert displacement to meters per second using the supplied
        frame rate and numerical frame interval.

    Input:
        A leading null, a six-meter movement over three frames, and a
        three-meter displacement with a zero frame interval.

    Expected:
        The valid segment produces 60 meters per second. Leading and
        nonpositive frame intervals produce null speed values.

    Guards against:
        Using a hidden frame rate, assuming every interval is one frame,
        or dividing by zero for duplicate frames.
    """
    displacement = pl.DataFrame(
        {
            "ball_displacement": [None, 6.0, 3.0],
            "delta_frame": [None, 3, 0],
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    result = add_ball_speed(
        displacement,
        frame_rate=30.0,
    ).collect()

    assert result["ball_speed"].to_list() == [None, 60.0, None]
    assert result["frame_rate"].to_list() == [30.0, 30.0, 30.0]
    assert result.schema["frame_rate"] == pl.Float64
    assert result.schema["ball_speed"] == pl.Float64


def test_ball_speed_resolves_rate_for_each_game() -> None:
    """Use explicit per-game rates in a shared LazyFrame.

    Behavior:
        Resolve frame-rate metadata by game before calculating speed.

    Input:
        Equal two-meter, two-frame movements in games sampled at 25 and
        50 frames per second.

    Expected:
        The rows retain rates 25 and 50 and produce speeds 25 and 50
        meters per second respectively.

    Guards against:
        Applying one game's sampling rate to every game in a dataset.
    """
    displacement = pl.DataFrame(
        {
            "game_id": [1, 2],
            "ball_displacement": [2.0, 2.0],
            "delta_frame": [2, 2],
        }
    ).lazy()

    result = add_ball_speed(
        displacement,
        frame_rate={1: 25.0, 2: 50.0},
    ).collect()

    assert result["frame_rate"].to_list() == [25.0, 50.0]
    assert result["ball_speed"].to_list() == [25.0, 50.0]


def test_ball_speed_rejects_incomplete_game_rate_mapping() -> None:
    """Fail when per-game frame-rate metadata is incomplete.

    Behavior:
        Resolve mappings strictly for every non-null game in the input.

    Input:
        Rows for games one and two with a rate supplied only for game
        one.

    Expected:
        Collection raises Polars' incomplete-mapping error.

    Guards against:
        Silently assigning a default rate or null speed to an unmapped
        game.
    """
    displacement = pl.DataFrame(
        {
            "game_id": [1, 2],
            "ball_displacement": [2.0, 2.0],
            "delta_frame": [1, 1],
        }
    ).lazy()

    result = add_ball_speed(
        displacement,
        frame_rate={1: 25.0},
    )

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="incomplete mapping",
    ):
        result.collect()


def test_ball_metrics_orchestrates_displacement_and_speed() -> None:
    """Compose displacement and speed through one public function.

    Behavior:
        Add displacement first and then calculate speed with the
        caller's
        explicitly supplied frame rate.

    Input:
        One possession moving six meters between frames one and three at
        30 frames per second.

    Expected:
        The second row has six meters displacement, a two-frame
        interval,
        and 90 meters per second speed.

    Guards against:
        Reintroducing hidden coupling or failing to pass frame rate from
        the orchestration function into speed calculation.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1],
            "framenum": [1, 3],
            "team_possession_id": ["team_a", "team_a"],
            "balls_smooth": [
                {"x": 0.0, "y": 0.0},
                {"x": 6.0, "y": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = add_ball_metrics(
        frames,
        "team_possession_id",
        frame_rate=30.0,
    ).collect()

    assert result["ball_displacement"].to_list() == [None, 6.0]
    assert result["delta_frame"].to_list() == [None, 2]
    assert result["ball_speed"].to_list() == [None, 90.0]


@pytest.mark.parametrize(
    "frame_rate",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_ball_speed_rejects_invalid_frame_rates(frame_rate: float) -> None:
    """Reject frame rates that cannot produce meaningful speed.

    Behavior:
        Require a finite and strictly positive scalar frame rate.

    Input:
        An empty typed displacement frame and zero, negative, NaN, or
        infinite frame rates.

    Expected:
        Every invalid rate raises the same clear ``ValueError``.

    Guards against:
        Zero division, negative speeds, and NaN or infinite output.
    """
    displacement = pl.DataFrame(
        schema={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        }
    ).lazy()

    with pytest.raises(
        ValueError,
        match="frame_rate values must be finite and greater than 0",
    ):
        add_ball_speed(displacement, frame_rate=frame_rate)


@pytest.mark.parametrize(
    ("schema", "error_type", "message"),
    [
        (
            {"delta_frame": pl.Int64},
            ValueError,
            "Missing ball speed columns: ball_displacement",
        ),
        (
            {
                "ball_displacement": pl.Float64,
                "delta_frame": pl.String,
            },
            TypeError,
            "Ball speed columns must be numeric: delta_frame",
        ),
    ],
)
def test_ball_speed_rejects_invalid_schemas(
    schema: dict[str, pl.DataType],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject missing or nonnumeric speed inputs clearly.

    Behavior:
        Validate displacement and frame-interval columns before building
        the speed expression.

    Input:
        Typed empty frames with a missing displacement or string frame
        interval.

    Expected:
        Each invalid schema raises its documented exception.

    Guards against:
        Deferred expression errors and accidental arithmetic on strings.
    """
    displacement = pl.DataFrame(schema=schema).lazy()

    with pytest.raises(error_type, match=message):
        add_ball_speed(displacement, frame_rate=30.0)


@pytest.mark.parametrize(
    ("schema", "error_type", "message"),
    [
        (
            {
                "game_id": pl.Int64,
                "team_possession_id": pl.String,
                "balls_smooth": BALL_DTYPE,
            },
            ValueError,
            "Missing ball metric columns: framenum",
        ),
        (
            {
                "game_id": pl.Int64,
                "framenum": pl.Int64,
                "team_possession_id": pl.String,
            },
            ValueError,
            "Missing ball metric columns: balls_smooth",
        ),
        (
            {
                "game_id": pl.Int64,
                "framenum": pl.String,
                "team_possession_id": pl.String,
                "balls_smooth": BALL_DTYPE,
            },
            TypeError,
            "framenum must be numeric",
        ),
        (
            {
                "game_id": pl.Int64,
                "framenum": pl.Int64,
                "team_possession_id": pl.String,
                "balls_smooth": pl.Float64,
            },
            TypeError,
            "balls_smooth must be a struct column",
        ),
        (
            {
                "game_id": pl.Int64,
                "framenum": pl.Int64,
                "team_possession_id": pl.String,
                "balls_smooth": pl.Struct({"x": pl.Float64}),
            },
            ValueError,
            "balls_smooth is missing fields: y",
        ),
        (
            {
                "game_id": pl.Int64,
                "framenum": pl.Int64,
                "team_possession_id": pl.String,
                "balls_smooth": pl.Struct({"x": pl.String, "y": pl.Float64}),
            },
            TypeError,
            "balls_smooth fields must be numeric: x",
        ),
        (
            {
                "game_id": pl.Int64,
                "framenum": pl.Int64,
                "team_possession_id": pl.String,
                "balls_smooth": BALL_DTYPE,
                "is_synthetic_pass_end": pl.String,
            },
            TypeError,
            "is_synthetic_pass_end must be Boolean",
        ),
    ],
)
def test_displacement_rejects_invalid_metric_schemas(
    schema: dict[str, pl.DataType],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject schemas incompatible with ball movement calculations.

    Behavior:
        Validate required metric columns and their nested dtypes before
        constructing coordinate or synthetic-row expressions.

    Input:
        Typed empty LazyFrames with missing columns, invalid frames,
        malformed ball structs, or a non-Boolean synthetic marker.

    Expected:
        Each invalid schema raises a specific and readable exception.

    Guards against:
        Deferred Polars errors that obscure the actual input-contract
        violation or silently permit unsupported coordinate types.
    """
    frames = pl.DataFrame(schema=schema).lazy()

    with pytest.raises(error_type, match=message):
        add_ball_displacement(
            frames,
            "team_possession_id",
        )


@pytest.mark.parametrize(
    ("possession_groups", "error_type", "message"),
    [
        ([], ValueError, "At least one possession group is required"),
        ([""], ValueError, "Possession group names cannot be empty"),
        (
            ["team_possession_id", "team_possession_id"],
            ValueError,
            "Possession group names must be unique",
        ),
        ([1], TypeError, "Possession group names must be strings"),
        (
            ["missing_possession_id"],
            ValueError,
            "Missing ball metric group columns: missing_possession_id",
        ),
    ],
)
def test_displacement_rejects_invalid_possession_groups(
    possession_groups: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject unusable possession grouping configurations clearly.

    Behavior:
        Validate group names and their presence before constructing the
        displacement expressions.

    Input:
        An empty frame with empty, duplicate, non-string, or missing
        possession group configurations.

    Expected:
        Every invalid configuration raises its documented exception.

    Guards against:
        Ambiguous partitions and delayed Polars errors that do not
        identify the invalid grouping configuration.
    """
    frames = pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "framenum": pl.Int64,
            "team_possession_id": pl.String,
            "balls_smooth": BALL_DTYPE,
        }
    ).lazy()

    with pytest.raises(error_type, match=message):
        add_ball_displacement(
            frames,
            possession_groups,  # type: ignore[arg-type]
        )


def test_team_tempo_uses_valid_observed_frame_intervals() -> None:
    """Aggregate team tempo from valid observed movement only.

    Behavior:
        Partition by game and team possession, then divide valid
        displacement by the frame intervals that produced it.

    Input:
        Two games sharing a possession ID. The first includes two valid
        segments plus a duplicate-frame segment; the second has one
        valid segment. Repeated and null player-possession IDs and a
        null
        team-possession row are also present.

    Expected:
        Game one totals ten meters over four frames for 75 m/s at 30
        fps. Game two totals two meters over two frames for 30 m/s. The
        games contain two and one unique player possessions
        respectively.
        The null team possession is excluded.

    Guards against:
        Cross-game aggregation, row-count denominators, duplicate-frame
        inflation, null-key groups, null-player inflation, and unstable
        output ordering.
    """
    frames = pl.DataFrame(
        {
            "game_id": [2, 1, 1, 1, 1, 1],
            "dev_match_team_possession_id": [
                "shared",
                "shared",
                "shared",
                "shared",
                "shared",
                None,
            ],
            "dev_match_team_player_possession_id": [
                "game_2_player",
                None,
                "game_1_player_a",
                "game_1_player_b",
                "game_1_player_b",
                "orphan_player",
            ],
            "framenum": [12, 1, 3, 4, 6, 8],
            "pitch_third": ["Middle"] * 6,
            "ball_displacement": [2.0, None, 6.0, 100.0, 4.0, 9.0],
            "delta_frame": [2, None, 2, 0, 2, 1],
            "frame_rate": [30.0] * 6,
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    result = aggregate_possession_tempo(
        frames,
        "team",
    ).collect()

    assert result.columns == [
        "game_id",
        "match_team_possession_id",
        "start_frame",
        "end_frame",
        "unique_player_possession_count",
        "starting_pitch_third",
        "frame_rate",
        "valid_segment_count",
        "total_ball_displacement",
        "elapsed_frames",
        "elapsed_duration",
        "ball_speed_tempo_team",
    ]
    assert result.to_dicts() == [
        {
            "game_id": 1,
            "match_team_possession_id": "shared",
            "start_frame": 1,
            "end_frame": 6,
            "unique_player_possession_count": 2,
            "starting_pitch_third": "Middle",
            "frame_rate": 30.0,
            "valid_segment_count": 2,
            "total_ball_displacement": 10.0,
            "elapsed_frames": 4.0,
            "elapsed_duration": 0.13333333333333333,
            "ball_speed_tempo_team": 75.0,
        },
        {
            "game_id": 2,
            "match_team_possession_id": "shared",
            "start_frame": 12,
            "end_frame": 12,
            "unique_player_possession_count": 1,
            "starting_pitch_third": "Middle",
            "frame_rate": 30.0,
            "valid_segment_count": 1,
            "total_ball_displacement": 2.0,
            "elapsed_frames": 2.0,
            "elapsed_duration": 0.06666666666666667,
            "ball_speed_tempo_team": 30.0,
        },
    ]
    assert result.schema["valid_segment_count"] == pl.UInt32
    assert result.schema["unique_player_possession_count"] == pl.UInt32
    assert result.schema["frame_rate"] == pl.Float64
    assert result.schema["total_ball_displacement"] == pl.Float64
    assert result.schema["elapsed_frames"] == pl.Float64
    assert result.schema["elapsed_duration"] == pl.Float64
    assert result.schema["ball_speed_tempo_team"] == pl.Float64


def test_player_tempo_credits_synthetic_pass_to_passer() -> None:
    """Preserve passer and receiver ownership during aggregation.

    Behavior:
        Aggregate the team-level displacement rows by player possession
        without reassigning a synthetic pass endpoint to the receiver.

    Input:
        A passer with a four-meter synthetic segment and a receiver with
        one null duplicate row followed by a one-meter segment.

    Expected:
        The passer tempo is 120 m/s and the receiver tempo is 30 m/s at
        30 fps, with one valid segment assigned to each possession.

    Guards against:
        Crediting the incoming pass to the receiver or counting the
        receiver's duplicate frame in the elapsed-time denominator.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "dev_match_team_possession_id": ["team_a"] * 4,
            "dev_match_team_player_possession_id": [
                "passer",
                "receiver",
                "passer",
                "receiver",
            ],
            "framenum": [1, 2, 2, 3],
            "pitch_third": ["Middle"] * 4,
            "ball_displacement": [None, None, 4.0, 1.0],
            "delta_frame": [None, None, 1, 1],
            "frame_rate": [30.0] * 4,
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    result = aggregate_possession_tempo(
        frames,
        "player",
    ).collect()

    assert "dev_match_team_possession_id" not in result.columns
    assert "dev_match_team_player_possession_id" not in result.columns
    assert result.select(
        "match_team_player_possession_id",
        "valid_segment_count",
        "total_ball_displacement",
        "elapsed_frames",
        "ball_speed_tempo_player",
    ).to_dicts() == [
        {
            "match_team_player_possession_id": "passer",
            "valid_segment_count": 1,
            "total_ball_displacement": 4.0,
            "elapsed_frames": 1.0,
            "ball_speed_tempo_player": 120.0,
        },
        {
            "match_team_player_possession_id": "receiver",
            "valid_segment_count": 1,
            "total_ball_displacement": 1.0,
            "elapsed_frames": 1.0,
            "ball_speed_tempo_player": 30.0,
        },
    ]


def test_starting_pitch_third_follows_frame_order() -> None:
    """Retain each possession's earliest pitch third.

    Behavior:
        Select the pitch third from the earliest chronological frame for
        both team and player possessions.

    Input:
        Unsorted rows containing two player spells within one team
        spell.

    Expected:
        The team and first player start in the middle third. The second
        player starts in the defensive third.

    Guards against:
        Selecting the first physical row instead of ordering by frame.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1] * 7,
            "dev_match_team_possession_id": ["team_a"] * 7,
            "dev_match_team_player_possession_id": [
                "player_1",
                "player_1",
                "player_2",
                "player_2",
                "player_1",
                "player_2",
                "player_2",
            ],
            "framenum": [3, 1, 7, 4, 2, 6, 5],
            "pitch_third": [
                "Middle",
                "Middle",
                "Attacking",
                "Defensive",
                "Attacking",
                "Attacking",
                None,
            ],
            "ball_displacement": [None] * 7,
            "delta_frame": [None] * 7,
            "frame_rate": [30.0] * 7,
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    team_result = aggregate_possession_tempo(frames, "team").collect()
    player_result = aggregate_possession_tempo(frames, "player").collect()

    assert team_result["starting_pitch_third"].to_list() == ["Middle"]
    assert player_result.select(
        "match_team_player_possession_id",
        "starting_pitch_third",
    ).to_dicts() == [
        {
            "match_team_player_possession_id": "player_1",
            "starting_pitch_third": "Middle",
        },
        {
            "match_team_player_possession_id": "player_2",
            "starting_pitch_third": "Defensive",
        },
    ]


def test_add_possession_sequence_numbers_unique_starts() -> None:
    """Number unique player-possession starts within each team spell.

    Behavior:
        Rank player possessions one-based by start frame and invalidate
        the complete team sequence when two starts are tied.

    Input:
        One team with two unsorted unique starts and another team with
        two player possessions sharing frame ten.

    Expected:
        The first team's physical rows receive sequences two and one.
        Both rows in the team with an ambiguous start receive null.

    Guards against:
        Sequencing by input order or silently inventing an order for
        player possessions that begin on the same frame.
    """
    possessions = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "dev_match_team_possession_id": [
                "team_a",
                "team_a",
                "team_b",
                "team_b",
            ],
            "dev_match_team_player_possession_id": [
                "player_2",
                "player_1",
                "player_3",
                "player_4",
            ],
            "start_frame": [5, 1, 10, 10],
        }
    ).lazy()

    result = _add_possession_sequence(possessions).collect()

    assert result["player_possession_sequence_number"].to_list() == [
        2,
        1,
        None,
        None,
    ]
    assert result.schema["player_possession_sequence_number"] == pl.UInt32


def test_add_possession_time_elapsed_uses_frame_difference() -> None:
    """Convert player starts into elapsed team-possession time.

    Behavior:
        Subtract the true team start numerically and divide the result
        by
        each match's frame rate.

    Input:
        Valid starts five and ten frames into team possessions sampled
        at
        20 and 25 fps, plus a player start before its team start.

    Expected:
        Valid rows produce 5 frames and 0.25 seconds, then 10 frames and
        0.4 seconds. The impossible negative interval produces nulls.

    Guards against:
        Counting rows, assuming one global FPS, or reporting negative
        elapsed possession time.
    """
    possessions = pl.DataFrame(
        {
            "start_frame": [105, 200, 299],
            "__team_possession_start_frame": [100, 190, 300],
            "frame_rate": [20.0, 25.0, 30.0],
        }
    ).lazy()

    result = _add_possession_time_elapsed(possessions).collect()

    assert result["elapsed_frames_team_possession"].to_list() == [
        5,
        10,
        None,
    ]
    assert result["elapsed_seconds_team_possession"].to_list() == [
        0.25,
        0.4,
        None,
    ]
    assert result.schema["elapsed_frames_team_possession"] == pl.Int64
    assert result.schema["elapsed_seconds_team_possession"] == pl.Float64


def test_player_aggregation_uses_true_team_start_and_sequence() -> None:
    """Add chronological team context to player-level aggregates.

    Behavior:
        Derive the team start before excluding unassigned player rows,
        then add sequence and elapsed-time columns after aggregation.

    Input:
        One team beginning at frame 100 without a player ID, followed by
        three player possessions starting at frames 102, 105, and 110.
        A player possession without a team ID is also included.

    Expected:
        The three valid players receive sequences 1–3, elapsed frames 2,
        5, and 10, and elapsed seconds 0.1, 0.25, and 0.5 at 20 fps. The
        player without a team possession is excluded.

    Guards against:
        Treating the first identified player as the team start, grouping
        orphan players, or sequencing by physical row order.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1, 1],
            "dev_match_team_possession_id": [
                "team_a",
                "team_a",
                "team_a",
                "team_a",
                "team_a",
                None,
            ],
            "dev_match_team_player_possession_id": [
                "player_2",
                None,
                "player_1",
                "player_1",
                "player_3",
                "orphan",
            ],
            "framenum": [105, 100, 103, 102, 110, 50],
            "pitch_third": ["Middle"] * 6,
            "ball_displacement": [1.0, None, 1.0, None, 2.0, 1.0],
            "delta_frame": [1, None, 1, None, 5, 1],
            "frame_rate": [20.0] * 6,
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    result = aggregate_possession_tempo(frames, "player").collect()

    assert result.select(
        "match_team_possession_id",
        "match_team_player_possession_id",
        "player_possession_sequence_number",
        "elapsed_frames_team_possession",
        "elapsed_seconds_team_possession",
    ).to_dicts() == [
        {
            "match_team_possession_id": "team_a",
            "match_team_player_possession_id": "player_1",
            "player_possession_sequence_number": 1,
            "elapsed_frames_team_possession": 2,
            "elapsed_seconds_team_possession": 0.1,
        },
        {
            "match_team_possession_id": "team_a",
            "match_team_player_possession_id": "player_2",
            "player_possession_sequence_number": 2,
            "elapsed_frames_team_possession": 5,
            "elapsed_seconds_team_possession": 0.25,
        },
        {
            "match_team_possession_id": "team_a",
            "match_team_player_possession_id": "player_3",
            "player_possession_sequence_number": 3,
            "elapsed_frames_team_possession": 10,
            "elapsed_seconds_team_possession": 0.5,
        },
    ]


def test_tempo_without_valid_segments_has_null_metrics() -> None:
    """Represent absent movement evidence as null rather than zero.

    Behavior:
        Retain a valid possession that has no measurable movement while
        reporting its aggregate movement metrics as unavailable.

    Input:
        One possession containing a leading row and invalid zero- and
        negative-frame intervals.

    Expected:
        The possession has zero valid segments and null distance,
        elapsed frames, and tempo values.

    Guards against:
        Treating Polars' all-null sum as measured zero movement or
        emitting misleading zero-speed possession metrics.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "dev_match_team_possession_id": ["team_a"] * 3,
            "dev_match_team_player_possession_id": ["player_a"] * 3,
            "framenum": [1, 2, 3],
            "pitch_third": ["Middle"] * 3,
            "ball_displacement": [None, 2.0, 3.0],
            "delta_frame": [None, 0, -1],
            "frame_rate": [30.0] * 3,
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()

    result = aggregate_possession_tempo(
        frames,
        "team",
    ).collect()

    assert result.select(
        "unique_player_possession_count",
        "valid_segment_count",
        "total_ball_displacement",
        "elapsed_frames",
        "ball_speed_tempo_team",
    ).row(0) == (1, 0, None, None, None)


def test_possession_tempo_nulls_inconsistent_frame_rates() -> None:
    """Reject conflicting frame-rate metadata within one game.

    Behavior:
        Require one finite, positive frame rate throughout a possession.

    Input:
        One team possession with valid segments labeled 25 and 30 fps.

    Expected:
        The output frame rate and tempo are null while the measurable
        distance and elapsed frames remain available.

    Guards against:
        Arbitrarily selecting the first rate in inconsistent metadata.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1],
            "dev_match_team_possession_id": ["team_a"] * 2,
            "dev_match_team_player_possession_id": ["player_a"] * 2,
            "framenum": [2, 3],
            "pitch_third": ["Middle"] * 2,
            "ball_displacement": [2.0, 3.0],
            "delta_frame": [1, 1],
            "frame_rate": [25.0, 30.0],
        }
    ).lazy()

    result = aggregate_possession_tempo(frames, "team").collect()

    assert result["total_ball_displacement"].to_list() == [5.0]
    assert result["elapsed_frames"].to_list() == [2.0]
    assert result["frame_rate"].to_list() == [None]
    assert result["ball_speed_tempo_team"].to_list() == [None]


@pytest.mark.parametrize(
    "frame_rate",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_possession_tempo_nulls_invalid_frame_rates(
    frame_rate: float,
) -> None:
    """Null tempo when stored frame-rate metadata is invalid.

    Behavior:
        Refuse to calculate tempo from a nonpositive or nonfinite stored
        frame rate.

    Input:
        One valid movement segment paired with an invalid frame rate.

    Expected:
        Distance and elapsed frames remain measurable, while the output
        frame rate and tempo are null.

    Guards against:
        Division by zero and negative, NaN, or infinite tempo values.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1],
            "dev_match_team_possession_id": ["team_a"],
            "dev_match_team_player_possession_id": ["player_a"],
            "framenum": [2],
            "pitch_third": ["Middle"],
            "ball_displacement": [2.0],
            "delta_frame": [1],
            "frame_rate": [frame_rate],
        },
        schema_overrides={"frame_rate": pl.Float64},
    ).lazy()

    result = aggregate_possession_tempo(frames, "team").collect()

    assert result["total_ball_displacement"].to_list() == [2.0]
    assert result["elapsed_frames"].to_list() == [1.0]
    assert result["frame_rate"].to_list() == [None]
    assert result["ball_speed_tempo_team"].to_list() == [None]


def test_possession_tempo_rejects_unknown_level() -> None:
    """Reject unsupported possession levels before schema planning.

    Behavior:
        Restrict the public API to the documented team and player
        levels.

    Input:
        An empty LazyFrame and an unsupported match-level value.

    Expected:
        A clear ``ValueError`` identifies the invalid level.

    Guards against:
        Unbound group columns and opaque downstream Polars errors.
    """
    with pytest.raises(
        ValueError,
        match="level must be either 'team' or 'player'",
    ):
        aggregate_possession_tempo(
            pl.DataFrame().lazy(),
            "match",  # type: ignore[arg-type]
        )


def test_possession_tempo_rejects_invalid_metric_schema() -> None:
    """Reject nonnumeric aggregation inputs with a readable error.

    Behavior:
        Validate required metric dtypes before constructing aggregation
        expressions.

    Input:
        A typed empty team-possession frame whose frame delta is text.

    Expected:
        A ``TypeError`` names the invalid numeric column.

    Guards against:
        Deferred Polars arithmetic errors that obscure the data
        contract.
    """
    frames = pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "dev_match_team_possession_id": pl.String,
            "dev_match_team_player_possession_id": pl.String,
            "framenum": pl.Int64,
            "pitch_third": pl.String,
            "ball_displacement": pl.Float64,
            "delta_frame": pl.String,
            "frame_rate": pl.Float64,
        }
    ).lazy()

    with pytest.raises(
        TypeError,
        match="Possession tempo columns must be numeric: delta_frame",
    ):
        aggregate_possession_tempo(
            frames,
            "team",
        )


def test_output_writer_materializes_frame_transform_once(
    tmp_path: Path,
) -> None:
    """Materialize expensive frame transformations only once.

    Behavior:
        Write frame-level data first, then scan that parquet to produce
        the team- and player-level tempo datasets.

    Input:
        Four tiny frame rows behind an execution-counting lazy batch,
        including a synthetic passer segment and receiver segment.

    Expected:
        The upstream lazy batch executes once. Three parquet files are
        written with four frame rows, one team row, and two player rows.
        Their known tempo values are 75, 120, and 30 m/s respectively.

    Guards against:
        Re-executing the complete frame transformation separately for
        every output or changing passer and receiver metric ownership.
    """
    execution_count = 0

    def count_execution(batch: pl.DataFrame) -> pl.DataFrame:
        """Record each execution of the upstream lazy frame."""
        nonlocal execution_count
        execution_count += 1
        return batch

    frame_df = (
        pl.DataFrame(
            {
                "game_id": [1, 1, 1, 1],
                "dev_match_team_possession_id": ["team_a"] * 4,
                "dev_match_team_player_possession_id": [
                    "passer",
                    "receiver",
                    "passer",
                    "receiver",
                ],
                "framenum": [1, 2, 2, 3],
                "pitch_third": ["Middle"] * 4,
                "ball_displacement": [None, None, 4.0, 1.0],
                "delta_frame": [None, None, 1, 1],
                "frame_rate": [30.0] * 4,
                "is_ball_release_attempt": [False, False, True, False],
            },
            schema_overrides={
                "ball_displacement": pl.Float64,
                "delta_frame": pl.Int64,
            },
        )
        .lazy()
        .map_batches(count_execution)
    )

    write_possession_outputs(
        frame_df,
        "frames.parquet",
        output_dir=tmp_path,
    )

    written_frames = pl.read_parquet(tmp_path / "frames.parquet")
    team_tempo = pl.read_parquet(tmp_path / "team_frames.parquet")
    player_tempo = pl.read_parquet(tmp_path / "player_frames.parquet")

    assert execution_count == 1
    assert written_frames.height == 4
    assert team_tempo.height == 1
    assert team_tempo["ball_speed_tempo_team"].to_list() == [75.0]
    assert player_tempo.height == 2
    assert player_tempo.select(
        "match_team_player_possession_id",
        "ball_speed_tempo_player",
    ).to_dicts() == [
        {
            "match_team_player_possession_id": "passer",
            "ball_speed_tempo_player": 120.0,
        },
        {
            "match_team_player_possession_id": "receiver",
            "ball_speed_tempo_player": 30.0,
        },
    ]


def test_output_writer_restores_prior_files_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the complete prior output set after replacement fails.

    Behavior:
        Stage all new outputs, back up existing files during
        publication,
        and roll every published file back after a replacement error.

    Input:
        Three existing parquet files, a valid new frame dataset, and an
        injected failure while publishing the new team output.

    Expected:
        The write raises its original error. Every public output still
        contains the old marker, and the temporary directory is removed.

    Guards against:
        Leaving a new frame file alongside old team and player files
        when
        publication fails partway through the three replacements.
    """
    output_paths = (
        tmp_path / "frames.parquet",
        tmp_path / "team_frames.parquet",
        tmp_path / "player_frames.parquet",
    )
    for output_path in output_paths:
        pl.DataFrame({"version": ["old"]}).write_parquet(output_path)

    frame_df = pl.DataFrame(
        {
            "game_id": [1, 1],
            "dev_match_team_possession_id": ["team_a"] * 2,
            "dev_match_team_player_possession_id": ["player_a"] * 2,
            "framenum": [1, 2],
            "pitch_third": ["Middle"] * 2,
            "ball_displacement": [None, 2.0],
            "delta_frame": [None, 1],
            "frame_rate": [30.0, 30.0],
            "is_ball_release_attempt": [False, False],
        },
        schema_overrides={
            "ball_displacement": pl.Float64,
            "delta_frame": pl.Int64,
        },
    ).lazy()
    original_replace = Path.replace
    replacement_failed = False

    def fail_team_publication(
        source: Path,
        target: str | Path,
    ) -> Path:
        """Fail once when the staged team file reaches its public path.
        """
        nonlocal replacement_failed
        target_path = Path(target)
        is_team_publication = (
            not replacement_failed
            and source.name == "team_frames.parquet"
            and source.parent.name.startswith(".possession-outputs-")
            and target_path.parent == tmp_path
        )
        if is_team_publication:
            replacement_failed = True
            raise OSError("injected team publication failure")

        return original_replace(source, target_path)

    monkeypatch.setattr(Path, "replace", fail_team_publication)

    with pytest.raises(
        OSError,
        match="injected team publication failure",
    ):
        write_possession_outputs(
            frame_df,
            "frames.parquet",
            output_dir=tmp_path,
        )

    assert replacement_failed
    for output_path in output_paths:
        assert pl.read_parquet(output_path).to_dicts() == [{"version": "old"}]
    assert list(tmp_path.glob(".possession-outputs-*")) == []
