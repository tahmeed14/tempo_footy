"""Test tracking-data interpolation helpers."""

import polars as pl
import pytest

from tempoctrl.gradient_sports.interpolations import (
    _CAN_IMPUTE_COLUMN,
    _IS_FIRST_FRAME_COLUMN,
    _NEXT_FRAME_COLUMN,
    _NEXT_X_COLUMN,
    _NEXT_Y_COLUMN,
    _PREVIOUS_FRAME_COLUMN,
    _PREVIOUS_X_COLUMN,
    _PREVIOUS_Y_COLUMN,
    _ROW_ORDER_COLUMN,
    _SOURCE_X_COLUMN,
    _SOURCE_Y_COLUMN,
    _TEMPORARY_COLUMNS,
    _X_CANDIDATE_COLUMN,
    _Y_CANDIDATE_COLUMN,
    _add_ball_gap_context,
    _add_interpolation_eligibility,
    _add_linear_interpolation_candidates,
    _add_source_coordinates,
    _apply_interpolation_candidates,
    _drop_interpolation_columns,
    _validate_interpolation_input,
    interpolate_ball_coordinates,
)

BALL_DTYPE = pl.Struct(
    {
        "visibility": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
    }
)


def test_interpolates_short_bounded_gap_in_frame_order() -> None:
    """Interpolate a short bounded gap while preserving input order.

    Behavior:
        Linearly interpolate x and y using frame order within one
        possession and mark only the filled rows as imputed.

    Input:
        Four unsorted rows containing two missing coordinate pairs
        bounded by observations at frames one and four.

    Expected:
        Frames two and three receive explicit linear values. Original
        order, other fields, row count, and numeric dtypes remain.

    Guards against:
        Interpolating in input order, sorting output, losing struct
        fields, marking observations as imputed, or changing dtype.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "framenum": [3, 1, 2, 4],
            "dev_match_team_player_possession_id": ["possession_a"] * 4,
            "balls_smooth": [
                {
                    "visibility": "estimated",
                    "x": None,
                    "y": None,
                    "z": 0.0,
                },
                {
                    "visibility": "visible",
                    "x": 0.0,
                    "y": 10.0,
                    "z": 1.0,
                },
                {
                    "visibility": "estimated",
                    "x": None,
                    "y": None,
                    "z": 0.5,
                },
                {
                    "visibility": "visible",
                    "x": 3.0,
                    "y": 13.0,
                    "z": 2.0,
                },
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = interpolate_ball_coordinates(frames, max_gap=2).collect()

    assert result.height == 4
    assert result["framenum"].to_list() == [3, 1, 2, 4]
    assert result["balls_smooth"].struct.field("x").to_list() == [
        2.0,
        0.0,
        1.0,
        3.0,
    ]
    assert result["balls_smooth"].struct.field("y").to_list() == [
        12.0,
        10.0,
        11.0,
        13.0,
    ]
    assert result["balls_smooth"].struct.field("visibility").to_list() == [
        "estimated",
        "visible",
        "estimated",
        "visible",
    ]
    assert result["balls_smooth"].struct.field("z").to_list() == [
        0.0,
        1.0,
        0.5,
        2.0,
    ]
    assert result["balls_smooth"].struct.field("is_imputed").to_list() == [
        True,
        False,
        True,
        False,
    ]
    assert result.schema["balls_smooth"] == pl.Struct(
        {
            "visibility": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
            "is_imputed": pl.Boolean,
        }
    )


def test_leaves_long_unbounded_and_partial_gaps_unchanged() -> None:
    """Reject gaps outside the existing interpolation eligibility rules.

    Behavior:
        Leave long gaps, leading and trailing gaps, and partially
        missing coordinate pairs unchanged.

    Input:
        One possession with leading/trailing nulls and a two-row
        internal gap over a one-row limit; another has only x missing.

    Expected:
        Every missing value remains null, the partial y value remains
        one, and no row is marked as imputed.

    Guards against:
        Exceeding ``max_gap``, extrapolating at boundaries, or replacing
        a partially observed coordinate pair.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1] * 9,
            "framenum": [1, 2, 3, 4, 5, 6, 1, 2, 3],
            "dev_match_team_player_possession_id": [
                "long_gap",
                "long_gap",
                "long_gap",
                "long_gap",
                "long_gap",
                "long_gap",
                "partial_gap",
                "partial_gap",
                "partial_gap",
            ],
            "balls_smooth": [
                None,
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                None,
                None,
                {"visibility": "v", "x": 3.0, "y": 3.0, "z": 0.0},
                None,
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                {"visibility": "v", "x": None, "y": 1.0, "z": 0.0},
                {"visibility": "v", "x": 2.0, "y": 2.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = interpolate_ball_coordinates(frames, max_gap=1).collect()
    x_values = result["balls_smooth"].struct.field("x").to_list()
    y_values = result["balls_smooth"].struct.field("y").to_list()

    assert x_values == [None, 0.0, None, None, 3.0, None, 0.0, None, 2.0]
    assert y_values == [None, 0.0, None, None, 3.0, None, 0.0, 1.0, 2.0]
    assert result["balls_smooth"].struct.field("is_imputed").sum() == 0


def test_isolates_games_possessions_and_null_possession_ids() -> None:
    """Prevent interpolation across game and possession boundaries.

    Behavior:
        Partition interpolation by game and the configured possession
        ID, and require a non-null ID before filling a row.

    Input:
        Matching custom IDs in separate games with complementary open
        gaps, one bounded unusual ID, and one bounded null-ID sequence.

    Expected:
        Only the bounded ``unusual/id`` row is interpolated. Open gaps
        and the bounded null-possession row remain null.

    Guards against:
        Cross-game leakage, treating null IDs as one possession,
        ignoring a custom column, or rejecting unexpected ID strings.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
            "framenum": [1, 2, 1, 2, 1, 2, 3, 1, 2, 3],
            "custom_possession": [
                "shared",
                "shared",
                "shared",
                "shared",
                "unusual/id",
                "unusual/id",
                "unusual/id",
                None,
                None,
                None,
            ],
            "balls_smooth": [
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                None,
                None,
                {"visibility": "v", "x": 2.0, "y": 2.0, "z": 0.0},
                {"visibility": "v", "x": 0.0, "y": 2.0, "z": 0.0},
                None,
                {"visibility": "v", "x": 2.0, "y": 4.0, "z": 0.0},
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                None,
                {"visibility": "v", "x": 2.0, "y": 2.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = interpolate_ball_coordinates(
        frames,
        possession_col="custom_possession",
        max_gap=1,
    ).collect()

    assert result["balls_smooth"].struct.field("x").to_list() == [
        0.0,
        None,
        None,
        2.0,
        0.0,
        1.0,
        2.0,
        0.0,
        None,
        2.0,
    ]
    assert result["balls_smooth"].struct.field("is_imputed").to_list() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("max_gap", [0, -1])
def test_rejects_nonpositive_max_gap(max_gap: int) -> None:
    """Reject a maximum gap that cannot admit any missing rows.

    Behavior:
        Validate ``max_gap`` before constructing the lazy interpolation.

    Input:
        An empty, typed LazyFrame and a zero or negative gap limit.

    Expected:
        A clear ``ValueError`` is raised immediately.

    Guards against:
        Silently disabling interpolation or producing misleading lazy
        execution errors for invalid configuration.
    """
    frames = pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "framenum": pl.Int64,
            "dev_match_team_player_possession_id": pl.String,
            "balls_smooth": BALL_DTYPE,
        }
    ).lazy()

    with pytest.raises(ValueError, match="max_gap must be at least 1"):
        interpolate_ball_coordinates(frames, max_gap=max_gap)


def test_gap_context_uses_ordered_partition_boundaries() -> None:
    """Expose gap bounds without crossing game partitions.

    Behavior:
        Calculate ordered positions and nearest observed positions
        within each partition while retaining input row order.

    Input:
        One unsorted bounded gap and a second game's leading gap using
        the same possession ID.

    Expected:
        The bounded row has positions zero and two as its bounds. The
        second game's leading row has no previous observed position.

    Guards against:
        Using input order for bounds or leaking observations between
        games that share a possession identifier.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "framenum": [3, 1, 2, 1, 2],
            "possession_id": ["shared"] * 5,
            "balls_smooth": [
                {"visibility": "v", "x": 2.0, "y": 4.0, "z": 0.0},
                {"visibility": "v", "x": 0.0, "y": 2.0, "z": 0.0},
                None,
                None,
                {"visibility": "v", "x": 8.0, "y": 6.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = (
        frames.pipe(_add_source_coordinates, overwrite=False)
        .pipe(
            _add_ball_gap_context,
            partition_columns=("game_id", "possession_id"),
        )
        .collect()
    )

    assert result[_PREVIOUS_FRAME_COLUMN].to_list() == [
        3,
        1,
        1,
        None,
        2,
    ]
    assert result[_NEXT_FRAME_COLUMN].to_list() == [3, 1, 3, 2, 2]
    assert result["framenum"].to_list() == [3, 1, 2, 1, 2]


def test_eligibility_marks_only_short_complete_bounded_gaps() -> None:
    """Separate interpolation policy from coordinate calculation.

    Behavior:
        Mark only fully missing rows in bounded gaps no larger than the
        configured limit and having a non-null possession ID.

    Input:
        A one-row eligible gap, a partial coordinate, and a bounded row
        whose possession ID is null, with explicit boundary positions.

    Expected:
        Only the fully missing row with a real possession ID is marked.

    Guards against:
        Filling partial coordinates or treating null IDs as a valid
        possession partition.
    """
    frames = pl.DataFrame(
        {
            "possession_id": ["p", "p", None],
            "framenum": [2, 2, 2],
            _IS_FIRST_FRAME_COLUMN: [True, True, True],
            _SOURCE_X_COLUMN: [None, None, None],
            _SOURCE_Y_COLUMN: [None, 1.0, None],
            _PREVIOUS_FRAME_COLUMN: [1, 1, 1],
            _NEXT_FRAME_COLUMN: [3, 3, 3],
        }
    ).lazy()

    result = _add_interpolation_eligibility(
        frames,
        possession_col="possession_id",
        max_gap=1,
    ).collect()

    assert result[_CAN_IMPUTE_COLUMN].to_list() == [True, False, False]


def test_linear_candidate_helper_only_calculates_coordinates() -> None:
    """Keep linear interpolation behind the candidate interface.

    Behavior:
        Produce linear x and y candidates in frame order without
        modifying the ball struct.

    Input:
        Three unsorted frames with a missing midpoint bounded by known
        coordinates.

    Expected:
        The midpoint candidates are two and four, and the source ball
        coordinates remain null.

    Guards against:
        Coupling the methodology helper to eligibility or applying its
        candidates before the shared application step.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "framenum": [3, 1, 2],
            "possession_id": ["p", "p", "p"],
            "balls_smooth": [
                {"visibility": "v", "x": 4.0, "y": 6.0, "z": 0.0},
                {"visibility": "v", "x": 0.0, "y": 2.0, "z": 0.0},
                None,
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = (
        frames.pipe(_add_source_coordinates, overwrite=False)
        .pipe(
            _add_ball_gap_context,
            partition_columns=("game_id", "possession_id"),
        )
        .pipe(_add_linear_interpolation_candidates)
        .collect()
    )

    assert result[_X_CANDIDATE_COLUMN].to_list() == [4.0, 0.0, 2.0]
    assert result[_Y_CANDIDATE_COLUMN].to_list() == [6.0, 2.0, 4.0]
    assert result["balls_smooth"].struct.field("x").to_list() == [
        4.0,
        0.0,
        None,
    ]


def test_application_and_cleanup_follow_candidate_contract() -> None:
    """Apply only eligible candidates and remove internal state.

    Behavior:
        Merge method-neutral candidates into eligible ball structs,
        preserve other fields, add markers, and drop temporary columns.

    Input:
        One eligible missing row and one ineligible observed row with
        deliberately different candidate values.

    Expected:
        Only the eligible row receives its candidates. Visibility and z
        remain unchanged, and no internal columns remain after cleanup.

    Guards against:
        Applying every candidate, dropping unrelated struct fields, or
        leaking implementation columns into downstream pipeline stages.
    """
    frames = pl.DataFrame(
        {
            "balls_smooth": [
                {"visibility": "estimated", "x": None, "y": None, "z": 1.0},
                {"visibility": "visible", "x": 9.0, "y": 8.0, "z": 2.0},
            ],
            _CAN_IMPUTE_COLUMN: [True, False],
            _SOURCE_X_COLUMN: [None, 9.0],
            _SOURCE_Y_COLUMN: [None, 8.0],
            _X_CANDIDATE_COLUMN: [3.0, 30.0],
            _Y_CANDIDATE_COLUMN: [4.0, 40.0],
            _ROW_ORDER_COLUMN: [0, 1],
            _IS_FIRST_FRAME_COLUMN: [True, True],
            _PREVIOUS_FRAME_COLUMN: [1, 2],
            _NEXT_FRAME_COLUMN: [3, 2],
            _PREVIOUS_X_COLUMN: [1.0, 9.0],
            _NEXT_X_COLUMN: [5.0, 9.0],
            _PREVIOUS_Y_COLUMN: [2.0, 8.0],
            _NEXT_Y_COLUMN: [6.0, 8.0],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = (
        frames.pipe(_apply_interpolation_candidates)
        .pipe(_drop_interpolation_columns)
        .collect()
    )

    assert result["balls_smooth"].struct.field("x").to_list() == [3.0, 9.0]
    assert result["balls_smooth"].struct.field("y").to_list() == [4.0, 8.0]
    assert result["balls_smooth"].struct.field("visibility").to_list() == [
        "estimated",
        "visible",
    ]
    assert result["balls_smooth"].struct.field("z").to_list() == [1.0, 2.0]
    assert result["balls_smooth"].struct.field("is_imputed").to_list() == [
        True,
        False,
    ]
    assert not set(_TEMPORARY_COLUMNS).intersection(result.columns)


def test_uses_numerical_frame_distance_for_limits_and_values() -> None:
    """Use frame numbers for both gap limits and linear position.

    Behavior:
        Measure the gap and interpolation fraction from numerical frame
        differences rather than the number and position of input rows.

    Input:
        Frames one, two, and five, with only frame two missing between
        endpoint x values zero and eight.

    Expected:
        A limit of two rejects the three-frame gap. A limit of three
        fills frame two with x two, one quarter of the endpoint change.

    Guards against:
        Treating one missing row as a one-frame gap or placing it at the
        midpoint merely because it is the middle dataset row.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "framenum": [1, 2, 5],
            "dev_match_team_player_possession_id": ["p", "p", "p"],
            "balls_smooth": [
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                None,
                {"visibility": "v", "x": 8.0, "y": 4.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    rejected = interpolate_ball_coordinates(frames, max_gap=2).collect()
    accepted = interpolate_ball_coordinates(frames, max_gap=3).collect()

    assert rejected["balls_smooth"].struct.field("x").to_list() == [
        0.0,
        None,
        8.0,
    ]
    assert accepted["balls_smooth"].struct.field("x").to_list() == [
        0.0,
        2.0,
        8.0,
    ]
    assert accepted["balls_smooth"].struct.field("y").to_list() == [
        0.0,
        1.0,
        4.0,
    ]


def test_only_first_duplicate_frame_participates() -> None:
    """Use the first row when a frame number appears more than once.

    Behavior:
        Allow only the first duplicate row to be an anchor or eligible
        interpolation target while preserving every duplicate row.

    Input:
        Frame three appears twice. Its first row is missing and its
        second row has coordinates far from the surrounding trajectory.

    Expected:
        The first frame-three row is interpolated to four. The later
        duplicate remains at one hundred and does not become an anchor.

    Guards against:
        Dropping duplicate rows, using a later duplicate as the frame's
        representative, or modifying every row at the duplicate frame.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "framenum": [1, 3, 3, 5],
            "dev_match_team_player_possession_id": ["p"] * 4,
            "balls_smooth": [
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                None,
                {
                    "visibility": "duplicate",
                    "x": 100.0,
                    "y": 100.0,
                    "z": 0.0,
                },
                {"visibility": "v", "x": 8.0, "y": 4.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()

    result = interpolate_ball_coordinates(frames, max_gap=3).collect()

    assert result.height == 4
    assert result["balls_smooth"].struct.field("x").to_list() == [
        0.0,
        4.0,
        100.0,
        8.0,
    ]
    assert result["balls_smooth"].struct.field("is_imputed").to_list() == [
        False,
        True,
        False,
        False,
    ]


def test_requires_overwrite_to_recalculate_existing_imputations() -> None:
    """Protect prior interpolation results unless overwrite is explicit.

    Behavior:
        Detect the existing imputation marker, reject an accidental
        rerun, and treat previously imputed coordinates as missing when
        overwrite is enabled.

    Input:
        A frame-two value first imputed across frames one and four, then
        rerun with both the original and a smaller maximum gap.

    Expected:
        The default rerun raises. Explicit overwrite reproduces the
        value and marker; reducing the limit clears the old estimate.

    Guards against:
        Treating estimates as observations, silently resetting their
        markers, or retaining stale estimates that no longer qualify.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "framenum": [1, 2, 4],
            "dev_match_team_player_possession_id": ["p", "p", "p"],
            "balls_smooth": [
                {"visibility": "v", "x": 0.0, "y": 0.0, "z": 0.0},
                None,
                {"visibility": "v", "x": 6.0, "y": 3.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": BALL_DTYPE},
    ).lazy()
    first_result = interpolate_ball_coordinates(frames, max_gap=2)

    with pytest.raises(ValueError, match="already been interpolated"):
        interpolate_ball_coordinates(first_result, max_gap=2)

    recalculated = interpolate_ball_coordinates(
        first_result,
        max_gap=2,
        overwrite=True,
    ).collect()
    disqualified = interpolate_ball_coordinates(
        first_result,
        max_gap=1,
        overwrite=True,
    ).collect()

    assert recalculated["balls_smooth"].struct.field("x").to_list() == [
        0.0,
        2.0,
        6.0,
    ]
    assert recalculated["balls_smooth"].struct.field(
        "is_imputed"
    ).to_list() == [False, True, False]
    assert disqualified["balls_smooth"].struct.field("x").to_list() == [
        0.0,
        None,
        6.0,
    ]
    assert disqualified["balls_smooth"].struct.field(
        "is_imputed"
    ).to_list() == [False, False, False]


@pytest.mark.parametrize(
    ("ball_dtype", "error_type", "message"),
    [
        (pl.Float64, TypeError, "balls_smooth must be a struct"),
        (
            pl.Struct({"x": pl.Float64}),
            ValueError,
            "balls_smooth is missing fields: y",
        ),
        (
            pl.Struct({"x": pl.String, "y": pl.Float64}),
            TypeError,
            "balls_smooth fields must be numeric: x",
        ),
    ],
)
def test_validation_rejects_incompatible_ball_schemas(
    ball_dtype: pl.DataType,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject schemas unsupported by every interpolation methodology.

    Behavior:
        Validate that the ball column is a struct with numeric x and y
        fields before building a lazy interpolation query.

    Input:
        Typed empty frames whose ball column is scalar, lacks y, or has
        a string x field.

    Expected:
        Each invalid schema raises a specific, readable exception.

    Guards against:
        Delayed Polars errors that obscure which shared input contract
        was violated.
    """
    frames = pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "framenum": pl.Int64,
            "possession_id": pl.String,
            "balls_smooth": ball_dtype,
        }
    ).lazy()

    with pytest.raises(error_type, match=message):
        _validate_interpolation_input(
            frames,
            possession_col="possession_id",
            max_gap=1,
        )
