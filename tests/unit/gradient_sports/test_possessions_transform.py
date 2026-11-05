"""Test Gradient Sports possession transformations."""

import polars as pl
import pytest

import tempoctrl.gradient_sports.possessions_transform as transform_module
from tempoctrl.gradient_sports.possessions_transform import (
    _flip_if_attacking_left,
    _successful_delivery_id,
    create_synthetic_final_pass_frame,
    fill_bounded_possessions_id,
    label_pitch_thirds,
    normalize_ball_coordinates,
    propagate_successful_delivery_possession,
    transform_possessions,
)


def test_fill_bounded_possessions_id_fills_matching_internal_gap() -> None:
    """Fill a multi-row gap enclosed by matching possession IDs.

    Behavior:
        Fill every null in an internal gap whose nearest IDs match.

    Input:
        Unsorted frames with two nulls enclosed by ``possession_a``.

    Expected:
        The development column is frame ordered and contains the ID for
        both null rows. The source column and its string dtype remain
        unchanged.

    Guards against:
        Filling only one null, mutating source data, losing dtype, or
        evaluating bounds in input order instead of frame order.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "framenum": [4, 2, 1, 3],
            "possession_id": [
                "possession_a",
                None,
                "possession_a",
                None,
            ],
        },
        schema_overrides={"possession_id": pl.String},
    ).lazy()

    result = fill_bounded_possessions_id(
        frames,
        "possession_id",
    ).collect()

    assert result["framenum"].to_list() == [1, 2, 3, 4]
    assert result["possession_id"].to_list() == [
        "possession_a",
        None,
        None,
        "possession_a",
    ]
    assert result["dev_possession_id"].to_list() == [
        "possession_a",
        "possession_a",
        "possession_a",
        "possession_a",
    ]
    assert result.schema["dev_possession_id"] == pl.String


def test_fill_bounded_possessions_id_leaves_unbounded_gaps_null() -> None:
    """Leave open and differently bounded possession gaps unchanged.

    Behavior:
        Impute only gaps with two equal, non-null bounds.

    Input:
        Leading and trailing nulls plus gaps between several distinct,
        otherwise unrestricted possession ID strings.

    Expected:
        Every null remains null and each non-null category is preserved.

    Guards against:
        Unconditional forward/backward filling, treating unequal IDs as
        equivalent, or rejecting unexpected ID values.
    """
    possession_ids = [
        None,
        "possession_a",
        None,
        "possession_b",
        None,
        "unexpected/id",
        None,
    ]
    frames = pl.DataFrame(
        {
            "game_id": [1] * len(possession_ids),
            "framenum": list(range(1, len(possession_ids) + 1)),
            "possession_id": possession_ids,
        },
        schema_overrides={"possession_id": pl.String},
    ).lazy()

    result = fill_bounded_possessions_id(
        frames,
        "possession_id",
    ).collect()

    assert result["dev_possession_id"].to_list() == possession_ids


def test_fill_bounded_possessions_id_does_not_cross_game_boundaries() -> None:
    """Keep matching IDs from bounding nulls in another game.

    Behavior:
        Calculate preceding and following possession IDs per game.

    Input:
        Game one ends with a null and game two starts with a null, with
        the same non-null ID on the outer sides of those nulls.

    Expected:
        Both nulls remain null because neither is bounded within its
        game.

    Guards against:
        Forward or backward fill state leaking across game boundaries.
    """
    frames = pl.DataFrame(
        {
            "game_id": [2, 1, 2, 1],
            "framenum": [2, 2, 1, 1],
            "possession_id": ["shared", None, None, "shared"],
        },
        schema_overrides={"possession_id": pl.String},
    ).lazy()

    result = fill_bounded_possessions_id(
        frames,
        "possession_id",
    ).collect()

    assert result["game_id"].to_list() == [1, 1, 2, 2]
    assert result["dev_possession_id"].to_list() == [
        "shared",
        None,
        None,
        "shared",
    ]


def test_fill_bounded_possessions_id_fills_multiple_columns() -> None:
    """Fill multiple possession columns in one helper call.

    Behavior:
        Apply bounded-gap filling independently to every requested
        column.

    Input:
        One team gap with matching bounds and one player gap with
        different bounds.

    Expected:
        The team gap is filled while the player gap remains null, and a
        development column is produced for each source column.

    Guards against:
        Coupling possession columns or requiring repeated helper calls
        and repeated frame sorting in the main transform.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1],
            "framenum": [1, 2, 3],
            "team_possession_id": ["team_a", None, "team_a"],
            "player_possession_id": ["player_a", None, "player_b"],
        },
        schema_overrides={
            "team_possession_id": pl.String,
            "player_possession_id": pl.String,
        },
    ).lazy()

    result = fill_bounded_possessions_id(
        frames,
        ("team_possession_id", "player_possession_id"),
    ).collect()

    assert result["dev_team_possession_id"].to_list() == [
        "team_a",
        "team_a",
        "team_a",
    ]
    assert result["dev_player_possession_id"].to_list() == [
        "player_a",
        None,
        "player_b",
    ]


def test_fill_bounded_possessions_id_shares_same_frame_annotation() -> None:
    """Share a possession annotation across duplicate tracking rows.

    Behavior:
        Treat rows with the same game and frame as one observation when
        exactly one non-null possession ID is present.

    Input:
        Two rows for frame two: one null annotation and one ``player_a``
        annotation, enclosed by unrelated IDs on adjacent frames.

    Expected:
        Both frame-two development values are ``player_a`` while the raw
        null remains unchanged and every source row is retained.

    Guards against:
        Input-order-dependent results for duplicate tracking frames or
        accidental row loss while normalizing frame-level annotations.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "framenum": [1, 2, 2, 3],
            "possession_id": [
                "previous",
                None,
                "player_a",
                "next",
            ],
        },
        schema_overrides={"possession_id": pl.String},
    ).lazy()

    result = fill_bounded_possessions_id(
        frames,
        "possession_id",
    ).collect()

    assert result.height == 4
    assert result["possession_id"].to_list() == [
        "previous",
        None,
        "player_a",
        "next",
    ]
    assert result["dev_possession_id"].to_list() == [
        "previous",
        "player_a",
        "player_a",
        "next",
    ]


def test_successful_delivery_id_tracks_valid_seed_per_game() -> None:
    """Track successful delivery IDs without crossing game boundaries.

    Behavior:
        Retain the latest successful delivery ID within each game while
        ignoring successful flags on rows without a possession ID.

    Input:
        Game one starts with a valid successful delivery. Game two
        starts with a true flag and null ID, followed by an unexpected
        ID value.

    Expected:
        Game one's delivery ID is forwarded. Every game-two result stays
        null, and the expression retains string dtype.

    Guards against:
        Cross-game propagation, null IDs becoming seeds, false flags
        becoming seeds, or dtype changes caused by null handling.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2, 2],
            "framenum": [1, 2, 3, 1, 2, 3],
            "player_possession_id": [
                "passer",
                None,
                None,
                None,
                "unexpected/id",
                None,
            ],
            "successful_pass_or_cross": [
                True,
                None,
                False,
                True,
                False,
                None,
            ],
        },
        schema_overrides={"player_possession_id": pl.String},
    ).lazy()

    result = frames.with_columns(
        _successful_delivery_id("player_possession_id")
    ).collect()

    assert result["__successful_delivery_possession_id"].to_list() == [
        "passer",
        "passer",
        "passer",
        None,
        None,
        None,
    ]
    assert result.schema["__successful_delivery_possession_id"] == pl.String


def test_propagate_successful_delivery_preserves_bounded_fills() -> None:
    """Extend successful deliveries without erasing bounded gap fills.

    Behavior:
        Fill nulls after a successful delivery until a different
        observed possession starts, while retaining earlier bounded
        imputations.

    Input:
        A successful passer and open gap, a receiver and open gap, a
        bounded gap with an unexpected ID, a failed passer and open gap,
        plus a second game beginning with a null.

    Expected:
        Only the successful passer's open gap is newly filled. The
        bounded gap remains filled; other open gaps remain null. Source
        values, row count, output schema, and string dtype are
        preserved.

    Guards against:
        Overwriting bounded fills, extending failed deliveries, leaking
        across possession or game boundaries, or exposing work columns.
    """
    source_ids = [
        "passer",
        None,
        "receiver",
        None,
        "unexpected/id",
        None,
        "unexpected/id",
        "failed_passer",
        None,
        None,
        "game_two_player",
    ]
    bounded_ids = [
        "passer",
        None,
        "receiver",
        None,
        "unexpected/id",
        "unexpected/id",
        "unexpected/id",
        "failed_passer",
        None,
        None,
        "game_two_player",
    ]
    frames = pl.DataFrame(
        {
            "game_id": [1] * 9 + [2, 2],
            "framenum": list(range(1, 10)) + [1, 2],
            "match_team_player_possession_id": source_ids,
            "dev_match_team_player_possession_id": bounded_ids,
            "successful_pass_or_cross": [
                True,
                None,
                None,
                None,
                False,
                None,
                None,
                False,
                None,
                None,
                None,
            ],
        },
        schema_overrides={
            "match_team_player_possession_id": pl.String,
            "dev_match_team_player_possession_id": pl.String,
        },
    ).lazy()

    result = propagate_successful_delivery_possession(frames).collect()

    assert result.height == 11
    assert result["match_team_player_possession_id"].to_list() == source_ids
    assert result["dev_match_team_player_possession_id"].to_list() == [
        "passer",
        "passer",
        "receiver",
        None,
        "unexpected/id",
        "unexpected/id",
        "unexpected/id",
        "failed_passer",
        None,
        None,
        "game_two_player",
    ]
    assert result.columns == [
        "game_id",
        "framenum",
        "match_team_player_possession_id",
        "dev_match_team_player_possession_id",
        "successful_pass_or_cross",
    ]
    assert result.schema["dev_match_team_player_possession_id"] == pl.String


def test_create_synthetic_pass_rows_for_gapped_and_immediate_receptions() -> (
    None
):
    """Create passer endpoints for gapped and immediate receptions.

    Behavior:
        Insert one synthetic passer row at the receiver frame after
        every successful delivery, whether or not null frames separate
        them.

    Input:
        Game one has a successful pass, two imputed frames, and a
        receiver. Game two has an immediately received successful pass.
        Each receiver frame has distinct nested tracking values.

    Expected:
        Two synthetic rows use the previous derived possession IDs and
        receiver tracking. Other source fields are null, originals
        remain unchanged, and synthetic rows precede receivers at shared
        frames.

    Guards against:
        Omitting immediate receptions, copying receiver metadata, using
        raw null possession IDs, or taking tracking from the passer
        frame.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2],
            "framenum": [10, 11, 12, 13, 20, 21],
            "match_team_possession_id": [
                "team_1",
                None,
                None,
                "team_1",
                "team_2",
                "team_2",
            ],
            "match_team_player_possession_id": [
                "passer_1",
                None,
                None,
                "receiver_1",
                "passer_2",
                "receiver_2",
            ],
            "dev_match_team_possession_id": [
                "team_1",
                "team_1",
                "team_1",
                "team_1",
                "team_2",
                "team_2",
            ],
            "dev_match_team_player_possession_id": [
                "passer_1",
                "passer_1",
                "passer_1",
                "receiver_1",
                "passer_2",
                "receiver_2",
            ],
            "successful_pass_or_cross": [
                True,
                None,
                None,
                None,
                True,
                True,
            ],
            "balls_smooth": [
                {"x": 10.0},
                {"x": 11.0},
                {"x": 12.0},
                {"x": 13.0},
                {"x": 20.0},
                {"x": 21.0},
            ],
            "away_players_smooth": [[10], [11], [12], [13], [20], [21]],
            "home_players_smooth": [[110], [111], [112], [113], [120], [121]],
            "event_metadata": [
                "pass_1",
                None,
                None,
                "receive_1",
                "pass_2",
                "receive_2",
            ],
        },
        schema_overrides={
            "balls_smooth": pl.Struct({"x": pl.Float64}),
            "away_players_smooth": pl.List(pl.Int64),
            "home_players_smooth": pl.List(pl.Int64),
        },
    ).lazy()

    result = create_synthetic_final_pass_frame(frames).collect()
    synthetic = result.filter("is_synthetic_pass_end")

    assert result.height == 8
    assert result.select(
        "game_id",
        "framenum",
        "is_synthetic_pass_end",
    ).rows() == [
        (1, 10, False),
        (1, 11, False),
        (1, 12, False),
        (1, 13, True),
        (1, 13, False),
        (2, 20, False),
        (2, 21, True),
        (2, 21, False),
    ]
    assert synthetic["dev_match_team_possession_id"].to_list() == [
        "team_1",
        "team_2",
    ]
    assert synthetic["dev_match_team_player_possession_id"].to_list() == [
        "passer_1",
        "passer_2",
    ]
    assert synthetic["match_team_possession_id"].null_count() == 2
    assert synthetic["match_team_player_possession_id"].null_count() == 2
    assert synthetic["balls_smooth"].to_list() == [
        {"x": 13.0},
        {"x": 21.0},
    ]
    assert synthetic["away_players_smooth"].to_list() == [[13], [21]]
    assert synthetic["home_players_smooth"].to_list() == [[113], [121]]
    assert synthetic["event_metadata"].null_count() == 2
    assert result.schema["is_synthetic_pass_end"] == pl.Boolean


def test_create_synthetic_pass_rows_handles_boundaries_and_duplicates() -> (
    None
):
    """Avoid invalid or duplicated synthetic delivery endpoints.

    Behavior:
        Require a successful delivery and a following new possession,
        and produce at most one synthetic row per receiver frame.

    Input:
        A failed pass with a receiver, a successful pass without a
        receiver, and a successful pass with two duplicate receiver
        rows.

    Expected:
        Only the final game receives one synthetic row. It uses the
        first non-null receiver-frame tracking state and preserves all
        originals.

    Guards against:
        Extending failed passes, crossing game ends, or multiplying
        synthetic rows when event joins duplicate a receiver frame.
    """
    frames = pl.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 3],
            "framenum": [1, 2, 1, 2, 1, 2, 2],
            "match_team_possession_id": [
                "team_1",
                "team_1",
                "team_2",
                None,
                "team_3",
                "team_3",
                "team_3",
            ],
            "match_team_player_possession_id": [
                "failed",
                "receiver_1",
                "no_receiver",
                None,
                "passer_3",
                "receiver_3",
                "receiver_3",
            ],
            "dev_match_team_possession_id": [
                "team_1",
                "team_1",
                "team_2",
                "team_2",
                "team_3",
                "team_3",
                "team_3",
            ],
            "dev_match_team_player_possession_id": [
                "failed",
                "receiver_1",
                "no_receiver",
                "no_receiver",
                "passer_3",
                "receiver_3",
                "receiver_3",
            ],
            "successful_pass_or_cross": [
                False,
                None,
                True,
                None,
                True,
                True,
                True,
            ],
            "balls_smooth": [
                {"x": 1.0},
                {"x": 2.0},
                {"x": 3.0},
                {"x": 4.0},
                {"x": 5.0},
                None,
                {"x": 6.0},
            ],
            "away_players_smooth": [
                [1],
                [2],
                [3],
                [4],
                [5],
                None,
                [6],
            ],
            "home_players_smooth": [
                [11],
                [12],
                [13],
                [14],
                [15],
                None,
                [16],
            ],
        },
        schema_overrides={
            "balls_smooth": pl.Struct({"x": pl.Float64}),
            "away_players_smooth": pl.List(pl.Int64),
            "home_players_smooth": pl.List(pl.Int64),
        },
    ).lazy()

    result = create_synthetic_final_pass_frame(frames).collect()
    synthetic = result.filter("is_synthetic_pass_end")

    assert result.filter(~pl.col("is_synthetic_pass_end")).height == 7
    assert synthetic.height == 1
    assert synthetic["game_id"].item() == 3
    assert synthetic["framenum"].item() == 2
    assert (
        synthetic["dev_match_team_player_possession_id"].item() == "passer_3"
    )
    assert synthetic["balls_smooth"].item() == {"x": 6.0}
    assert synthetic["away_players_smooth"].to_list() == [[6]]
    assert synthetic["home_players_smooth"].to_list() == [[16]]


def test_transform_possessions_adds_synthetic_pass_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run synthetic endpoint creation in the main transform pipeline.

    Behavior:
        Invoke bounded filling, delivery propagation, and synthetic-row
        creation through ``transform_possessions``, then calculate ball
        metrics using the explicitly supplied frame rate.

    Input:
        A successful passer immediately followed by a receiver, with the
        minimal required possession and tracking columns.

    Expected:
        Both source rows remain and one marked synthetic passer row is
        inserted at the receiver frame using its ball coordinates. The
        synthetic row receives the pass displacement and speed.

    Guards against:
        Implementing the helper without connecting it to production flow
        or invoking it before development possession IDs are available.
    """
    ball_dtype = pl.Struct(
        {
            "visibility": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        }
    )
    frames = pl.DataFrame(
        {
            "game_id": [1, 1],
            "framenum": [100, 101],
            "match_team_possession_id": ["team_1", "team_1"],
            "match_team_player_possession_id": ["passer", "receiver"],
            "possession_event_type": ["PA", "IT"],
            "successful_pass_or_cross": [True, None],
            "balls_smooth": [
                {
                    "visibility": "visible",
                    "x": 10.0,
                    "y": 5.0,
                    "z": 0.0,
                },
                {
                    "visibility": "visible",
                    "x": 25.0,
                    "y": 8.0,
                    "z": 0.0,
                },
            ],
            "away_players_smooth": [[1], [2]],
            "home_players_smooth": [[11], [12]],
        },
        schema_overrides={
            "balls_smooth": ball_dtype,
            "away_players_smooth": pl.List(pl.Int64),
            "home_players_smooth": pl.List(pl.Int64),
        },
    ).lazy()

    monkeypatch.setattr(
        transform_module,
        "append_attacking_direction",
        lambda frame: frame.with_columns(
            pl.lit("R").alias("attacking_team_direction")
        ),
    )

    result = transform_possessions(
        frames,
        frame_rate=29.97,
    ).collect()
    synthetic = result.filter("is_synthetic_pass_end")

    assert result.height == 3
    assert result["is_synthetic_pass_end"].to_list() == [
        False,
        True,
        False,
    ]
    assert synthetic["framenum"].item() == 101
    assert synthetic["dev_match_team_player_possession_id"].item() == "passer"
    assert synthetic["balls_smooth"].item() == {
        "visibility": "visible",
        "x": 25.0,
        "y": 8.0,
        "z": 0.0,
        "is_imputed": False,
    }
    assert synthetic["ball_displacement"].item() == pytest.approx(
        15.297058540778355
    )
    assert synthetic["ball_speed"].item() == pytest.approx(458.4528444671273)


def test_flip_if_attacking_left_handles_direction_categories() -> None:
    """Flip one coordinate only for a left-attacking team.

    Behavior:
        Negate a selected struct field when direction is ``L`` and leave
        it unchanged for right, null, and unexpected directions.

    Input:
        Four ball structs paired with ``L``, ``R``, null, and
        ``UNKNOWN`` direction values, including one null coordinate.

    Expected:
        Only the left-attacking coordinate changes sign. Null remains
        null and the result retains ``Float64`` dtype.

    Guards against:
        Flipping every direction, treating unknown direction as left,
        filling null coordinates, or changing numeric dtype.
    """
    ball_dtype = pl.Struct(
        {
            "visibility": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        }
    )
    frames = pl.DataFrame(
        {
            "attacking_team_direction": ["L", "R", None, "UNKNOWN"],
            "balls_smooth": [
                {"visibility": "visible", "x": 10.0, "y": 1.0, "z": 0.0},
                {"visibility": "visible", "x": -3.0, "y": 2.0, "z": 0.0},
                {"visibility": "visible", "x": None, "y": 3.0, "z": 0.0},
                {"visibility": "visible", "x": 4.0, "y": 4.0, "z": 0.0},
            ],
        },
        schema_overrides={"balls_smooth": ball_dtype},
    ).lazy()

    result = frames.with_columns(
        _flip_if_attacking_left("balls_smooth", "x")
    ).collect()

    assert result["x"].to_list() == [-10.0, -3.0, None, 4.0]
    assert result.schema["x"] == pl.Float64


def test_normalize_ball_coordinates_flips_xy_and_preserves_fields() -> None:
    """Normalize left-attacking ball coordinates as one typed struct.

    Behavior:
        Negate both planar coordinates for ``L`` while retaining
        right-attacking coordinates, visibility, height, and unrelated
        columns.

    Input:
        One left-attacking and one right-attacking ball record with
        distinct coordinates, visibility values, heights, and row IDs.

    Expected:
        The left row has negated x and y; the right row is unchanged.
        Struct field order and dtypes match the input schema.

    Guards against:
        Flipping z, losing visibility, dropping unrelated columns,
        changing schema, or normalizing right-attacking coordinates.
    """
    ball_dtype = pl.Struct(
        {
            "visibility": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        }
    )
    frames = pl.DataFrame(
        {
            "row_id": [1, 2],
            "attacking_team_direction": ["L", "R"],
            "balls_smooth": [
                {
                    "visibility": "visible",
                    "x": 12.5,
                    "y": -7.0,
                    "z": 1.5,
                },
                {
                    "visibility": "occluded",
                    "x": -4.0,
                    "y": 8.5,
                    "z": 0.25,
                },
            ],
        },
        schema_overrides={"balls_smooth": ball_dtype},
    ).lazy()

    result = normalize_ball_coordinates(frames).collect()

    assert result["row_id"].to_list() == [1, 2]
    assert result["balls_smooth"].to_list() == [
        {
            "visibility": "visible",
            "x": -12.5,
            "y": 7.0,
            "z": 1.5,
        },
        {
            "visibility": "occluded",
            "x": -4.0,
            "y": 8.5,
            "z": 0.25,
        },
    ]
    assert result.schema["balls_smooth"] == ball_dtype


def test_normalize_ball_coordinates_handles_null_ball_values() -> None:
    """Retain null coordinate fields without raising arithmetic errors.

    Behavior:
        Reconstruct null and partially null ball records safely while an
        unknown attacking direction leaves available coordinates intact.

    Input:
        A null ball struct with direction ``L`` and a partially
        populated struct with an unexpected direction category.

    Expected:
        The null record contains four null fields. The partial record
        remains unchanged, and the ball struct dtype stays stable.

    Guards against:
        Exceptions during null negation, implicit value imputation,
        treating unknown categories as left, or schema degradation.
    """
    ball_dtype = pl.Struct(
        {
            "visibility": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        }
    )
    frames = pl.DataFrame(
        {
            "attacking_team_direction": ["L", "UNKNOWN"],
            "balls_smooth": [
                None,
                {
                    "visibility": None,
                    "x": None,
                    "y": 2.0,
                    "z": None,
                },
            ],
        },
        schema_overrides={"balls_smooth": ball_dtype},
    ).lazy()

    result = normalize_ball_coordinates(frames).collect()

    assert result["balls_smooth"].to_list() == [
        {
            "visibility": None,
            "x": None,
            "y": None,
            "z": None,
        },
        {
            "visibility": None,
            "x": None,
            "y": 2.0,
            "z": None,
        },
    ]
    assert result.schema["balls_smooth"] == ball_dtype


def test_label_pitch_thirds_assigns_boundaries_explicitly() -> None:
    """Assign normalized x coordinates to deterministic pitch thirds.

    Behavior:
        Label coordinates through ``-17.5`` defensive, coordinates above
        ``-17.5`` through ``17.5`` middle, and larger values attacking.

    Input:
        Seven ball records covering both exact boundaries, values
        directly beside each boundary, the center, and both endpoints.

    Expected:
        Labels follow the inclusive-boundary convention. Input row IDs
        and ball structs remain unchanged, and output is string.

    Guards against:
        Boundary gaps or overlaps, reversed thirds, source mutation, row
        loss, and unstable output dtype.
    """
    ball_dtype = pl.Struct({"x": pl.Float64})
    coordinates = [-52.5, -17.5, -17.49, 0.0, 17.5, 17.51, 52.5]
    frames = pl.DataFrame(
        {
            "row_id": list(range(1, 8)),
            "balls_smooth": [{"x": value} for value in coordinates],
        },
        schema_overrides={"balls_smooth": ball_dtype},
    ).lazy()

    result = label_pitch_thirds(frames).collect()

    assert result.height == 7
    assert result["row_id"].to_list() == list(range(1, 8))
    assert result["balls_smooth"].struct.field("x").to_list() == coordinates
    assert result["pitch_third"].to_list() == [
        "Defensive",
        "Defensive",
        "Middle",
        "Middle",
        "Middle",
        "Attacking",
        "Attacking",
    ]
    assert result.schema["pitch_third"] == pl.Enum(
        ["Defensive", "Middle", "Attacking"]
    )


def test_label_pitch_thirds_rejects_invalid_coordinates() -> None:
    """Leave missing, non-finite, and out-of-pitch values unlabeled.

    Behavior:
        Return null unless x is finite and inside the centered pitch.

    Input:
        A null struct, a null x, coordinates outside both pitch ends,
        NaN, and positive and negative infinity.

    Expected:
        Every invalid coordinate remains unlabeled and output uses the
        configured pitch-third enum dtype.

    Guards against:
        Mislabeling invalid tracking as attacking, raising on null
        structs, or accepting impossible pitch coordinates.
    """
    ball_dtype = pl.Struct({"x": pl.Float64})
    frames = pl.DataFrame(
        {
            "balls_smooth": [
                None,
                {"x": None},
                {"x": -60.0},
                {"x": 60.0},
                {"x": float("nan")},
                {"x": float("inf")},
                {"x": float("-inf")},
            ]
        },
        schema_overrides={"balls_smooth": ball_dtype},
    ).lazy()

    result = label_pitch_thirds(frames).collect()

    assert result["pitch_third"].to_list() == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert result.schema["pitch_third"] == pl.Enum(
        ["Defensive", "Middle", "Attacking"]
    )


def test_label_pitch_thirds_returns_typed_empty_output() -> None:
    """Preserve schema when labeling an empty possession frame.

    Behavior:
        Add the derived pitch-third column without requiring any rows.

    Input:
        An empty LazyFrame with a typed ball-coordinate struct.

    Expected:
        Output remains empty and includes an enum ``pitch_third``
        column.

    Guards against:
        Schema inference failures or accidental collection assumptions
        on empty partitions.
    """
    frames = pl.DataFrame(
        schema={"balls_smooth": pl.Struct({"x": pl.Float64})}
    ).lazy()

    result = label_pitch_thirds(frames).collect()

    assert result.height == 0
    assert result.schema == {
        "balls_smooth": pl.Struct({"x": pl.Float64}),
        "pitch_third": pl.Enum(["Defensive", "Middle", "Attacking"]),
    }
