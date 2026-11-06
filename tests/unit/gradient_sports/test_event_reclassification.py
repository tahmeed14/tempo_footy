"""Test Null-dtype defaults for event reclassification."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports import event_transform


@pytest.mark.parametrize("source_dtype", [pl.Null, pl.String])
@pytest.mark.parametrize("row_count", [0, 2])
def test_setpiece_reclassification_accepts_null_columns(
    source_dtype: pl.DataType,
    row_count: int,
) -> None:
    """Map null set-piece columns to String N/A values.

    Behavior:
        Support Null dtype and String-typed nulls, including empty data.

    Input:
        Zero or two null set-piece values alongside UInt32 event
        numbers in unsorted order.

    Expected:
        Keep the existing columns and their order. Replace set-piece
        values with String N/A, preserving event numbers and row count.

    Guards against:
        Null-dtype replacement errors, fabricated empty rows, column
        renames, row reordering, and output dtype changes.
    """
    event_numbers = pl.Series([2, 1][:row_count], dtype=pl.UInt32)
    frame = pl.DataFrame(
        {
            "event_number": event_numbers,
            "ge_setpiecetype": pl.Series(
                [None] * row_count, dtype=source_dtype
            ),
        }
    )
    expected = pl.DataFrame(
        {
            "event_number": event_numbers,
            "ge_setpiecetype": pl.Series(
                ["N/A"] * row_count, dtype=pl.String
            ),
        }
    )

    result = event_transform.rename_classes_setpieces(frame)
    assert_frame_equal(result, expected)


def test_setpiece_reclassification_preserves_existing_mappings() -> None:
    """Retain set-piece labels when adding Null-dtype support.

    Behavior:
        Map all known codes and default unknown and null String values.

    Input:
        Ten rows containing O, D, F, G, K, P, T, C, UNKNOWN, and null.

    Expected:
        Explicit readable labels in source order, ending in two N/A
        values. Retain the ge_setpiecetype column with String dtype.

    Guards against:
        Changing existing mappings, unknown/null defaults, or schema.
    """
    frame = pl.DataFrame(
        {"ge_setpiecetype": [
            "O", "D", "F", "G", "K", "P", "T", "C", "UNKNOWN", None
        ]}
    )
    expected = pl.DataFrame(
        {"ge_setpiecetype": [
            "Open Play", "Drop Kick", "Free Kick", "Goal Kick", "Kickoff",
            "Penalty", "Throw In", "Corner", "N/A", "N/A",
        ]}
    )

    result = event_transform.rename_classes_setpieces(frame)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("function_name", "sources", "targets", "string_default"),
    [
        (
            "reclassify_pressuretype",
            ("pe_pressuretype", "it_initialpressuretype"),
            ("defender_pressure_type", "first_touch_defender_pressure_type"),
            "No Pressure",
        ),
        (
            "reclassify_linesbrokentype",
            ("pe_linesbrokentype",),
            ("lines_broken_type",),
            "None",
        ),
        (
            "reclassify_firsttouch",
            ("it_initialbodytype",),
            ("first_touch_bodypart",),
            "N/A",
        ),
    ],
)
@pytest.mark.parametrize("source_dtype", [pl.Null, pl.String])
@pytest.mark.parametrize("row_count", [0, 2])
def test_reclassification_distinguishes_null_dtype_from_null_cells(
    function_name: str,
    sources: tuple[str, ...],
    targets: tuple[str, ...],
    string_default: str,
    source_dtype: pl.DataType,
    row_count: int,
) -> None:
    """Use N/A only for Null dtype while retaining existing defaults.

    Behavior:
        Reclassify null-only columns according to their declared dtype.

    Input:
        Zero or two null rows in each source column, with either Null
        or String dtype and unsorted UInt32 event numbers.

    Expected:
        Null dtype yields String N/A values. String nulls retain each
        function's existing default. Source columns are replaced and
        event numbers, order, row counts, and empty schemas survive.

    Guards against:
        Null-dtype conversion errors, changing ordinary null handling,
        fabricating empty rows, or dropping and reordering events.
    """
    event_numbers = pl.Series(
        "event_number", [2, 1][:row_count], dtype=pl.UInt32
    )
    frame = pl.DataFrame(
        {
            "event_number": event_numbers,
            **{
                source: pl.Series([None] * row_count, dtype=source_dtype)
                for source in sources
            },
        }
    )
    label = "N/A" if source_dtype == pl.Null else string_default
    expected = pl.DataFrame(
        {
            "event_number": event_numbers,
            **{
                target: pl.Series([label] * row_count, dtype=pl.String)
                for target in targets
            },
        }
    )

    result = getattr(event_transform, function_name)(frame)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "null_source", ["pe_pressuretype", "it_initialpressuretype"]
)
def test_pressure_reclassification_checks_each_column_independently(
    null_source: str,
) -> None:
    """Keep pressure defaults independent when source dtypes differ.

    Behavior:
        Apply the Null-dtype fallback to only its own pressure column.

    Input:
        Three rows with one Null-typed pressure column and the other
        containing String values P, null, and UNKNOWN. Swap their roles.

    Expected:
        Only the Null column becomes N/A. The String column yields
        Pressured followed by two No Pressure values in original order.

    Guards against:
        One pressure column's dtype overriding the other's mapping or
        null-cell behavior.
    """
    sources = ("pe_pressuretype", "it_initialpressuretype")
    targets = ("defender_pressure_type", "first_touch_defender_pressure_type")
    frame = pl.DataFrame(
        {
            source: (
                pl.Series([None, None, None], dtype=pl.Null)
                if source == null_source
                else pl.Series(["P", None, "UNKNOWN"], dtype=pl.String)
            )
            for source in sources
        }
    )
    expected = pl.DataFrame(
        {
            target: (
                ["N/A", "N/A", "N/A"]
                if source == null_source
                else [
                    "Pressured (inc. Attempted Pressure)",
                    "No Pressure",
                    "No Pressure",
                ]
            )
            for source, target in zip(sources, targets, strict=True)
        }
    )

    result = event_transform.reclassify_pressuretype(frame)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("function_name", "sources", "targets", "codes", "labels"),
    [
        (
            "reclassify_pressuretype",
            ("pe_pressuretype", "it_initialpressuretype"),
            ("defender_pressure_type", "first_touch_defender_pressure_type"),
            ["L", "P", "A", "N", "UNKNOWN", None],
            [
                "No Pressure", "Pressured (inc. Attempted Pressure)",
                "Pressured (inc. Attempted Pressure)", "No Pressure",
                "No Pressure", "No Pressure",
            ],
        ),
        (
            "reclassify_linesbrokentype",
            ("pe_linesbrokentype",),
            ("lines_broken_type",),
            ["A", "AD", "AM", "AMD", "D", "M", "MD", "UNKNOWN", None],
            [
                "Attack", "Attack & Defense (Midfield Bypassed)",
                "Attack & Midfield", "Attack, Midfield, & Defense",
                "Defense", "Midfield", "Midfield & Defense", "None", "None",
            ],
        ),
        (
            "reclassify_firsttouch",
            ("it_initialbodytype",),
            ("first_touch_bodypart",),
            ["2H", "HE", "CH", "LK", "RF", "VM", "UNKNOWN", None],
            [
                "Hands/Arms", "Head", "Torso/Other", "Leg (not Foot)",
                "Foot", "N/A", "N/A", "N/A",
            ],
        ),
    ],
)
def test_reclassification_preserves_string_mappings(
    function_name: str,
    sources: tuple[str, ...],
    targets: tuple[str, ...],
    codes: list[str | None],
    labels: list[str],
) -> None:
    """Preserve explicit known, unknown, and null String mappings.

    Behavior:
        Limit the new fallback to Null dtype without changing code maps.

    Input:
        String pressure and lines-broken codes or representative body
        part codes, followed by an unknown code and null.

    Expected:
        Explicit labels cover each pressure and lines-broken mapping,
        each body-part group, and the existing unknown/null defaults.
        Output columns are String and retain source order and count.

    Guards against:
        Changing ordinary mappings while adding the dtype fallback.
    """
    frame = pl.DataFrame({source: codes for source in sources})
    expected = pl.DataFrame({target: labels for target in targets})

    result = getattr(event_transform, function_name)(frame)
    assert_frame_equal(result, expected)
