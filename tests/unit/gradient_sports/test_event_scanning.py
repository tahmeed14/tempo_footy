"""Test lazy scanning of selected staged event files."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports.ingest import scan_events


@pytest.mark.parametrize(
    "input_form",
    ["list", "tuple", "directory", "wildcard", "combined", "combined_list"],
)
@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.String])
def test_scan_events_numbers_matches_consistently_across_input_forms(
    tmp_path: Path,
    input_form: str,
    id_dtype: pl.DataType,
) -> None:
    """Number matches consistently regardless of input packaging.

    Behavior:
        Restart numbering within each gameId for every supported input
        form, preserving row order and numbering before filters.

    Input:
        Four interleaved events from matches 202 and 101, with integer
        or string IDs. Supply two files through a list, tuple,
        directory, or wildcard, or one combined file directly or in a
        list.

    Expected:
        Four unchanged rows with leading UInt32 numbers 1, 1, 2, 2.
        Filtering to the last two markers retains numbers 2, 2.
        No joins occur and no unmatched rows are removed.

    Guards against:
        Global numbering for directory, wildcard, or combined-file
        scans, row reordering, dtype changes, or filter renumbering.
    """
    source = pl.DataFrame(
        {"gameId": [202, 101, 202, 101], "marker": [40, 30, 20, 10]}
    ).with_columns(pl.col("gameId").cast(id_dtype))
    selection: str | Path | list[Path] | tuple[Path, ...]
    if input_form in ("combined", "combined_list"):
        combined = tmp_path / "combined.parquet"
        source.write_parquet(combined)
        selection = combined if input_form == "combined" else [combined]
    else:
        first = tmp_path / "a.parquet"
        second = tmp_path / "b.parquet"
        source.head(2).write_parquet(first)
        source.tail(2).write_parquet(second)
        if input_form == "list":
            selection = [first, second]
        elif input_form == "tuple":
            selection = (first, second)
        elif input_form == "directory":
            selection = tmp_path
        else:
            selection = str(tmp_path / "*.parquet")

    result = scan_events(selection)
    expected = pl.DataFrame(
        {
            "event_number": pl.Series([1, 1, 2, 2], dtype=pl.UInt32),
            "gameId": pl.Series([202, 101, 202, 101]).cast(id_dtype),
            "marker": [40, 30, 20, 10],
        }
    )
    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(result.collect(), expected)
    assert_frame_equal(
        result.filter(pl.col("marker") <= 20).collect(), expected.tail(2)
    )


def test_scan_events_preserves_empty_single_match_schema(
    tmp_path: Path,
) -> None:
    """Keep typed empty single-file scans empty after window numbering.

    Behavior:
        Apply per-match numbering to an empty file with a gameId column.

    Input:
        One empty Parquet with String gameId and Int64 marker columns.

    Expected:
        Zero rows with a leading UInt32 event_number column and the
        original source schema.

    Guards against:
        Fabricated rows or schema changes in the new single-file window.
    """
    path = tmp_path / "empty.parquet"
    source = pl.DataFrame(schema={"gameId": pl.String, "marker": pl.Int64})
    source.write_parquet(path)
    expected = pl.DataFrame(
        schema={
            "event_number": pl.UInt32,
            "gameId": pl.String,
            "marker": pl.Int64,
        }
    )

    assert_frame_equal(scan_events(path).collect(), expected)


@pytest.mark.parametrize("use_tuple", [False, True])
def test_scan_events_numbers_matches_across_selected_files(
    tmp_path: Path,
    use_tuple: bool,
) -> None:
    """Number events per match without changing source row order.

    Behavior:
        Scan mixed string/Path selections lazily and number each game
        before downstream filters or projections.

    Input:
        Two Parquet files with six unsorted events from three matches.
        Matches 101 and 202 span both files. The selected filenames are
        intentionally in reverse lexical order.

    Expected:
        Six rows in supplied file and source row order, with event
        numbers 1, 1, 2, 2, 1, 3 and the original game IDs and markers.
        Filtering markers at most 3 retains numbers 2, 1, 3. No joins
        occur, and no unmatched rows are removed.

    Guards against:
        Global numbering, numbering by file, sorting selected paths,
        losing source order, or renumbering after filters.
    """
    first_path = tmp_path / "z.parquet"
    second_path = tmp_path / "a.parquet"
    pl.DataFrame(
        {"gameId": ["202", "101", "202"], "marker": [9, 4, 8]}
    ).write_parquet(first_path)
    pl.DataFrame(
        {"gameId": ["101", "303", "202"], "marker": [3, 2, 1]}
    ).write_parquet(second_path)
    paths: list[str | Path] = [str(first_path), second_path]
    selection = tuple(paths) if use_tuple else paths

    result = scan_events(selection)

    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(
        result.collect(),
        pl.DataFrame(
            {
                "event_number": pl.Series(
                    [1, 1, 2, 2, 1, 3], dtype=pl.UInt32
                ),
                "gameId": ["202", "101", "202", "101", "303", "202"],
                "marker": [9, 4, 8, 3, 2, 1],
            }
        ),
    )
    assert_frame_equal(
        result.filter(pl.col("marker") <= 3)
        .select("event_number", "marker")
        .collect(),
        pl.DataFrame(
            {
                "event_number": pl.Series([2, 1, 3], dtype=pl.UInt32),
                "marker": [3, 2, 1],
            }
        ),
    )


def test_scan_events_combines_compatible_nested_schemas(
    tmp_path: Path,
) -> None:
    """Preserve nested values when matches have differing schemas.

    Behavior:
        Promote compatible numeric and nested dtypes and retain columns
        missing from one input as nulls.

    Input:
        Two events from match 101 with Int32 markers and null pressure,
        then one event from match 202 with an Int64 marker, populated
        pressure, an unknown category, and additional nested/top fields.

    Expected:
        All three source rows, per-match numbering, Int64 markers, and
        the union of struct fields. Null structs remain null; missing
        fields become null without dropping the additional values.

    Guards against:
        Schema mismatch failures, lost late fields, fabricated values,
        or row loss while combining independently inferred JSON schemas.
    """
    first_path = tmp_path / "101.parquet"
    second_path = tmp_path / "202.parquet"
    pl.DataFrame(
        {
            "gameId": ["101", "101"],
            "marker": pl.Series([30, 10], dtype=pl.Int32),
            "event": [{"type": "PASS", "pressure": None}, None],
        },
        schema_overrides={
            "event": pl.Struct({"type": pl.String, "pressure": pl.Null}),
        },
    ).write_parquet(first_path)
    pl.DataFrame(
        {
            "gameId": ["202"],
            "marker": [20],
            "event": [
                {"type": "UNKNOWN", "pressure": 3.5, "detail": "late"}
            ],
            "optional": ["present"],
        }
    ).write_parquet(second_path)

    result = scan_events([first_path, second_path]).collect()

    expected = pl.DataFrame(
        {
            "event_number": [1, 2, 1],
            "gameId": ["101", "101", "202"],
            "marker": [30, 10, 20],
            "event": [
                {"type": "PASS", "pressure": None, "detail": None},
                None,
                {"type": "UNKNOWN", "pressure": 3.5, "detail": "late"},
            ],
            "optional": [None, None, "present"],
        },
        schema={
            "event_number": pl.UInt32,
            "gameId": pl.String,
            "marker": pl.Int64,
            "event": pl.Struct(
                {
                    "type": pl.String,
                    "pressure": pl.Float64,
                    "detail": pl.String,
                }
            ),
            "optional": pl.String,
        },
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("use_tuple", [False, True])
def test_scan_events_single_file_sequence_preserves_existing_behavior(
    tmp_path: Path,
    use_tuple: bool,
) -> None:
    """Retain single-file behavior for a one-path sequence.

    Behavior:
        Treat one selected file like the existing scalar-path scan.

    Input:
        Two source rows with markers 20 and 10, without a gameId column,
        as supported by the existing single-file scanner.

    Expected:
        The same rows and schema as a scalar-path scan, with leading
        UInt32 event numbers 1 and 2.

    Guards against:
        Breaking single-file callers or requiring new metadata for them.
    """
    path = tmp_path / "events.parquet"
    pl.DataFrame({"marker": [20, 10]}).write_parquet(path)
    selection = (path,) if use_tuple else [path]
    expected = pl.DataFrame(
        {
            "event_number": pl.Series([1, 2], dtype=pl.UInt32),
            "marker": [20, 10],
        }
    )

    assert_frame_equal(scan_events(selection).collect(), expected)
    assert_frame_equal(scan_events(path).collect(), expected)


@pytest.mark.parametrize("include_rows", [False, True])
def test_scan_events_preserves_empty_inputs_in_multifile_selection(
    tmp_path: Path,
    include_rows: bool,
) -> None:
    """Handle empty files without changing other matches' numbering.

    Behavior:
        Combine typed empty scans with empty or populated event scans.

    Input:
        An empty gameId String/marker Int64 file and a second file
        containing zero or two events from match 202.

    Expected:
        The same schema in both cases. A populated second file retains
        both rows numbered 1 and 2; two empty files yield no rows.

    Guards against:
        Empty-window errors, fabricated rows, or index dtype drift.
    """
    first_path = tmp_path / "empty.parquet"
    second_path = tmp_path / "selected.parquet"
    schema = {"gameId": pl.String, "marker": pl.Int64}
    pl.DataFrame(schema=schema).write_parquet(first_path)
    second = pl.DataFrame(
        {
            "gameId": ["202", "202"] if include_rows else [],
            "marker": [20, 10] if include_rows else [],
        },
        schema=schema,
    )
    second.write_parquet(second_path)
    expected = second.insert_column(
        0,
        pl.Series(
            "event_number", [1, 2] if include_rows else [], dtype=pl.UInt32
        ),
    )

    assert_frame_equal(
        scan_events([first_path, second_path]).collect(), expected
    )


@pytest.mark.parametrize("paths", [[], ()])
def test_scan_events_rejects_empty_selection(
    paths: list[Path] | tuple[Path, ...],
) -> None:
    """Reject an empty path selection before constructing a scan.

    Behavior:
        Require at least one staged event file.

    Input:
        An empty list or tuple of paths, with no DataFrame input.

    Expected:
        ValueError explains that a staged event path is required.

    Guards against:
        Silent no-op batches or obscure concatenation errors.
    """
    with pytest.raises(ValueError, match="At least one staged event"):
        scan_events(paths)


def test_scan_events_requires_match_ids_for_multifile_numbering(
    tmp_path: Path,
) -> None:
    """Require match identity for numbering across multiple files.

    Behavior:
        Validate the schema of every file in a multi-file selection.

    Input:
        One valid gameId file and one marker-only file.

    Expected:
        ValueError identifies the missing gameId and the offending path.

    Guards against:
        Global numbering or assigning unidentified events to one group.
    """
    valid_path = tmp_path / "valid.parquet"
    invalid_path = tmp_path / "invalid.parquet"
    pl.DataFrame({"gameId": ["101"], "marker": [1]}).write_parquet(
        valid_path
    )
    pl.DataFrame({"marker": [2]}).write_parquet(invalid_path)

    with pytest.raises(ValueError, match="missing gameId") as error_info:
        scan_events([valid_path, invalid_path])

    assert str(invalid_path) in str(error_info.value)


def test_scan_events_reports_missing_selected_file(tmp_path: Path) -> None:
    """Report a missing input instead of processing a partial selection.

    Behavior:
        Require every selected staged file to exist.

    Input:
        One valid Parquet file and a nonexistent second path.

    Expected:
        FileNotFoundError occurs before any batch rows are returned.

    Guards against:
        Silently omitting requested matches during a combined scan.
    """
    valid_path = tmp_path / "valid.parquet"
    pl.DataFrame({"gameId": ["101"]}).write_parquet(valid_path)

    with pytest.raises(FileNotFoundError):
        scan_events([valid_path, tmp_path / "missing.parquet"])
