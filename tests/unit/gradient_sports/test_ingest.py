"""Test raw Gradient Sports event-file discovery."""

from pathlib import Path

import pytest

from tempoctrl.gradient_sports.ingest import discover_event_files


@pytest.mark.parametrize("use_string_path", [False, True])
def test_discover_event_files_returns_numeric_order(
    tmp_path: Path,
    use_string_path: bool,
) -> None:
    """Discover only immediate JSON paths in numeric match order.

    Behavior:
        Accept string or Path directories and return match/path tuples.

    Input:
        Files 10.json, 2.json, and 1.json, an unrelated text file, and
        a nested JSON file. Contents are deliberately not valid JSON.

    Expected:
        A tuple containing only matches 1, 2, and 10 in numeric order,
        with their original Path objects.

    Guards against:
        Lexical ordering, recursive discovery, unwanted extensions,
        or parsing file contents during discovery.
    """
    for filename in ("10.json", "2.json", "1.json", "notes.txt"):
        (tmp_path / filename).write_text("not JSON", encoding="utf-8")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "3.json").touch()
    input_dir = str(tmp_path) if use_string_path else tmp_path

    result = discover_event_files(input_dir)

    assert result == (
        (1, tmp_path / "1.json"),
        (2, tmp_path / "2.json"),
        (10, tmp_path / "10.json"),
    )


def test_discover_event_files_uses_default_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the default raw-event directory from the working folder.

    Behavior:
        Support discovery without an explicit input directory.

    Input:
        A temporary working folder with one event file at the default
        data/raw/gradient_sports/events/10517.json path.

    Expected:
        One tuple with match 10517 and its relative event-file path.

    Guards against:
        Changing the pipeline's default input location during the move.
    """
    event_path = Path("data/raw/gradient_sports/events/10517.json")
    (tmp_path / event_path.parent).mkdir(parents=True)
    (tmp_path / event_path).touch()
    monkeypatch.chdir(tmp_path)

    assert discover_event_files() == ((10517, event_path),)


@pytest.mark.parametrize("existing_file", [False, True])
def test_discover_event_files_requires_directory(
    tmp_path: Path,
    existing_file: bool,
) -> None:
    """Reject absent directories and paths pointing to regular files.

    Behavior:
        Require a directory before attempting to discover event files.

    Input:
        A nonexistent path, or a regular file at that same path.

    Expected:
        FileNotFoundError identifies the invalid raw-event directory.

    Guards against:
        Treating invalid input paths as empty successful discovery.
    """
    input_path = tmp_path / "events"
    if existing_file:
        input_path.touch()

    with pytest.raises(
        FileNotFoundError,
        match="Raw event directory does not exist:",
    ) as error_info:
        discover_event_files(input_path)

    assert str(input_path) in str(error_info.value)


@pytest.mark.parametrize("non_json_file", [False, True])
def test_discover_event_files_requires_json_files(
    tmp_path: Path,
    non_json_file: bool,
) -> None:
    """Reject directories that contain no matching raw-event files.

    Behavior:
        Report missing JSON inputs explicitly.

    Input:
        An empty directory, or one containing only a text file.

    Expected:
        FileNotFoundError names the directory with no event JSON files.

    Guards against:
        A pipeline run silently doing no work or selecting other files.
    """
    if non_json_file:
        (tmp_path / "notes.txt").touch()

    with pytest.raises(
        FileNotFoundError,
        match="No raw event JSON files found in:",
    ) as error_info:
        discover_event_files(tmp_path)

    assert str(tmp_path) in str(error_info.value)


@pytest.mark.parametrize("filename", ["final.json", "null.json"])
def test_discover_event_files_rejects_nonnumeric_match_ids(
    tmp_path: Path,
    filename: str,
) -> None:
    """Reject event filenames that cannot identify a numeric match.

    Behavior:
        Validate every discovered JSON filename before returning inputs.

    Input:
        One valid numeric event filename and one nonnumeric filename.

    Expected:
        ValueError names the invalid file and retains its parsing cause.

    Guards against:
        Silently skipping malformed names or returning partial inputs.
    """
    (tmp_path / "10517.json").touch()
    (tmp_path / filename).touch()

    with pytest.raises(
        ValueError,
        match="Raw event filenames must be numeric match IDs:",
    ) as error_info:
        discover_event_files(tmp_path)

    assert filename in str(error_info.value)
    assert isinstance(error_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("filenames", "expected_id"),
    [
        (("1.json", "01.json"), 1),
        (("10517.json", "010517.json"), 10517),
    ],
)
def test_discover_event_files_rejects_duplicate_numeric_ids(
    tmp_path: Path,
    filenames: tuple[str, str],
    expected_id: int,
) -> None:
    """Reject different filenames that resolve to one numeric match ID.

    Behavior:
        Require unique numeric IDs before returning discovered files.

    Input:
        Two empty JSON files whose stems differ only by leading zeros,
        plus one unrelated match file.

    Expected:
        ValueError identifies the duplicate ID and both source paths.

    Guards against:
        Processing a match twice or overwriting its staged output.
    """
    (tmp_path / "2.json").touch()
    for filename in filenames:
        (tmp_path / filename).touch()

    with pytest.raises(
        ValueError, match="Duplicate raw event match ID"
    ) as error_info:
        discover_event_files(tmp_path)

    message = str(error_info.value)
    assert f"match ID {expected_id}:" in message
    for filename in filenames:
        assert str(tmp_path / filename) in message


@pytest.mark.parametrize(
    "directory_name", ["202.json", "backup.json", "0101.json"]
)
def test_discover_event_files_ignores_json_named_directories(
    tmp_path: Path,
    directory_name: str,
) -> None:
    """Ignore directories even when their names match the JSON pattern.

    Behavior:
        Validate and return only regular event files.

    Input:
        Valid files 101.json and 10.json, plus a JSON-named directory
        whose stem is numeric, nonnumeric, or aliases match 101.
        The directory also contains an otherwise valid nested JSON file.

    Expected:
        Only files 10.json and 101.json are returned in numeric order.

    Guards against:
        Staging directories, false filename or duplicate-ID errors, or
        recursively including files outside the raw input directory.
    """
    (tmp_path / "101.json").touch()
    (tmp_path / "10.json").touch()
    nested_dir = tmp_path / directory_name
    nested_dir.mkdir()
    (nested_dir / "303.json").touch()

    assert discover_event_files(tmp_path) == (
        (10, tmp_path / "10.json"),
        (101, tmp_path / "101.json"),
    )


def test_discover_event_files_requires_files_when_json_directories_exist(
    tmp_path: Path,
) -> None:
    """Report missing events when only JSON-named directories exist.

    Behavior:
        Treat directories as non-inputs before checking for empty input.

    Input:
        Directories named 101.json and backup.json, with no regular
        JSON files at the discovery level.

    Expected:
        FileNotFoundError reports no raw event JSON files in the path.

    Guards against:
        Returning directories as matches or raising a misleading
        nonnumeric-filename error instead of reporting missing inputs.
    """
    (tmp_path / "101.json").mkdir()
    (tmp_path / "backup.json").mkdir()

    with pytest.raises(
        FileNotFoundError, match="No raw event JSON files found in:"
    ) as error_info:
        discover_event_files(tmp_path)

    assert str(tmp_path) in str(error_info.value)
