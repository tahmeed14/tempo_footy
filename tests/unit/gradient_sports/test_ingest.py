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
