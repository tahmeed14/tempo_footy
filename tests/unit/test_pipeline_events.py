"""Test staged event inputs at the executable pipeline boundary."""

import importlib.util
import json
import logging
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tempoctrl.gradient_sports import event_load
from tempoctrl.gradient_sports import join as join_module


@pytest.fixture
def pipeline_events() -> ModuleType:
    """Load the event pipeline without executing its entry point.

    Behavior:
        Expose pipeline functions for isolated orchestration tests.

    Input:
        The repository's event pipeline script, without event data.

    Expected:
        An imported module without processing files or writing outputs.

    Guards against:
        Tests triggering a complete production run during import.
    """
    pipeline_path = (
        Path(__file__).parents[2] / "pipelines" / "pipeline_events.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipeline_events_under_test", pipeline_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load event pipeline: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("custom_source", [False, True])
@pytest.mark.parametrize("overwrite_stage", [False, True])
def test_run_pipeline_preserves_processing_and_controls_restaging(
    pipeline_events: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    custom_source: bool,
    overwrite_stage: bool,
) -> None:
    """Use staged events while retaining the eager processing contract.

    Behavior:
        Stage and scan before transform, features, cleanup, and writing.
        Forward the restaging override.

    Input:
        Two raw events with markers 20 and 10, then replacement marker
        99 with an older source timestamp. The source is either the
        default path or a custom JSON file. Processing steps and the
        writer are mocked; ingestion uses real temporary files.

    Expected:
        No joins occur. Both initial rows reach the eager transforms
        with UInt32 event numbers 1 and 2, including their source order.
        No unmatched filtering occurs. Replacement marker 99 appears
        only when restaging is forced. Processing steps run in order.
        The original two writer paths are returned and logged.

    Guards against:
        Passing a LazyFrame downstream, parsing JSON on cache reuse,
        ignoring restaging overrides, or changing output return values.
    """
    monkeypatch.chdir(tmp_path)
    raw_path = (
        tmp_path / "custom.json"
        if custom_source
        else tmp_path / "data/raw/gradient_sports/events/101.json"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        '[{"marker": 20, "gameId": 101}, {"marker": 10, "gameId": 101}]',
        encoding="utf-8",
    )
    event_path = str(raw_path) if custom_source else None
    output_paths = (
        Path("data/processed/gradient_sports/events/101.parquet"),
        Path("data/curated/gradient_sports/lookup.parquet"),
    )
    steps = Mock()
    steps.transform.side_effect = lambda df: df.with_columns(
        pl.lit(True).alias("transformed")
    )
    steps.features.side_effect = lambda df: df.with_columns(
        pl.lit(True).alias("featured")
    )
    steps.cleanup.side_effect = lambda df: df.with_columns(
        pl.lit(True).alias("cleaned")
    )
    steps.load.return_value = output_paths
    monkeypatch.setattr(pipeline_events, "transform_events", steps.transform)
    monkeypatch.setattr(pipeline_events, "features_events", steps.features)
    monkeypatch.setattr(pipeline_events, "cleanup_events", steps.cleanup)
    monkeypatch.setattr(pipeline_events, "load_events", steps.load)

    with caplog.at_level(logging.INFO, logger=pipeline_events.logger.name):
        assert pipeline_events.run_pipeline(101, event_path) == output_paths

    assert [entry[0] for entry in steps.mock_calls] == [
        "transform", "features", "cleanup", "load",
    ]
    assert_frame_equal(
        steps.transform.call_args.args[0],
        pl.DataFrame(
            {
                "event_number": pl.Series([1, 2], dtype=pl.UInt32),
                "marker": [20, 10],
                "gameId": [101, 101],
            }
        ),
    )
    assert_frame_equal(
        steps.load.call_args.args[0],
        pl.DataFrame(
            {
                "event_number": pl.Series([1, 2], dtype=pl.UInt32),
                "marker": [20, 10],
                "gameId": [101, 101],
                "transformed": [True, True],
                "featured": [True, True],
                "cleaned": [True, True],
            }
        ),
    )
    staged_path = Path("data/staged/gradient_sports/events/101.parquet")
    assert staged_path.is_file()
    assert "event_number" not in pl.read_parquet(staged_path).columns
    assert f"Loading staged event file: {staged_path}" in caplog.text
    for output_path in output_paths:
        assert f"Event pipeline output: {output_path}" in caplog.text

    raw_path.write_text(
        '[{"marker": 99, "gameId": 101}]', encoding="utf-8"
    )
    os.utime(raw_path, (1000, 1000))
    os.utime(staged_path, (2000, 2000))
    steps.reset_mock()

    assert pipeline_events.run_pipeline(
        101, event_path, overwrite_stage=overwrite_stage
    ) == output_paths

    expected_markers = [99] if overwrite_stage else [20, 10]
    assert steps.transform.call_args.args[0]["marker"].to_list() == (
        expected_markers
    )


@pytest.fixture
def raw_event_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Provide complete raw event JSON for two selected matches.

    Behavior:
        Isolate staging and output writes from repository data.

    Input:
        Three events per match: kickoff, completed pass, and END. Both
        matches reuse player and team IDs. Only match 101 scores.

    Expected:
        Two default raw paths, with all nested transform fields present
        and null player/team metadata on each END row.

    Guards against:
        Tests depending on local datasets or mocking away batch joins,
        score isolation, event numbering, or output writing.
    """
    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "data/raw/gradient_sports/events"
    raw_dir.mkdir(parents=True)
    paths = []
    for game_id in (101, 202):
        rows = []
        for index, event_type in enumerate(("FIRSTKICKOFF", "PASS", "END")):
            is_end = event_type == "END"
            rows.append(
                {
                    "gameId": game_id,
                    "gameEventId": index + 1,
                    "possessionEventId": index + 1,
                    "duration": 1.0,
                    "initialTouch": {
                        "initialPressureType": "L",
                        "initialHeightType": "G",
                        "initialBodyType": "RF",
                    },
                    "gameEvents": {
                        "gameEventType": event_type,
                        "playerId": None if is_end else 11,
                        "playerName": None if is_end else "Player",
                        "teamId": None if is_end else 1,
                        "teamName": None if is_end else "Shared",
                        "homeTeam": None if is_end else True,
                        "setpieceType": "K" if index == 0 else "O",
                        "outType": (
                            "H" if game_id == 101 and index == 1 else None
                        ),
                        "endType": "G" if is_end else None,
                        "period": 1,
                        "touches": 1,
                    },
                    "possessionEvents": {
                        "possessionEventType": ("IT", "PA", None)[index],
                        "formattedGameClock": f"00:0{index}",
                        "linesBrokenType": "A",
                        "pressureType": "L",
                        "crossOutcomeType": None,
                        "passOutcomeType": "C" if index == 1 else None,
                    },
                    "stadiumMetadata": {"teamAttackingDirection": "R"},
                }
            )
        path = raw_dir / f"{game_id}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


@pytest.mark.parametrize("overwrite_stage", [False, True])
def test_run_pipeline_processes_selected_matches_once(
    pipeline_events: ModuleType,
    raw_event_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    overwrite_stage: bool,
) -> None:
    """Stage selected sources and execute processing once per batch.

    Starting datasets:
        Two three-event raw JSON files selected in reversed match order,
        with explicit mixed string/Path source arguments. Processing and
        writing are mocked; staging and scanning use real files.

    Expected:
        Two staging calls receive the overwrite flag. One combined scan
        sends all six rows to transform, features, cleanup, and write,
        each called once in order. Event numbers restart within each
        game. No joins or unmatched-row filtering occur. The three
        writer paths are returned unchanged.

    Guards against:
        Retaining a per-match processing loop, global numbering, ignored
        custom sources, or forwarding overwrite to only one match.
    """
    steps = Mock()
    steps.transform.side_effect = lambda frame: frame
    steps.features.side_effect = lambda frame: frame
    steps.cleanup.side_effect = lambda frame: frame
    output_paths = (
        Path("101.parquet"), Path("202.parquet"), Path("lookup.parquet")
    )
    steps.load.return_value = output_paths
    monkeypatch.setattr(pipeline_events, "transform_events", steps.transform)
    monkeypatch.setattr(pipeline_events, "features_events", steps.features)
    monkeypatch.setattr(pipeline_events, "cleanup_events", steps.cleanup)
    monkeypatch.setattr(pipeline_events, "load_events", steps.load)

    with (
        patch.object(
            pipeline_events, "stage_events", wraps=pipeline_events.stage_events
        ) as stage,
        patch.object(
            pipeline_events, "scan_events", wraps=pipeline_events.scan_events
        ) as scan,
    ):
        result = pipeline_events.run_pipeline(
            [202, 101],
            [str(raw_event_sources[1]), raw_event_sources[0]],
            overwrite_stage=overwrite_stage,
        )

    assert result == output_paths
    assert stage.call_count == 2
    assert [call.args[0] for call in stage.call_args_list] == [202, 101]
    assert all(
        call.kwargs["overwrite"] is overwrite_stage
        for call in stage.call_args_list
    )
    assert scan.call_count == 1
    assert scan.call_args.args[0] == [
        Path("data/staged/gradient_sports/events/202.parquet"),
        Path("data/staged/gradient_sports/events/101.parquet"),
    ]
    assert [call[0] for call in steps.mock_calls] == [
        "transform", "features", "cleanup", "load",
    ]
    assert steps.transform.call_args.args[0].select(
        "gameId", "event_number"
    ).rows() == [(202, 1), (202, 2), (202, 3), (101, 1), (101, 2), (101, 3)]


@pytest.mark.parametrize("selection", [None, [202, 101], 101])
def test_event_pipeline_outputs_are_ready_for_downstream_joins(
    pipeline_events: ModuleType,
    raw_event_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    selection: list[int] | int | None,
) -> None:
    """Run the real pipeline and consume its outputs downstream.

    Starting datasets:
        Two complete raw matches, no optional player metadata, and no
        prior staged or processed data. Select all discovered matches,
        an explicit reversed subset, or a single match. Synthetic
        tracking has two matching keys and one unmatched key per match.

    Expected:
        All selected sources are staged. Processing and lookup writing
        each run once. Every selected match has a three-row, 31-column
        event file with numbers 1, 2, 3 and isolated scores. The shared
        lookup has one six-column identity row per match. The existing
        downstream many-to-one join keeps all three tracking rows and
        leaves unmatched event fields null, without multiplying rows.

    Guards against:
        Incorrect discovery, changed default destinations or schemas,
        repeated lookup writes, cross-match scores, or output files that
        downstream tracking/event integration cannot consume.
    """
    with (
        patch.object(
            pipeline_events,
            "transform_events",
            wraps=pipeline_events.transform_events,
        ) as transform,
        patch.object(
            pipeline_events,
            "features_events",
            wraps=pipeline_events.features_events,
        ) as features,
        patch.object(
            pipeline_events,
            "cleanup_events",
            wraps=pipeline_events.cleanup_events,
        ) as cleanup,
        patch.object(
            event_load,
            "write_match_possession_lookup",
            wraps=event_load.write_match_possession_lookup,
        ) as lookup_writer,
    ):
        output_paths = pipeline_events.run_pipeline(selection)

    assert transform.call_count == 1
    assert features.call_count == 1
    assert cleanup.call_count == 1
    assert lookup_writer.call_count == 1
    selected_ids = [101] if selection == 101 else [101, 202]
    expected_event_paths = tuple(
        Path(f"data/processed/gradient_sports/events/{game_id}.parquet")
        for game_id in selected_ids
    )
    expected_lookup_path = Path(
        "data/curated/gradient_sports/possession_lookup/"
        "match_possession_lookup.parquet"
    )
    assert output_paths == (*expected_event_paths, expected_lookup_path)
    lookup = pl.read_parquet(expected_lookup_path)
    assert lookup.shape == (len(selected_ids), 6)
    assert lookup["match_team_player_possession_id"].to_list() == [
        f"{game_id}_Shared_11_1" for game_id in selected_ids
    ]
    for game_id, path in zip(selected_ids, expected_event_paths, strict=True):
        staged_path = Path(
            f"data/staged/gradient_sports/events/{game_id}.parquet"
        )
        assert staged_path.is_file()
        output = pl.read_parquet(path)
        assert output.shape == (3, 31)
        assert output["game_id"].to_list() == [game_id] * 3
        assert output["event_number"].to_list() == [1, 2, 3]
        assert output["home_score"].to_list() == (
            [0, 1, 1] if game_id == 101 else [0, 0, 0]
        )
        assert output["away_score"].to_list() == [0, 0, 0]
        tracking = pl.DataFrame(
            {
                "game_id": [game_id] * 3,
                "game_event_id": [1, 2, 999],
                "possession_event_id": [1, 2, 999],
                "framenum": [10, 20, 30],
            }
        ).lazy()
        monkeypatch.setattr(join_module, "scan_tracking", lambda _: tracking)

        joined = join_module.possession_join(game_id).collect()

        assert joined.select(
            "framenum", "event_number", "player_id"
        ).rows() == [(10, 1, 11), (20, 2, 11), (30, None, None)]
    if selection == 101:
        assert not Path(
            "data/staged/gradient_sports/events/202.parquet"
        ).exists()
        assert not Path(
            "data/processed/gradient_sports/events/202.parquet"
        ).exists()


@pytest.mark.parametrize(
    ("match_id", "event_path", "message"),
    [
        ([], None, "At least one match ID"),
        ([101, 101], None, "must be unique"),
        ([101, 202], "custom.json", "counts must match"),
        (101, ["first.json", "second.json"], "counts must match"),
        ([101, 202], [], "counts must match"),
        (None, "custom.json", "requires explicit match IDs"),
    ],
)
def test_run_pipeline_rejects_invalid_selection_before_staging(
    pipeline_events: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    match_id: int | list[int] | None,
    event_path: str | list[str] | None,
    message: str,
) -> None:
    """Reject invalid selections before staging or processing.

    Behavior:
        Validate selected match IDs and corresponding source paths.

    Input:
        Empty or duplicate match IDs, unequal ID/path counts, or custom
        paths without explicit match IDs. No DataFrame is supplied.

    Expected:
        A descriptive ValueError and no staging or scan calls.

    Guards against:
        Duplicate cached inputs, silently dropped matches, or partially
        staging an invalid request before discovering its error.
    """
    stage = Mock()
    scan = Mock()
    monkeypatch.setattr(pipeline_events, "stage_events", stage)
    monkeypatch.setattr(pipeline_events, "scan_events", scan)

    with pytest.raises(ValueError, match=message):
        pipeline_events.run_pipeline(match_id, event_path)

    stage.assert_not_called()
    scan.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_main_runs_one_batch_and_logs_runtime_on_failure(
    pipeline_events: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fails: bool,
) -> None:
    """Run one discovered batch and retain timing when processing fails.

    Behavior:
        Invoke the batch pipeline once from the executable entry point.

    Input:
        A mocked batch run, successful or raising RuntimeError, and
        monotonic clock readings 65 seconds apart. No data is written.

    Expected:
        One run_pipeline call with default discovery. Both cases log
        1 minute, 5 seconds; the failure propagates to the caller.

    Guards against:
        Keeping the old per-match entry-point loop, losing runtime logs,
        or swallowing processing errors.
    """
    run = Mock(side_effect=RuntimeError("batch failed") if fails else None)
    clock_readings = iter((10.0, 75.0))
    monkeypatch.setattr(pipeline_events, "run_pipeline", run)
    monkeypatch.setattr(
        pipeline_events, "perf_counter", lambda: next(clock_readings)
    )

    with caplog.at_level(logging.INFO, logger=pipeline_events.logger.name):
        if fails:
            with pytest.raises(RuntimeError, match="batch failed"):
                pipeline_events.main()
        else:
            pipeline_events.main()

    run.assert_called_once_with()
    assert (
        "Event pipeline runtime: 0 hours, 1 minute, 5.000 seconds"
    ) in caplog.text


@pytest.mark.parametrize(
    "mismatch", ["numeric_filename", "wrong_content", "mixed_ids", "null_id"]
)
def test_pipeline_rejects_match_identity_mismatches_before_processing(
    pipeline_events: ModuleType,
    raw_event_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    """Stop mismatched inputs before publishing stages or event outputs.

    Starting datasets:
        Two valid raw matches. Request match 101 with source 202.json,
        or change the 101.json contents to identify only 202, include
        both matches, or contain a null gameId row.

    Expected:
        Staging raises ValueError before transforms or the writer run.
        No joins occur and no event rows are published. No staged,
        processed, or curated output directories are created.

    Guards against:
        Producing another match's files, accepting mixed IDs in one
        source, or creating mislabeled cached stages.
    """
    transform = Mock()
    writer = Mock()
    monkeypatch.setattr(pipeline_events, "transform_events", transform)
    monkeypatch.setattr(pipeline_events, "load_events", writer)
    if mismatch != "numeric_filename":
        rows = json.loads(raw_event_sources[0].read_text(encoding="utf-8"))
        if mismatch == "wrong_content":
            for row in rows:
                row["gameId"] = 202
        else:
            rows[-1]["gameId"] = 202 if mismatch == "mixed_ids" else None
        raw_event_sources[0].write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="match ID mismatch"):
        if mismatch == "numeric_filename":
            pipeline_events.run_pipeline(101, raw_event_sources[1])
        else:
            pipeline_events.run_pipeline()

    transform.assert_not_called()
    writer.assert_not_called()
    assert not Path("data/staged").exists()
    assert not Path("data/processed").exists()
    assert not Path("data/curated").exists()


def test_pipeline_repairs_cached_match_identity_before_processing(
    pipeline_events: ModuleType,
    raw_event_sources: tuple[Path, Path],
) -> None:
    """Repair a mislabeled cache before running the real event pipeline.

    Starting datasets:
        Valid raw matches 101 and 202, but a newer staged 101.parquet
        containing all three rows from match 202. Select only match 101.

    Expected:
        The pipeline restages valid raw 101 automatically. Its staged
        file has only gameId 101, and its processed file has three rows
        and 31 columns identifying game_id 101. The shared lookup has
        one match-101 identity row. No joins or unmatched-row filtering
        occur and no match-202 event file is written.

    Guards against:
        Reusing polluted cache identities or requiring manual overwrite
        before valid sources can produce the correct downstream files.
    """
    staged_path = Path("data/staged/gradient_sports/events/101.parquet")
    staged_path.parent.mkdir(parents=True)
    pl.read_json(raw_event_sources[1], infer_schema_length=None).write_parquet(
        staged_path
    )
    os.utime(raw_event_sources[0], (1000, 1000))
    os.utime(staged_path, (2000, 2000))

    paths = pipeline_events.run_pipeline(101)

    assert paths[0] == Path(
        "data/processed/gradient_sports/events/101.parquet"
    )
    assert pl.read_parquet(staged_path)["gameId"].to_list() == [101, 101, 101]
    output = pl.read_parquet(paths[0])
    assert output.shape == (3, 31)
    assert output["game_id"].to_list() == [101, 101, 101]
    lookup = pl.read_parquet(paths[1])
    assert lookup.shape == (1, 6)
    assert lookup["game_id"].item() == 101
    assert not Path(
        "data/processed/gradient_sports/events/202.parquet"
    ).exists()
