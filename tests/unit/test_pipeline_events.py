"""Test staged event inputs at the executable pipeline boundary."""

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_frame_equal


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
        '[{"marker": 20}, {"marker": 10}]', encoding="utf-8"
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
            }
        ),
    )
    assert_frame_equal(
        steps.load.call_args.args[0],
        pl.DataFrame(
            {
                "event_number": pl.Series([1, 2], dtype=pl.UInt32),
                "marker": [20, 10],
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

    raw_path.write_text('[{"marker": 99}]', encoding="utf-8")
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
