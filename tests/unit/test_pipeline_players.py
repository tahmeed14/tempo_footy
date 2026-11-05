"""Test the player-data pipeline."""

import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest


def _load_pipeline_module() -> ModuleType:
    """Load the executable player pipeline for isolated unit tests."""
    pipeline_path = (
        Path(__file__).parents[2] / "pipelines" / "pipeline_players.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "pipeline_players_under_test",
        pipeline_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            f"Could not load player pipeline module: {pipeline_path}"
        )

    pipeline_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(pipeline_module)
    return pipeline_module


pipeline_players = _load_pipeline_module()


def test_run_pipeline_builds_writes_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Test player lookup pipeline orchestration and logging.

    Behavior:
        The pipeline builds one lookup, writes it with the requested
        overwrite setting, and logs its source, size, and destination.

    Input:
        Three rows with two unique players across two games, temporary
        source and output paths, and `overwrite=False`.

    Expected:
        The builder and writer each run once with explicit arguments;
        the output path is returned and unique player counts are logged.

    Guards against:
        A command-line pipeline that writes the wrong artifact, ignores
        overwrite configuration, or provides no useful run visibility.
    """
    roster_dir = tmp_path / "rosters"
    output_path = tmp_path / "player_lookup.parquet"
    df_lookup = pl.DataFrame(
        {
            "game_id": [101, 101, 202],
            "player_id": [1, 2, 1],
        }
    )
    build_calls: list[str | Path] = []
    write_calls: list[tuple[pl.DataFrame, str | Path, bool]] = []

    def fake_build(path: str | Path) -> pl.DataFrame:
        """Record the roster source and return the test lookup."""
        build_calls.append(path)
        return df_lookup

    def fake_write(
        df_in: pl.DataFrame,
        path: str | Path,
        *,
        overwrite: bool,
    ) -> Path:
        """Record the requested write and return its destination."""
        write_calls.append((df_in, path, overwrite))
        return Path(path)

    monkeypatch.setattr(
        pipeline_players,
        "build_player_game_lookup",
        fake_build,
    )
    monkeypatch.setattr(
        pipeline_players,
        "write_player_game_lookup",
        fake_write,
    )

    with caplog.at_level(
        logging.INFO,
        logger=pipeline_players.logger.name,
    ):
        result = pipeline_players.run_pipeline(
            roster_dir,
            output_path,
            overwrite=False,
        )

    assert result == output_path
    assert build_calls == [roster_dir]
    assert len(write_calls) == 1
    assert write_calls[0][0] is df_lookup
    assert write_calls[0][1:] == (output_path, False)
    assert f"Building player lookup from: {roster_dir}" in caplog.text
    assert (
        "Built player lookup with 2 unique players across 2 games"
    ) in caplog.text
    assert f"Player lookup pipeline output: {output_path}" in caplog.text


@pytest.mark.parametrize("fails", [False, True])
def test_main_logs_runtime_even_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fails: bool,
) -> None:
    """Log the full pipeline duration and propagate processing failures.
    """
    clock_readings = iter((10.0, 75.0))
    monkeypatch.setattr(
        pipeline_players, "perf_counter", lambda: next(clock_readings)
    )

    def fake_run_pipeline() -> None:
        """Simulate a completed or failed pipeline without writing data.
        """
        if fails:
            raise RuntimeError("pipeline failed")

    monkeypatch.setattr(pipeline_players, "run_pipeline", fake_run_pipeline)
    with caplog.at_level(logging.INFO, logger=pipeline_players.logger.name):
        if fails:
            with pytest.raises(RuntimeError, match="pipeline failed"):
                pipeline_players.main()
        else:
            pipeline_players.main()

    assert (
        "Player lookup pipeline runtime: 0 hours, 1 minute, 5.000 seconds"
    ) in caplog.text
