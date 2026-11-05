"""Build processed event datasets for configured matches."""

import logging
from pathlib import Path
from time import perf_counter

from tempoctrl.gradient_sports.event_features import features_events
from tempoctrl.gradient_sports.event_load import load_events
from tempoctrl.gradient_sports.event_transform import (
    cleanup_events,
    transform_events,
)
from tempoctrl.gradient_sports.ingest import (
    RAW_EVENTS_DIRECTORY,
    discover_event_files,
    read_events,
)
from tempoctrl.pipeline_runtime import format_pipeline_runtime

logger = logging.getLogger(__name__)
PIPELINE_DIVIDER = "=" * 72


def configure_logging() -> None:
    """Configure logging at the executable pipeline boundary."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(
    match_id: int,
    event_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build processed event data for one match."""
    input_path = (
        Path(event_path)
        if event_path is not None
        else RAW_EVENTS_DIRECTORY / f"{match_id}.json"
    )
    df = read_events(input_path)
    df = transform_events(df)
    df = features_events(df)
    df = cleanup_events(df)
    output_paths = load_events(df)
    for output_path in output_paths:
        logger.info("Event pipeline output: %s", output_path)
    return output_paths


def main() -> None:
    """Build event outputs and log the complete pipeline runtime."""
    configure_logging()
    logger.info(PIPELINE_DIVIDER)
    logger.info("EVENT PIPELINE")
    logger.info(PIPELINE_DIVIDER)
    started_at = perf_counter()
    try:
        for match_id, event_path in discover_event_files():
            logger.info("Processing event file: %s", event_path)
            run_pipeline(match_id, event_path)
    finally:
        logger.info(
            "Event pipeline runtime: %s",
            format_pipeline_runtime(perf_counter() - started_at),
        )
        logger.info(PIPELINE_DIVIDER + "\n")


if __name__ == "__main__":
    main()
