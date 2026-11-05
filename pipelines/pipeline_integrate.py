"""Build integrated event and tracking datasets."""

import logging
from pathlib import Path
from time import perf_counter

from tempoctrl.gradient_sports.join import possession_load
from tempoctrl.pipeline_runtime import format_pipeline_runtime

logger = logging.getLogger(__name__)
PIPELINE_DIVIDER = "=" * 72
PROCESSED_EVENTS_DIRECTORY = Path("data/processed/gradient_sports/events")
PROCESSED_TRACKING_DIRECTORY = Path("data/processed/gradient_sports/tracking")


def configure_logging() -> None:
    """Configure logging at the executable pipeline boundary."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


OVERWRITE = True


def _discover_processed_match_ids(
    input_dir: str | Path,
    dataset_name: str,
) -> set[int]:
    """Discover numeric match IDs from processed parquet filenames."""
    input_directory = Path(input_dir)
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Processed {dataset_name} directory does not exist: "
            f"{input_directory}"
        )

    parquet_files = sorted(input_directory.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No processed {dataset_name} parquet files found in: "
            f"{input_directory}"
        )

    match_ids: set[int] = set()
    for parquet_path in parquet_files:
        try:
            match_ids.add(int(parquet_path.stem))
        except ValueError as error:
            raise ValueError(
                f"Processed {dataset_name} filenames must be numeric "
                f"match IDs: {parquet_path.name}"
            ) from error
    return match_ids


def discover_match_ids(
    events_dir: str | Path = PROCESSED_EVENTS_DIRECTORY,
    tracking_dir: str | Path = PROCESSED_TRACKING_DIRECTORY,
) -> tuple[int, ...]:
    """Discover matches available in both processed input directories."""
    event_match_ids = _discover_processed_match_ids(
        events_dir,
        "event",
    )
    tracking_match_ids = _discover_processed_match_ids(
        tracking_dir,
        "tracking",
    )

    if event_match_ids != tracking_match_ids:
        events_only = sorted(event_match_ids - tracking_match_ids)
        tracking_only = sorted(tracking_match_ids - event_match_ids)
        raise ValueError(
            "Processed event and tracking match IDs do not match; "
            f"events only: {events_only}; tracking only: {tracking_only}"
        )

    return tuple(sorted(event_match_ids))


def run_pipeline(match_id: int) -> Path:
    """Build one integrated match dataset and log its output path."""
    output_path = possession_load(match_id, overwrite=OVERWRITE)
    logger.info("Integration pipeline output: %s", output_path)
    return output_path


def main() -> None:
    """Build integrated possession files for configured matches."""
    configure_logging()
    logger.info(PIPELINE_DIVIDER)
    logger.info("INTEGRATION PIPELINE")
    logger.info(PIPELINE_DIVIDER)
    started_at = perf_counter()
    try:
        for match_id in discover_match_ids():
            logger.info("Integrating processed match: %d", match_id)
            run_pipeline(match_id)
    finally:
        logger.info(
            "Integration pipeline runtime: %s",
            format_pipeline_runtime(perf_counter() - started_at),
        )
        logger.info(PIPELINE_DIVIDER + "\n")


if __name__ == "__main__":
    main()
