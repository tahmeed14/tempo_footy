"""Build processed tracking datasets for configured matches."""

import logging
from pathlib import Path
from time import perf_counter

from tempoctrl.gradient_sports.ingest import (
    resolve_tracking_paths,
    scan_tracking,
    stage_tracking,
    tracking_stage_is_current,
)
from tempoctrl.gradient_sports.tracking_load import load_tracking
from tempoctrl.gradient_sports.tracking_transform import transform_tracking
from tempoctrl.pipeline_runtime import format_pipeline_runtime

logger = logging.getLogger(__name__)
PIPELINE_DIVIDER = "=" * 72
RAW_TRACKING_DIRECTORY = Path("data/raw/gradient_sports/tracking")


def configure_logging() -> None:
    """Configure logging at the executable pipeline boundary."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


OVERWRITE = True


def discover_tracking_files(
    input_dir: str | Path = RAW_TRACKING_DIRECTORY,
) -> tuple[tuple[int, Path], ...]:
    """Discover raw tracking files and their numeric match IDs."""
    input_directory = Path(input_dir)
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Raw tracking directory does not exist: {input_directory}"
        )

    tracking_files = sorted(input_directory.glob("*.jsonl.bz2"))
    if not tracking_files:
        raise FileNotFoundError(
            f"No raw tracking JSONL.BZ2 files found in: {input_directory}"
        )

    discovered_files: list[tuple[int, Path]] = []
    suffix = ".jsonl.bz2"
    for tracking_path in tracking_files:
        match_id_text = tracking_path.name.removesuffix(suffix)
        try:
            match_id = int(match_id_text)
        except ValueError as error:
            raise ValueError(
                "Raw tracking filenames must be numeric match IDs: "
                f"{tracking_path.name}"
            ) from error
        discovered_files.append((match_id, tracking_path))

    return tuple(sorted(discovered_files))


def run_pipeline(
    match_id: int,
    tracking_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build processed tracking data for one match."""
    raw_path, expected_staged_path = resolve_tracking_paths(
        match_id,
        tracking_path,
    )
    if tracking_stage_is_current(match_id, raw_path=raw_path):
        logger.info(
            "Loading staged tracking file: %s",
            expected_staged_path,
        )
    else:
        logger.info(
            "Processing raw tracking file (staging may take longer): %s",
            raw_path,
        )

    staged_path = stage_tracking(match_id, raw_path=tracking_path)
    df_out = scan_tracking(staged_path)
    df_out = transform_tracking(df_out)
    processed_path = load_tracking(
        df_out,
        match_id,
        overwrite=OVERWRITE,
    )
    output_paths = staged_path, processed_path
    for output_path in output_paths:
        logger.info("Tracking pipeline output: %s", output_path)
    return output_paths


def main() -> None:
    """Build tracking outputs and log the complete pipeline runtime."""
    configure_logging()
    logger.info(PIPELINE_DIVIDER)
    logger.info("TRACKING PIPELINE")
    logger.info(PIPELINE_DIVIDER)
    started_at = perf_counter()
    try:
        for match_id, tracking_path in discover_tracking_files():
            run_pipeline(match_id, tracking_path)
    finally:
        logger.info(
            "Tracking pipeline runtime: %s",
            format_pipeline_runtime(perf_counter() - started_at),
        )
        logger.info(PIPELINE_DIVIDER + "\n")


if __name__ == "__main__":
    main()
