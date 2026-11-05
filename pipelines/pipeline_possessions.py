"""Build frame- and possession-level model datasets."""

import logging
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from tempoctrl.gradient_sports.frame_rates import (
    GRADIENT_SPORTS_DEFAULT_FPS,
    FrameRateSpec,
    resolve_gradient_sports_frame_rates,
)
from tempoctrl.gradient_sports.possessions_load import possessions_load
from tempoctrl.pipeline_runtime import format_pipeline_runtime

logger = logging.getLogger(__name__)
PIPELINE_DIVIDER = "=" * 72


def configure_logging() -> None:
    """Configure logging at the executable pipeline boundary."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


OUTPUT_DIRECTORY = Path("data/analysis")
METADATA_DIRECTORY = Path("data/raw/gradient_sports/metadata")


def run_pipeline(
    df_path: str | Path | Sequence[str | Path],
    *,
    metadata_dir: str | Path,
    output_dir: str | Path = OUTPUT_DIRECTORY,
    output_name: str = "possessions.parquet",
    default_frame_rate: float = GRADIENT_SPORTS_DEFAULT_FPS,
) -> tuple[Path, ...]:
    """Run possession processing with match-specific FPS metadata."""
    resolutions = resolve_gradient_sports_frame_rates(
        df_path,
        metadata_dir,
        default_frame_rate=default_frame_rate,
    )
    for resolution in resolutions:
        log = (
            logger.info if resolution.source == "metadata" else logger.warning
        )
        log(
            "Using %.3f FPS for match %d (source: %s)",
            resolution.frame_rate,
            resolution.game_id,
            resolution.source,
        )

    frame_rates: FrameRateSpec = {
        resolution.game_id: resolution.frame_rate for resolution in resolutions
    }
    output_paths = possessions_load(
        df_path=df_path,
        output_name=output_name,
        output_dir=output_dir,
        frame_rate=frame_rates,
    )
    for output_path in output_paths:
        logger.info("Possession pipeline output: %s", output_path)
    return output_paths


def main() -> None:
    """Build possession outputs and log the complete pipeline runtime."""
    configure_logging()
    logger.info(PIPELINE_DIVIDER)
    logger.info("POSSESSION & TEMPO METRICS PIPELINE")
    logger.info(PIPELINE_DIVIDER)
    started_at = perf_counter()
    try:
        run_pipeline(
            df_path="data/integrated/gradient_sports",
            metadata_dir=METADATA_DIRECTORY,
        )
    finally:
        logger.info(
            "Possession & Tempo Metrics pipeline runtime: %s",
            format_pipeline_runtime(perf_counter() - started_at),
        )
        logger.info(PIPELINE_DIVIDER + "\n")


if __name__ == "__main__":
    main()
