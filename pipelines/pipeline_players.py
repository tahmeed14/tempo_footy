"""Build the processed Gradient Sports player-game lookup."""

import logging
from pathlib import Path
from time import perf_counter

from tempoctrl.gradient_sports.players import (
    PLAYER_LOOKUP_PATH,
    build_player_game_lookup,
    write_player_game_lookup,
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


ROSTER_DIRECTORY = Path("data/raw/gradient_sports/roster")
OVERWRITE = True


def run_pipeline(
    roster_dir: str | Path = ROSTER_DIRECTORY,
    output_path: str | Path = PLAYER_LOOKUP_PATH,
    *,
    overwrite: bool = OVERWRITE,
) -> Path:
    """Build and write the processed player-game lookup.

    Args:
        roster_dir: Directory containing raw game roster JSON files.
        output_path: Destination for the processed lookup Parquet file.
        overwrite: Whether to replace an existing lookup artifact.

    Returns:
        The path of the written or preserved lookup artifact.

    """
    logger.info("Building player lookup from: %s", roster_dir)
    df_lookup = build_player_game_lookup(roster_dir)
    logger.info(
        "Built player lookup with %d players across %d games",
        df_lookup.height,
        df_lookup.get_column("game_id").n_unique(),
    )

    written_path = write_player_game_lookup(
        df_lookup,
        output_path,
        overwrite=overwrite,
    )
    logger.info("Player lookup pipeline output: %s", written_path)
    return written_path


def main() -> None:
    """Run the player lookup pipeline with runtime logging."""
    configure_logging()
    logger.info(PIPELINE_DIVIDER)
    logger.info("PLAYER LOOKUP PIPELINE")
    logger.info(PIPELINE_DIVIDER)
    started_at = perf_counter()
    try:
        run_pipeline()
    finally:
        logger.info(
            "Player lookup pipeline runtime: %s",
            format_pipeline_runtime(perf_counter() - started_at),
        )
        logger.info(PIPELINE_DIVIDER + "\n")


if __name__ == "__main__":
    main()
