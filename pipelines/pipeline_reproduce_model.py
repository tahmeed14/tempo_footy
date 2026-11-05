"""Run the model-reproduction pipeline."""

import logging
from pathlib import Path

import polars as pl

from tempoctrl.model.load import load_curated_model_df

logger = logging.getLogger(__name__)
PIPELINE_DIVIDER = "=" * 72

REPRODUCIBLE_DATA_DIR = Path("data/analysis/modeldata_v0.parquet")
REPRODUCIBLE_DATA_OUT = Path("data/analysis/model_data_vFINAL.parquet")


def configure_logging() -> None:
    """Configure logging at the executable pipeline boundary."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline() -> pl.DataFrame:
    """Run the model-reproduction workflow."""
    df = load_curated_model_df(REPRODUCIBLE_DATA_DIR)

    return df


def main() -> None:
    """Run the pipeline from the command line."""
    configure_logging()

    logger.info(PIPELINE_DIVIDER)
    logger.info("Running model reproduction pipeline")
    logger.info(PIPELINE_DIVIDER)

    try:
        df = run_pipeline()
        df.write_parquet(
            REPRODUCIBLE_DATA_OUT, compression="zstd", compression_level=3
        )
    finally:
        logger.info("Model reproduction pipeline complete")
        logger.info("File: %s", REPRODUCIBLE_DATA_OUT)
        logger.info(PIPELINE_DIVIDER)


if __name__ == "__main__":
    main()
