"""Build processed event datasets for configured matches."""

import logging
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from tempoctrl.gradient_sports.event_features import features_events
from tempoctrl.gradient_sports.event_load import load_events
from tempoctrl.gradient_sports.event_transform import (
    cleanup_events,
    transform_events,
)
from tempoctrl.gradient_sports.ingest import (
    discover_event_files,
    scan_events,
    stage_events,
)
from tempoctrl.pipeline_runtime import format_pipeline_runtime

logger = logging.getLogger(__name__)
PIPELINE_DIVIDER = "=" * 72
OVERWRITE_STAGE = False


def configure_logging() -> None:
    """Configure logging at the executable pipeline boundary."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(
    match_id: int | Sequence[int] | None = None,
    event_path: str | Path | Sequence[str | Path] | None = None,
    *,
    overwrite_stage: bool = OVERWRITE_STAGE,
) -> tuple[Path, ...]:
    """Stage selected matches and process them together as one batch.

    Args:
        match_id: One match ID, a sequence, or None to discover all raw
            event files. Explicit match IDs must be unique.
        event_path: Optional source JSON path or matching sequence of
            paths. Requires explicit match IDs; paths follow ID order.
        overwrite_stage: Force replacement of selected staged files.

    Returns:
        Event paths ordered by match ID, then the shared lookup path.
        A single-match call retains its two-path return value.

    Raises:
        ValueError: If IDs are empty or repeated, path counts differ, or
            custom paths are supplied without explicit IDs.

    """
    if match_id is None:
        if event_path is not None:
            raise ValueError("event_path requires explicit match IDs")
        event_files = discover_event_files()
    else:
        match_ids = (
            (match_id,) if isinstance(match_id, int) else tuple(match_id)
        )
        if not match_ids:
            raise ValueError("At least one match ID is required")
        if len(set(match_ids)) != len(match_ids):
            raise ValueError("Selected match IDs must be unique")
        if event_path is None:
            raw_paths = (None,) * len(match_ids)
        elif isinstance(event_path, (str, Path)):
            raw_paths = (event_path,)
        else:
            raw_paths = tuple(event_path)
        if len(raw_paths) != len(match_ids):
            raise ValueError("Match ID and event path counts must match")
        event_files = tuple(zip(match_ids, raw_paths, strict=True))

    staged_paths = []
    for selected_id, raw_path in event_files:
        staged_path = stage_events(
            selected_id,
            overwrite=overwrite_stage,
            raw_path=raw_path,
        )
        logger.info("Loading staged event file: %s", staged_path)
        staged_paths.append(staged_path)

    df = scan_events(staged_paths).collect()
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
        run_pipeline()
    finally:
        logger.info(
            "Event pipeline runtime: %s",
            format_pipeline_runtime(perf_counter() - started_at),
        )
        logger.info(PIPELINE_DIVIDER + "\n")


if __name__ == "__main__":
    main()
