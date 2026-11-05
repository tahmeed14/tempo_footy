"""Historical tests for the removed frame-rate logging helper."""

import logging

import pytest

from tempoctrl.gradient_sports.frame_rates import (
    GameFrameRate,
    log_frame_rate_resolutions,
)

def test_logs_selected_fps_for_every_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log metadata and fallback FPS decisions for every match.

    Behavior:
        Log each resolved rate at info level for metadata and warning
        level for the default fallback.

    Input:
        Resolutions specifying 50-fps metadata for one match and a
        24-fps default for another.

    Expected:
        Logs identify 50 fps as metadata and 24 fps as the default for
        their respective match IDs.

    Guards against:
        Unlogged fallback behavior or ambiguous frame-rate provenance.
    """
    resolutions = (
        GameFrameRate(
            game_id=10516,
            frame_rate=24.0,
            source="default",
        ),
        GameFrameRate(
            game_id=10517,
            frame_rate=50.0,
            source="metadata",
        ),
    )
    logger = logging.getLogger("test_frame_rate_resolutions")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_frame_rate_resolutions(resolutions, logger)

    assert "Using 24.000 FPS for match 10516 (source: default)" in (
        caplog.text
    )
    assert "Using 50.000 FPS for match 10517 (source: metadata)" in (
        caplog.text
    )
