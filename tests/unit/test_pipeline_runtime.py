"""Tests for shared executable-pipeline runtime formatting."""

import pytest

from tempoctrl.pipeline_runtime import format_pipeline_runtime


@pytest.mark.parametrize(
    ("elapsed_seconds", "expected"),
    [
        (0, "0 hours, 0 minutes, 0.000 seconds"),
        (65, "0 hours, 1 minute, 5.000 seconds"),
        (3661.25, "1 hour, 1 minute, 1.250 seconds"),
        (8284.5, "2 hours, 18 minutes, 4.500 seconds"),
    ],
)
def test_format_pipeline_runtime(
    elapsed_seconds: float,
    expected: str,
) -> None:
    """Preserve units, singular/plural labels, and fractional seconds.
    """
    assert format_pipeline_runtime(elapsed_seconds) == expected


@pytest.mark.parametrize(
    ("elapsed_seconds", "expected"),
    [
        (0.0004, "0 hours, 0 minutes, 0.000 seconds"),
        (0.0006, "0 hours, 0 minutes, 0.001 seconds"),
        (59.9994, "0 hours, 0 minutes, 59.999 seconds"),
        (59.9996, "0 hours, 1 minute, 0.000 seconds"),
        (60, "0 hours, 1 minute, 0.000 seconds"),
        (3599.9994, "0 hours, 59 minutes, 59.999 seconds"),
        (3599.9996, "1 hour, 0 minutes, 0.000 seconds"),
        (3659.9996, "1 hour, 1 minute, 0.000 seconds"),
        (86399.9996, "24 hours, 0 minutes, 0.000 seconds"),
    ],
)
def test_runtime_rounding_carries_across_unit_boundaries(
    elapsed_seconds: float,
    expected: str,
) -> None:
    """Round elapsed time before splitting hours, minutes, and seconds.

    Behavior:
        Carry rounded seconds into minutes and minutes into hours.

    Input:
        Nonnegative durations just above or below millisecond, minute,
        hour, and day boundaries, including an exact minute.

    Expected:
        Explicit strings retain three decimals, singular/plural units,
        and seconds below 60 after every rollover.

    Guards against:
        Displaying 60.000 seconds or failing to carry a rounded minute
        into the next hour, including durations exceeding one day.
    """
    assert format_pipeline_runtime(elapsed_seconds) == expected
