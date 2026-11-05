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
