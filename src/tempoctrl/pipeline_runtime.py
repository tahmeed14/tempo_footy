"""Formatting helpers for executable pipeline runtimes."""

from __future__ import annotations


def format_pipeline_runtime(elapsed_seconds: float) -> str:
    """Format elapsed time after rounding to the nearest millisecond."""
    elapsed_milliseconds = round(elapsed_seconds * 1_000)
    hour_count, remaining_milliseconds = divmod(
        elapsed_milliseconds, 3_600_000
    )
    minute_count, second_milliseconds = divmod(remaining_milliseconds, 60_000)
    hour_unit = "hour" if hour_count == 1 else "hours"
    minute_unit = "minute" if minute_count == 1 else "minutes"
    return (
        f"{hour_count} {hour_unit}, "
        f"{minute_count} {minute_unit}, "
        f"{second_milliseconds / 1_000:.3f} seconds"
    )
