"""Historical release-marker tests for an API absent from this version.
"""

import polars as pl

from tempoctrl.gradient_sports.possessions_transform import (
    mark_ball_release_attempts,
)

def test_mark_ball_release_attempts_includes_passes_and_crosses() -> None:
    """Mark pass and cross frames without marking other event types.

    Behavior:
        Identify both successful and unsuccessful delivery attempts from
        the provider possession-event type before synthetic rows exist.

    Input:
        Pass, cross, initial-touch, and null event-type rows.

    Expected:
        Only the pass and cross rows are marked as release attempts.

    Guards against:
        Restricting release time to successful deliveries or treating
        reception and missing annotations as release attempts.
    """
    frames = pl.DataFrame(
        {"possession_event_type": ["PA", "CR", "IT", None]},
        schema_overrides={"possession_event_type": pl.String},
    ).lazy()

    result = mark_ball_release_attempts(frames).collect()

    assert result["is_ball_release_attempt"].to_list() == [
        True,
        True,
        False,
        False,
    ]
    assert result.schema["is_ball_release_attempt"] == pl.Boolean
