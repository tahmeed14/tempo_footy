# Historical tests

These files preserve existing scenarios whose implementations are absent
from the current tracked source. They deliberately have no `test_` filename
prefix and are not part of the runnable suite. This is an archive, not
passing coverage or an expected-failure list.

- `ball_release_markers.py` and `ball_release_metrics.py` target removed
  release-attempt and release-time APIs. The current model uses ball-speed
  tempo. Active tempo and possession tests remain in `unit/gradient_sports/`.
- `frame_rate_logging.py` targets the removed
  `log_frame_rate_resolutions` helper. Active frame-rate tests verify the
  returned rate and its metadata/default provenance.
- `possession_gap_examples.py` targets an ignored developer script absent
  from Git. The local copy also imports the removed `scan_integrated` API.
- `possession_start_draft.py` preserves an entirely commented-out test draft
  for the former provider package. Active event and possession-identifier
  tests cover the current `tempoctrl.gradient_sports` implementation.

Before restoring a scenario, implement or identify its supported API,
update its imports and expected schema, move it into a `test_*.py` module
under `unit/`, and run the complete suite. Do not restore private match data.
