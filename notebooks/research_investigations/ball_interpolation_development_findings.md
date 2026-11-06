# Ball interpolation: development findings

Date: 2026-11-05. Status: completed broad development screen and dense policy
validation; final confirmation and production promotion remain pending.

The leading development candidate is a hybrid: linear for one missing frame,
PCHIP for longer permitted gaps, and explicit linear fallback when PCHIP lacks
enough observed context. All five development folds selected this family at
every studied maximum gap. The evidence is stronger for longer gaps; choosing
an acceptable maximum gap still requires scientific absolute-error limits.

Interactive investigation: [ball_interpolation_model_comparison.py](ball_interpolation_model_comparison.py).
Protocol: [ball_interpolation_model_comparison_plan.md](ball_interpolation_model_comparison_plan.md).

## Decision-focused policy validation

Use the completed run
`data/investigate/ball_interpolation/validation_20261105T175545_234867Z/`
for this section. The earlier `validation_20261105T175124_712861Z/` run did not
include joint-mask scenarios at every candidate cap and is superseded here.

The final follow-up evaluated all 51 development matches, three new mask seeds
(101, 211, 307), every integer single-gap length from 1 through 45 frames, and
two- and four-block joint masks with block lengths sampled uniformly from
1 through each cap (5, 10, 15, 20, 30, 45). There were 12,393 mask realizations
and 235,467 policy evaluations. These are repeated measurements within 51
matches, not independent samples. All requested mask cells had full support,
and the 13 reserved matches were not opened.

At every cap, linear, PCHIP and the hybrid see identical eligible cases. Each
policy also appears on a common full design so abstentions and represented time
remain visible. Every joint reconstruction receives the same fully masked
input; an earlier prediction never becomes an observation for another block.

| Maximum gap | Experiment | Linear ADE, m | Hybrid ADE, m | Linear player-tempo MAE, m/s | Hybrid player-tempo MAE, m/s |
| --- | --- | --- | --- | --- | --- |
| 5 | Single gap | 0.011116 | 0.010100 | 0.000397 | 0.000377 |
| 5 | Joint gaps | 0.010906 | 0.010069 | 0.002434 | 0.002295 |
| 15 | Single gap | 0.047103 | 0.030080 | 0.004342 | 0.004052 |
| 15 | Joint gaps | 0.030187 | 0.022024 | 0.007805 | 0.007325 |
| 30 | Single gap | 0.130795 | 0.076073 | 0.012709 | 0.011787 |
| 30 | Joint gaps | 0.063160 | 0.041372 | 0.013629 | 0.012821 |
| 45 | Single gap | 0.287326 | 0.160250 | 0.026740 | 0.024740 |
| 45 | Joint gaps | 0.101710 | 0.064762 | 0.018562 | 0.017271 |

Single-gap summaries balance integer lengths within a match. Joint summaries
balance the declared scenarios up to the cap. Tempo averages masks within
affected possessions before scenarios and matches; it uses the actual full
possession, not merely the reconstructed interval. Team metrics describe
corrupting one player possession within its team possession. Each cap covers
a different cohort, so this table cannot choose the cap by comparing its means.

At cap 30, the hybrid-minus-linear coordinate ADE difference has an exploratory
paired-match 95% interval of [-0.060104, -0.049184] m for single gaps and
[-0.024418, -0.019290] m for joint gaps. At cap 5, the corresponding intervals
include zero: [-0.002806, 0.000534] m and [-0.001832, 0.000190] m. Thus the tiny
short-gap coordinate advantage remains uncertain under new masks despite its
favorable point estimate. All five folds choosing the hybrid does not turn
these development results into independent confirmation.

Tempo MAE, signed bias, mean match RMSE, and within-match/scenario
possession-weighted absolute-error P95 are reported for both player and team
possessions. The reported P95 averages within-match/scenario percentiles; it is
not a pooled global percentile. At cap 30, joint-gap player-tempo P95 decreases
from 0.085697 to 0.082294 m/s. At cap 45, it is 0.110590 versus 0.105457 m/s.
These tails and the absolute duration-specific coordinate errors matter when
choosing the cap, even though the hybrid improves over linear.

VISIBLE-only diagnostics retain the same direction of improvement. For cap-30
joint masks, all-context-VISIBLE cases cover 51 matches and 1,927 realizations:
linear ADE is 0.073936 m versus hybrid 0.047304 m. Reference-fast-motion,
reference-sharp-turn, all-targets-VISIBLE, and provider event strata are also
saved with their support. These are overlapping diagnostic subsets, not
deployment predictors. The full-segment VISIBLE restriction for joint masks is
stricter than the local-window restriction for single masks.

Natural-gap coverage is a separate issue. Of 1,651 bounded natural gaps in the
derived cohort, 355 (21.5%) are at most five frames, 918 (55.6%) at most 30, and
1,094 (66.3%) at most 45. Those caps account for only 0.77%, 9.98%, and 16.52%
of missing frame indices within that bounded-gap cohort: the longest gaps
contain much of the missing duration. These are length-only eligibility
ceilings, not demonstrated successful reconstruction or overall match coverage.

The walkthrough now replays named saved runs by default and saves coordinate,
speed, reference-versus-reconstructed tempo, and tempo-error distribution plots.
Explicit controls can rerun either development stage. The confirmation runner
requires one frozen candidate and complete `DecisionCriteria`, checks source,
split and artifact identity, and records a single holdout opening. A failed
opening is recorded as consumed rather than silently repeated. It evaluates
coordinate improvement, absolute error limits, speed/tempo noninferiority and
tail guardrails in both experiments; diagnostic review remains required.

No final-test run or production change has been performed. The next decision
is the scientific maximum acceptable coordinate/speed/tempo error and the
associated candidate cap, followed by confirmation on the 13 reserved matches.
Validation completed: 139 relevant research/interpolation/tempo tests passed;
the complete walkthrough replayed successfully from both named saved runs;
saved masks, predictions and numerical-helper source hashes were verified.
Independent physical-reference validation, naturally informative missingness,
simultaneous corruption across multiple player possessions within a team
possession, and boundary/partial-axis/absent-row recovery remain limitations.

## Initial model-screen evidence and reproducibility

The principal run is
`data/investigate/ball_interpolation/development_20261105T171259_421871Z/`.
An earlier screening run is
`data/investigate/ball_interpolation/development_20261105T170943_839776Z/`.
Use the principal run for the numbers below; the screening run has fewer masks
and does not contain the later whole-window visibility classification.

The principal run used:

- 51 development matches, with 13 other matches reserved before evaluation.
  The investigation has not opened the reserved matches' tracking values.
- Five fixed mask seeds and three nonoverlapping target blocks per gap length
  per seed per match. Every requested mask cell had enough support.
- 6,885 isolated single-gap perturbations: 3,825 primary cases of lengths
  1–5 and 3,060 stress cases of lengths 8, 15, 30 and 45 frames.
- 43 configurations: the actual production linear function at seven gap limits,
  missing/hold/nearest controls, PCHIP, natural cubic and Hermite interpolation,
  CV/CA RTS smoothing, and local Matérn GP regression.
- 296,055 method–gap evaluations, with saved case coordinates/masks, predictions,
  source hashes, metadata, configurations, code snapshots and failure/support
  records. These are repeated measurements within 51 matches, not 296,055
  independent observations for confidence intervals.

For each gap we measured coordinate error and production-derived changes in
speed, path length, valid time, player tempo and team tempo. The short-gap
comparison includes exactly the same hidden coordinates for all methods.
Paired bootstrap intervals resample matches rather than frames. Each model
family's configuration is also selected on training matches and assessed on
disjoint development folds.

## Short-gap reconstruction

The following scores average gaps within length, then lengths within match,
then matches equally. Weighted ADE instead uses natural 1–5-frame gap frequencies
from development data; it is descriptive here. Fold-selection results use only
training-match frequencies.

| Method | Balanced position ADE, m | Natural-length-weighted ADE, m | Speed MAE, m/s | Player-tempo MAE, m/s |
| --- | --- | --- | --- | --- |
| Current linear, maximum gap 5 | 0.010955 | 0.009210 | 0.216122 | 0.000612 |
| PCHIP, 0.25 s context plus linear fallback | 0.010096 | 0.009133 | 0.206041 | 0.000587 |
| Matérn 3/2 GP, 0.25 s context, 0.5 s scale | 0.010519 | 0.009286 | 0.213725 | 0.001106 |
| Nearest observed anchor | 0.295704 | 0.264661 | 9.500930 | 0.000612 |

PCHIP reduces balanced ADE by approximately 0.000860 m, or 0.86 mm (7.8%). The
fold-selected PCHIP minus linear difference has an exploratory paired-match 95%
interval of about [-0.001777, -0.000030] m. This is development evidence of a
small difference, not independent confirmation after comparing model families.
Whether a submillimetre gain matters scientifically remains unresolved.

The natural-length-weighted reduction is only about 0.8%. There were 355 natural
bounded gaps of lengths 1–5 in the derived development possession cohort:
147 of length 1, 84 of length 2, 50 of length 3, 55 of length 4 and 19 of length 5.
This mix favors the shortest gaps, where linear is competitive.

| Gap length | Linear ADE, m | PCHIP ADE, m |
| --- | --- | --- |
| 1 | 0.005763 | 0.007392 |
| 2 | 0.009668 | 0.008738 |
| 3 | 0.011946 | 0.011553 |
| 4 | 0.013679 | 0.011410 |
| 5 | 0.013721 | 0.011385 |

PCHIP's 0.25, 0.5 and 1-second context configurations produce the same short-gap
result because its local slopes depend on nearby observations. In the primary
cohort, 108 of 3,825 cases required its explicit linear fallback (2.8%). The
result therefore describes the complete PCHIP-plus-fallback policy.

Nearest/hold methods can preserve endpoint-to-endpoint distance for certain
trajectories while badly distorting intermediate position and instantaneous
speed. Their tempo result alone must not be taken as evidence of good recovery.

## Longer gaps deserve a separate decision

| Gap length | Linear with limit 30: ADE, m | PCHIP: ADE, m | Linear player-tempo MAE, m/s | PCHIP player-tempo MAE, m/s |
| --- | --- | --- | --- | --- |
| 8 | 0.032688 | 0.023132 | 0.003331 | 0.003122 |
| 15 | 0.076813 | 0.048249 | 0.006600 | 0.005926 |
| 30 | 0.402269 | 0.220558 | 0.039558 | 0.037023 |
| 45 | Does not fill | 0.459632 | Availability changes | 0.063916 |

PCHIP looks more promising as gap duration increases, but absolute errors also
increase. These results come from complete reference trajectories with artificial
interior gaps; they do not establish accuracy during genuine long occlusions.
The four stress lengths are probes, not validation of every length up to 45.
The production five-frame limit should therefore not be expanded from this
table alone.

## Reference quality and uncertainty

Of the 6,885 case windows, 3,639 have provider `VISIBLE` labels throughout the
entire reference/context window; 3,246 include another visibility state.
All hidden target positions are marked `VISIBLE` in 4,982 cases. Finite provider
coordinates can thus include estimates and cannot be treated as independent
physical truth.

On supported visible-only match/length cells, mean ADE is about 0.011255 m for
linear and 0.010440 m for PCHIP. The same direction appears in the expanded
quality stratum, but these stratum summaries have different support and are
descriptive, not a new balanced-cohort primary endpoint. See
`quality_stratum_metrics.csv` for counts.

The selected GP example has roughly 99.3% joint coverage for its nominal 95%
observation ellipse and a mean marginal 95% interval width of about 0.338 m in
the primary cohort. Its assumed noise/amplitude grid therefore needs calibration
against the reference and its intended uncertainty target. High coverage alone
does not justify replacing the simpler deterministic methods.

The development audit contains 2,393 physical keys with conflicting source
coordinate versions; such keys are excluded as reference targets. It also finds
12,570 derived-possession rows without a recognized direction. These are audit
signals for later boundary/alignment work, not automatic evidence that all those
rows are erroneous. Out-of-pitch positions may represent valid out-of-play motion.

## Interpretation and next decisions

1. Shortlist the hybrid policy, with its explicit linear fallback, for confirmation.
   Dense single-gap and joint-gap development comparisons are complete. The
   production linear implementation and five-frame limit remain provisional
   while the final criteria and candidate cap are selected.
2. Distinguish choosing the interpolation family from expanding the maximum gap.
   Favorable longer-gap comparisons with linear do not by themselves establish
   an acceptable absolute reconstruction error.
3. Decide an absolute coordinate improvement that matters scientifically, plus
   allowable tempo/tail degradation. The earlier 0.1 m/s tempo tolerance was an
   illustrative proposal, not an agreed or established standard. On these
   isolated short gaps, average tempo errors are far smaller than that value.
4. Review the completed joint-mask, visibility, motion and event diagnostics.
   Assess transfer to natural gaps using independent reference evidence where
   possible. Partial-axis recovery, missing-row insertion, boundary extrapolation
   and simultaneous corruption across multiple player possessions remain separate
   tasks.
5. Freeze the final candidate, gap policy, practical criteria and test masks,
   then evaluate the 13 reserved matches once. Implement the supported method
   directly in the production pipeline after that decision, with regression and
   integration checks.

The existing production pipeline remains on linear interpolation. This is the
completed development phase of the first investigation, not a claim that the
entire interpolation study or pipeline migration is complete.
