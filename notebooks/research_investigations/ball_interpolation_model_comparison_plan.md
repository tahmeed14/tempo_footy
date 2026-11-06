# Ball-coordinate interpolation model comparison

Status: broad development screen and decision-focused policy validation implemented
and executed; scientific acceptance limits and reserved-match confirmation pending.
Prepared: 2026-11-04.
Updated: 2026-11-05. See the executable
[investigation](ball_interpolation_model_comparison.py) and
[development findings](ball_interpolation_development_findings.md) for completed runs.
Related: [spatiotemporal issue hypotheses](spatiotemporal_data_issue_hypotheses.md).

## 1. Research question and scope

Which method reconstructs short missing ball trajectories accurately enough to
preserve our ball-speed and possession-tempo measurements, and when should we
leave a gap unfilled?

The primary task is retrospective, two-sided interpolation of `balls_smooth.x`
and `y` in Gradient Sports tracking. Observations before and after a gap are
available. Online prediction, player-coordinate imputation, and 3D trajectory
reconstruction are separate tasks. We may investigate them later, but should
not combine their scores with this comparison.

The repository README describes the 64-match 2022 World Cup dataset. Inventory
the usable matches before fixing sample sizes; this plan does not assume that
every downloaded match or every trajectory is suitable for evaluation.

## 2. Repository baseline and implications

These are code observations, not conclusions from exploring tracking values:

- [`interpolate_ball_coordinates`](../../src/tempoctrl/gradient_sports/interpolations.py)
  uses linear interpolation based on numerical frame distance. Its default
  `max_gap=5` counts intervening frame indices, not the number of existing rows.
  It fills existing rows only when both `x` and `y` are null and complete anchors
  exist on both sides within `(game_id, dev_match_team_player_possession_id)`.
  Only the first duplicate-frame row participates; null possession IDs are
  ineligible. It adds `is_imputed`, preserves other ball fields, and does not
  reconstruct absent frame rows or impute `z`.
- [`transform_possessions`](../../src/tempoctrl/gradient_sports/possessions_transform.py)
  derives possession IDs, copies receiver tracking into synthetic pass-end rows,
  normalizes ball direction, interpolates, then labels pitch thirds and computes
  ball metrics. Masking only after copying rows could leave the hidden target
  visible in another possession.
- [`frame_rates.py`](../../src/tempoctrl/gradient_sports/frame_rates.py) resolves
  per-game FPS and records a default of 29.97 when metadata is absent. Five
  missing frames at that rate represent about 0.167 seconds of missing samples;
  the surrounding anchors are six frame intervals, about 0.200 seconds, apart.
- [`ball_metrics.py`](../../src/tempoctrl/gradient_sports/ball_metrics.py) uses 2D
  displacement and numerical frame intervals. It gives synthetic pass endpoints
  priority when assigning movement metrics to duplicate frames.
- [`tempo_metrics.py`](../../src/tempoctrl/gradient_sports/tempo_metrics.py) computes
  tempo as total valid displacement divided by the time covered by valid
  segments. Filling a gap can change both numerator and denominator.
- [`filter_model_data`](../../src/tempoctrl/model/transform.py) filters player tempo
  to values strictly between 0 and 25 m/s, among other criteria. Evaluate before
  this filtering and report changed inclusion decisions. This cutoff is not a
  maximum physically possible instantaneous ball speed.

The interpolation partition does not explicitly include `period`, and anchor
checks use non-null rather than finite-coordinate tests. Investigate whether
those choices matter on our data before using the baseline as a quality filter.

## 3. Candidate methods

Begin with a small comparison whose results can be inspected trajectory by
trajectory. All methods receive the same masked inputs and context budget, use
actual frame times, preserve unmasked coordinates, and disable extrapolation.

| Candidate | Purpose and expected strength | Parameters or limitations to examine |
| --- | --- | --- |
| Leave missing | Downstream coverage and selection-bias control | Has no position-error score; never treat missing predictions as zero error |
| Nearest observed position / forward hold | Simple diagnostic baselines | Expect steps and distorted speed; forward hold uses only past information |
| Current linear implementation, `max_gap=5` | Exact production reference | Preserve its eligibility and duplicate policies; report exclusions and failures |
| Linear with candidate gap limits | Separate the gap-policy question from the model question | Evaluate 1, 2, 3, 5, 10, 15, 30-frame limits and duration equivalents |
| Coordinate-wise PCHIP | Local curvature with shape preservation along each coordinate | Context size; no guarantee of physically plausible 2D motion |
| Local cubic spline or cubic Hermite bridge | Curvature and endpoint-velocity continuity | Boundary condition, context size, endpoint velocity estimation, overshoot |
| Constant-velocity and constant-acceleration state-space models with RTS smoothing | Joint position/velocity dynamics and model-based uncertainty | Process/observation noise, time-step handling, response to touches and deflections |
| Local Gaussian-process regression | Flexible smooth trajectories and predictive uncertainty | Matérn 3/2 versus 5/2 kernel, length scale, noise, context size and runtime |

PCHIP preserves one-dimensional monotonicity; ordinary cubic splines can
overshoot. Those properties motivate including both, but do not establish which
works best for soccer trajectories. See the [SciPy interpolation guide](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html)
and [PCHIP reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

Use a smoother for the primary state-space comparison because later observations
are available; a causal Kalman filter answers a different deployment question.
See [Särkkä, Bayesian Filtering and Smoothing, chapter 8](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).
For GP choices, consult [Rasmussen and Williams, chapters 2, 4 and 5](https://gaussianprocess.org/gpml/chapters/).

For the initial search, allow up to 0.25, 0.5, or 1 second of observed context on
each side, restricted to the same continuous segment. Document minimum anchors
and a linear fallback when higher-order methods lack sufficient context. Tune
all other parameters on development data only. State-space and GP methods may
estimate latent values at observed times internally, but their output must
replace only the masked coordinates for this interpolation benchmark.

Defer neural sequence models, interaction-aware models using players, and
ballistic 3D models until simple baselines reveal a material unmet need. They
require more independent training sequences or validated contextual/height data.
Any context-aware extension must disclose its extra inputs and include an
ablation using ball coordinates alone.

## 4. Construct an honest reconstruction benchmark

### 4.1 Inventory and establish a reference

1. Inventory match IDs, metadata FPS and its provenance, period conventions,
   units, coordinate orientation, duplicate/conflicting keys, `visibility`, `z`,
   null/NaN/infinite coordinates, and absent frame indices. Use the hypothesis
   register to organize later diagnostics.
2. Reserve test matches using match metadata before exploring model errors.
   Learn gap distributions and quality thresholds on development matches.
3. Start from pre-imputation tracking/integrated data. Keep immutable source
   coordinates and provenance; never use previously imputed coordinates as
   reference truth. Exclude synthetic copies as independent reference samples.
4. Define clean, fully observed continuous windows using finite values and
   verified keys, units, and boundaries. Record all exclusions and retain an
   expanded-quality sensitivity set so screening does not erase difficult play.
5. Treat provider-smoothed coordinates as a reconstruction reference, not exact
   physical truth. Upstream smoothing may already encode nearby observations;
   masking after vendor smoothing cannot undo that dependence. If unsmoothed
   data become available, mask before any refitted smoothing as a sensitivity
   analysis. Otherwise state this limit explicitly.

### 4.2 Hide contiguous blocks, then reconstruct

- Primary masks: both coordinates missing for 1, 2, 3, 4, or 5 consecutive
  frames, matching current production scope. Stress masks: 6–10, 11–15,
  16–30, and more than 30 frames where sufficient clean windows exist. Record
  both missing-sample duration and anchor-to-anchor duration in seconds.
- Run a balanced length benchmark and a deployment-weighted benchmark. Estimate
  natural gap-length/context weights from training matches only; freeze and
  reuse them for validation/test. Report lack of observed-window support for
  naturally difficult situations rather than inventing reference coordinates.
- Add contextual masks near touches, passes, sharp changes of direction,
  crowded areas, aerial events, and field boundaries when labels are available.
  These stress tests probe informative missingness; they do not prove that the
  natural missingness mechanism is missing-not-at-random.
- Evaluate partial-coordinate missingness, whole-row dropout, boundary gaps,
  missing possession IDs, non-finite values, and duplicate conflicts separately.
  They require policy changes beyond the current function. An absent-row task
  must specify the time grid and row-insertion policy before comparison.
- Mask each physical ball observation in every duplicate, synthetic copy and
  transformed representation. Prefer masking source coordinates before running
  possession preparation; retain period/frame provenance through synthetic rows.
  Recompute all ball-derived speed, direction, distance and spatial labels from
  masked data. Keep hidden `z`, visibility or event-derived location proxies out
  of candidate inputs unless explicitly justified as available at real gaps.
- Keep an unmasked reference branch for scoring only. Independent event labels
  and observed player positions may be usable covariates, but distinguish them
  from fields computed from hidden ball coordinates. Full-data motion labels
  may define diagnostic strata; they cannot choose the deployed method.
- Use the same mask manifest for every candidate, initially with five fixed mask
  seeds. Avoid overlapping masks within a realization. Additional seeds reduce
  mask-sampling noise but do not create independent matches.

Retain both an exact production-baseline result and a controlled algorithm
comparison on a common, valid cohort. The latter enforces period and continuous
segment boundaries, finite anchors, verified orientation, and identical target
eligibility for all methods. Label any repaired preprocessing explicitly; do not
attribute its benefit to a different interpolator.

## 5. Evaluation splits: holdout plus grouped K-fold

Recommended default: a locked match-level test set plus five-fold grouped
cross-validation on development matches. If all 64 matches are usable, reserve
13 matches for final testing and use 51 for development. Select whole matches
with a fixed seed and metadata-based balance across tournament stages and
available acquisition conditions. Save the IDs, not just the seed.

| Design | Use in this investigation |
| --- | --- |
| Random frame-level train/test or ordinary K-fold | Reject for model selection: overlapping windows and same-match information cross the split |
| Match-level holdout | Final assessment on untouched matches after freezing the method, gap policy and thresholds |
| Five-fold `GroupKFold`, grouped by `game_id` | Primary development comparison and hyperparameter/policy selection; every match appears in validation once |
| Leave-one-match-out | Alternative development design when fewer than about 20 matches are usable; informative but more expensive and not equivalent to many independent frames |
| Leave-one-frame-out (LOO) | Optional isolated-single-frame diagnostic only; does not represent contiguous gaps or new-match generalization |
| Leave-one-gap-out | Optional within-match interpolation diagnostic; insufficient as the main generalization estimate |
| Chronological holdout / rolling origin | Secondary check for later-match deployment or drift; sort actual match dates, never numeric match IDs |
| Leave-team-out or acquisition-group-out | Optional domain-transfer challenge; remove every match involving a held-out team from training |

The match-group recommendation applies the dependence-aware validation
principles in [Roberts et al. (2017)](https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file).
It is a proposed design for this repository, not a result reported by that paper.
[Scikit-learn documents group K-fold and leave-one-group-out](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
Match grouping tests new matches from this dataset; it does not guarantee new
teams, venues, providers or competitions.

Execution rules:

1. Assign matches to splits before window extraction, mask generation or learned
   preprocessing. Every overlapping window, derived copy and seed realization
   from one match stays in that match's split.
2. Within each development fold, fit model parameters, scaling, noise estimates,
   missingness weights and any learned gating on training matches. Use validation
   reconstruction errors to choose hyperparameters and gap thresholds.
3. At validation/test inference, observed anchors on both sides of a hidden gap
   are legitimate inputs for retrospective interpolation. Local spline fitting,
   GP conditioning and smoother state inference may use those anchors with the
   frozen configuration. Hidden coordinates must never enter fitting or tuning.
4. Freeze the selected method, fallback/abstention rules, masks, metrics and
   acceptance criteria, fit global parameters on all development matches, and
   evaluate once on the locked test set. Report the baseline and a predeclared
   shortlist; do not choose a new winner from test scores.
5. If no separate test set is feasible, use nested grouped CV: outer held-out
   matches estimate performance, inner grouped folds tune all choices. With
   very few matches, use leave-one-match-out outer folds where feasible and
   clearly label inference as preliminary. A single match supports only a
   within-match pilot, not a new-match performance claim.

For a within-match pilot, split contiguous time blocks at possession boundaries
and purge overlapping context windows using a buffer at least as large as the
maximum allowed context. Never reuse an anchor or derived copy across those
train/validation blocks. For a genuinely online experiment, disallow future
anchors and use causal features; report it as forecasting rather than smoothing.

## 6. Metrics and aggregation

For hidden time `t`, let `p_t=(x_t,y_t)` be reference position and `p_hat_t`
the reconstruction. Define `e_t = ||p_hat_t - p_t||_2`, in verified meters.
Score hidden coordinates only; copied endpoints must not dilute the error.

| Metric | Definition and purpose |
| --- | --- |
| Primary: gap-average Euclidean error | `ADE_g = mean(e_t)` within gap `g`; measures typical spatial reconstruction error |
| Euclidean RMSE | `sqrt(mean(e_t^2))`; gives larger misses more weight |
| Median, P90/P95 error and maximum error per gap | Detect tails and isolated trajectory failures; report definitions and denominators |
| Signed x/y bias | Mean coordinate residuals, in meters; reveals directional drift |
| Recovery within tolerance | Fraction of hidden points within 0.25, 0.5, 1 and 2 m, as descriptive thresholds pending reference-quality assessment |
| Velocity and speed error | Vector-velocity error and speed MAE/RMSE in m/s using actual `delta_frame/FPS`; include entry and exit segments touching a gap |
| Path-length bias | Signed and absolute difference in summed segment distances across the gap, including both anchor connections |
| Possession-tempo error | MAE and signed bias for `ball_speed_tempo_player` and team tempo on affected possessions; recompute through the production metric functions |
| Downstream category/inclusion changes | Pitch-third disagreement, changed model-data eligibility, and possession/ranking changes; report counts and rates |
| Fill coverage and failure rate | Filled eligible coordinates/gaps divided by attempted coordinates/gaps; separate abstention, insufficient context, invalid output and exceptions |
| Plausibility and continuity | Non-finite output, boundary overshoot, velocity jumps and excessive acceleration/jerk; review with event context |
| Uncertainty, where provided | Per-axis 50/80/95% interval coverage and width; assess joint 2D coverage separately and proper predictive scores if distributions are available |
| Operational cost | Fit time, inference time per million frames, peak memory, dependency/maintenance cost and reproducibility |

Do not use MAPE for coordinates, speed or acceleration near zero. Do not make
R-squared or endpoint error the headline metric: the pitch extent can obscure
small local errors, and observed anchors are already supplied to the method.
Acceleration/jerk are diagnostics, not objectives to minimize blindly, because
touches cause real changes and differentiation amplifies tracking noise.

Aggregate gap errors within each match and length stratum, apply the frozen
training-derived stratum weights, then average matches equally. Report balanced
length-stratum scores alongside the deployment-weighted primary score, plus a
frame-weighted secondary score. Give match/gap/frame counts in every stratum;
flag empty or low-support strata instead of silently changing weights.

Evaluate downstream error twice: on the fixed affected segments/possessions
shared by methods, and under each method's actual availability policy. Show
valid time and distance coverage in both analyses. The leave-missing control
can look accurate simply by omitting difficult segments; it has no spatial
reconstruction score and must remain visible in the coverage comparison.

For uncertainty methods, specify whether the interval predicts the latent
trajectory or a noisy reference observation. Score against the corresponding
reference; latent-state intervals need not cover noisy measurements at their
nominal rate. Never pool coordinate-wise coverage and call it joint coverage.

## 7. Fair comparison, inference and selection

- Compare algorithms on the identical 1–5-frame cohort first. Then compare
  longer-gap policies separately. Report standalone-method coverage and the
  performance of complete candidate-plus-linear-fallback pipelines.
- Show error versus coverage when methods abstain; compare at matched coverage
  and show all attempted gaps. Accuracy on the successful subset alone cannot
  establish superiority. Report pairwise common-support scores as supplementary.
- Stratify by gap length/duration, available context, motion speed/curvature,
  event type, aerial/ground labels, pressure/crowding, pitch region, match/period,
  visibility and FPS source wherever supported. Keep hidden-target-derived
  strata diagnostic rather than runtime selection features.
- Compute paired method-minus-linear differences per match using identical
  masks. Use a paired match-cluster bootstrap, initially 2,000 resamples, for
  95% intervals; keep every gap and seed realization attached to its match.
  Report match-level differences and sensitivity to repeated teams/acquisition
  groups. Frame-wise confidence intervals would exaggerate sample size.
- Do not treat fold scores, overlapping windows or repeated masks as independent
  replications. Label small-test-set intervals as uncertain and exploratory
  subgroup comparisons as exploratory; avoid selecting from dozens of p-values.

Proposed promotion gate: a practically meaningful reduction in primary spatial
error at matched coverage, supported across matches, with no material worsening
of P95 error, possession-tempo MAE/bias or failure rate. As starting discussion
values, consider at least 5% relative ADE improvement and no more than 0.1 m/s
worsening in tempo MAE; these are proposed tolerances, not established domain
standards. Finalize them from development/reference-quality analysis before
opening test results, and include an absolute position-error tolerance.

Prefer linear when gains are small, unstable, or outweighed by speed/tempo bias
or operational cost. A duration-dependent method or abstention policy is allowed
only if selected and validated within development folds. Longer gaps may remain
unfilled. Do not silently clip trajectories to the pitch to improve scores:
out-of-play positions can be real, and clipping changes distances.

## 8. Execution stages and deliverables

1. **Audit and preregistration:** usable-match inventory, development-only gap
   profile, quality/provenance rules, split manifest, and fixed primary endpoint.
   Resolve period/key/unit issues before interpreting model differences.
2. **Benchmark harness:** immutable reference, reproducible masked copies,
   coordinate-copy leakage assertions, and a baseline reproduction check.
   Include small known-trajectory checks for irregular timing, duplicates,
   partial gaps, boundaries and synthetic endpoints.
3. **Development comparison:** linear and simple baselines, then PCHIP/Hermite
   or spline, state-space smoothers, and local GP. Save predictions and per-gap
   scores, paired match summaries, coverage curves and trajectory plots.
4. **Sensitivity studies:** natural-gap weighting, difficult-context masks,
   reference-quality tiers, longer gaps, FPS/orientation uncertainty and domain
   holdouts. Review naturally missing gaps qualitatively; without independent
   coordinates or calibrated video annotations, do not assign them position RMSE.
5. **Locked evaluation and recommendation:** final test report, uncertainty,
   downstream effects, limitations, and a recommended method/gap/abstention
   policy. If independent video is available, use a blinded annotated subset
   with annotation uncertainty to assess real-gap transfer.

Future experiment artifacts should live under an ignored data directory such
as `data/investigate/ball_interpolation/`; keep the protocol and concise reports
in `research_investigations/`. Record source file hashes, code revision and any
dirty patch, package versions, random seeds, split/mask IDs, model configuration,
FPS provenance, excluded observations and failure reasons. A per-gap record
should identify match, period, segment/possession, anchor/target frames, mask
seed, reference quality, method, predictions, imputation status and timing.

This plan proposes experiments only; it does not change the production pipeline
or assert that any candidate outperforms the existing interpolator.
