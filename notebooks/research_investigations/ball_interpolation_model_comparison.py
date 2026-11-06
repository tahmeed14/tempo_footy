# %% [markdown]
# ---
# title: "Ball-coordinate interpolation: reconstruction and tempo preservation"
# author: "Tahmeed Tureen"
# format:
#   pdf:
#     pdf-engine: lualatex
# execute:
#   echo: false
# ---
#
# We investigate the first item in our pipeline research roadmap:
#
# $$
# \text{missing ball coordinates}\longrightarrow\widehat{\mathbf p}_t
# \longrightarrow\widehat v_t\longrightarrow\widehat T_{possession}.
# $$
# The question is which reconstruction method preserves our measurements, and
# for which gap durations recovery is defensible. A smooth trajectory alone is
# not evidence of accuracy. We compare against the current production linear
# interpolator and against leaving a gap missing.
#
# This script follows `ball_interpolation_model_comparison_plan.md`. Execute the
# cells sequentially. Tables and examples are calculated from a named saved run;
# no illustrative numbers are presented as empirical findings. Final test matches
# stay sealed while we inspect development tradeoffs and define practical
# tolerances. Helper functions live in `src/tempoctrl/research`.
#
# Run from the repository root with
# `uv run python notebooks/research_investigations/ball_interpolation_model_comparison.py`.
# Defaults replay the two explicitly named completed development runs. Set
# RUN_EXPERIMENTS=True to repeat the broad screen, or RUN_POLICY_EXPERIMENTS=True
# to repeat the dense policy validation. The final test has its own control.

# %%
from pathlib import Path
import sys

_origins = [Path.cwd(), Path(globals().get("__file__", Path.cwd())).resolve().parent]
ROOT = next(
    candidate
    for origin in _origins
    for candidate in (origin, *origin.parents)
    if (candidate / "pyproject.toml").exists() and (candidate / "src/tempoctrl").is_dir()
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dataclasses import replace
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display, Markdown

from tempoctrl.research.interpolation_workflow import (
    InterpolationConfig, load_development_run, run_development, start_study,
)
from tempoctrl.research.interpolation_evaluation import (
    grouped_selection, match_summary, paired_bootstrap,
)
from tempoctrl.research.interpolation_visualize import (
    plot_error_by_length, plot_gap_example, plot_paired_differences, representative_cases,
    plot_policy_validation,
)
from tempoctrl.research.interpolation_validation import (
    DecisionCriteria, ValidationConfig, decision_frontier, freeze_confirmation,
    load_policy_validation, run_confirmation, run_policy_validation,
)

CONFIG = InterpolationConfig()
# The default has three masks per length/seed on all development matches. For a
# quick screening comparison use replace(CONFIG, gaps_per_length=1).
RUN_EXPERIMENTS = globals().get("RUN_EXPERIMENTS", False)
# To inspect existing results, set RUN_EXPERIMENTS=False and paste the exact run
# directory here. We never choose an old run automatically by filename recency.
RUN_DIRECTORY = globals().get(
    "RUN_DIRECTORY", "data/investigate/ball_interpolation/development_20261105T171259_421871Z",
)
# The initial broad model screen above is a named completed experiment. The
# decision-focused follow-up below runs separately, without repeating that fit.
POLICY_CONFIG = globals().get("POLICY_CONFIG", ValidationConfig())
RUN_POLICY_EXPERIMENTS = globals().get("RUN_POLICY_EXPERIMENTS", False)
POLICY_RUN_DIRECTORY = globals().get(
    "POLICY_RUN_DIRECTORY", "data/investigate/ball_interpolation/validation_20261105T175545_234867Z",
)
# Enter scientific limits explicitly after reviewing development evidence.
# DecisionCriteria defines all required limits, in metres or metres/second.
SCIENTIFIC_CRITERIA = globals().get("SCIENTIFIC_CRITERIA", None)
CONFIRMATION_CANDIDATE = globals().get("CONFIRMATION_CANDIDATE", None)
FREEZE_CONFIRMATION = globals().get("FREEZE_CONFIRMATION", False)
RUN_FINAL_TEST = globals().get("RUN_FINAL_TEST", False)
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

# %% [markdown]
# ## What is the reference, and what is available at inference?
#
# Let $\mathbf p_t=(x_t,y_t)^\top$ be the provider-smoothed ball position in
# verified meters. We remove both coordinates over a contiguous block
# $G=\{a+1,\ldots,b-1\}$ while retaining the bounding observations at $a,b$.
# Model inputs contain the visible context and a missingness mask, never hidden
# positions, speeds, derivatives, visibility, height, or labels derived from
# hidden ball coordinates. Event/visibility and reference-motion labels below
# are diagnostic strata, not predictors or rules for selecting a method.
#
# This is retrospective interpolation:
#
# $$\widehat{\mathbf p}_G=f_\theta(\{(t,\mathbf p_t):t\notin G\}).$$
#
# Both past and future anchors are legitimate because reconstruction happens
# after the match. Forecasting with only past observations is a separate task.
# Provider smoothing may itself contain information from neighboring frames;
# masked reconstruction measures recovery of that supplied signal, not exact
# physical ball motion. Independent video/raw tracking would strengthen the
# reference and is not available to this initial benchmark.

# %% [markdown]
# ## Why a match holdout plus grouped folds?
#
# Adjacent frames, overlapping context windows, and repeated masks from the same
# match are dependent. Random frame K-fold would measure recovery in nearly
# duplicated local conditions and would not estimate transfer to a new match.
# Before reading tracking values, we reserve 20% of whole matches (13 of 64),
# stratified proportionally by metadata week. The remaining 51 matches are split
# into five balanced match folds. Small studies use leave-one-match-out folds.
#
# Within development fold $k$, natural gap-length weights and each method family's
# configuration are chosen from training matches $\mathcal D_{-k}$ only:
#
# $$
# \theta^{*}_{F,-k}=\arg\min_{\theta\in\Theta_F}
# \frac{1}{|\mathcal D_{-k}|}\sum_{m\in\mathcal D_{-k}}
# \sum_{\ell=1}^{5}w_{\ell,-k}\overline e_{m\ell\theta}.
# $$
#
# The chosen configuration is then assessed on matches in $\mathcal D_k$.
# All candidates are local models with predeclared parameters; conditioning a
# spline, GP or smoother on visible held-out anchors is inference, not learning
# from the hidden target. Candidate-wise tables are exploratory development
# comparisons. Fold-isolated selections reduce configuration-selection optimism;
# deciding between families still needs the locked final holdout.
#
# Isolated-frame LOO is only the one-frame diagnostic within this broader task.
# It does not represent longer contiguous occlusions. Match holdout also does not
# establish transfer to new competitions, providers, or unseen teams.

# %%
splits = start_study(CONFIG, root=ROOT)
display(splits.groupby(["split", "week"], observed=True).size().rename("matches").reset_index())
display(splits[["game_id", "date", "week", "split", "fold"]])

# %% [markdown]
# ## Source preparation and integrity checks
#
# We read pre-imputation integrated tracking and reuse our possession ID filling,
# delivery propagation, synthetic endpoint, coordinate normalization, pitch-third,
# ball-metric and possession aggregation functions. For the controlled comparison
# we isolate periods and namespace derived possession IDs. That extra safeguard
# is disclosed; improvements from it are not credited to an interpolator.
#
# Reference targets must be finite, have known direction, and lie inside a
# continuous player/team segment. Conflicting duplicate coordinates and synthetic
# or multiply represented physical target frames are excluded. Their frequencies
# remain visible in the audit. A separate helper masks all copies of a physical
# `(game_id, period, framenum)` key and is tested against synthetic copies.
#
# Each case removes one gap in isolation. Across a seed realization target blocks
# do not overlap. Context windows may overlap; those repetitions remain attached
# to their match for uncertainty estimation. Joint corruption of several gaps in
# a possession is an additional sensitivity before pipeline promotion.

# %%
if RUN_EXPERIMENTS:
    RUN_DIRECTORY = run_development(CONFIG, root=ROOT)
elif RUN_DIRECTORY is None:
    raise ValueError("Specify a saved RUN_DIRECTORY or enable RUN_EXPERIMENTS.")

RUN_DIRECTORY = Path(RUN_DIRECTORY)
if not RUN_DIRECTORY.is_absolute():
    RUN_DIRECTORY = ROOT / RUN_DIRECTORY
FIGURES = RUN_DIRECTORY / "figures"
FIGURES.mkdir(exist_ok=True)
results = load_development_run(RUN_DIRECTORY)
scores = results["scores"]
display(Markdown(f"Results: `{RUN_DIRECTORY}`"))
display(results["audit"])
display(results["support"].groupby("gap_length").agg(
    requested=("requested", "sum"), sampled=("sampled", "sum"),
    match_count=("game_id", "nunique"),
))

# %% [markdown]
# ## Candidate models and their assumptions
#
# Linear interpolation uses the actual production helper, not a reimplementation:
#
# $$
# \widehat{\mathbf p}_t=(1-u)\mathbf p_a+u\mathbf p_b,
# \qquad u=\frac{t-a}{b-a}.
# $$
#
# It is inexpensive and preserves the endpoints, but imposes constant velocity
# across the gap. We compare gap limits 1, 2, 3, 5, 10, 15 and 30 frames; the
# five-frame version is the production reference. Limits are based on numerical
# frame distance. Primary gaps have lengths 1–5; 8, 15, 30 and 45 are stress
# probes, not evidence that every duration in each broad bin has been validated.
#
# Nearest-anchor and forward-hold models diagnose the value of movement recovery.
# PCHIP preserves shape along each coordinate. Natural cubic splines permit
# smooth curvature but can overshoot. An endpoint Hermite bridge uses velocities
# estimated only from observed context:
#
# $$\widehat{\mathbf p}(u)=h_{00}(u)\mathbf p_a+h_{10}(u)\Delta\mathbf v_a
# +h_{01}(u)\mathbf p_b+h_{11}(u)\Delta\mathbf v_b,$$
#
# where $h_{00}=2u^3-3u^2+1$, $h_{10}=u^3-2u^2+u$,
# $h_{01}=-2u^3+3u^2$, $h_{11}=u^3-u^2$, and $\Delta=t_b-t_a$.
# Context budgets are .25, .5 and 1 second on each side. Higher-order methods
# with fewer than two observations on either side explicitly fall back to linear;
# the fallback rate is reported. Unmasked coordinates are always preserved.
#
# ### State-space smoothing
#
# Per axis, the CV state is $s_t=(p_t,v_t)$ and CA state is $(p_t,v_t,a_t)$:
#
# $$s_{t+1}=F(\Delta t)s_t+\eta_t,\quad\eta_t\sim N(0,Q(\Delta t)),
# \qquad y_t=Hs_t+\epsilon_t,\quad\epsilon_t\sim N(0,r).$$
#
# $F_{ij}=\Delta t^{j-i}/(j-i)!$ for $j\ge i$. For state dimension $d$,
# an integrated white-noise process gives
#
# $$Q_{ij}=q\frac{\Delta t^{2d-1-i-j}}
# {(2d-1-i-j)(d-1-i)!(d-1-j)!}.$$
#
# The Kalman filter skips updates at hidden observations. RTS backward smoothing
# uses the later anchors: $m^s_t=m_t+C_t(m^s_{t+1}-m^-_{t+1})$,
# $C_t=P_tF_t^\top(P^-_{t+1})^{-1}$. We compare $q=1,100$ with observation SD
# 0.05 m. These are explicit initial candidate values, not established tracking
# noise estimates. The two axes have independent identical noise assumptions.
#
# ### Gaussian processes
#
# Local GP regression models residuals around the observed-endpoint linear mean
# with Matérn 3/2 or 5/2 covariance, length scale .15 or .5 s and 1 m residual SD:
#
# $$\mu_* = m_*+K_{*o}(K_{oo}+rI)^{-1}(y_o-m_o),$$
# $$\Sigma_* = K_{**}-K_{*o}(K_{oo}+rI)^{-1}K_{o*}.$$
#
# Predictive observation variance adds $r$ back. The initial grid fixes noise
# and GP amplitude, so conclusions apply to that grid; broaden it if diagnostics
# show miscalibration. Neither GP nor state-space Gaussian intervals include all
# provider preprocessing uncertainty or uncertainty in chosen hyperparameters.

# %% [markdown]
# ## Why these evaluation metrics?
#
# For gap $g$ containing $L_g$ hidden positions, our primary position error is
#
# $$ADE_g=\frac1{L_g}\sum_{t\in g}\|\widehat{\mathbf p}_t-\mathbf p_t\|_2.$$
#
# We average gaps within length, lengths within match, and matches equally.
# This prevents long gaps and matches with many observations from dominating:
#
# $$R_{balanced}=\frac1M\sum_m\frac15\sum_{\ell=1}^5\overline{ADE}_{m\ell}.$$
#
# A second score replaces $1/5$ with training-only natural-gap frequencies
# $w_\ell=n_{\ell,train}/\sum_{j=1}^5n_{j,train}$. It targets the observed
# missing-gap mixture. If natural gaps or required strata lack support, the score
# is explicitly unavailable; the code does not silently invent uniform weights.
# RMSE and gap-level tail errors expose large misses. Coverage is displayed beside
# error because a method must not win by abstaining on its difficult cases.
#
# Speed and full-possession tempo are different quantities:
#
# $$v_t=\frac{\|\mathbf p_t-\mathbf p_{t-1}\|_2}{\Delta t_t},\qquad
# T_P=\frac{\sum_{t\in V_P}d_t}{\sum_{t\in V_P}\Delta t_t}.$$
#
# Here $V_P$ is the set of valid movement segments. We reuse `add_ball_metrics`
# and `aggregate_possession_tempo` to measure bridge changes $\delta D,\delta S$;
# the affected complete-possession value becomes
# $\widehat T_P=(D_P+\delta D)/(S_P+\delta S)$. Both boundary connections of the
# gap count. Missing predictions reduce represented time; they never receive
# zero coordinate error. Player and team tempo are evaluated before model filters.
# `tempo_range_inclusion_changed` concerns only the existing 0–25 m/s outcome
# filter, not all contextual eligibility rules.
#
# For example, if reference tempo is 8.0 m/s, linear gives 7.8 and another method
# gives 7.7, absolute errors are .2 and .3 m/s: the second method worsens error by
# .1. A possible noninferiority tolerance controls this *average error increase*;
# it is not a coordinate tolerance or an automatic acceptable scientific effect.
# We leave it undecided until reviewing these development tradeoffs.

# %%
display(results["leaderboard"])
display(results["paired"])
fig = plot_paired_differences(results["paired"])
fig.savefig(FIGURES / "paired_match_differences.png", dpi=160, bbox_inches="tight")
plt.show()

# %% [markdown]
# Paired confidence intervals resample complete matches, keeping every gap,
# method and mask seed together:
#
# $$\widehat\Delta^*=\frac1M\sum_{b=1}^{M}
# (R_{I_b,candidate}-R_{I_b,linear}),\qquad I_b\sim Unif\{1,\ldots,M\}.$$
#
# Negative values favor the candidate. Five mask seeds improve stability to
# which gaps were selected; they do not multiply the independent sample size.
# Match bootstrap intervals do not remove dependence from repeated teams, and
# development selection means these intervals are not final confirmation.

# %%
gaps = pd.read_csv(RUN_DIRECTORY / "natural_gaps.csv")
selections, selected_oof = grouped_selection(scores, splits, gaps)
display(selections)
if len(selected_oof):
    display(selected_oof.groupby("family").agg(
        matches=("game_id", "nunique"), balanced_ade=("balanced_ade", "mean"),
        weighted_ade=("weighted_ade", "mean"),
        player_tempo_mae=("player_tempo_abs_error", "mean"),
    ))
    baseline_matches = pd.read_csv(RUN_DIRECTORY / "match_metrics.csv").query("method == 'linear_5'")
    selected_paired = paired_bootstrap(
        pd.concat([baseline_matches, selected_oof], ignore_index=True),
        draws=CONFIG.bootstrap_draws, seed=CONFIG.split_seed,
    )
    display(selected_paired)

# %% [markdown]
# ## How far can we extend reconstruction?
#
# The next curves include long gaps and fill coverage. An accurate one-frame
# reconstruction does not establish recovery across a one-second occlusion.
# Boundary extrapolation, partial-axis gaps and missing-row insertion are separate
# tasks and remain outside the controlled comparison. Natural gap profiles also
# describe what is missing, not what the unobserved motion truly was.

# %%
complete = results["leaderboard"].query("complete_matches == matches and balanced_ade == balanced_ade")
shortlist = complete.method.head(5).tolist()
shortlist = list(dict.fromkeys(["linear_5", "linear_30", *shortlist]))
fig = plot_error_by_length(scores, shortlist)
fig.savefig(FIGURES / "error_and_coverage_by_length.png", dpi=160, bbox_inches="tight")
plt.show()
display(scores.groupby(["method", "gap_length"]).agg(
    attempts=("case_id", "count"), fill_fraction=("fill_fraction", "mean"),
    player_tempo_mae=("player_tempo_abs_error", "mean"),
    player_time_coverage=("player_time_coverage", "mean"),
    tempo_filter_changes=("tempo_range_inclusion_changed", "sum"),
))

# %% [markdown]
# ## Check difficult contexts and uncertainty
#
# Tail and curvature strata are calculated using the reference for diagnosis
# only. They cannot choose a method at a real hidden gap. Natural missingness
# may depend on unobserved motion; even good artificial-mask results do not prove
# performance during real occlusions.
#
# For Gaussian observation predictions we report negative log predictive density,
# $-\log N(\mathbf p_t;\mu_t,\Sigma_t)$, and marginal 50/80/95% coverage/width.
# Joint 2D coverage uses the Mahalanobis distance and a $\chi^2_2$ threshold;
# two marginal 95% intervals do not define a joint 95% region. These intervals
# predict provider observations conditional on the local model, not calibrated
# physical truth. Excess width and poor coverage both warrant investigation.

# %%
primary = scores[scores.gap_length <= 5].copy()
primary["motion_stratum"] = np.select(
    [primary.turn_radians > np.pi / 4, primary.max_reference_speed > 15],
    ["reference sharp turn", "reference fast motion"], default="other",
)
display(primary.groupby(["method", "motion_stratum"]).agg(
    gaps=("case_id", "count"), matches=("game_id", "nunique"),
    ade=("ade", "mean"), speed_mae=("speed_mae", "mean"),
    player_tempo_mae=("player_tempo_abs_error", "mean"),
))
uncertainty_columns = [c for c in primary if c.startswith(("coverage_", "width_")) or c == "observation_nlpd"]
display(primary.groupby("method")[uncertainty_columns].mean().dropna(how="all"))
display(primary.groupby("method")[uncertainty_columns].count().rename(
    columns={column: f"evaluated_gaps_{column}" for column in uncertainty_columns}
))

# %% [markdown]
# The provider distinguishes VISIBLE and ESTIMATED coordinates. Our expanded
# cohort retains both rather than assuming non-null means independently observed.
# The next table separates windows whose complete context is labelled VISIBLE.
# Each quality/match/length cell is reported with its sample support; absence of
# a stratum must not be interpreted as zero error or full representativeness.

# %%
quality = pd.read_csv(RUN_DIRECTORY / "quality_stratum_metrics.csv")
display(quality.groupby(["reference_quality", "method"]).agg(
    supported_match_length_cells=("ade", "count"), matches=("game_id", "nunique"),
    gaps=("gaps", "sum"), mean_cell_ade=("ade", "mean"),
    player_tempo_mae=("player_tempo_abs_error", "mean"),
))

# %% [markdown]
# ## Inspect typical and difficult trajectories
#
# We mechanically select cases near the median and lower decile of improvement
# over linear. This avoids showing only attractive reconstructions. A method
# can improve position error yet smooth a true touch, inflate the path, or distort
# a short possession's tempo. Inspect the trajectory and time traces together.

# %%
candidate = next((name for name in shortlist if not name.startswith("linear_")), None)
if candidate is not None:
    for label, case_id in representative_cases(scores, candidate).items():
        display(Markdown(f"### {label}"))
        fig = plot_gap_example(RUN_DIRECTORY, case_id, ["linear_5", candidate])
        fig.savefig(FIGURES / f"example_{label.split()[0].lower()}.png", dpi=160, bbox_inches="tight")
        plt.show()

# %% [markdown]
# ## Evidence and implementation status
#
# | Plan element | Initial implementation |
# | --- | --- |
# | Source audit, immutable match split, FPS provenance | Implemented; test tracking unopened |
# | Contiguous 1–5-frame masks, longer-gap stress probes, five seeds | Implemented with per-length support counts |
# | Linear, hold/nearest, PCHIP, cubic/Hermite, CV/CA RTS, Matérn GP | Implemented on a predeclared local parameter grid |
# | Actual production linear and downstream metric functions | Reused; controlled preprocessing is explicitly period-isolated |
# | Training-only natural weights/configuration selection | Implemented with disjoint match validation folds |
# | Paired match bootstrap, saved masks/predictions/provenance | Implemented |
# | Independent video truth and informative missingness validation | Pending external/reference evidence |
# | Joint multi-gap masks, dense lengths, hybrid and maximum-gap policies | Implemented in the follow-up cells below |
# | Partial-axis, absent-row and boundary extrapolation tasks | Separate investigations |
# | Noise/amplitude expansion, height/pressure/crowding strata | Follow-up where data/support justify them |
# | Locked final test | Executable below after freezing one candidate and scientific limits |
# | Pipeline promotion | Follows confirmation and diagnostic review |
#
# Development results determine whether we need a denser grid or follow-up masks.
# A promotion decision requires meaningful coordinate improvement at matched
# coverage, acceptable speed/tempo/tail behavior, repeatable match performance,
# and independent confirmation. If the differences are negligible, retaining
# linear is a useful result. The final change belongs directly in the production
# pipeline after this evidence is established; research code alone is not deployment.
#
# References: [dependence-aware validation, Roberts et al.](https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file),
# [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data),
# [SciPy PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html),
# [Särkkä: filtering and smoothing, chapter 8](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf),
# [Rasmussen and Williams: GP regression](https://gaussianprocess.org/gpml/chapters/).

# %%
display(Markdown((RUN_DIRECTORY / "results.md").read_text()))

# %% [markdown]
# ## Decision-focused follow-up: method AND maximum gap
#
# The screen identified PCHIP as promising, especially for longer gaps, while
# linear did better for a single missing frame. We now compare three complete
# policies at each declared cap $c\in\{5,10,15,20,30,45\}$:
#
# $$f_{hybrid,c}(G)=\begin{cases}
# f_{linear}(G),& |G|=1,\\
# f_{PCHIP}(G),&2\leq |G|\leq c,\\
# \text{leave missing},&|G|>c.
# \end{cases}$$
#
# Pure linear and pure PCHIP use the same cap. PCHIP has an explicit linear
# fallback if fewer than two observed context points are available on either
# side. All policies preserve observed coordinates and require two-sided
# anchors. The hybrid uses only the observed gap length, never hidden curvature.
# The .25-second context is fixed from the screening investigation; this is
# an adaptive development follow-up, not a new independent confirmation.
#
# Single-gap masks cover EVERY integer length from 1 through 45, using three
# new reproducible seeds. This separates a local improvement at a tested length
# from a defensible maximum-gap policy. Durations in seconds are recorded using
# match-specific FPS. If the provider frame rate changes, frame caps must be
# reconsidered in time units.
#
# At a given cap all methods receive identical eligible cases. Across caps,
# eligible cohorts differ: a five-frame score cannot establish superiority over
# a thirty-frame policy. The common-design coverage table exposes that difference.

# %%
display(pd.DataFrame([vars(POLICY_CONFIG)]).T.rename(columns={0: "setting"}))
if RUN_POLICY_EXPERIMENTS:
    POLICY_RUN_DIRECTORY = run_policy_validation(POLICY_CONFIG, study=CONFIG, root=ROOT)
elif POLICY_RUN_DIRECTORY is None:
    raise ValueError("Specify POLICY_RUN_DIRECTORY or enable RUN_POLICY_EXPERIMENTS.")

POLICY_RUN_DIRECTORY = Path(POLICY_RUN_DIRECTORY)
if not POLICY_RUN_DIRECTORY.is_absolute():
    POLICY_RUN_DIRECTORY = ROOT / POLICY_RUN_DIRECTORY
policy_results = load_policy_validation(POLICY_RUN_DIRECTORY)
policy_scores = policy_results["scores"]
POLICY_FIGURES = POLICY_RUN_DIRECTORY / "figures"
POLICY_FIGURES.mkdir(exist_ok=True)
display(Markdown(f"Policy validation: `{POLICY_RUN_DIRECTORY}`"))
policy_support = pd.read_csv(POLICY_RUN_DIRECTORY / "mask_support.csv")
display(policy_support.groupby("experiment").agg(
    requested=("requested", "sum"), sampled=("sampled", "sum"),
    supported_matches=("game_id", "nunique"),
))
display(policy_support[policy_support.sampled < policy_support.requested])

# %% [markdown]
# ## Multiple gaps: measure cumulative possession distortion
#
# For one player possession, hide $B\in\{2,4\}$ separated blocks together:
#
# $$G=G_1\cup\cdots\cup G_B,\qquad
# \widehat{\mathbf p}_{G_b}=f(\mathbf p_{\setminus G},G_b).$$
#
# Every block sees the same fully masked input. Predictions from an earlier
# block never become an observed input for another block. Blocks retain at least
# one observed frame between them and cannot cross a player/team/period boundary.
# Sample block lengths uniformly from 1 through each candidate cap (5, 10, 15,
# 20, 30 and 45 frames) in separate scenarios. These are stress designs;
# they do not claim to reproduce the natural
# joint missingness process. Report requested and achieved support explicitly.
#
# We reuse production displacement and tempo functions over the entire affected
# bridge, including its observed interior and both boundary connections, and
# apply the resulting distance/time change to the original full possession.
# Frame-speed error uses ONLY segments touching hidden coordinates, preventing
# long untouched spans between blocks from diluting the speed error.
#
# Team-tempo results describe the effect of corrupting one player possession
# within that team possession. Simultaneous corruption across multiple player
# possessions and possession-boundary errors remain additional pipeline questions.

# %%
tempo_columns = [
    "gap_limit", "method", "experiment", "ade", "speed_mae", "fill_fraction",
    "player_tempo_abs_error", "player_tempo_bias", "player_tempo_rmse", "player_tempo_p95",
    "team_tempo_abs_error", "team_tempo_bias", "team_tempo_rmse", "team_tempo_p95",
    "player_tempo_available_fraction", "team_tempo_available_fraction",
]
display(policy_results["leaderboard"][tempo_columns])

# %% [markdown]
# ## What does possession-tempo accuracy mean?
#
# For each possession and mask realization, compute reference tempo from the
# ORIGINAL supplied coordinates and reconstructed tempo after masking/filling:
#
# $$e_{pr}=\widehat T_{pr}-T_p^{ref},\qquad
# MAE=E[|e_{pr}|],\quad Bias=E[e_{pr}],\quad RMSE=\sqrt{E[e_{pr}^2]}.$$
#
# MAE measures typical distortion, signed bias detects systematic speed-up or
# slow-down, and RMSE emphasizes large errors. In the equations above the
# expectation is within a match; the table averages match RMSEs equally:
# $RMSE_{reported}=M^{-1}\sum_m\sqrt{MSE_m}$. This differs from taking the
# square root after pooling all matches. We also report a 95th percentile
# of absolute tempo error. For MAE/bias/MSE we first average mask realizations
# within a possession, then possessions within a scenario, scenarios within a
# match, and finally matches equally. Team metrics use team-possession weights.
#
# The reported `player_tempo_p95` and `team_tempo_p95` first calculate a
# possession-weighted 95th percentile of realization errors WITHIN each
# match/scenario and then average those percentiles. They are not the pooled
# global 95th percentile. Repeated seeds do not create independent possessions.
# `p95_error` similarly summarizes the within-mask coordinate percentile.
#
# Undefined tempo is never assigned zero error: availability and represented-time
# coverage accompany every score. A low MAE with missing predictions cannot
# establish a better policy. Conversely, aggregate tempo can remain accurate
# while the intermediate trajectory is wrong, so retain coordinate and affected-
# segment speed scores alongside tempo.
#
# Single-gap effects may be diluted over a long possession. The joint-mask
# experiments show whether multiple reconstructions accumulate enough error to
# change the scientific outcome. These metrics use the supplied signal as the
# reference; the provider's own smoothing/estimation remains a limitation.

# %%
coverage_columns = ["fill_fraction", "complete_case_fraction", "player_time_coverage",
                    "team_time_coverage", "player_tempo_available_fraction", "team_tempo_available_fraction"]
display(policy_results["common"].groupby(["method", "experiment"])[coverage_columns].mean())
for displayed_candidate, cap in (("hybrid_5", 5), ("hybrid_30", 30)):
    if displayed_candidate in set(policy_scores.method):
        fig = plot_policy_validation(policy_scores, displayed_candidate, limit=cap)
        fig.savefig(POLICY_FIGURES / f"{displayed_candidate}_coordinates_speed_tempo.png", dpi=160, bbox_inches="tight")
        plt.show()

# %% [markdown]
# ## Paired uncertainty and development selection
#
# Compare each candidate against linear at the SAME cap, using identical cases.
# Paired intervals resample entire matches; all masks, methods and realizations
# within a match remain together. A negative difference favors the candidate.
# We calculate intervals for coordinate ADE/tails, speed MAE, player/team tempo
# MAE and player/team tempo tails. This makes the downstream tradeoff explicit.
#
# Five development folds choose the family independently at each FIXED cap,
# minimizing the equally weighted mean of single-gap and joint-gap coordinate
# ADE on training matches. The held-out fold assesses that choice. This objective
# is declared here; tempo and tails are scientific guardrails, not silently
# optimized after looking at validation outcomes. Cap selection still requires
# absolute acceptable error limits because the covered data change with the cap.
# The earlier screen's natural-gap-weighted short-gap results remain relevant;
# these follow-up joint scenarios deliberately stress missingness rather than
# estimating its prevalence. All-development family comparisons remain exploratory.

# %%
display(policy_results["paired"])
display(pd.read_json(POLICY_RUN_DIRECTORY / "fold_policy_choices.json"))
selected_oof_policy = pd.read_csv(POLICY_RUN_DIRECTORY / "selected_policy_oof.csv")
if len(selected_oof_policy):
    display(selected_oof_policy.groupby(["gap_limit", "experiment"])[
        ["ade", "speed_mae", "player_tempo_abs_error", "team_tempo_abs_error"]].mean())
frontier = decision_frontier(policy_results["scoped"])
display(frontier[~frontier.dominated])

# %% [markdown]
# The frontier retains methods for which no other method at the same cap improves
# every displayed coordinate/speed/tempo metric. It describes tradeoffs rather
# than declaring a winner. We do not choose among caps using the smallest error:
# filling fewer, easier gaps naturally produces smaller errors.

# %% [markdown]
# ## Difficult contexts and visibly observed references
#
# Inspect separately: all-context VISIBLE, all-targets VISIBLE, fast reference
# motion (>15 m/s), sharp reference turns (>45 degrees), and provider possession
# event labels. Strata overlap. Speed/turn labels use hidden reference values
# only for diagnosis; they cannot choose a method at an actual missing gap.
# Event labels are not independently annotated release/contact times.
#
# Check both average performance and support. A missing stratum is unknown
# performance. The all-context VISIBLE restriction is stricter for the full
# segment retained in joint cases than for a local single-gap window. Even
# VISIBLE provider-smoothed coordinates are not independent physical truth.
# Natural occlusions may depend on the unobserved trajectory. Independent video
# annotation or another reference system is still needed to quantify that gap.

# %%
stratum_summary = policy_results["strata"].groupby(
    ["stratum", "gap_limit", "method", "experiment"], observed=True,
).agg(matches=("game_id", "nunique"), cases=("cases", "sum"),
      ade=("ade", "mean"), speed_mae=("speed_mae", "mean"),
      player_tempo_mae=("player_tempo_abs_error", "mean"),
      player_tempo_p95=("player_tempo_p95", "mean"),
      fill_fraction=("fill_fraction", "mean"))
display(stratum_summary)
natural_profile = pd.read_csv(POLICY_RUN_DIRECTORY / "natural_gaps.csv")
display(natural_profile.groupby("gap_length").agg(
    gaps=("game_id", "count"), matches=("game_id", "nunique"),
    mean_missing_seconds=("missing_seconds", "mean"),
))
cap_eligibility = pd.DataFrame([
    {"gap_limit": cap,
     "bounded_natural_gaps_within_cap": int((natural_profile.gap_length <= cap).sum()),
     "bounded_natural_gap_fraction_within_cap": float((natural_profile.gap_length <= cap).mean()),
     "bounded_missing_frame_fraction_within_cap": float(
         natural_profile.loc[natural_profile.gap_length <= cap, "gap_length"].sum()
         / natural_profile.gap_length.sum())}
    for cap in policy_results["manifest"]["config"]["limits"]
])
display(cap_eligibility)

# %% [markdown]
# The cap-eligibility table is an optimistic ceiling based only on gap length
# within the derived bounded-gap cohort. It is not measured successful recovery:
# natural gaps may lack suitable context, and their true coordinates are unknown.
# Gap-count coverage and missing-frame coverage answer different questions. A
# small number of very long gaps can contain most missing frames, so filling a
# majority of gaps does not imply recovering a majority of missing tracking time.

# %% [markdown]
# ## Freeze a scientific decision, then confirm once
#
# Choose one complete candidate policy and numerical limits BEFORE opening the
# 13 reserved matches. `DecisionCriteria` requires an absolute meaningful ADE
# gain, absolute coordinate/speed/tempo limits, and allowed paired speed/tempo
# degradation. No default scientific tolerance is invented. The earlier .1 m/s
# example was explanatory, not an acceptance criterion.
#
# A replacement must satisfy the predeclared criteria in BOTH single-gap and
# joint-gap experiments. Paired upper 95% interval bounds assess improvement and
# noninferiority against linear at the same cap. Absolute error guardrails use
# one-sided 95% match-bootstrap upper means. These are simultaneous requirements,
# not a search for any favorable metric. Visible-only/event strata require review
# even when aggregate criteria pass. Match resampling does not remove dependence
# between matches sharing teams or establish transfer to another competition.
#
# Expanding the cap is a separate scientific decision: inspect absolute long-gap
# errors and common-design coverage relative to production `linear_5`. The cap
# comparator `linear_c` isolates the interpolation method's contribution.
#
# Set SCIENTIFIC_CRITERIA to a `DecisionCriteria(...)`, CONFIRMATION_CANDIDATE
# to a studied policy name, and FREEZE_CONFIRMATION=True to write an immutable
# protocol. This operation does not read test frames. Set RUN_FINAL_TEST=True
# only when ready to consume the frozen holdout. The runner refuses changed
# code/splits/protocols and a second holdout opening, including after a failed run.
# A successful statistical assessment still requires diagnostic review before
# integrating the supported policy into the production pipeline.

# %%
protocol_path = ROOT / CONFIG.output / "confirmation_protocol.json"
if FREEZE_CONFIRMATION:
    if SCIENTIFIC_CRITERIA is None or CONFIRMATION_CANDIDATE is None:
        raise ValueError("Supply explicit scientific criteria and one candidate before freezing.")
    protocol_path = freeze_confirmation(
        POLICY_RUN_DIRECTORY, CONFIRMATION_CANDIDATE, SCIENTIFIC_CRITERIA, root=ROOT,
    )
if RUN_FINAL_TEST:
    if not protocol_path.exists():
        raise ValueError("Freeze a complete confirmation protocol first.")
    confirmation_directory = run_confirmation(protocol_path, root=ROOT)
    display(pd.read_csv(confirmation_directory / "confirmation_assessment.csv"))
else:
    display(Markdown(
        "Development policy validation is complete for the displayed run. "
        "Final confirmation awaits one frozen candidate, scientific error limits, "
        "and review of the diagnostic strata."
    ))
