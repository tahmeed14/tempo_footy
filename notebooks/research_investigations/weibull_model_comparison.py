# %% [markdown]
# # Weibull BST: distribution and prior sensitivity
#
# Implements the initial modeling/evaluation stages of
# `bst_distribution_comparison_plan.md`. Helpers live in `tempoctrl.research`.
# The CMSAC workflow and its posterior are independent of these experiments.
#
# Use the project environment, for example from the repository root:
#
# ```bash
# .venv/bin/python notebooks/research_investigations/weibull_model_comparison.py --mode prepare
# .venv/bin/python notebooks/research_investigations/weibull_model_comparison.py --mode prior --prior all
# .venv/bin/python notebooks/research_investigations/weibull_model_comparison.py --mode fit --prior all
# .venv/bin/python notebooks/research_investigations/weibull_model_comparison.py --mode evaluate --run /path/to/saved/run
# ```
#
# The default only prepares data and specifications. `prior` simulates the
# proposed priors; `fit` runs prior checks, MCMC and evaluation. Inspect prior
# plausibility before starting substantive fits. `evaluate` reads the saved
# snapshot and draws and never refits. All paths resolve from the repository,
# even when launched from this directory. Imports never start sampling.

# %%
import sys

from tempoctrl.research.bst_workflow import RunConfig, cli, run_family
from tempoctrl.research.bst_evaluation import compare_saved_runs

# %% [markdown]
# ## Data and reproducibility
#
# `--data` selects any updated processed parquet file; the default is
# `data/analysis/model_data_vFINAL.parquet`. Expected units are m/s, following
# the upstream metric. The audit records this assumption, not a unit test of
# physical accuracy. Observational keys must be unique. Missing model inputs
# and nonpositive/nonfinite outcomes are excluded and logged; no arbitrary
# positive offset or additional outlier cutoff is applied.
#
# Every family uses the same canonical row ordering and categorical contrasts.
# The log-sequence z score is recomputed after exclusions. An input hash,
# cohort hash, row IDs, exclusions, formulas, priors, package versions, source
# snapshots, seeds and sampling settings identify each run. New input files
# produce new artifact paths; repeated executions also receive unique paths.
# Artifacts live in `paper/results/bst_distribution_comparison/`.

# %% [markdown]
# ## Model formulation
#
# Following the manuscript's indexing, let $j[i]$ identify the player and
# $k[i]$ the opposition team for possession $i$:
#
# $$\log(\mu_i)=\beta_0+\mathbf{x}_i^{\mathsf{T}}\boldsymbol{\beta}
# +u_{j[i]}+v_{k[i]},$$
# $$\zeta_i=\gamma_0+\mathbf{z}_i^{\mathsf{T}}\boldsymbol{\gamma}+a_{j[i]}.$$
# The family-specific definition of $\zeta_i$ is given below; in the
# manuscript's Gamma model it is $\log(\alpha_i)$.
#
# $X$ contains goal difference, ball height, pressure, defender challenges,
# pitch third, position group, standardized log possession sequence and period.
# $Z$ contains pressure, defender challenges, pitch third and position group.
# The player and opponent effects use independent noncentered hierarchies.
# Shared predictors and arithmetic-mean semantics isolate distribution choice.
#
# $Y_i\mid k_i,\lambda_i\sim\operatorname{Weibull}(k_i,\lambda_i)$,
# with $\log k_i=\zeta_i$ and
# $$\lambda_i=\frac{\mu_i}{\Gamma(1+1/k_i)}.$$
#
# $$E[Y_i]=\mu_i,\qquad
# CV[Y_i]=\sqrt{\frac{\Gamma(1+2/k_i)}{\Gamma(1+1/k_i)^2}-1},\qquad
# SD[Y_i]=\mu_i CV[Y_i].$$
#
# Here $\Gamma$ is the mathematical gamma function. This scale correction
# keeps the arithmetic mean predictor comparable to the other families.
# Higher log-shape means lower relative variability.

# %% [markdown]
# ## Proposed priors and sensitivity configurations
#
# $$\beta_0\sim N(\log 7,s_0),\quad \beta_j\sim N(0,s_\beta),\quad
# u_j\sim N(0,\sigma_{\text{player}}),\quad
# v_k\sim N(0,\sigma_{\text{opponent}}),\quad
# \sigma_{\text{player}},\sigma_{\text{opponent}}
# \sim\operatorname{HalfNormal}(s_\beta).$$
# Each Normal's second argument denotes its standard deviation.
#
# Each group has its own scale parameter. Dispersion uses analogous Normal
# priors for its intercept/context effects and HalfNormal group-scale priors.
# Baseline centers come from the existing CMSAC priors, possibly informed by
# earlier analysis of these outcomes; they are not independent elicitation.
# The mean center is 7 m/s and the CV center is $1/\sqrt{3}$ in every family.
#
# | Configuration | Mean intercept SD | Mean effect / group-scale SD | Approx. log-CV intercept SD | Approx. log-CV effect / group-scale SD |
# | --- | --- | --- | --- | --- |
# | baseline | 0.40 | 0.30 | 0.25 | 0.150 |
# | stronger | 0.25 | 0.15 | 0.15 | 0.075 |
# | weaker | 0.60 | 0.45 | 0.35 | 0.225 |
#
# Dispersion centers solve the common reference CV; widths divide the above
# log-CV widths by $|d\log(CV)/d\zeta|$ at that center. This is a local
# calibration, not equivalence of complete prior distributions. Exact values
# for all nine family/configuration combinations are saved in `priors.csv`.
# Hold centers fixed when testing widths. Review near-zero behavior, tails,
# contextual contrasts and player variability in `prior_checks.md` and its
# CSV/figure before accepting a configuration. Revise implausible priors and
# record outcome-informed calibration as such. Weaker priors are not assumed
# acceptable merely because a sampler can fit them.

# %%
# Interactive settings: edit these before running the final cell. In a terminal,
# use the corresponding CLI flags instead. Full posterior fits are opt-in.
CONFIG = RunConfig(
    family="weibull",
    mode="prepare",  # "prior" or "fit" when ready
    prior="all",  # "baseline", "stronger", "weaker", or "all"
    data="data/analysis/model_data_vFINAL.parquet",
    seed=42,
    draws=1000,
    tune=1500,
    chains=4,
    cores=4,
    target_accept=0.90,
    prior_draws=200,
    predictive_draws=200,  # per chain; LL uses every posterior draw
)

# %% [markdown]
# ## Evaluation: metric and validation rationale
#
# The primary predictive metric is expected log predictive density:
# $$\operatorname{ELPD}_{LOO}=\sum_i\log p(y_i\mid y_{-i}).$$
# Higher is better. This proper score evaluates the full predictive density,
# including dispersion and tails, which point-prediction RMSE cannot assess.
# All likelihoods are evaluated in original BST units. Score differences need
# paired uncertainty, not just an ordering of totals.
#
# Observation-level PSIS-LOO is an efficient approximation for another
# possession conditional on represented players/opponents and observed contexts.
# Record Pareto-k, divergences, R-hat and ESS. Require four chains, zero
# divergences, R-hat <=1.01, bulk/tail ESS >=400 and acceptable Pareto-k before
# interpreting comparisons. Bad Pareto-k requires influential-observation
# refits or suitable K-fold validation; automatic refits are deferred.
#
# New-match inference requires match-grouped refits; future-match inference
# requires chronological splits. Ordinary LOO supports neither claim. Fit
# learned preprocessing within training folds for those future extensions.
# The present z score uses only predictors in the frozen observed-design cohort.
# Shared effects and rare players may make individual-observation deletion
# difficult; inspect rather than suppress the diagnostics.
#
# Also inspect pooled/tail and context-specific replicated statistics, player
# empirical CV checks, and 50/80/95% predictive interval coverage. Coverage here
# is in-sample PPC, not held-out validation. Predictive draw subsampling is
# recorded; likelihood draws are never subsampled. Memory is estimated before
# allocating the large arrays; override `--max-working-gib` deliberately.

# %% [markdown]
# ## Results and inferences
#
# Each completed fit writes `results.md` with actual diagnostics, ELPD,
# draw-wise reference mean/SD/CV summaries and detected predictive discrepancies.
# No results are invented before fitting. Use `methods.md` for full assumptions.
# A fit with failed diagnostic gates cannot establish a preferred distribution.
# Favorable relative ELPD does not erase predictive deficiencies.
#
# For saved fits on the same snapshot and implementation, evaluate prior
# sensitivity within a family and family sensitivity within a prior separately:
#
# ```python
# comparison = compare_saved_runs(
#     ["/path/to/baseline/run", "/path/to/alternative/run"],
#     "/path/to/comparison/output",
# )
# ```
#
# This function checks fingerprints and row alignment, reports paired score
# differences and diagnostic gates, and never fits a model. Do not mix old
# interpolation/outlier snapshots into a family comparison.
#
# Context-adjusted dispersion does not imply deliberate tempo changes or
# tactical adaptability. SD/CV describe outcome variation; credible intervals
# describe uncertainty about those quantities. Reference moments transform
# every draw before summarizing. Player empirical CVs in PPCs mix contexts;
# common-context player ranking/archetype inference, sparse-player sensitivity,
# targeted component refits and match-grouped validation remain later stages
# of the plan. A full sensitivity analysis needs all nine reliable fits.
#
# Distribution formulas: [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/api/distributions.html).
# Validation: [PSIS-LOO and diagnostics](https://mc-stan.org/loo/articles/loo2-example.html).

# %%
if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        RUNS = run_family(CONFIG)
        from IPython.display import Markdown, display

        for run in RUNS:
            report = run / ("results.md" if CONFIG.mode == "fit" else "prior_checks.md" if CONFIG.mode == "prior" else "methods.md")
            display(Markdown(report.read_text()))
    else:
        cli("weibull")
