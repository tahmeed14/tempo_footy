# %% [markdown]
# # Gamma BST model: a sequential walkthrough
#
# Work through these cells in order, as in `notebooks/cmsac_model.py`.
# The sequence is: inspect data -> choose priors -> construct the model ->
# simulate before fitting -> fit -> diagnose -> predict -> evaluate -> interpret.
#
# Model construction and sampler calls are visible below. Shared helpers keep
# data checks, contrasts and saved artifacts consistent with the comparison
# scripts. We use PyMC directly to give every family the same arithmetic-mean
# predictor and a separate regression for dispersion.
#
# **First pass:** the defaults inspect the data and build the model without
# sampling. Turn on `RUN_PRIOR_PREDICTIVE`, run the prior-check cells and review
# them. Then turn on `RUN_MCMC` and continue from the fitting section.
# For another prior configuration, change `PRIOR_NAME` and rerun from settings.
# Each fresh run gets its own directory. If you edit a model cell, rebuild from
# "Start the model" before rerunning its subsequent construction cells.
#
# Updated outlier filtering or interpolation belongs in the processed parquet.
# Change `DATA_PATH` as needed and rerun from settings to create a new snapshot.
# A saved fit can be explored using `SAVED_RUN_DIRECTORY` without refitting.

# %%
# Installed packages
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from IPython.display import Image, Markdown, display

# Shared research helpers; importing these never starts fitting.
from tempoctrl.research.bst_models import (
    OBSERVATION_KEY,
    OUTCOME_VARIABLE,
    REFERENCE_LEVELS,
    conditional_moments,
    prepare_data,
    prior_spec,
    repository_root,
)
from tempoctrl.research.bst_workflow import RunConfig, memory_estimate, prior_table
from tempoctrl.research.bst_evaluation import (
    compare_saved_runs,
    diagnose,
    loo_diagnostics,
    predictive_array,
    predictive_checks,
    reference_moments,
    save_predictive_plot,
)
from tempoctrl.research.bst_interactive import (
    check_interactive_memory,
    load_interactive_snapshot,
    start_interactive_run,
    update_interactive_status,
    write_interactive_results,
)

# %% [markdown]
# ## 1. Settings and a reproducible analysis snapshot
#
# Start with one named prior so that each intermediate result is easy to inspect.
# Baseline, stronger and weaker configurations will eventually need separate
# fits. A complete distribution/prior comparison has $3\times3=9$ fits.
# Seeds reproduce random simulations; fingerprints identify the actual data
# and implementation. A seed alone does not identify an analysis.
#
# To revisit a fit, set `SAVED_RUN_DIRECTORY` to its run directory and leave
# both sampling flags False. Its original data and settings will be restored,
# even if the current processed file has changed.

# %%
FAMILY = "gamma"
PRIOR_NAME = "baseline"  # "stronger" or "weaker" for a separate sensitivity run
ROOT = repository_root()
DATA_PATH = ROOT / "data/analysis/model_data_vFINAL.parquet"
SAVED_RUN_DIRECTORY = None  # e.g. Path("/absolute/path/to/a/saved/run")
RUN_PRIOR_PREDICTIVE = False
RUN_MCMC = False
COMPARISON_RUNS = []  # Optional: at least two completed, aligned run directories

CONFIG = RunConfig(
    family=FAMILY,
    prior=PRIOR_NAME,
    data=DATA_PATH,
    output=ROOT / "paper/results/bst_distribution_comparison",
    seed=42,
    draws=1000,
    tune=1500,
    chains=4,
    cores=4,
    target_accept=0.90,
    prior_draws=200,
    predictive_draws=200,  # Per chain; all posterior draws are used for LOO.
    max_working_gib=8.0,
)

# %% [markdown]
# ### Read, validate and freeze the data
#
# `prepare_data` reuses the module loader, category orders, and log-sequence
# transformation. It checks unique possession IDs; logs exclusions for missing
# inputs, invalid sequences and nonpositive/nonfinite BST; sorts a common cohort;
# and recomputes $z_i=[\log(1+s_i)-\overline{\log(1+s)}]/SD[\log(1+s)]$.
# Units are assumed to be m/s, following the upstream metric.
#
# We apply no additional outlier cutoff here. This lets revised upstream
# filtering/interpolation carry through every family on the same observations.
# `start_interactive_run` only freezes artifacts and source files; it does not
# construct or fit the model. The learning steps below do that explicitly.

# %%
if SAVED_RUN_DIRECTORY is None:
    prepared = prepare_data(CONFIG.data)
    RUN_DIRECTORY = start_interactive_run(prepared, CONFIG)
else:
    RUN_DIRECTORY = Path(SAVED_RUN_DIRECTORY).resolve()
    prepared, CONFIG = load_interactive_snapshot(RUN_DIRECTORY, FAMILY)
    PRIOR_NAME = CONFIG.prior

model_pd_df = prepared.frame
print(f"Artifacts: {RUN_DIRECTORY}")
display(pd.Series(prepared.audit, name="audit"))
display(prepared.exclusions.head(10))

# %% [markdown]
# ## 2. Descriptive statistics
#
# Check the sample size, outcome range, tails and player observation counts
# before modeling. A right-skewed positive histogram motivates these families,
# but cannot select a conditional model: contexts and players also shape the
# pooled distribution. Sparse players will depend more on hierarchical shrinkage.

# %%
display(pd.Series({
    "possessions": len(model_pd_df),
    "matches": model_pd_df["game_id"].nunique(),
    "players": model_pd_df["player_id"].nunique(),
    "opponents": model_pd_df["opponent_id"].nunique(),
}))
display(model_pd_df[[OUTCOME_VARIABLE, "player_possession_sequence_log_z"]].describe())
display(model_pd_df.groupby("player_id", observed=True).size().describe().rename("rows_per_player"))

# %%
model_pd_df[OUTCOME_VARIABLE].plot.hist(bins=60, figsize=(8, 4))
plt.xlabel("Ball-speed tempo (m/s)")
plt.ylabel("Number of possessions")
plt.title("Observed BST in this frozen analysis cohort")
plt.tight_layout()
plt.show()
plt.close()

# %% [markdown]
# ### Inspect categorical reference levels and design matrices
#
# A reference level gets no dummy coefficient. A coefficient for another level
# describes its contrast with that reference, holding other variables fixed.
# Ordered pandas categories preserve the repository's reference choices.
# Unobserved nonreference levels have no likelihood information and are omitted.
#
# $X$ contains goal difference, log-standardized possession sequence, first-touch
# height and pressure, defender challenges, pitch third, position and period.
# $Z$ contains pressure, defender challenges, pitch third and position.

# %%
for column, levels in REFERENCE_LEVELS.items():
    print(f"{column}: reference={levels[0]!r}; levels={levels}")

X = prepared.mean_design.to_numpy(dtype=float)
Z = prepared.dispersion_design.to_numpy(dtype=float)
y = model_pd_df[OUTCOME_VARIABLE].to_numpy(dtype=float)
display(prepared.mean_design.head())
display(prepared.dispersion_design.head())
print(f"X: {X.shape}; Z: {Z.shape}; outcome: {y.shape}")

# %% [markdown]
# ## 3. Define what the model should estimate
#
# $Y_i$ is BST for possession $i$, $p[i]$ identifies its player, and $o[i]$
# identifies its opponent. We model **conditional arithmetic mean** and
# **conditional dispersion** separately:
#
# $$\eta_i=\log\mu_i=\beta_0+X_i\beta+u_{p[i]}+v_{o[i]},$$
# $$\zeta_i=\delta_0+Z_i\delta+w_{p[i]}.$$
#
# Mean and dispersion player effects are distinct, independent hierarchies.
# Thus a player can have higher adjusted mean BST without necessarily having
# higher relative variability. We retain the CMSAC predictor sets to isolate
# distribution and prior sensitivity from predictor selection.
#
# $$Y_i\sim\operatorname{Gamma}(\alpha_i,\alpha_i/\mu_i),\qquad
# \alpha_i=\exp(\zeta_i).$$
#
# The second Gamma parameter is a **rate**. This gives
# $$E[Y_i]=\mu_i,\quad SD[Y_i]=\mu_i/\sqrt{\alpha_i},\quad
# CV[Y_i]=1/\sqrt{\alpha_i}.$$
# Increasing log-shape reduces relative variability. The mean stays fixed
# when shape changes, because the rate changes with it.

# %% [markdown]
# ## 4. Specify and inspect the priors
#
# $$\beta_0\sim N(\log 7,s_0),\quad \beta_j\sim N(0,s_\beta),$$
# $$u_p\sim N(0,\tau_p^2),\quad v_o\sim N(0,\tau_o^2),\quad
# \tau_p,\tau_o\sim\operatorname{HalfNormal}(s_\beta).$$
#
# Dispersion uses analogous priors on its natural log-parameter. Each group
# scale is a separate random variable. The prior center 7 is the median of the
# induced prior for the reference conditional mean, not $E[\mu]=7$.
# Centers come from CMSAC's existing mean 7 / Gamma shape 3 specification;
# they may reflect previous analysis of these outcomes, rather than independent
# elicitation. All families have reference CV center $1/\sqrt3$.
#
# | Prior | Mean intercept SD | Mean effect/group SD | Approx. log-CV intercept SD | Approx. log-CV effect/group SD |
# | --- | --- | --- | --- | --- |
# | baseline | 0.40 | 0.30 | 0.25 | 0.150 |
# | stronger | 0.25 | 0.15 | 0.15 | 0.075 |
# | weaker | 0.60 | 0.45 | 0.35 | 0.225 |
#
# Shape and log-sigma are different scales. `prior_spec` solves for the common
# CV center and converts widths using the local slope
# $|d\log(CV)/d\zeta|$. Local width matching does not make full prior predictive
# distributions equivalent. These specifications remain proposed until their
# original-scale predictions are judged plausible. Centers stay fixed while
# widths change; outcome-informed calibration must be labeled as such.

# %%
priors = prior_spec(FAMILY, PRIOR_NAME)
display(pd.Series(priors.as_dict(), name="selected_prior"))
all_priors = prior_table()
display(all_priors.loc[all_priors["family"] == FAMILY])

# %% [markdown]
# ## 5. Start the model and define mean priors
#
# Named coordinates map coefficient arrays to terms, players and observations.
# PyMC registers random variables inside `with model:`. We return to this same
# model across several cells so each piece can be inspected separately.
# Recreate `model` here before rerunning construction cells after an edit.

# %%
coords = {
    "observation": model_pd_df[OBSERVATION_KEY].to_list(),
    "mean_term": list(prepared.mean_design),
    "dispersion_term": list(prepared.dispersion_design),
    "player": prepared.players,
    "opponent": prepared.opponents,
}
model = pm.Model(coords=coords)

with model:
    mean_intercept = pm.Normal(
        "mean_intercept", mu=priors.mean_intercept_center,
        sigma=priors.mean_intercept_sd,
    )
    mean_beta = pm.Normal(
        "mean_beta", mu=0, sigma=priors.mean_effect_sd, dims="mean_term",
    )

# %% [markdown]
# ### Add player and opponent effects to the mean
#
# Noncentered effects are written as $u_p=\tau_p z_p$, where $z_p\sim N(0,1)$.
# The implied prior is still $u_p\sim N(0,\tau_p^2)$. This representation often
# makes hierarchical sampling easier. Learning $\tau_p$ controls how much player
# estimates are partially pooled toward the shared contextual regression.

# %%
with model:
    mean_player_sd = pm.HalfNormal("mean_player_sd", sigma=priors.mean_group_sd)
    mean_player_z = pm.Normal("mean_player_z", mu=0, sigma=1, dims="player")
    mean_player = pm.Deterministic(
        "mean_player", mean_player_sd * mean_player_z, dims="player",
    )

    mean_opponent_sd = pm.HalfNormal("mean_opponent_sd", sigma=priors.mean_group_sd)
    mean_opponent_z = pm.Normal("mean_opponent_z", mu=0, sigma=1, dims="opponent")
    mean_opponent = pm.Deterministic(
        "mean_opponent", mean_opponent_sd * mean_opponent_z, dims="opponent",
    )

# %% [markdown]
# ### Define the dispersion priors and player effects
#
# Context-adjusted possession-to-possession dispersion can vary by player.
# This describes outcome variation, not uncertainty about a player's estimate,
# deliberate tempo changes, or tactical adaptability.

# %%
with model:
    dispersion_intercept = pm.Normal(
        "dispersion_intercept", mu=priors.dispersion_intercept_center,
        sigma=priors.dispersion_intercept_sd,
    )
    dispersion_beta = pm.Normal(
        "dispersion_beta", mu=0, sigma=priors.dispersion_effect_sd,
        dims="dispersion_term",
    )
    dispersion_player_sd = pm.HalfNormal(
        "dispersion_player_sd", sigma=priors.dispersion_group_sd,
    )
    dispersion_player_z = pm.Normal(
        "dispersion_player_z", mu=0, sigma=1, dims="player",
    )
    dispersion_player = pm.Deterministic(
        "dispersion_player", dispersion_player_sd * dispersion_player_z,
        dims="player",
    )

# %% [markdown]
# ### Connect each possession to its predictors and group effects
#
# Matrix multiplication produces the contextual predictor for every row.
# Integer indexes select each observation's player and opponent effect.
# Exponentiating $\eta_i$ ensures a positive mean. These are symbolic tensors;
# no posterior values have been estimated yet. We avoid storing observation-wide
# deterministic arrays for every draw, which would add substantial memory.

# %%
with model:
    eta = (
        mean_intercept + pt.dot(X, mean_beta)
        + mean_player[prepared.player_index]
        + mean_opponent[prepared.opponent_index]
    )
    zeta = (
        dispersion_intercept + pt.dot(Z, dispersion_beta)
        + dispersion_player[prepared.player_index]
    )
    mu = pt.exp(eta)

# %% [markdown]
# ### Add the Gamma likelihood
#
# The likelihood states how BST varies around its conditional mean at a given
# dispersion. Its `observed=y` argument links the generative model to our data.
# PyMC accepts a mean and a standard deviation rather than a mean and shape
# together. We therefore supply $SD_i=\mu_i/\sqrt{\alpha_i}$. This implements
# the manuscript's mean-shape likelihood exactly; $\alpha_i$ is still the
# independently modeled shape. `sigma` below is this derived outcome SD,
# not an additional model parameter or a player-effect standard deviation.

# %%
with model:
    alpha = pt.exp(zeta)
    likelihood = pm.Gamma(
        OUTCOME_VARIABLE,
        mu=mu,
        sigma=mu / pt.sqrt(alpha),
        observed=y,
        dims="observation",
    )

print(model)
print("Estimated array sizes (GiB):")
display(pd.Series(memory_estimate(len(model_pd_df), CONFIG)))

# %% [markdown]
# ## 6. Prior predictive checks: simulate before learning from BST
#
# Draw parameters from the priors, then simulate BST using this observed design.
# This asks what the specification can generate before conditioning parameters
# on the outcome. It is different from asking whether posterior results change
# under another prior; that requires separate fitted sensitivity runs.
#
# Set `RUN_PRIOR_PREDICTIVE=True` and run this section. Inspect near-zero values,
# upper tails and contextual/player spread. Observed values are a descriptive
# reference, not a held-out test set. Revisit implausible priors before fitting.

# %%
prior_idata = None
prior_path = RUN_DIRECTORY / "prior.nc"
if SAVED_RUN_DIRECTORY is not None and prior_path.exists():
    prior_idata = xr.open_datatree(prior_path, engine="h5netcdf")
elif RUN_PRIOR_PREDICTIVE:
    check_interactive_memory(prepared, CONFIG, stage="prior")
    with model:
        prior_idata = pm.sample_prior_predictive(
            draws=CONFIG.prior_draws,
            random_seed=CONFIG.seed,
            var_names=[OUTCOME_VARIABLE, "mean_intercept", "dispersion_intercept"],
        )
    prior_idata.to_netcdf(prior_path, engine="h5netcdf")
else:
    print("Prior simulation is off. Enable RUN_PRIOR_PREDICTIVE to run this section.")

# %%
if prior_idata is not None:
    prior_draws = predictive_array(prior_idata, model_pd_df, "prior_predictive")
    prior_checks, prior_coverage = predictive_checks(model_pd_df, prior_draws)
    display(prior_checks.query("variable == 'overall'"))
    display(prior_checks.query("variable == 'player_position_group'").head(18))
    if SAVED_RUN_DIRECTORY is None:
        prior_checks.to_csv(RUN_DIRECTORY / "prior_predictive_checks.csv", index=False)
        prior_coverage.to_csv(RUN_DIRECTORY / "prior_predictive_coverage.csv", index=False)
        save_predictive_plot(model_pd_df, prior_draws, RUN_DIRECTORY / "prior_predictive.png")
        update_interactive_status(RUN_DIRECTORY, "prior_checked")
    if (RUN_DIRECTORY / "prior_predictive.png").exists():
        display(Image(filename=str(RUN_DIRECTORY / "prior_predictive.png")))
    del prior_draws

# %% [markdown]
# ## 7. Fit the posterior, then save it
#
# Bayes' rule combines the prior with the observed likelihood:
# $$p(\theta\mid y)\propto p(y\mid\theta)p(\theta).$$
# NUTS explores this posterior. Tuning adapts the sampler; retained draws
# approximate posterior uncertainty. Multiple chains help diagnose mixing.
# `target_accept=0.90` requests smaller steps than the default; it does not
# guarantee convergence. We compute observation-level log likelihood explicitly
# after sampling because LOO needs it, unlike the original CMSAC fit.
#
# After reviewing prior checks, set `RUN_MCMC=True` and run this section.
# A pre-existing posterior is loaded for exploration. To fit a new specification,
# rerun from settings with `SAVED_RUN_DIRECTORY=None` to create a fresh run.

# %%
idata = None
posterior_path = RUN_DIRECTORY / "posterior.nc"
if posterior_path.exists():
    idata = xr.open_datatree(posterior_path, engine="h5netcdf")
    print(f"Loaded posterior: {posterior_path}")
elif RUN_MCMC:
    if SAVED_RUN_DIRECTORY is not None:
        raise ValueError("This saved run has no posterior. Start a fresh run to fit it.")
    if prior_idata is None:
        raise ValueError("Run and inspect the prior predictive section before fitting.")
    check_interactive_memory(prepared, CONFIG, stage="posterior")
    update_interactive_status(RUN_DIRECTORY, "sampling")
    with model:
        idata = pm.sample(
            draws=CONFIG.draws,
            tune=CONFIG.tune,
            chains=CONFIG.chains,
            cores=min(CONFIG.cores, CONFIG.chains),
            target_accept=CONFIG.target_accept,
            random_seed=CONFIG.seed,
        )
        pm.compute_log_likelihood(idata)
    idata.to_netcdf(posterior_path, engine="h5netcdf")
    update_interactive_status(RUN_DIRECTORY, "posterior_saved")
else:
    print("Posterior fitting is off. Continue here after reviewing prior predictions.")

# %% [markdown]
# ## 8. Diagnose sampling before interpreting coefficients
#
# Trace plots should show chains exploring the same region. The shared checks
# require four chains, no divergences, $\widehat R\le1.01$, and bulk/tail
# effective sample sizes at least 400. These are screening criteria, not proof
# that every posterior feature is estimated accurately. Small pilot fits may
# exercise the code successfully while failing scientific diagnostic gates.

# %%
if idata is not None:
    posterior_summary, sampling_diagnostics = diagnose(idata)
    display(pd.Series(sampling_diagnostics, name="sampling_diagnostics"))
    display(posterior_summary.head(20))
    posterior_summary.to_csv(RUN_DIRECTORY / "posterior_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for axis, variable in zip(axes, ["mean_intercept", "dispersion_intercept"]):
        traces = idata.posterior[variable].transpose("chain", "draw").values
        for chain_number, trace in enumerate(traces):
            axis.plot(trace, alpha=0.7, label=f"chain {chain_number + 1}")
        axis.set(title=variable, xlabel="Posterior draw")
        axis.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    if not sampling_diagnostics["sampling_diagnostics_pass"]:
        print("Diagnostic gate failed: remaining estimates are exploratory checks.")

# %% [markdown]
# ## 9. Interpret posterior effects on useful scales
#
# A mean coefficient $\beta_j$ gives a multiplicative contrast $\exp(\beta_j)$,
# or $100[\exp(\beta_j)-1]$ percent, per unit/dummy contrast. Transform **each
# draw**, then summarize; a nonlinear transform of the posterior mean is a
# different quantity. These are adjusted associations, not causal effects.
#
# Reference mean/SD/CV use the reference categories, goal difference 0,
# standardized log-sequence 0, and player/opponent effects 0. Their intervals
# describe uncertainty about conditional moments. SD/CV describe outcome spread.
# They are not population-mixture moments or standardized player rankings.

# %%
if idata is not None:
    percent_draws = 100 * np.expm1(
        idata.posterior["mean_beta"].stack(sample=("chain", "draw"))
        .transpose("sample", "mean_term").values
    )
    mean_contrasts = pd.DataFrame({
        "term": idata.posterior["mean_term"].values,
        "posterior_mean_percent_change": percent_draws.mean(axis=0),
        "q025": np.quantile(percent_draws, .025, axis=0),
        "q975": np.quantile(percent_draws, .975, axis=0),
    })
    display(mean_contrasts)
    mean_contrasts.to_csv(RUN_DIRECTORY / "mean_context_contrasts.csv", index=False)
    moments = reference_moments(idata, FAMILY)
    display(moments)
    moments.to_csv(RUN_DIRECTORY / "reference_moments.csv", index=False)

# %% [markdown]
# ### A worked dispersion contrast: first-touch pressure
#
# Raw dispersion coefficients have different meanings across families. For one
# pressure contrast, compute CV with and without the coefficient for every draw
# in a common reference context. Report the percent CV change with uncertainty.
# This illustrates how to compare predicted quantities rather than shape/log-sigma
# coefficients. A pressure level absent from the cohort has no estimated contrast.

# %%
if idata is not None:
    pressure_term = "first_touch_defender_pressure_type[Pressured (inc. Attempted Pressure)]"
    if pressure_term in prepared.dispersion_design:
        spread_reference = np.exp(idata.posterior["dispersion_intercept"].values)
        pressure_effect = idata.posterior["dispersion_beta"].sel(dispersion_term=pressure_term).values
        spread_pressure = spread_reference * np.exp(pressure_effect)
        cv_reference = conditional_moments(FAMILY, np.ones_like(spread_reference), spread_reference)["cv"]
        cv_pressure = conditional_moments(FAMILY, np.ones_like(spread_pressure), spread_pressure)["cv"]
        cv_percent_change = 100 * (cv_pressure / cv_reference - 1)
        display(pd.Series({
            "posterior_mean_percent_CV_change": cv_percent_change.mean(),
            "q025": np.quantile(cv_percent_change, .025),
            "q975": np.quantile(cv_percent_change, .975),
        }, name="Pressure versus no pressure at reference context"))

# %% [markdown]
# ## 10. Posterior predictive checks
#
# Sample $y^{rep}\sim p(y^{rep}\mid\theta)$ using posterior parameter draws.
# These simulations include outcome variation as well as parameter uncertainty.
# Keep only evenly spaced posterior draws for prediction to reduce storage;
# retain every draw for log-likelihood/LOO. Do not judge a family solely by the
# pooled histogram: inspect tails, positions, pitch thirds and pressure too.
#
# Interval coverage below uses the same observations used for fitting: it is
# an **in-sample model check**, not held-out validation. The player CV checks
# mix each player's observed contexts and are not adjusted player CV estimates.

# %%
predictive_idata = None
predictive_path = RUN_DIRECTORY / "posterior_predictive.nc"
if idata is not None:
    if predictive_path.exists():
        predictive_idata = xr.open_datatree(predictive_path, engine="h5netcdf")
    elif SAVED_RUN_DIRECTORY is None:
        retained_draws = idata.posterior.sizes["draw"]
        draw_indices = np.unique(np.linspace(
            0, retained_draws - 1, min(retained_draws, CONFIG.predictive_draws), dtype=int,
        ))
        with model:
            predictive_idata = pm.sample_posterior_predictive(
                idata.sel(draw=draw_indices),
                var_names=[OUTCOME_VARIABLE],
                random_seed=CONFIG.seed + 1,
            )
        predictive_idata.to_netcdf(predictive_path, engine="h5netcdf")
    else:
        print("Saved posterior predictive draws are absent; this load-only session will not resimulate.")

# %%
if predictive_idata is not None:
    posterior_draws = predictive_array(predictive_idata, model_pd_df, "posterior_predictive")
    posterior_checks, posterior_coverage = predictive_checks(model_pd_df, posterior_draws)
    display(posterior_checks.query("variable == 'overall'"))
    display(posterior_coverage)
    posterior_checks.to_csv(RUN_DIRECTORY / "posterior_predictive_checks.csv", index=False)
    posterior_coverage.to_csv(RUN_DIRECTORY / "posterior_predictive_coverage.csv", index=False)
    save_predictive_plot(model_pd_df, posterior_draws, RUN_DIRECTORY / "posterior_predictive.png")
    display(Image(filename=str(RUN_DIRECTORY / "posterior_predictive.png")))
    del posterior_draws

# %% [markdown]
# ## 11. Predictive evaluation: why ELPD and why PSIS-LOO?
#
# $$\operatorname{ELPD}_{LOO}=\sum_i\log p(y_i\mid y_{-i}).$$
# Higher is better. This proper logarithmic score evaluates the full predictive
# density, including dispersion and tails. Point-mean RMSE would overlook the
# variability behavior central to this investigation. All three likelihoods
# score the original BST scale, including the log-normal Jacobian automatically.
#
# PSIS approximates leaving one observation out without thousands of refits.
# Our prediction target is another possession conditional on represented
# players/opponents and observed contexts. Check Pareto-$k$ against
# $\min(0.7,1-1/\log_{10}S)$. Large values make the approximation unreliable:
# use influential-observation refits or suitable K-fold validation before ranking.
# Rare groups can be particularly problematic when their observations are left out.
#
# This is not validation on new matches, future matches or unseen players.
# Those targets require match-grouped refits, chronological splits or integration
# over new group effects. Estimate preprocessing within training folds for such
# extensions. Here the sequence scaler uses only the frozen observed predictors;
# the analysis targets that observed design. In saved-fit comparisons, paired
# match-cluster uncertainty can reveal dependence in ordinary LOO differences,
# but it does not turn them into leave-match-out scores.

# %%
if idata is not None:
    loo_pointwise, loo_results = loo_diagnostics(idata, model_pd_df)
    display(pd.Series(loo_results, name="PSIS_LOO"))
    display(loo_pointwise.sort_values("pareto_k", ascending=False).head(10))
    loo_pointwise.to_csv(RUN_DIRECTORY / "loo_pointwise.csv", index=False)
    if not loo_results["loo_diagnostics_pass"]:
        print("PSIS diagnostic gate failed: resolve influential observations before model ranking.")

# %% [markdown]
# ## 12. Record results and qualified inferences
#
# The report below uses this run's actual numerical results. Read its
# diagnostics first, then reference moments and predictive discrepancies.
# A favorable relative score cannot make a predictively inadequate model adequate.
# One fit cannot establish prior robustness or a preferred distribution.
#
# Compare baseline/stronger/weaker fits within each family, then compare
# families under a fixed named prior configuration. Judge paired ELPD differences
# against uncertainty. Common-context player means/CVs, sparse-player sensitivity
# and archetype membership probabilities remain later stages of the plan.
# Outcome variability alone does not establish deliberate tempo switching.

# %%
if idata is not None and predictive_idata is not None:
    report_path = write_interactive_results(
        RUN_DIRECTORY, sampling_diagnostics, loo_results, moments, posterior_checks,
    )
    display(Markdown(report_path.read_text()))
else:
    print("A results report will be available after posterior and predictive draws are available.")

# %% [markdown]
# ### Optional comparison with other completed runs
#
# Populate `COMPARISON_RUNS` with aligned saved directories. The helper checks
# fingerprints, row ordering and implementation, and never starts a refit.
# Positive ELPD differences favor a run over the first supplied run. Small
# differences relative to their uncertainty are inconclusive. Diagnostic
# failures are retained in the comparison report. Keep data snapshots separate.

# %%
if len(COMPARISON_RUNS) >= 2:
    comparison = compare_saved_runs(COMPARISON_RUNS, RUN_DIRECTORY / "comparisons")
    display(comparison)
    display(Markdown((RUN_DIRECTORY / "comparisons/comparison.md").read_text()))

# %% [markdown]
# ## Continue the investigation
#
# Revisit any failed diagnostics or predictive deficiencies before making
# scientific claims. Repeat the walkthrough for the other two prior settings
# and outcome families on the same processed snapshot. Saved source files,
# settings, data audit, row IDs, draws and tables preserve the path from model
# formulation to the reported result. For a batch run, use the original
# `gamma_model_comparison.py`; this teaching copy leaves it unchanged.
