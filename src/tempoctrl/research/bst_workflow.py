"""Explicit preparation, prior checks, fitting and artifact-only evaluation.

Example (no fitting)::

    run_family(RunConfig(family="gamma", mode="prepare"))

Full fits are opt-in via ``mode="fit"``. Each execution writes a distinct run
directory. Nothing loads a previous posterior based only on its filename.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from tempoctrl.research.bst_models import (
    FAMILIES, OBSERVATION_KEY, OUTCOME_VARIABLE, PRIOR_NAMES,
    build_model, file_sha256, prepare_data, prior_spec, repository_root, specification_hash,
)
from tempoctrl.research.bst_evaluation import (
    diagnose, loo_diagnostics, markdown_table, predictive_array,
    predictive_checks, reference_moments, save_predictive_plot,
)


@dataclass(frozen=True)
class RunConfig:
    family: str
    mode: str = "prepare"
    prior: str = "all"
    data: str | Path = "data/analysis/model_data_vFINAL.parquet"
    output: str | Path = "paper/results/bst_distribution_comparison"
    seed: int = 42
    draws: int = 1000
    tune: int = 1500
    chains: int = 4
    cores: int = 4
    target_accept: float = .90
    prior_draws: int = 200
    predictive_draws: int = 200
    max_working_gib: float = 8.0


def memory_estimate(rows: int, config: RunConfig) -> dict:
    """Approximate float64 storage; LOO can need several LL-sized temporaries."""
    gib = 1024**3
    ll = rows * config.draws * config.chains * 8 / gib
    ppc = rows * min(config.predictive_draws, config.draws) * config.chains * 8 / gib
    prior = rows * config.prior_draws * 8 / gib
    return {
        "log_likelihood_gib": ll, "posterior_predictive_gib": ppc,
        "prior_predictive_gib": prior,
        "approximate_working_gib": max(4 * ll + 2 * ppc, 3 * prior),
        "caveat": "Estimate excludes sampler/compiler overhead and is not a hard memory limit.",
    }


def prior_table() -> pd.DataFrame:
    """All nine predeclared specifications, including local CV calibration widths."""
    rows = []
    for family in FAMILIES:
        for name in PRIOR_NAMES:
            spec = prior_spec(family, name)
            rows.append({
                **spec.as_dict(), "reference_mean_center": 7.0,
                "reference_cv_center": float(1 / np.sqrt(3)),
                "approx_log_cv_intercept_sd": spec.dispersion_intercept_sd * abs(spec.log_cv_slope),
                "approx_log_cv_effect_sd": spec.dispersion_effect_sd * abs(spec.log_cv_slope),
                "calibration_status": "proposed; inspect prior predictive checks",
            })
    return pd.DataFrame(rows)


def methods_markdown(family: str) -> str:
    formulations = {
        "gamma": r"$Y_i\sim\operatorname{Gamma}(\alpha_i,\alpha_i/\mu_i)$, $\log\alpha_i=\zeta_i$, $CV_i=\alpha_i^{-1/2}$.",
        "lognormal": r"$Y_i\sim\operatorname{LogNormal}(\log\mu_i-\sigma_i^2/2,\sigma_i)$, $\log\sigma_i=\zeta_i$, $CV_i=\sqrt{\exp(\sigma_i^2)-1}$.",
        "weibull": r"$Y_i\sim\operatorname{Weibull}(k_i,\lambda_i)$, $\log k_i=\zeta_i$, $\lambda_i=\mu_i/\Gamma(1+1/k_i)$, $CV_i=\sqrt{\Gamma(1+2/k_i)/\Gamma(1+1/k_i)^2-1}$.",
    }
    return "\n\n".join([
        f"# {family.capitalize()} BST research model",
        r"The outcome $Y_i>0$ is possession BST in metres/second. All families model the same conditional arithmetic mean: $\log\mu_i=\beta_0+X_i\beta+u_{p[i]}+v_{o[i]}$. Dispersion follows $\zeta_i=\delta_0+Z_i\delta+w_{p[i]}$.",
        formulations[family],
        "X preserves CMSAC predictors: goal difference, first-touch height and pressure, defender challenges, pitch third, position, log-standardized possession sequence, and game period; player and opponent intercepts. Z uses pressure, challenges, pitch third, position, and a player intercept. Contrasts use the module's explicit repository reference levels.",
        "## Proposed priors",
        r"$\beta_0\sim N(\log 7,s_0)$; contextual mean effects $\beta_j\sim N(0,s_\beta)$; player and opponent effects are independent $N(0,\tau^2)$ with $\tau\sim\operatorname{HalfNormal}(s_\beta)$. Dispersion has the same hierarchy on its natural log-parameter, with family-specific centers and widths in priors.csv. Effects use noncentered parameterizations.",
        "Baseline, stronger and weaker regularization keep centers fixed. The mean 7 and Gamma shape 3 originate in the existing CMSAC specification, rather than independent domain elicitation; they may reflect earlier analysis of these outcomes. The common CV center is 1/sqrt(3). Dispersion widths use the local derivative of log(CV) at that center: this matches local uncertainty, not full prior distributions. All contextual and group-scale widths change jointly; stronger/weaker mean-intercept SDs are .25/.60 versus .40 baseline. These are proposed configurations pending original-scale prior predictive review; implausible tails require revision before confirmatory inference.",
        markdown_table(prior_table().query("family == @family").drop(columns=["family", "calibration_status"])),
        "## Evaluation metric and validation",
        r"Primary predictive metric: expected log predictive density (ELPD), $\sum_i\log p(y_i\mid y_{-i})$. A proper logarithmic score assesses the entire predictive density, rewarding calibrated uncertainty and exposing overconfident tail predictions. Higher is better. RMSE of a point mean would omit the dispersion behavior central to this investigation. All three likelihoods are on original BST units; the log-normal distribution includes the Jacobian automatically.",
        "Observation-level PSIS-LOO targets an additional possession conditional on represented players/opponents and observed contexts. It approximates leave-one-observation-out without thousands of full refits. Inspect Pareto-k against min(0.7, 1-1/log10(S)), R-hat <=1.01, bulk/tail ESS >=400, four chains and zero divergences before ranking. Poor k requires exact influential-observation refits or suitable K-fold validation; this initial workflow flags that work and does not implement automatic refits. A sparsely observed group whose sole observation is omitted may make this approximation unreliable.",
        "Ordinary LOO does not establish new-match, future-match, or new-player generalization. Those targets require common match-grouped refits, chronological splits, or new-level integration respectively, with data-derived preprocessing fitted on training data. Those extensions are deferred. The present deterministic log-sequence scaler uses only predictors from the frozen cohort; it targets the observed-design conditional analysis. For strict prospective CV, estimate its mean and SD within each training fold.",
        "Report paired ELPD differences and uncertainty, alongside distribution, tail and contextual predictive checks. In-sample predictive interval coverage is a posterior predictive check, not held-out validation. Descriptive low/high thresholds (1 and 20 m/s) are checks, not exclusion rules. Player empirical CVs mix their observed contexts and are only PPC statistics. The reference-moment table transforms each posterior draw before summarizing; its intervals are equal-tail 95% intervals.",
        "## Scope of scientific inference",
        "Variability means context-adjusted possession-to-possession dispersion, not deliberate changes or tactical adaptability. Scientific claims about player rankings/archetypes require common supported contexts, draw-wise mean/SD/CV contrasts, membership uncertainty and sensitivity by player count across all nine fits. Those Phase 6 comparisons and targeted prior-component refits remain future work. No empirical winner is asserted by this implementation.",
        "## References",
        "[PyMC Gamma](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Gamma.html), [log-normal](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LogNormal.html), [Weibull](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Weibull.html), [PSIS-LOO and diagnostics](https://mc-stan.org/loo/articles/loo2-example.html).",
    ]) + "\n"


def _write_json(path: Path, value: dict) -> None:
    # JSON has no NaN/Inf. Keep a failed numeric diagnostic explicit as null.
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, Path):
            return str(item)
        return item
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def _implementation() -> tuple[dict, str]:
    directory = Path(__file__).parent
    hashes = {name: file_sha256(directory / name) for name in ("bst_models.py", "bst_evaluation.py", "bst_workflow.py")}
    return hashes, specification_hash(hashes)


def evaluate_run(directory: str | Path) -> dict:
    """Evaluate a saved run and its frozen observations without reading current data."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_sha256(directory / "observations.parquet") != manifest["snapshot_sha256"]:
        raise ValueError("Saved observation snapshot changed; evaluate the original frozen artifact.")
    frame = pd.read_parquet(directory / "observations.parquet")
    with xr.open_datatree(directory / "posterior.nc", engine="h5netcdf") as idata, xr.open_datatree(directory / "posterior_predictive.nc", engine="h5netcdf") as predictive:
        summary, sampling = diagnose(idata)
        summary.to_csv(directory / "posterior_summary.csv")
        moments = reference_moments(idata, manifest["family"])
        moments.to_csv(directory / "reference_moments.csv", index=False)
        pointwise, loo = loo_diagnostics(idata, frame)
        pointwise.to_csv(directory / "loo_pointwise.csv", index=False)
        draws = predictive_array(predictive, frame, "posterior_predictive")
        checks, coverage = predictive_checks(frame, draws)
        checks.to_csv(directory / "posterior_predictive_checks.csv", index=False)
        coverage.to_csv(directory / "posterior_predictive_coverage.csv", index=False)
        save_predictive_plot(frame, draws, directory / "posterior_predictive.png")
    evaluation = {**sampling, **loo}
    healthy = sampling["sampling_diagnostics_pass"] and loo["loo_diagnostics_pass"]
    pooled = checks.query("variable == 'overall'")
    deficiencies = pooled[(pooled.observed < pooled.predictive_q025) | (pooled.observed > pooled.predictive_q975)]
    report = [
        "# Results and qualified inferences", "",
        f"Family: {manifest['family']}; prior: {manifest['prior']['name']}; observations: {len(frame)}.", "",
        "Sampling and PSIS diagnostics passed." if healthy else "At least one sampling or PSIS diagnostic failed. Resolve these issues before interpreting estimates or ranking models.", "",
        markdown_table(pd.DataFrame([evaluation])), "",
        "## Reference conditional moments", "", markdown_table(moments), "",
        "These are population reference-context estimates with group effects set to zero. Intervals represent posterior uncertainty about the conditional mean, SD or CV; SD/CV represent outcome variability.", "",
        "## Overall predictive checks", "", markdown_table(pooled), "",
        ("Observed summaries outside the 95% predictive replicate interval: " + ", ".join(deficiencies.statistic) + ". Investigate these discrepancies even if relative ELPD is favorable.")
        if len(deficiencies) else "No listed pooled summary falls outside its 95% predictive replicate interval. This does not establish conditional adequacy; inspect subgroup, tail and player checks.", "",
        "## What can be inferred", "",
        "This run quantifies one family's fit under one prior configuration. Its ELPD cannot establish a preferred family in isolation. Compare aligned saved fits using compare_saved_runs; small paired differences relative to their uncertainty are inconclusive. Prior configurations remain proposed until predictive plausibility is reviewed. The three-by-three sensitivity analysis, shared-context player comparisons and archetypes are required before claiming robust player conclusions.", "",
        "See methods.md for metric rationale, validation target, assumptions and deferred work.",
    ]
    (directory / "results.md").write_text("\n".join(report) + "\n")
    manifest.update(status="complete", evaluation=evaluation, evaluation_implementation_sha256=_implementation()[1])
    _write_json(directory / "manifest.json", manifest)
    return evaluation


def _validate_config(config: RunConfig) -> None:
    if config.family not in FAMILIES or config.prior not in (*PRIOR_NAMES, "all"):
        raise ValueError("Unknown family or prior configuration.")
    if config.mode not in ("prepare", "prior", "fit"):
        raise ValueError("Use prepare, prior or fit; use evaluate_run for saved artifacts.")
    if any(getattr(config, name) < 1 for name in ("draws", "tune", "chains", "cores", "prior_draws", "predictive_draws")):
        raise ValueError("Sampling counts must be positive.")
    if not .5 < config.target_accept < 1 or config.max_working_gib <= 0:
        raise ValueError("Invalid target_accept or memory budget.")


def run_family(config: RunConfig) -> list[Path]:
    """Prepare, check or fit named configurations with unique, auditable artifacts."""
    _validate_config(config)
    root = repository_root()
    data_path = Path(config.data)
    output = Path(config.output)
    if not data_path.is_absolute():
        data_path = root / data_path
    if not output.is_absolute():
        output = root / output
    data = prepare_data(data_path)
    estimate = memory_estimate(len(data.frame), config)
    required = (estimate["approximate_working_gib"] if config.mode == "fit" else 3 * estimate["prior_predictive_gib"])
    if config.mode != "prepare" and required > config.max_working_gib:
        raise MemoryError(f"Estimated working arrays need {required:.2f} GiB; budget is {config.max_working_gib:.2f}. Reduce draws or explicitly increase --max-working-gib.")
    prior_names = PRIOR_NAMES if config.prior == "all" else (config.prior,)
    code_hashes, code_hash = _implementation()
    paths = []
    for name in prior_names:
        priors = prior_spec(config.family, name)
        settings = asdict(config)
        settings["data"], settings["output"] = str(data_path), str(output)
        identity = {"data": data.audit, "prior": priors.as_dict(), "settings": settings, "code": code_hash}
        run_hash = specification_hash(identity)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = output / data.audit["source_sha256"][:12] / config.family / name / f"{stamp}_{run_hash[:12]}"
        directory.mkdir(parents=True, exist_ok=False)
        paths.append(directory)
        manifest = {
            "status": "prepared", "created_utc": stamp, "family": config.family,
            "prior": priors.as_dict(), "data": data.audit, "settings": settings,
            "run_sha256": run_hash, "implementation_sha256": code_hash,
            "implementation_files": code_hashes, "memory_estimate": estimate,
            "prior_calibration_status": "proposed; inspect prior predictive checks",
            "packages": {package: version(package) for package in ("pymc", "arviz", "bambi", "numpy", "pandas", "polars", "scipy", "pytensor", "xarray")},
            "posterior_predictive_subsampling": "Evenly spaced draws within each chain; LL retains all posterior draws.",
            "validation": "observation-level PSIS-LOO; no new-match or future-match claim",
        }
        # Read-only Git commands also work when this workflow is invoked outside root.
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        manifest["git_revision"] = revision.stdout.strip() or None
        patch = subprocess.run(["git", "diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"], cwd=root, capture_output=True, text=True)
        (directory / "working_tree.patch").write_text(patch.stdout)
        for filename in code_hashes:
            shutil.copyfile(Path(__file__).parent / filename, directory / filename)
        data.frame.to_parquet(directory / "observations.parquet", index=False)
        manifest["snapshot_sha256"] = file_sha256(directory / "observations.parquet")
        data.frame[[OBSERVATION_KEY, "game_id", "player_id", "opponent_id"]].to_csv(directory / "observation_ids.csv", index=False)
        data.frame.groupby("player_id", observed=True).size().rename("observations").to_csv(directory / "player_counts.csv")
        data.exclusions.to_csv(directory / "exclusions.csv", index=False)
        prior_table().to_csv(directory / "priors.csv", index=False)
        (directory / "methods.md").write_text(methods_markdown(config.family))
        _write_json(directory / "manifest.json", manifest)
        print(f"{config.family}/{name}: {len(data.frame)} observations; artifacts: {directory}", flush=True)
        if config.mode == "prepare":
            continue
        try:
            import pymc as pm

            model = build_model(data, priors)
            with model:
                prior = pm.sample_prior_predictive(draws=config.prior_draws, random_seed=config.seed, var_names=[OUTCOME_VARIABLE, "mean_intercept", "dispersion_intercept"])
            prior.to_netcdf(directory / "prior.nc", engine="h5netcdf")
            prior_values = predictive_array(prior, data.frame, "prior_predictive")
            checks, coverage = predictive_checks(data.frame, prior_values)
            checks.to_csv(directory / "prior_predictive_checks.csv", index=False)
            coverage.to_csv(directory / "prior_predictive_coverage.csv", index=False)
            save_predictive_plot(data.frame, prior_values, directory / "prior_predictive.png")
            manifest["status"] = "prior_checked"
            (directory / "prior_checks.md").write_text(
                "# Prior predictive checks\n\nThese are proposed priors. Observed values are shown only as a descriptive reference, not a validation set. "
                "Assess low values, upper tails, contexts and player variability for substantive plausibility. "
                "Local CV matching does not guarantee equivalent tail behavior. Revisions informed by these outcomes are data-informed.\n\n"
                + markdown_table(checks.query("variable == 'overall'")) + "\n"
            )
            _write_json(directory / "manifest.json", manifest)
            del prior_values, prior
            if config.mode == "prior":
                continue
            manifest["status"] = "sampling"
            _write_json(directory / "manifest.json", manifest)
            with model:
                idata = pm.sample(
                    draws=config.draws, tune=config.tune, chains=config.chains,
                    cores=min(config.cores, config.chains), target_accept=config.target_accept,
                    random_seed=config.seed,
                )
                pm.compute_log_likelihood(idata)
                # Subsample predictions, never likelihoods used for PSIS-LOO.
                indices = np.unique(np.linspace(0, config.draws - 1, min(config.draws, config.predictive_draws), dtype=int))
                predictive = pm.sample_posterior_predictive(
                    idata.sel(draw=indices), var_names=[OUTCOME_VARIABLE], random_seed=config.seed + 1,
                )
            idata.to_netcdf(directory / "posterior.nc", engine="h5netcdf")
            predictive.to_netcdf(directory / "posterior_predictive.nc", engine="h5netcdf")
            # Close in-memory arrays before evaluation opens saved artifacts.
            del idata, predictive, model
            evaluate_run(directory)
        except Exception as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_json(directory / "manifest.json", manifest)
            raise
    return paths


def cli(family: str) -> None:
    """Shared command-line entry point for the three percent-cell scripts."""
    parser = argparse.ArgumentParser(description=f"{family} BST distribution/prior comparison")
    parser.add_argument("--mode", choices=("prepare", "prior", "fit", "evaluate"), default="prepare")
    parser.add_argument("--run", type=Path, help="Saved run directory; required for evaluate")
    parser.add_argument("--data", type=Path, default=RunConfig.__dataclass_fields__["data"].default)
    parser.add_argument("--output", type=Path, default=RunConfig.__dataclass_fields__["output"].default)
    parser.add_argument("--prior", choices=(*PRIOR_NAMES, "all"), default="all")
    for name in ("seed", "draws", "tune", "chains", "cores", "prior_draws", "predictive_draws"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=RunConfig.__dataclass_fields__[name].default)
    parser.add_argument("--target-accept", type=float, default=.90)
    parser.add_argument("--max-working-gib", type=float, default=8.0)
    arguments = vars(parser.parse_args())
    directory = arguments.pop("run")
    if arguments["mode"] == "evaluate":
        if directory is None:
            parser.error("--mode evaluate requires --run PATH")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["family"] != family:
            parser.error(f"Saved run belongs to {manifest['family']}, not {family}")
        print(json.dumps(evaluate_run(directory), indent=2))
    else:
        if directory is not None:
            parser.error("--run is only used with --mode evaluate")
        run_family(RunConfig(family=family, **arguments))
