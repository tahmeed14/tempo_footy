"""Diagnostics and original-scale predictive evaluation for BST research fits."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from tempoctrl.research.bst_models import OBSERVATION_KEY, OUTCOME_VARIABLE, conditional_moments

DIAGNOSTIC_VARIABLES = (
    "mean_intercept", "mean_beta", "mean_player_sd", "mean_opponent_sd",
    "mean_player", "mean_opponent", "dispersion_intercept", "dispersion_beta",
    "dispersion_player_sd", "dispersion_player",
)


def markdown_table(frame: pd.DataFrame) -> str:
    """Small Markdown tables without an optional tabulate dependency."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)),
    ])


def predictive_array(tree, frame: pd.DataFrame, group: str) -> np.ndarray:
    """Require exact observation identity and ordering before calculating scores."""
    values = tree[group][OUTCOME_VARIABLE]
    expected = frame[OBSERVATION_KEY].astype(str).to_numpy()
    if not np.array_equal(values.observation.values.astype(str), expected):
        raise ValueError("Predictive draws do not match the frozen observation IDs/order.")
    return values.stack(sample=("chain", "draw")).transpose("sample", "observation").values


def predictive_checks(frame: pd.DataFrame, draws: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled/tail/player checks and in-sample interval calibration by context.

    Statistics are computed per predictive replicate, preserving simulation
    uncertainty. Interval coverage here is a PPC, not held-out validation.
    """
    y = frame[OUTCOME_VARIABLE].to_numpy(dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(y):
        raise ValueError("Expected predictive draws shaped (sample, observation).")
    if not np.isfinite(draws).all() or (draws < 0).any():
        raise ValueError("Predictive draws contain non-finite or negative values; inspect priors/numerics.")
    strata = [("overall", "all", np.ones(len(y), dtype=bool))]
    for column in ("player_position_group", "starting_pitch_third", "first_touch_defender_pressure_type"):
        for level in sorted(frame[column].astype(str).unique()):
            strata.append((column, level, (frame[column].astype(str) == level).to_numpy()))
    statistics, calibration = [], []
    for variable, level, mask in strata:
        observed, simulated = y[mask], draws[:, mask]
        functions = {
            "mean": lambda a: np.mean(a, axis=-1),
            "sd": lambda a: np.std(a, axis=-1),
            "q01": lambda a: np.quantile(a, .01, axis=-1),
            "q05": lambda a: np.quantile(a, .05, axis=-1),
            "median": lambda a: np.quantile(a, .5, axis=-1),
            "q95": lambda a: np.quantile(a, .95, axis=-1),
            "q99": lambda a: np.quantile(a, .99, axis=-1),
            "fraction_below_1": lambda a: np.mean(a < 1, axis=-1),
            "fraction_above_20": lambda a: np.mean(a > 20, axis=-1),
        }
        for statistic, function in functions.items():
            values = function(simulated)
            statistics.append({
                "variable": variable, "level": level, "n": int(mask.sum()),
                "statistic": statistic, "observed": float(function(observed)),
                "predictive_median": float(np.median(values)),
                "predictive_q025": float(np.quantile(values, .025)),
                "predictive_q975": float(np.quantile(values, .975)),
            })
        for probability in (.50, .80, .95):
            lower, upper = np.quantile(simulated, [(1-probability)/2, (1+probability)/2], axis=0)
            calibration.append({
                "variable": variable, "level": level, "n": int(mask.sum()),
                "nominal_coverage": probability,
                "in_sample_coverage": float(np.mean((observed >= lower) & (observed <= upper))),
                "mean_interval_width": float(np.mean(upper - lower)),
            })
    # Observed within-player spread mixes contexts. Compare like-for-like
    # replicated mixtures, without interpreting these as adjusted player CVs.
    for player, indices in frame.groupby("player_id", observed=True).indices.items():
        if len(indices) < 2:
            continue
        obs = y[indices]
        sim = draws[:, indices]
        values = sim.std(axis=1) / sim.mean(axis=1)
        statistics.append({
            "variable": "player_id", "level": player, "n": len(indices),
            "statistic": "empirical_mixture_cv", "observed": float(obs.std() / obs.mean()),
            "predictive_median": float(np.median(values)),
            "predictive_q025": float(np.quantile(values, .025)),
            "predictive_q975": float(np.quantile(values, .975)),
        })
    return pd.DataFrame(statistics), pd.DataFrame(calibration)


def save_predictive_plot(frame: pd.DataFrame, draws: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    y = frame[OUTCOME_VARIABLE].to_numpy()
    # Identical bins across families for a given cohort. Values outside the
    # plotting range are still included in numerical tail diagnostics.
    edges = np.linspace(0, max(30, np.ceil(y.max())), 65)
    hist = np.array([np.histogram(row, bins=edges, density=False)[0] / len(y) / np.diff(edges) for row in draws])
    lower, median, upper = np.quantile(hist, [.025, .5, .975], axis=0)
    centers = (edges[1:] + edges[:-1]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    observed_hist = np.histogram(y, bins=edges)[0] / len(y) / np.diff(edges)
    axes[0].stairs(observed_hist, edges, label="Observed")
    axes[0].plot(centers, median, label="Predictive median")
    axes[0].fill_between(centers, lower, upper, alpha=.25, label="95% predictive histogram band")
    axes[0].set(xlabel="BST (m/s)", ylabel="Density")
    axes[0].legend(fontsize=8)
    probabilities = np.array([.01, .05, .25, .5, .75, .95, .99])
    quantiles = np.quantile(draws, probabilities, axis=1)
    qlo, qmed, qhi = np.quantile(quantiles, [.025, .5, .975], axis=1)
    axes[1].plot(probabilities, np.quantile(y, probabilities), "o-", label="Observed")
    axes[1].errorbar(probabilities, qmed, yerr=[qmed-qlo, qhi-qmed], fmt="o", label="Predictive 95% interval")
    axes[1].set(xlabel="Quantile probability", ylabel="BST (m/s)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reference_moments(idata, family: str) -> pd.DataFrame:
    """Population reference moments, transforming each draw before summarizing.

    All categorical reference levels, goal difference 0, sequence z=0 and
    player/opponent effects 0. This is a reference conditional distribution,
    not a marginal population mixture or a standardized player comparison.
    """
    posterior = idata.posterior
    moments = conditional_moments(
        family, np.exp(posterior["mean_intercept"].values),
        np.exp(posterior["dispersion_intercept"].values),
    )
    return pd.DataFrame([
        {"quantity": name, "posterior_mean": float(values.mean()),
         "q025": float(np.quantile(values, .025)), "q975": float(np.quantile(values, .975))}
        for name, values in moments.items()
    ])


def diagnose(idata) -> tuple[pd.DataFrame, dict]:
    """Require four chains and conservative diagnostic thresholds for interpretation."""
    summary = az.summary(idata, var_names=list(DIAGNOSTIC_VARIABLES), ci_kind="hdi", ci_prob=.95, round_to=6)
    def extreme(column, operation):
        values = summary[column].to_numpy(dtype=float)
        return float(operation(values)) if np.isfinite(values).all() else None

    max_rhat = extreme("r_hat", np.max)
    min_bulk, min_tail = extreme("ess_bulk", np.min), extreme("ess_tail", np.min)
    divergences = int(idata.sample_stats["diverging"].sum())
    chains = idata.posterior.sizes["chain"]
    healthy = (chains >= 4 and divergences == 0 and max_rhat is not None and max_rhat <= 1.01
               and min_bulk is not None and min_bulk >= 400 and min_tail is not None and min_tail >= 400)
    return summary, {
        "chains": chains, "divergences": divergences, "max_rhat": max_rhat,
        "min_ess_bulk": min_bulk, "min_ess_tail": min_tail,
        "sampling_diagnostics_pass": bool(healthy),
    }


def loo_diagnostics(idata, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Observation-level PSIS-LOO for another possession in represented groups."""
    values = idata.log_likelihood[OUTCOME_VARIABLE]
    if not np.array_equal(values.observation.values.astype(str), frame[OBSERVATION_KEY].astype(str).to_numpy()):
        raise ValueError("Log likelihood does not match frozen observations.")
    if not np.isfinite(values.values).all():
        raise ValueError("Log likelihood contains non-finite entries.")
    result = az.loo(idata, var_name=OUTCOME_VARIABLE, pointwise=True)
    # ArviZ >=1 uses ELPDData.elpd and .elpd_i; the project pins >=1.2.
    pointwise = frame[[OBSERVATION_KEY, "game_id", "player_id"]].copy()
    pointwise["elpd_loo"] = np.asarray(result.elpd_i).reshape(-1)
    pointwise["pareto_k"] = np.asarray(result.pareto_k).reshape(-1)
    threshold = float(result.good_k)
    bad = int((~np.isfinite(pointwise.pareto_k) | (pointwise.pareto_k > threshold)).sum())
    return pointwise, {
        "elpd_loo": float(result.elpd), "elpd_se": float(result.se),
        "p_loo": float(result.p), "pareto_k_threshold": threshold,
        "bad_pareto_k_count": bad,
        "max_pareto_k": float(pointwise.pareto_k.max()),
        "loo_diagnostics_pass": bad == 0,
        "validation_target": "additional possession conditional on represented players/opponents and observed contexts",
    }


def compare_saved_runs(run_directories: list[str | Path], output: str | Path) -> pd.DataFrame:
    """Compare completed artifacts only; never refit or mix different snapshots.

    Positive paired ELPD differences favor the current run over the first
    supplied run. Differences are descriptive when diagnostic gates fail.
    """
    if len(run_directories) < 2:
        raise ValueError("Supply at least two completed runs.")
    runs, points = [], []
    for directory in map(Path, run_directories):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Run is not complete: {directory}")
        runs.append(manifest)
        points.append(pd.read_csv(directory / "loo_pointwise.csv", dtype={OBSERVATION_KEY: str, "player_id": str}))
    baseline = runs[0]
    for manifest, point in zip(runs, points):
        if (manifest["data"]["source_sha256"] != baseline["data"]["source_sha256"]
                or manifest["data"]["cohort_sha256"] != baseline["data"]["cohort_sha256"]
                or manifest["implementation_sha256"] != baseline["implementation_sha256"]
                or manifest.get("evaluation_implementation_sha256") != baseline.get("evaluation_implementation_sha256")
                or manifest.get("packages") != baseline.get("packages")):
            raise ValueError("Runs differ in data or implementation; rerun a matched comparison.")
        if not point[[OBSERVATION_KEY, "game_id", "player_id"]].equals(points[0][[OBSERVATION_KEY, "game_id", "player_id"]]):
            raise ValueError("Observation IDs or ordering differ across runs.")
    rows = []
    for directory, manifest, point in zip(run_directories, runs, points):
        delta = point.elpd_loo - points[0].elpd_loo
        # This paired observation SE assumes exchangeable pointwise errors. A
        # cluster summary quantifies match dependence but is still ordinary LOO,
        # never a substitute for actual leave-match-out refits.
        cluster = delta.groupby(point.game_id).sum()
        rows.append({
            "run_directory": str(directory), "family": manifest["family"], "prior": manifest["prior"]["name"],
            "elpd_loo": manifest["evaluation"]["elpd_loo"],
            "elpd_difference_vs_first": float(delta.sum()),
            "paired_observation_se": float(np.sqrt(len(delta) * delta.var(ddof=1))),
            "paired_match_cluster_se": float(np.sqrt(len(cluster) * cluster.var(ddof=1))) if len(cluster) > 1 else np.nan,
            "diagnostics_pass": manifest["evaluation"]["sampling_diagnostics_pass"] and manifest["evaluation"]["loo_diagnostics_pass"],
        })
    table = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "predictive_comparison.csv", index=False)
    all_healthy = bool(table.diagnostics_pass.all())
    report = (
        "# Saved BST fit comparison\n\n" + markdown_table(table.drop(columns="run_directory"))
        + "\n\nPositive differences favor a run over the first supplied run; higher ELPD is better. "
        "The score evaluates a full predictive density on the original BST scale. "
        "The paired match cluster SE checks dependence of LOO differences; it is not leave-match-out validation. "
        "Small differences relative to uncertainty do not establish a preferred family.\n\n"
        + ("Sampling and importance-sampling diagnostic gates passed. " if all_healthy else "At least one diagnostic gate failed. Do not rank these fits; resolve the flagged fits or refit influential observations. ")
        + "Prior calibration, predictive deficiencies and common-context player/archetype sensitivity must also be reviewed before scientific selection.\n"
    )
    (output / "comparison.md").write_text(report)
    return table
