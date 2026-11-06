"""Inspectable development plots, without hidden-data-based deployment rules."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def plot_error_by_length(scores, methods=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = scores if methods is None else scores[scores.method.isin(methods)]
    # Equal-match curves; all lengths including unfilled cases remain visible.
    grouped = data.groupby(["method", "game_id", "gap_length"])[["ade", "fill_fraction"]].mean()
    for method, frame in grouped.groupby(level="method"):
        curve = frame.groupby(level="gap_length").mean()
        axes[0].plot(curve.index, curve.ade, "o-", label=method)
        axes[1].plot(curve.index, curve.fill_fraction, "o-", label=method)
    axes[0].set(xlabel="Consecutive hidden frames", ylabel="Mean gap Euclidean error (m)")
    axes[1].set(xlabel="Consecutive hidden frames", ylabel="Filled target fraction", ylim=(-.02, 1.02))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_paired_differences(paired):
    import matplotlib.pyplot as plt

    frame = paired[paired.method != "linear_5"].sort_values("difference")
    fig, ax = plt.subplots(figsize=(9, max(4, len(frame) * .22)))
    ax.errorbar(frame.difference, np.arange(len(frame)),
                xerr=np.stack([frame.difference - frame.lower_95, frame.upper_95 - frame.difference]), fmt="o")
    ax.set_yticks(np.arange(len(frame)), frame.method)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Candidate minus linear_5 ADE (m); paired match bootstrap 95% interval")
    fig.tight_layout()
    return fig


def plot_gap_example(directory, case_id, methods):
    """A saved reference/prediction trajectory and time traces, including anchors."""
    import matplotlib.pyplot as plt

    directory = Path(directory) / "matches" / case_id.split(":")[0]
    case = pl.read_parquet(directory / "cases.parquet").filter(pl.col("case_id") == case_id).row(0, named=True)
    predictions = pl.read_parquet(directory / "predictions.parquet").filter(
        (pl.col("case_id") == case_id) & pl.col("method").is_in(methods))
    xy, target = np.asarray(case["reference_xy"]), np.asarray(case["target"], dtype=int)
    time = (np.asarray(case["frames"]) - case["frames"][target[0]]) / case["fps"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(xy[:, 0], xy[:, 1], ".-", color="gray", label="Reference context")
    axes[0].plot(xy[target, 0], xy[target, 1], "ko", label="Hidden reference")
    for j, axis in enumerate(("x", "y")):
        axes[j + 1].plot(time, xy[:, j], ".-", color="gray", label="Reference")
        axes[j + 1].axvspan(time[target[0]], time[target[-1]], alpha=.1, color="black")
        axes[j + 1].set(xlabel="Seconds from first hidden frame", ylabel=f"{axis} (m)")
    for row in predictions.iter_rows(named=True):
        pred = np.asarray(row["predicted_xy"])
        axes[0].plot(pred[:, 0], pred[:, 1], "o--", label=row["method"])
        for j in range(2):
            axes[j + 1].plot(time[target], pred[:, j], "o--", label=row["method"])
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title=case_id)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def representative_cases(scores, candidate, baseline="linear_5"):
    """Typical and difficult comparisons, selected by paired error quantiles."""
    primary = scores[scores.gap_length <= 5]
    paired = primary.pivot(index="case_id", columns="method", values="ade")[[candidate, baseline]].dropna()
    if paired.empty:
        return {}
    ordered = (paired[baseline] - paired[candidate]).sort_values()
    return {"Difficult relative to linear": ordered.index[int(.1 * (len(ordered) - 1))],
            "Typical relative to linear": ordered.index[len(ordered) // 2]}


def plot_policy_validation(scores, candidate, *, limit):
    """Equal-match single-gap curves and paired possession-tempo error diagnostics."""
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys([f"linear_{limit}", candidate]))
    data = scores[scores.method.isin(methods) & (scores.scenario_cap <= limit)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    single = data[data.experiment == "single_gap"]
    for method in methods:
        curve = single[single.method == method].groupby(["game_id", "gap_length"])[
            ["ade", "speed_mae"]].mean().groupby("gap_length").mean()
        axes[0, 0].plot(curve.index, curve.ade, "o-", label=method, markersize=3)
        axes[0, 1].plot(curve.index, curve.speed_mae, "o-", label=method, markersize=3)
    axes[0, 0].set(xlabel="Consecutive hidden frames", ylabel="Coordinate ADE (m)", title="Single-gap recovery; equal-match means")
    axes[0, 1].set(xlabel="Consecutive hidden frames", ylabel="Affected-segment speed MAE (m/s)")
    for method in methods:
        joint = data[(data.method == method) & (data.experiment == "joint_gaps")]
        axes[1, 0].scatter(joint.player_tempo_reference, joint.player_tempo, s=7, alpha=.25, label=method)
        values = joint.player_tempo_bias.dropna().to_numpy()
        if len(values):
            axes[1, 1].plot(np.sort(values), np.arange(1, len(values) + 1) / len(values), label=method)
    low, high = axes[1, 0].get_xlim()
    axes[1, 0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1, 0].set(xlabel="Reference player-possession tempo (m/s)", ylabel="Reconstructed tempo (m/s)", title="Joint gaps; possession-mask realizations")
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set(xlabel="Reconstructed minus reference tempo (m/s)", ylabel="Empirical cumulative fraction", title="Descriptive repeated-mask distribution")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle(f"Interpolation policy comparison: maximum gap {limit} frames")
    fig.tight_layout()
    return fig
