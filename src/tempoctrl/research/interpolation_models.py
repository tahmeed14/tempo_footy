"""Local two-sided ball reconstructors; hidden reference values are never inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import factorial

import numpy as np
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.linalg import cho_factor, cho_solve

from tempoctrl.gradient_sports.interpolations import interpolate_ball_coordinates


@dataclass(frozen=True)
class Interpolator:
    name: str
    family: str
    context_seconds: float = .5
    max_gap: int = 5
    process_noise: float = 10.0
    observation_sd: float = .05
    length_scale: float = .2
    kernel_order: int = 3
    linear_through: int = 1


@dataclass
class Reconstruction:
    coordinates: np.ndarray
    observation_variance: np.ndarray | None = None
    status: str = "filled"


def candidate_interpolators(contexts=(.25, .5, 1.0)) -> tuple[Interpolator, ...]:
    """Predeclared grid; noise parameters are candidates, not fitted on test truth."""
    specs = [Interpolator("leave_missing", "missing"),
             Interpolator("nearest", "nearest", max_gap=45),
             Interpolator("forward_hold", "hold", max_gap=45)]
    for limit in (1, 2, 3, 5, 10, 15, 30):
        specs.append(Interpolator(f"linear_{limit}", "linear", max_gap=limit))
    for context in contexts:
        for family in ("pchip", "cubic", "hermite"):
            specs.append(Interpolator(f"{family}_c{context:g}", family, context, 45))
        for family in ("rts_cv", "rts_ca"):
            for noise in (1., 100.):
                specs.append(Interpolator(
                    f"{family}_c{context:g}_q{noise:g}", family, context, 45,
                    process_noise=noise,
                ))
        for order in (3, 5):
            for length in (.15, .5):
                specs.append(Interpolator(
                    f"gp{order}_c{context:g}_l{length:g}", "gp", context, 45,
                    length_scale=length, kernel_order=order,
                ))
    return tuple(specs)


def production_linear(frames, xy, target, *, max_gap=5) -> np.ndarray:
    """Call the actual production interpolator on a single isolated trajectory."""
    records = [{"x": None if not np.isfinite(p[0]) else float(p[0]),
                "y": None if not np.isfinite(p[1]) else float(p[1])} for p in xy]
    data = pl.DataFrame({
        "game_id": [1] * len(frames), "framenum": frames,
        "dev_match_team_player_possession_id": ["case"] * len(frames),
        "balls_smooth": records,
    }, schema_overrides={"balls_smooth": pl.Struct({"x": pl.Float64, "y": pl.Float64})})
    result = interpolate_ball_coordinates(data.lazy(), max_gap=max_gap).collect()
    return np.column_stack([
        result["balls_smooth"].struct.field(axis).to_numpy() for axis in ("x", "y")
    ])[target]


def transition(dt: float, derivatives: int, spectral_density: float):
    """Exact integrated-Wiener transition for CV (2) or CA (3) per axis."""
    d = derivatives
    f = np.zeros((d, d))
    q = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if j >= i:
                f[i, j] = dt ** (j - i) / factorial(j - i)
            power = 2 * d - 1 - i - j
            q[i, j] = spectral_density * dt**power / (
                power * factorial(d - 1 - i) * factorial(d - 1 - j)
            )
    return f, q


def _rts(times, xy, spec):
    d = 2 if spec.family == "rts_cv" else 3
    n = len(times)
    r = spec.observation_sd**2
    filtered = np.zeros((n, d, 2))
    cov = np.zeros((n, d, d))
    predicted = np.zeros_like(filtered)
    pred_cov = np.zeros_like(cov)
    transitions = np.zeros_like(cov)
    # First input is observed. Its position likelihood is used once.
    filtered[0, 0] = xy[0]
    cov[0] = np.diag([r] + [1e4] * (d - 1))
    for k in range(1, n):
        f, q = transition(times[k] - times[k - 1], d, spec.process_noise)
        transitions[k] = f
        predicted[k] = f @ filtered[k - 1]
        pred_cov[k] = f @ cov[k - 1] @ f.T + q
        filtered[k], cov[k] = predicted[k], pred_cov[k]
        if np.isfinite(xy[k]).all():
            gain = pred_cov[k, :, 0] / (pred_cov[k, 0, 0] + r)
            filtered[k] = predicted[k] + gain[:, None] * (xy[k] - predicted[k, 0])
            # Joseph form avoids negative covariance from cancellation.
            a = np.eye(d)
            a[:, 0] -= gain
            cov[k] = a @ pred_cov[k] @ a.T + r * np.outer(gain, gain)
    smoothed, variance = filtered.copy(), cov.copy()
    for k in range(n - 2, -1, -1):
        gain = np.linalg.solve(pred_cov[k + 1], transitions[k + 1] @ cov[k]).T
        smoothed[k] += gain @ (smoothed[k + 1] - predicted[k + 1])
        variance[k] += gain @ (variance[k + 1] - pred_cov[k + 1]) @ gain.T
    # Predict the provider observation, not only its latent state.
    observation_var = np.maximum(variance[:, 0, 0], 0) + r
    return smoothed[:, 0], np.repeat(observation_var[:, None], 2, axis=1)


def _gp(t, y, query, spec):
    # An observed-endpoint linear mean makes the local residual model translation
    # invariant. Fixed 1 m residual amplitude is part of this explicit initial grid.
    trend = y[0] + (t - t[0])[:, None] / (t[-1] - t[0]) * (y[-1] - y[0])
    query_trend = y[0] + (query - t[0])[:, None] / (t[-1] - t[0]) * (y[-1] - y[0])

    def kernel(a, b):
        distance = np.abs(a[:, None] - b[None, :]) / spec.length_scale
        z = np.sqrt(spec.kernel_order) * distance
        return (1 + z + (z*z / 3 if spec.kernel_order == 5 else 0)) * np.exp(-z)

    k = kernel(t, t) + np.eye(len(t)) * spec.observation_sd**2
    factor = cho_factor(k, lower=True, check_finite=False)
    cross = kernel(query, t)
    mean = query_trend + cross @ cho_solve(factor, y - trend, check_finite=False)
    variance = np.maximum(1 - np.sum(cross * cho_solve(factor, cross.T).T, axis=1), 0)
    return mean, np.repeat((variance + spec.observation_sd**2)[:, None], 2, axis=1)


def reconstruct(spec: Interpolator, frames, masked_xy, target, fps: float) -> Reconstruction:
    """Return target-only predictions from observed coordinates; never extrapolate.

    All target rows must already be fully missing. Partial gaps/non-finite anchors
    are outside this initial controlled cohort and cause explicit rejection.
    """
    frames = np.asarray(frames, dtype=float)
    xy = np.array(masked_xy, dtype=float, copy=True)
    target = np.asarray(target, dtype=int)
    if not np.isfinite(fps) or fps <= 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("Require positive FPS and strictly increasing frame numbers.")
    if not len(target) or np.any(np.isfinite(xy[target])):
        raise ValueError("Hidden targets must be masked before reconstruction.")
    observed = np.isfinite(xy).all(axis=1)
    bounds = np.flatnonzero(observed)
    if len(bounds) < 2 or target.min() <= bounds[0] or target.max() >= bounds[-1]:
        raise ValueError("Two-sided observed anchors are required.")
    left = bounds[bounds < target.min()][-1]
    right = bounds[bounds > target.max()][0]
    missing_frames = frames[right] - frames[left] - 1
    empty = Reconstruction(np.full((len(target), 2), np.nan), status="abstained")
    if spec.family == "missing" or missing_frames > spec.max_gap:
        return empty
    times = (frames - frames[left]) / fps
    keep = (times >= -spec.context_seconds) & (times <= times[right] + spec.context_seconds)
    obs = observed & keep
    t, y, query = times[obs], xy[obs], times[target]
    fraction = (query - times[left]) / (times[right] - times[left])
    linear = xy[left] + fraction[:, None] * (xy[right] - xy[left])
    if spec.family == "linear" or (spec.family == "hybrid" and missing_frames <= spec.linear_through):
        # This is the exact repository implementation, not a numpy replacement.
        return Reconstruction(production_linear(frames, xy, target, max_gap=spec.max_gap))
    if spec.family == "nearest":
        return Reconstruction(np.where((fraction <= .5)[:, None], xy[left], xy[right]))
    if spec.family == "hold":
        return Reconstruction(np.repeat(xy[left][None, :], len(target), axis=0))
    if len(t) < 4 or (t < query.min()).sum() < 2 or (t > query.max()).sum() < 2:
        return Reconstruction(linear, status="linear_fallback")
    if spec.family in ("pchip", "hybrid"):
        return Reconstruction(PchipInterpolator(t, y, axis=0, extrapolate=False)(query))
    if spec.family == "cubic":
        return Reconstruction(CubicSpline(t, y, axis=0, bc_type="natural", extrapolate=False)(query))
    if spec.family == "hermite":
        before, after = np.flatnonzero(obs & (times <= times[left])), np.flatnonzero(obs & (times >= times[right]))
        velocity = np.stack([(xy[before[-1]] - xy[before[-2]]) / (times[before[-1]] - times[before[-2]]),
                             (xy[after[1]] - xy[after[0]]) / (times[after[1]] - times[after[0]])])
        return Reconstruction(CubicHermiteSpline(
            times[[left, right]], xy[[left, right]], velocity, axis=0, extrapolate=False,
        )(query))
    if spec.family.startswith("rts_"):
        selected = np.flatnonzero(keep)
        mean, var = _rts(times[keep], xy[keep], spec)
        indices = np.searchsorted(selected, target)
        return Reconstruction(mean[indices], var[indices])
    if spec.family == "gp":
        mean, var = _gp(t, y, query, spec)
        return Reconstruction(mean, var)
    raise ValueError(f"Unknown interpolator {asdict(spec)}")


def reconstruct_blocks(spec, frames, masked_xy, target, fps):
    """Reconstruct separated blocks using the SAME fully masked input each time.

    A prediction for one block must never become an observed anchor for another.
    Maximum gap rules apply to each contiguous frame block, not their union.
    """
    target = np.asarray(target, dtype=int)
    if not len(target) or np.any(np.diff(target) <= 0):
        raise ValueError("Require nonempty, sorted, unique target indices.")
    frames = np.asarray(frames)
    cuts = np.flatnonzero((np.diff(target) != 1) | (np.diff(frames[target]) != 1)) + 1
    results = [reconstruct(spec, frames, masked_xy, block, fps)
               for block in np.split(target, cuts)]
    statuses = [result.status for result in results]
    status = "filled"
    if all(s == "abstained" for s in statuses):
        status = "abstained"
    elif "abstained" in statuses:
        status = "partially_filled"
    elif "linear_fallback" in statuses:
        status = "linear_fallback"
    variance = (np.concatenate([r.observation_variance for r in results])
                if all(r.observation_variance is not None for r in results) else None)
    return Reconstruction(np.concatenate([r.coordinates for r in results]), variance, status)
