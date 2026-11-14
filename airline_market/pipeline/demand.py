"""Declared priors for weakly identified booking and taste parameters."""

import numpy as np
from scipy.stats import gamma


def booking_curve(points, shape=2.0, scale=12.0):
    """
    Distribute arrivals over booking intervals using a declared gamma curve (model
    §§1–3, calibration assumptions).
    """
    points = np.asarray(points, dtype=float)
    mass = np.maximum(
        0.0,
        gamma.cdf(points[:-1], shape, scale=scale)
        - gamma.cdf(points[1:], shape, scale=scale),
    )
    return (
        mass / mass.sum()
        if mass.sum()
        else np.full(len(points) - 1, 1 / (len(points) - 1))
    )


def fare_mixture(fares, weights=None, seed=0):
    """
    Fit a two-component log-fare mixture or report the sparse-data prior (model §§1–3,
    calibration assumptions).
    """
    x = np.log(np.asarray(fares, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {
            "business_share": 0.3,
            "identified": False,
            "method": "declared_sparse_prior",
        }
    means = np.quantile(x, [0.25, 0.75])
    sd = np.full(2, max(0.1, x.std()))
    mix = np.array([0.7, 0.3])
    for _ in range(100):
        density = np.exp(-0.5 * ((x[:, None] - means) / sd) ** 2) / sd * mix
        responsibility = density / density.sum(1, keepdims=True)
        n = responsibility.sum(0)
        means = (responsibility * x[:, None]).sum(0) / n
        sd = np.sqrt(
            np.maximum(0.01, (responsibility * (x[:, None] - means) ** 2).sum(0) / n)
        )
        mix = n / len(x)
    return {
        "business_share": float(mix[np.argmax(means)]),
        "identified": True,
        "method": "two_log_normal_EM",
        "log_means": means.tolist(),
    }


def alpha_from_elasticity(elasticity, fare, share):
    """
    Translate an assumed elasticity and reference share to a logit sensitivity (model
    §§1–3, calibration assumptions).
    """
    return abs(elasticity) / (fare * max(0.01, 1 - share))
