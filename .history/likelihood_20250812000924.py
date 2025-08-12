"""Likelihood evaluation for strong-lens population parameters."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.stats import norm, skewnorm

from .cached_A import cached_A_interp
from .make_tabulate import LensGrid, tabulate_likelihood_grids
from .mock_generator.mass_sampler import MODEL_PARAMS
from .config import SCATTER

# Parameters describing the halo-mass relation used in the mock generator
MODEL_P = MODEL_PARAMS["deVauc"]
BETA_DM = MODEL_P["beta_h"]
SIGMA_DM = MODEL_P["sigma_h"]

# Convenience wrapper ---------------------------------------------------------

def precompute_grids(
    mock_observed_data,
    logMh_grid: Iterable[float],
    zl: float = 0.3,
    zs: float = 2.0,
    sigma_m: float | None = None,
) -> list[LensGrid]:
    """Wrapper around :func:`tabulate_likelihood_grids` with defaults."""

    if sigma_m is None:
        sigma_m = SCATTER.mag

    return tabulate_likelihood_grids(
        mock_observed_data,
        logMh_grid,
        zl=zl,
        zs=zs,
        sigma_m=sigma_m,
    )


# Priors ---------------------------------------------------------------------

def log_prior(theta: Sequence[float]) -> float:
    """Flat prior on ``(muDM, alpha)`` within broad bounds."""

    muDM, alpha = theta
    if not (10.0 < muDM < 16.0 and -0.5 < alpha < 1.0):
        return -np.inf
    return 0.0


# Single-lens integral -------------------------------------------------------

def _single_lens_likelihood(
    grid: LensGrid,
    logM_sps_obs: float,
    theta: Sequence[float],
) -> float:
    """Evaluate the likelihood contribution of one lens."""

    muDM, alpha = theta

    mask = np.isfinite(grid.logM_star) & np.isfinite(grid.sample_factor)
    if not np.any(mask):
        return 0.0

    logMh = grid.logMh_grid[mask]
    logM_star = grid.logM_star[mask]
    sample_factor = grid.sample_factor[mask]

    Msps = logM_star - alpha

    # Halo mass conditional on stellar mass
    p_logMh = norm.pdf(
        logMh,
        loc=muDM + BETA_DM * (Msps - 11.4),
        scale=SIGMA_DM,
    )

    scatter_Mstar = SCATTER.star

    # Likelihood for observed SPS mass
    p_Msps_obs = norm.pdf(logM_sps_obs, loc=Msps, scale=scatter_Mstar)

    # Prior on true SPS mass
    a = 10 ** MODEL_P["log_s_star"]
    loc = MODEL_P["mu_star"]
    scale = MODEL_P["sigma_star"]
    p_Msps_prior = skewnorm.pdf(Msps, a=a, loc=loc, scale=scale)

    # Size likelihood using same relation as mock generator
    mu_Re = MODEL_P["mu_R0"] + MODEL_P["beta_R"] * (Msps - 11.4)
    p_logRe = norm.pdf(grid.logRe, loc=mu_Re, scale=MODEL_P["sigma_R"])

    # sample_factor = np.ones_like(sample_factor)
    p

    Z = sample_factor * p_logMh * p_Msps_obs * p_Msps_prior * p_logRe

    integral = np.trapz(Z, logMh)
    return float(max(integral, 1e-300))


# Public API -----------------------------------------------------------------

def log_likelihood(
    theta: Sequence[float],
    grids: Sequence[LensGrid],
    logM_sps_obs: Sequence[float],
) -> float:
    """Joint log-likelihood for all lenses."""

    muDM, alpha = theta

    try:
        A_eta = cached_A_interp(muDM, alpha)
        if not np.isfinite(A_eta) or A_eta <= 0:
            return -np.inf
    except Exception:
        return -np.inf
    

    A_eta = 1

    logL = 0.0
    for grid, logM_obs in zip(grids, logM_sps_obs):
        L_i = _single_lens_likelihood(grid, float(logM_obs), theta)
        if not np.isfinite(L_i) or L_i <= 0:
            return -np.inf
        logL += np.log(L_i) - np.log(A_eta)

    return float(logL)


def log_posterior(
    theta: Sequence[float],
    grids: Sequence[LensGrid],
    logM_sps_obs: Sequence[float],
) -> float:
    """Posterior = prior + likelihood."""

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, grids, logM_sps_obs)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


__all__ = [
    "precompute_grids",
    "log_prior",
    "log_likelihood",
    "log_posterior",
]
