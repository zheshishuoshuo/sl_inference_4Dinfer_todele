"""Pre-compute lensing grids for likelihood evaluation.

This module tabulates, for each observed lens, the quantities that are
independent of the population hyper-parameters.  For every lens we solve the
lens equation on a grid of halo masses and store the inferred stellar mass and
an overall geometric + photometric factor used later in the likelihood
integral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .config import SCATTER
from .mock_generator.lens_solver import (
    compute_detJ,
    solve_lens_parameters_from_obs_yc,
)
from .mock_generator.lens_model import LensModel
from .utils import mag_likelihood, selection_function

# -----------------------------------------------------------------------------
# Source magnitude prior -- must be consistent with mock generation
# -----------------------------------------------------------------------------
ALPHA_S = -1.3
M_S_STAR = 24.5
MS_MIN, MS_MAX = 20.0, 30.0
MS_GRID = np.linspace(MS_MIN, MS_MAX, 100)


def _source_mag_prior(ms: np.ndarray) -> np.ndarray:
    L = 10 ** (-0.4 * (ms - M_S_STAR))
    return L ** (ALPHA_S + 1) * np.exp(-L)


P_MS = _source_mag_prior(MS_GRID)
P_MS /= np.trapz(P_MS, MS_GRID)


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------


@dataclass
class LensGrid:
    """Container holding pre-computed quantities for a single lens."""

    logMh_grid: np.ndarray
    logM_star: np.ndarray
    sample_factor: np.ndarray
    logRe: float


# -----------------------------------------------------------------------------
# Grid tabulation
# -----------------------------------------------------------------------------


def tabulate_likelihood_grids(
    mock_observed_data: pd.DataFrame,
    logMh_grid: Iterable[float],
    zl: float = 0.3,
    zs: float = 2.0,
    sigma_m: float = SCATTER.mag,
) -> List[LensGrid]:
    """Compute grids of quantities independent of hyper-parameters.

    Parameters
    ----------
    mock_observed_data:
        DataFrame containing observed image positions and magnitudes.  Required
        columns are ``xA``, ``xB``, ``logRe``, ``magnitude_observedA`` and
        ``magnitude_observedB``.
    logMh_grid:
        Sequence of halo masses (log10 scale) on which to evaluate the grids.
    zl, zs:
        Lens and source redshifts.
    sigma_m:
        Measurement scatter on the observed magnitudes.

    Returns
    -------
    list of :class:`LensGrid`
        One entry per lens with arrays evaluated on ``logMh_grid``.
    """

    logMh_grid = np.asarray(list(logMh_grid))
    results: List[LensGrid] = []

    for _, row in mock_observed_data.iterrows():
        xA = float(row["xA"])
        xB = float(row["xB"])
        logRe = float(row["logRe"])
        m1_obs = float(row["magnitude_observedA"])
        m2_obs = float(row["magnitude_observedB"])

        logMstar_list: List[float] = []
        sample_list: List[float] = []

        for logMh in logMh_grid:
            try:
                logM_star, beta_unit, yc = solve_lens_parameters_from_obs_yc(
                    xA, xB, logRe, logMh, zl, zs
                )
            except Exception:
                logM_star = np.nan
                beta_unit = np.nan
                yc = np.nan

            if np.isnan(logM_star) or np.isnan(beta_unit) or np.isnan(yc):
                logMstar_list.append(np.nan)
                sample_list.append(0.0)
                continue

            try:
                detJ = compute_detJ(xA, xB, logRe, logMh, zl, zs)
            except Exception:
                logMstar_list.append(np.nan)
                sample_list.append(0.0)
                continue

            model = LensModel(
                logM_star=logM_star,
                logM_halo=logMh,
                logRe=logRe,
                zl=zl,
                zs=zs,
            )
            mu1 = model.mu_from_rt(xA)
            mu2 = model.mu_from_rt(xB)
            beta = beta_unit * yc

            L1 = mag_likelihood(m1_obs, mu1, MS_GRID, sigma_m)
            L2 = mag_likelihood(m2_obs, mu2, MS_GRID, sigma_m)
            L3 = selection_function(mu1, m_lim=26.5, ms=MS_GRID, sigma_m=sigma_m)
            L4 = selection_function(mu2, m_lim=26.5, ms=MS_GRID, sigma_m=sigma_m)
            Lphot = np.trapz(P_MS * L1 * L2 , MS_GRID)

            sample_factor =  abs(detJ) * Lphot

            logMstar_list.append(logM_star)
            sample_list.append(sample_factor)

        results.append(
            LensGrid(
                logMh_grid=logMh_grid,
                logM_star=np.array(logMstar_list),
                sample_factor=np.array(sample_list),
                logRe=logRe,
            )
        )

    return results


__all__ = ["LensGrid", "tabulate_likelihood_grids"]
