"""Monte Carlo evaluation of the normalisation factor A(eta).

This module implements the algorithm described in the project notes for
estimating the selection-function normalisation that appears in the
likelihood.  The computation proceeds by Monte Carlo sampling of the lens
population and evaluating the integrand

    T = T1 * T2 * T3

for each sample.  Here:

* ``T1`` is the integral over source magnitude of the detection probability
  for the two lensed images weighted by the source magnitude prior.
* ``T2`` is the weighting from the random source position, proportional to
  the square of the caustic scale ``betamax`` times the uniform variate ``u``.
* ``T3`` is the (untruncated) halo–mass relation ``p(Mh | muDM, Msps)``
  evaluated at the sampled halo mass.

The final estimate of ``A`` is the average of ``T`` over all Monte Carlo
samples with an additional factor from importance sampling the halo mass with
an uninformative uniform proposal.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from .config import SCATTER
from .mock_generator.lens_model import LensModel
from .mock_generator.lens_solver import solve_single_lens
from .mock_generator.mass_sampler import MODEL_PARAMS, generate_samples
from .utils import selection_function

# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------


def sample_lens_population(n_samples: int, zl: float = 0.3, zs: float = 2.0):
    """Draw Monte Carlo samples of the lens population.

    Parameters
    ----------
    n_samples
        Number of Monte Carlo samples to draw.
    zl, zs
        Lens and source redshifts.

    Returns
    -------
    dict
        Dictionary containing sampled stellar masses, sizes, halo masses and
        source-position variables along with the bounds of the halo-mass
        proposal distribution.
    """

    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    # Random source position u \in [0,1]
    beta = np.random.rand(n_samples)
    # Uniform proposal for halo mass to allow importance reweighting
    logMh_min, logMh_max = 8.0, 18.0
    logMh = np.random.uniform(logMh_min, logMh_max, n_samples)

    return {
        "logM_star_sps": logM_star_sps,
        "logRe": logRe,
        "beta": beta,
        "logMh": logMh,
        "logMh_min": logMh_min,
        "logMh_max": logMh_max,
        "zl": zl,
        "zs": zs,
    }


# -----------------------------------------------------------------------------
# Lens-equation solver
# -----------------------------------------------------------------------------


def _solve_magnification(args):
    """Solve a single lens configuration returning magnifications and caustic."""

    logM_star, logRe, logMh, beta, zl, zs = args
    try:
        model = LensModel(
            logM_star=logM_star, logM_halo=logMh, logRe=logRe, zl=zl, zs=zs
        )
        xA, xB = solve_single_lens(model, beta)
        ycaustic = model.solve_ycaustic() or 0.0
        mu1 = model.mu_from_rt(xA)
        mu2 = model.mu_from_rt(xB)
        if not (np.isfinite(mu1) and np.isfinite(mu2) and ycaustic > 0):
            return (np.nan, np.nan, 0.0)
        return (mu1, mu2, ycaustic)
    except Exception:
        return (np.nan, np.nan, 0.0)


def compute_magnifications(
    logM_star: np.ndarray,
    logRe: np.ndarray,
    logMh: np.ndarray,
    beta: np.ndarray,
    zl: float,
    zs: float,
    n_jobs: int | None = None,
):
    """Compute magnifications for each Monte Carlo sample."""

    n = len(logM_star)
    args = zip(logM_star, logRe, logMh, beta, repeat(zl), repeat(zs))
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(
            tqdm(
                pool.map(_solve_magnification, args),
                total=n,
                desc="solving lenses",
                leave=False,
            )
        )
    mu1, mu2, betamax = map(np.array, zip(*results))
    return mu1, mu2, betamax


# -----------------------------------------------------------------------------
# Source magnitude prior
# -----------------------------------------------------------------------------


# def ms_distribution(ms_grid: np.ndarray, alpha_s: float = -1.3, ms_star: float = 24.5):
#     """Normalised PDF of the unlensed source magnitude."""

#     L = 10 ** (-0.4 * (ms_grid - ms_star))
#     pdf = L ** (alpha_s + 1) * np.exp(-L)
#     pdf /= np.trapz(pdf, ms_grid)
#     return pdf


def ms_distribution(ms_grid: np.ndarray, alpha_s: float = -1.3, ms_star: float = 24.5):
    "mockgaussian"
    return 


# -----------------------------------------------------------------------------
# Main A(eta) computation
# -----------------------------------------------------------------------------


def build_eta_grid():
    """Return default grids for ``mu_DM`` and ``alpha``."""

    mu_DM_grid = np.linspace(12.5, 13.5, 100)
    alpha_grid = np.linspace(0.0, 0.5, 100)
    return mu_DM_grid, alpha_grid


def compute_A_eta(
    n_samples: int = 5000,
    ms_points: int = 100,
    m_lim: float = 26.5,
    n_jobs: int | None = None,
):
    """Monte Carlo estimate of the normalisation grid ``A(eta)``.

    Parameters are chosen to mirror the pseudocode provided in the
    documentation.  The outer loop iterates over ``alpha`` and for each sample
    draws ``(Msps, Re, u, Mh)``.  For each ``muDM`` the halo-mass relation is
    evaluated and accumulated.
    """

    samples = sample_lens_population(n_samples)

    ms_grid = np.linspace(20.0, 30.0, ms_points)
    p_ms = ms_distribution(ms_grid)

    mu_DM_grid, alpha_grid = build_eta_grid()
    A_accum = np.zeros((mu_DM_grid.size, alpha_grid.size))

    # Parameters of the halo mass relation
    model_p = MODEL_PARAMS["deVauc"]
    beta_DM = model_p["beta_h"]
    sigma_DM = model_p["sigma_h"]

    for j, alpha in enumerate(tqdm(alpha_grid, desc="alpha loop")):
        # Mstar used in lensing is Msps + alpha
        logM_star = samples["logM_star_sps"] + alpha

        mu1, mu2, betamax = compute_magnifications(
            logM_star,
            samples["logRe"],
            samples["logMh"],
            samples["beta"],
            samples["zl"],
            samples["zs"],
            n_jobs=n_jobs,
        )

        mu1 = np.nan_to_num(mu1, nan=0.0, posinf=0.0, neginf=0.0)
        mu2 = np.nan_to_num(mu2, nan=0.0, posinf=0.0, neginf=0.0)
        betamax = np.nan_to_num(betamax, nan=0.0, posinf=0.0, neginf=0.0)

        mu1_clip = np.clip(mu1, 1e-6, 1e6)
        mu2_clip = np.clip(mu2, 1e-6, 1e6)
        valid = (mu1_clip > 0) & (mu2_clip > 0) & (betamax > 0)

        # ---- T1: integrate detection probability over source magnitude ----
        p_det = np.zeros((n_samples, ms_grid.size))
        if np.any(valid):
            sel1 = selection_function(
                mu1_clip[valid, None], m_lim, ms_grid[None, :], SCATTER.mag
            )
            sel2 = selection_function(
                mu2_clip[valid, None], m_lim, ms_grid[None, :], SCATTER.mag
            )
            p_det[valid] = sel1 * sel2
        p_det = np.clip(np.nan_to_num(p_det, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        T1 = np.trapz(p_det * p_ms[None, :], ms_grid, axis=1)
        T1 = np.nan_to_num(T1, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- T2: source position weighting ----
        T2 = betamax**2 * samples["beta"]

        # Combined static weight per Monte Carlo sample
        w_static = T1 * T2

        # ---- T3: halo-mass relation for each muDM ----
        for logM_sps_i, logMh_i, w_i in zip(
            samples["logM_star_sps"], samples["logMh"], w_static
        ):
            if not (np.isfinite(w_i) and w_i > 0):
                continue

            mean = mu_DM_grid + beta_DM * (logM_sps_i - 11.4)
            p_Mh = norm.pdf(logMh_i, loc=mean, scale=sigma_DM)
            A_accum[:, j] += w_i * p_Mh

    Mh_range = samples["logMh_max"] - samples["logMh_min"]
    A = Mh_range * A_accum / n_samples
    A = np.maximum(np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), 1e-300)

    mu_flat, alpha_flat = np.meshgrid(mu_DM_grid, alpha_grid, indexing="ij")
    df = pd.DataFrame(
        {"mu_DM": mu_flat.ravel(), "alpha": alpha_flat.ravel(), "A": A.ravel()}
    )
    path = os.path.join(os.path.dirname(__file__), "A_eta_table_alpha.csv")
    df.to_csv(path, index=False)
    return df


# Provide a convenience alias reflecting the terminology in the documentation
normfactor = compute_A_eta


if __name__ == "__main__":
    compute_A_eta()