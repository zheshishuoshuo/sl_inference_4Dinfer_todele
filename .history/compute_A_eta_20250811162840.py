import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from typing import Tuple

# --- Keep project-local imports identical to your codebase ---
from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
from .mock_generator.lens_solver import solve_single_lens
from .mock_generator.lens_model import LensModel
from .utils import selection_function
from .config import SCATTER

# =============================================================================
# Distributions & helpers (interfaces unchanged)
# =============================================================================

def sample_lens_population(n_samples: int, zl: float = 0.3, zs: float = 2.0):
    """Generate lens population samples.
    Matches your original sampling interfaces/columns, but we won't use the pre-sampled
    beta for u-integration (we will draw u ourselves per-sample).
    """
    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    # beta here is unused for the correct u-integral, but kept for cache parity
    beta = np.random.rand(n_samples)
    # Mh proposal (uniform) for IS
    logMh_min, logMh_max = 8.0, 18.0
    logMh = np.random.uniform(logMh_min, logMh_max, n_samples)
    return {
        "logM_star_sps": logM_star_sps,
        "logRe": logRe,
        "logMh": logMh,
        "beta": beta,
        "logMh_min": logMh_min,
        "logMh_max": logMh_max,
        "zl": zl,
        "zs": zs,
    }


def ms_distribution(ms_grid: np.ndarray, alpha_s: float = -1.3, ms_star: float = 24.5) -> np.ndarray:
    """Normalized PDF of source magnitude on a grid (unchanged form)."""
    L = 10 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


def build_eta_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Your default eta grid (kept)."""
    mu_DM_grid = np.linspace(10, 16, 100)
    alpha_grid = np.linspace(0.0, 0.5, 100)
    return mu_DM_grid, alpha_grid


# =============================================================================
# Main: A(eta) with the correct integral \int u P_det(beta_max u, psi) du
# =============================================================================

def compute_A_eta(
    n_samples: int = 5000,
    ms_points: int = 100,
    m_lim: float = 26.5,
    u_per_sample: int = 1,
    lens_file: str = None,
) -> pd.DataFrame:
    """Compute A(eta) using the integral form
        A(eta) = E_{Msps, Re, ms, Mh} [ beta_max(psi)^2 * \int_0^1 u P_det(beta_max u, psi) du ]
    with Mh drawn from a *uniform* proposal and reweighted by p(Mh|mu_DM, Msps).

    The distributions and function interfaces are kept identical to your old code.
    """
    rng = np.random.default_rng(42)

    if lens_file is None:
        lens_file = os.path.join(os.path.dirname(__file__), "lens_samples.csv")

    # --- Load or create cached lens population (and persist Mh proposal bounds) ---
    if os.path.exists(lens_file):
        lens_df = pd.read_csv(lens_file)
        if len(lens_df) != n_samples:
            # regenerate to match n_samples
            samples = sample_lens_population(n_samples)
            lens_df = pd.DataFrame(samples)
            lens_df.to_csv(lens_file, index=False)
    else:
        samples = sample_lens_population(n_samples)
        lens_df = pd.DataFrame(samples)
        lens_df.to_csv(lens_file, index=False)

    # Ensure proposal bounds exist and are used (not min/max of the realized sample)
    if not {"logMh_min", "logMh_max"}.issubset(lens_df.columns):
        raise RuntimeError("lens_samples.csv must contain logMh_min/logMh_max for IS.")
    Mh_min = float(lens_df["logMh_min"].iloc[0])
    Mh_max = float(lens_df["logMh_max"].iloc[0])
    Mh_range = max(1e-12, Mh_max - Mh_min)

    # ms grid
    ms_grid = np.linspace(20.0, 30.0, ms_points)
    pdf_ms = ms_distribution(ms_grid)

    # eta grid
    mu_DM_grid, alpha_grid = build_eta_grid()
    n_mu, n_alpha = mu_DM_grid.size, alpha_grid.size
    A_accum = np.zeros((n_mu, n_alpha))

    # Halo-mass relation params
    MODEL_P = MODEL_PARAMS["deVauc"]
    beta_DM = MODEL_P["beta_h"]
    sigma_DM = MODEL_P["sigma_h"]
    Msps_pivot = 11.4  # same as your old code

    # === Outer loop over alpha ===
    for j, alpha in enumerate(tqdm(alpha_grid, desc="alpha loop")):
        # Per-sample lens model ingredients
        logM_sps = lens_df["logM_star_sps"].values
        logRe = lens_df["logRe"].values
        logMh_samples = lens_df["logMh"].values  # uniform proposal draws
        zl = float(lens_df.get("zl", pd.Series([0.3])).iloc[0])
        zs = float(lens_df.get("zs", pd.Series([2.0])).iloc[0])

        # Stellar mass including alpha offset
        logM_star = logM_sps + alpha

        # For each sample, we need beta_max and u-average of P_det evaluated at beta = u * beta_max
        # We compute ycaustic once per sample; then loop over u.
        for i in range(n_samples):
            model = LensModel(
                logM_star=logM_star[i],
                logM_halo=logMh_samples[i],  # this draw will be reweighted via IS
                logRe=logRe[i],
                zl=zl,
                zs=zs,
            )

            # beta_max (caustic size)
            ycaustic = model.solve_ycaustic() or 0.0
            if not (np.isfinite(ycaustic) and ycaustic > 0):
                continue

            # u Monte Carlo integral: \int_0^1 u P_det(beta_max u, psi) du
            # We'll average over u_per_sample draws.
            u_vals = rng.random(u_per_sample)
            p_u = 0.0
            for u in u_vals:
                # solve_single_lens expects beta_unit in [0,1]
                xA, xB = solve_single_lens(model, u)
                mu1 = abs(model.mu_from_rt(xA))
                mu2 = abs(model.mu_from_rt(xB))
                if not (np.isfinite(mu1) and np.isfinite(mu2)):
                    continue

                # Detection probability integrated over ms
                sel1 = selection_function(mu1, m_lim, ms_grid, SCATTER.mag)
                sel2 = selection_function(mu2, m_lim, ms_grid, SCATTER.mag)
                p_det_ms = np.clip(sel1 * sel2, 0.0, 1.0)
                w_ms = np.trapz(p_det_ms * pdf_ms, ms_grid)
                if np.isfinite(w_ms) and w_ms > 0:
                    p_u += u * w_ms

            if p_u <= 0:
                continue
            p_u /= u_per_sample  # Monte Carlo average over u

            # Importance weight for Mh: (range) * p(Mh | mu_DM, Msps)
            for i_mu, mu_DM in enumerate(mu_DM_grid):
                mean = mu_DM + beta_DM * (logM_sps[i] - Msps_pivot)
                p_Mh = norm.pdf(logMh_samples[i], loc=mean, scale=sigma_DM)  # untruncated target
                w_is = Mh_range * p_Mh
                A_accum[i_mu, j] += (ycaustic ** 2) * p_u * w_is

    # Normalize by number of outer samples
    A = A_accum / n_samples
    A = np.maximum(A, 1e-300)

    # Flatten to table
    mu_flat, alpha_flat = np.meshgrid(mu_DM_grid, alpha_grid, indexing="ij")
    df = pd.DataFrame({
        "mu_DM": mu_flat.ravel(),
        "alpha": alpha_flat.ravel(),
        "A": A.ravel(),
    })

    out_path = os.path.join(os.path.dirname(__file__), "A_eta_table_alpha.csv")
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    compute_A_eta()
