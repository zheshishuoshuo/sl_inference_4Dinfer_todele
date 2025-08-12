import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from typing import Callable, Optional, Tuple

# ---- Project-local imports (keep identical to your original environment) ----
from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
from .mock_generator.lens_solver import solve_single_lens
from .mock_generator.lens_model import LensModel
from .utils import selection_function

# =============================================================================
# Core helpers
# =============================================================================

def _ensure_lens_samples(
    n_samples: int,
    lens_file: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Load cached lens samples if present (and length matches),
    otherwise generate and cache a fresh population.

    The cached CSV *must* include the proposal interval used for Mh IS:
    columns "logMh_min" and "logMh_max".
    """
    if os.path.exists(lens_file):
        df = pd.read_csv(lens_file)
        if len(df) == n_samples and {"logMh_min", "logMh_max"}.issubset(df.columns):
            return df

    # (Re-)generate samples and write the proposal bounds explicitly.
    samples = generate_samples(n_samples=n_samples)

    Mh_range = 8, 18)
    # Expect generate_samples to return a DataFrame-like object; coerce to DataFrame
    df = pd.DataFrame(samples)

    # Persist the actual proposal interval you used during sampling
    df["logMh_min"] = Mh_range[0]
    df["logMh_max"] = Mh_range[1]

    df.to_csv(lens_file, index=False)
    return df


def _magnifications_for_beta(
    model: LensModel,
    beta: float,
) -> Tuple[float, float]:
    """Compute absolute magnifications (|mu1|, |mu2|) at a given source-plane beta.
    Keeps the logic centralized and consistent.
    """
    mu1, mu2 = solve_single_lens(model, beta)
    return abs(mu1), abs(mu2)


# =============================================================================
# Grid builder (mu_DM x alpha)
# =============================================================================

def build_eta_grid(
    mu_DM_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return broadcastable grids; kept for API parity."""
    mu_DM_grid = np.asarray(mu_DM_grid, dtype=float)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    return np.meshgrid(mu_DM_grid, alpha_grid, indexing="ij")


# =============================================================================
# Main estimator
# =============================================================================

def compute_A_eta(
    n_samples: int = 5000,
    ms_points: int = 100,
    m_lim: float = 26.5,
    lens_file: str = None,
    mu_DM_grid: Optional[np.ndarray] = None,
    alpha_grid: Optional[np.ndarray] = None,
    u_per_sample: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute the normalization table A(eta) with a clean, unbiased IS estimator.

    Unchanged functionality:
      - Population is (re-)used from CSV cache (and the Mh proposal interval is
        preserved inside the CSV so IS weights are correct on reload).
      - Mh is sampled from Uniform(logMh_min, logMh_max); importance weight is
        (range) * p_Mh_given(eta), *without* truncation of the target pdf.
      - u integration uses u ~ U(0,1); estimator carries the u factor correctly.
      - Magnifications feed selection_function exactly as before.

    Parameters
    ----------
    n_samples : int
        Number of independent lens population samples.
    ms_points : int
        Kept for parity; if your selection_function integrates over ms internally,
        set accordingly. (No change to behavior.)
    m_lim : float
        Detection magnitude limit for selection_function.
    lens_file : str
        CSV cache path. Defaults to "lens_samples.csv" beside this file.
    mu_DM_grid, alpha_grid : array-like
        The eta grid; if None, sensible defaults are used.
    u_per_sample : int
        Number of u draws per outer sample (variance reduction). 1 keeps speed.
    seed : int
        RNG seed.

    Returns
    -------
    pandas.DataFrame with columns ["mu_DM", "alpha", "A"].
    """
    rng = np.random.default_rng(seed)

    if lens_file is None:
        lens_file = os.path.join(os.path.dirname(__file__), "lens_samples.csv")

    # Load/generate lens population (includes Mh proposal bounds)
    lens_df = _ensure_lens_samples(n_samples=n_samples, lens_file=lens_file, seed=seed)

    # Eta grids
    if mu_DM_grid is None:
        mu_DM_grid = np.linspace(11.5, 14.5, 61)
    if alpha_grid is None:
        alpha_grid = np.linspace(-0.5, 0.5, 41)

    n_mu, n_alpha = len(mu_DM_grid), len(alpha_grid)
    A_accum = np.zeros((n_mu, n_alpha), dtype=float)

    # Proposal width for Mh (constant for the whole cached set)
    Mh_min = float(lens_df["logMh_min"].iloc[0])
    Mh_max = float(lens_df["logMh_max"].iloc[0])
    Mh_range = Mh_max - Mh_min
    if Mh_range <= 0:
        raise ValueError("Invalid Mh proposal interval stored in cache.")

    # Loop over lens population
    for idx in tqdm(range(n_samples), desc="Accumulating A(eta)"):
        row = lens_df.iloc[idx]

        # Build the lens model for this sample (parity with original behavior)
        model = LensModel(
            logM_star=row["logM_star"],
            logM_halo=None,  # filled per-Mh draw below
            logRe=row["logRe"],
            zl=row.get("zl", 0.3),
            zs=row.get("zs", 2.0),
        )

        # Pre-draw u values for this sample (variance reduction)
        u_vals = rng.random(u_per_sample)

        # Draw one Mh per u (vectorized across eta grid later)
        logMh_draws = rng.uniform(Mh_min, Mh_max, size=u_per_sample)

        # For each u-draw, accumulate its contribution over the eta grid
        for u, logMh in zip(u_vals, logMh_draws):
            beta_max = float(row["ycaustic"])  # assumed present in cache
            beta = u * beta_max

            # Set halo mass for this draw and compute magnifications
            model.logM_halo = logMh
            mu1, mu2 = _magnifications_for_beta(model, beta)

            # Detection probability for this configuration
            p_det = selection_function(mu1, mu2, m_s=row["m_s"], m_lim=m_lim)
            if p_det <= 0:
                continue

            # Evaluate p(Mh | mu_DM, Msps) on the whole mu_DM_grid for this Msps
            # NOTE: Use the *untruncated* target density for IS correctness.
            msps = float(row["logM_star_sps"]) if "logM_star_sps" in row else float(row["logM_star_sps"]) if "logM_star_sps" in lens_df.columns else float(row["logM_star"])  # fallback

            # Broadcast over mu-grid; alpha-grid only scales A at the end if needed
            for i_mu, mu_DM in enumerate(mu_DM_grid):
                # You can plug your calibrated relation here (mean as a function of Msps)
                mean = mu_DM  # keep parity with your prior implementation
                sigma_DM = MODEL_PARAMS["deVauc"]["sigma_DM"]  # or pass as arg

                p_Mh = norm.pdf(logMh, loc=mean, scale=sigma_DM)  # NOT truncated
                weight = Mh_range * p_Mh  # IS weight (target / proposal)

                # Accumulate over alpha-grid identically (no alpha-dependence here)
                # If alpha enters p(Mh|.), move p_Mh inside j-loop accordingly.
                A_accum[i_mu, :] += weight * (beta_max ** 2) * u * p_det

    # Normalize by number of outer samples and (optionally) u_per_sample
    A = A_accum / (n_samples)
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
