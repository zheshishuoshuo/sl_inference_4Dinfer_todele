import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import os
import sys
import types

if __package__ is None or __package__ == "":
    # Allow running as a standalone script without installing the package
    package_dir = os.path.dirname(os.path.abspath(__file__))
    pkg = types.ModuleType("compute_norm_acc")
    pkg.__path__ = [package_dir]
    sys.modules["compute_norm_acc"] = pkg
    sys.path.insert(0, os.path.dirname(package_dir))
    from compute_norm_acc.mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
    from compute_norm_acc.mock_generator.lens_solver import solve_single_lens
    from compute_norm_acc.mock_generator.lens_model import LensModel
    from compute_norm_acc.utils import selection_function
    from compute_norm_acc.config import OBS_SCATTER_MAG
else:
    from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
    from .mock_generator.lens_solver import solve_single_lens
    from .mock_generator.lens_model import LensModel
    from .utils import selection_function
    from .config import OBS_SCATTER_MAG


def sample_lens_population(n_samples, zl=0.3, zs=2.0):
    """Generate lens population samples without alpha offset."""
    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    beta = np.random.rand(n_samples) ** 0.5
    # sample halo mass uniformly to allow importance reweighting later
    logMh_min, logMh_max = 11.0, 15.0
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


def _solve_magnification(args):
    """Helper to solve a single lens and return magnifications."""
    logM_star, logRe, logMh, beta, zl, zs = args
    model = LensModel(
        logM_star=logM_star,
        logM_halo=logMh,
        logRe=logRe,
        zl=zl,
        zs=zs,
    )
    xA, xB = solve_single_lens(model, beta)
    return model.mu_from_rt(xA), model.mu_from_rt(xB)


def compute_magnifications(logM_star, logRe, logMh, beta, zl, zs, n_jobs=None):
    """Solve lens equation for each sample and return two magnifications."""
    n = len(logM_star)
    args = zip(logM_star, logRe, logMh, beta, repeat(zl), repeat(zs))
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(
            tqdm(
                pool.map(_solve_magnification, args),
                total=n,
                desc="solving lenses",
            )
        )
    mu1, mu2 = map(np.array, zip(*results))
    return mu1, mu2


def ms_distribution(ms_grid, alpha_s=-1.3, ms_star=24.5):
    """Normalized PDF of source magnitude on a grid."""
    L = 10 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


    # if not (
    #     12.0 < mu0 < 14.0
    #     and 0 < sigmaDM < 1
    #     and 0. < sigma_alpha < 1
    #     and -0.3 < mu_alpha < 0.5
    #     and 0 < beta < 5
    # ):


def build_eta_grid():
    mu_DM_grid = np.linspace(12.5, 13.2, 100)
    alpha_grid = np.linspace(0.07, 0.13, 40)
    return mu_DM_grid, alpha_grid


def compute_A_eta(n_samples=5000, ms_points=15, m_lim=26.5, lens_file="lens_samples.csv"):
    """Compute normalization grid A(eta).

    If a cached lens sample file exists with the requested number of samples,
    reuse it instead of regenerating the lens population and recomputing
    magnifications.  The final A-table is accumulated per sample to avoid
    holding large 4D arrays in memory.
    """

    lens_df = None
    if os.path.exists(lens_file):
        cached = pd.read_csv(lens_file)
        if len(cached) == n_samples:
            lens_df = cached
            print(f"Loaded {n_samples} lens samples from {lens_file}.")

    if lens_df is None:
        samples = sample_lens_population(n_samples)
        lens_df = pd.DataFrame(
            {
                "logM_star_sps": samples["logM_star_sps"],
                "logRe": samples["logRe"],
                "logMh": samples["logMh"],
                "beta": samples["beta"],
                "zl": np.full(n_samples, samples["zl"]),
                "zs": np.full(n_samples, samples["zs"]),
            }
        )
        lens_df.to_csv(lens_file, index=False)
    else:
        samples = {
            "logM_star_sps": lens_df["logM_star_sps"].values,
            "logRe": lens_df["logRe"].values,
            "logMh": lens_df["logMh"].values,
            "beta": lens_df["beta"].values,
            "zl": lens_df.get("zl", pd.Series([0.3])).iloc[0],
            "zs": lens_df.get("zs", pd.Series([2.0])).iloc[0],
            "logMh_min": 11.0,
            "logMh_max": 15.0,
        }

    ms_grid = np.linspace(20.0, 30.0, ms_points)
    pdf_ms = ms_distribution(ms_grid)

    mu_DM_grid, alpha_grid = build_eta_grid()
    mu_grid = mu_DM_grid[:, None]
    A_accum = np.zeros((mu_DM_grid.size, alpha_grid.size))

    # Fixed halo-mass relation parameters from the default model
    MODEL_P = MODEL_PARAMS["deVauc"]
    beta_DM = MODEL_P["beta_h"]
    sigma_DM = MODEL_P["sigma_h"]

    for j, alpha in enumerate(tqdm(alpha_grid, desc="alpha loop")):
        # logM_star = samples["logM_star_sps"] + alpha
        logM_star = samples["logM_star_sps"] + alpha + np.random.normal(0, 0.01, size=n_samples)

        mu1, mu2 = compute_magnifications(
            logM_star,
            samples["logRe"],
            samples["logMh"],
            samples["beta"],
            samples["zl"],
            samples["zs"],
        )
        sel1 = selection_function(mu1[:, None], m_lim, ms_grid[None, :], OBS_SCATTER_MAG)
        sel2 = selection_function(mu2[:, None], m_lim, ms_grid[None, :], OBS_SCATTER_MAG)
        p_det = sel1 * sel2
        w_ms = np.trapz(p_det * pdf_ms[None, :], ms_grid, axis=1)
        w_static = w_ms

        for logM_sps_i, logMh_i, w_i in zip(
            samples["logM_star_sps"], samples["logMh"], w_static
        ):
            mean = mu_grid[:, 0] + beta_DM * (logM_sps_i - 11.4)
            p_Mh = norm.pdf(logMh_i, loc=mean, scale=sigma_DM)
            A_accum[:, j] += w_i * p_Mh

    Mh_range = samples.get("logMh_max", 15.0) - samples.get("logMh_min", 11.0)
    A = Mh_range * A_accum / n_samples

    mu_flat, alpha_flat = np.meshgrid(mu_DM_grid, alpha_grid, indexing="ij")
    df = pd.DataFrame(
        {
            "mu_DM": mu_flat.ravel(),
            "alpha": alpha_flat.ravel(),
            "A": A.ravel(),
        }
    )
    path = os
    df.to_csv("A_eta_table_alpha.csv", index=False)
    return df


if __name__ == "__main__":
    compute_A_eta()
