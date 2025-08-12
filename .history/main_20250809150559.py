from __future__ import annotations
import numpy as np
import seaborn as sns
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import os

from .mock_generator.mock_generator import run_mock_simulation
from .likelihood import precompute_grids, log_likelihood
from .run_mcmc import run_mcmc
from .config import SCATTER

# 如果用 dynesty / pymultinest，需要额外安装
import dynesty
try:
    import pymultinest
except ImportError:
    pymultinest = None

matplotlib.use("TkAgg")

# 选择采样方法: "emcee", "dynesty", "pymultinest"
SAMPLER_METHOD = "pymultinest"



def run_emcee_sampler(grids, logM_sps_obs):
    """原 emcee 版本"""
    nsteps = 5000
    sampler = run_mcmc(
        grids,
        logM_sps_obs,
        nsteps=nsteps,
        nwalkers=20,
        initial_guess=np.array([12.6, 0.2]),
        backend_file="chains_bless_new_test5.h5",
        parallel=True,
        nproc=mp.cpu_count() - 3,
    )
    chain = sampler.get_chain(discard=nsteps - 2000, flat=True)
    return chain.reshape(-1, chain.shape[-1])


def run_dynesty_sampler(grids, logM_sps_obs):
    """dynesty 全局采样"""
    def prior_transform(u):
        muDM = 10.0 + 6.0 * u[0]         # μ_DM in [10,16]
        alpha = -0.5 + 1.5 * u[1]        # α in [-0.5, 1.0]
        return muDM, alpha

    def loglike(theta):
        muDM, alpha = theta
        return log_likelihood(muDM, alpha, grids, logM_sps_obs)

    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=2, nlive=500)
    sampler.run_nested()
    return sampler.results.samples


def run_pymultinest_sampler(grids, logM_sps_obs):
    """pymultinest 多峰采样"""
    if pymultinest is None:
        raise ImportError("pymultinest is not installed. Please install it first.")

    def prior_transform(cube, ndim, nparams):
        cube[0] = 10.0 + 6.0 * cube[0]         # μ_DM
        cube[1] = -0.5 + 1.5 * cube[1]         # α

    def loglike(cube, ndim, nparams):
        muDM, alpha = cube[0], cube[1]
        return log_likelihood(muDM, alpha, grids, logM_sps_obs)

    out_dir = "chains_multinest"
    os.makedirs(out_dir, exist_ok=True)

    pymultinest.run(
        loglike, prior_transform, 2,
        outputfiles_basename=os.path.join(out_dir, "mn_"),
        n_live_points=500, resume=False, verbose=True
    )

    results = pymultinest.Analyzer(n_params=2, outputfiles_basename=os.path.join(out_dir, "mn_"))
    return results.get_equal_weighted_posterior()[:, :-1]  # 去掉最后一列loglike


def main() -> None:
    chain_file = os.path.join(os.path.dirname(__file__), "chains", "chains.h5")
    if os.path.exists(chain_file):
        os.remove(chain_file)

    # 模拟数据
    mock_lens_data, mock_observed_data = run_mock_simulation(1000, logalpha=0.1)
    logM_sps_obs = mock_observed_data["logM_star_sps_observed"].values
    mock_lens_data.to_csv("mock_lens_data.csv", index=False)

    # Halo质量grid
    logMh_grid = np.linspace(11.5, 14.0, 100)
    grids = precompute_grids(mock_observed_data, logMh_grid, sigma_m=SCATTER.mag)

    # 选择采样器
    if SAMPLER_METHOD == "emcee":
        samples = run_emcee_sampler(grids, logM_sps_obs)
    elif SAMPLER_METHOD == "dynesty":
        samples = run_dynesty_sampler(grids, logM_sps_obs)
    elif SAMPLER_METHOD == "pymultinest":
        samples = run_pymultinest_sampler(grids, logM_sps_obs)
    else:
        raise ValueError("Unknown SAMPLER_METHOD")

    # 画后验
    param_names = [r"$\mu_{DM}$", r"$\alpha$"]
    df_samples = pd.DataFrame(samples, columns=param_names)
    true_values = [12.91, 0.1]

    g = sns.pairplot(df_samples, diag_kind="kde", markers=".", plot_kws={"alpha": 0.1, "s": 10}, corner=True)

    # 真值线
    for i, ax in enumerate(np.diag(g.axes)):
        ax.axvline(true_values[i], color="red", linestyle="--", linewidth=1)
    for i in range(len(true_values)):
        for j in range(len(true_values)):
            if i > j:
                ax = g.axes[i, j]
                ax.axvline(true_values[j], color="red", linestyle="--", linewidth=1)
                ax.axhline(true_values[i], color="red", linestyle="--", linewidth=1)

    plt.show()
    print(f"Finished {SAMPLER_METHOD} sampling. Samples shape:", samples.shape)


if __name__ == "__main__":
    main()
