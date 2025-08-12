# main.py
from __future__ import annotations
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

from .mock_generator.mock_generator import run_mock_simulation
from .likelihood import precompute_grids
from .run_mcmc import run_mcmc
from .config import SCATTER

matplotlib.use("TkAgg")

# 选择采样器: "emcee", "dynesty", "pymultinest"
SAMPLER_METHOD = "dynesty"

def main() -> None:
    # 生成模拟数据
    mock_lens_data, mock_observed_data = run_mock_simulation(100, logalpha=0.1)
    logM_sps_obs = mock_observed_data["logM_star_sps_observed"].values

    # 预计算
    logMh_grid = np.linspace(11.5, 14.0, 100)
    grids = precompute_grids(mock_observed_data, logMh_grid, sigma_m=SCATTER.mag)

    # 运行采样
    samples = run_mcmc(
        grids,
        logM_sps_obs,
        method=SAMPLER_METHOD,
        nwalkers=20,
        nsteps=5000,
        initial_guess=np.array([12.6, 0.2]),
        parallel=True
    )

    # 画后验
    param_names = [r"$\mu_{DM}$", r"$\alpha$"]
    df_samples = pd.DataFrame(samples, columns=param_names)
    true_values = [12.91, 0.1]

    g = sns.pairplot(df_samples, diag_kind="kde", markers=".", plot_kws={"alpha": 0.1, "s": 10}, corner=True)
    for i, ax in enumerate(np.diag(g.axes)):
        ax.axvline(true_values[i], color="red", linestyle="--", linewidth=1)
    for i in range(len(true_values)):
        for j in range(len(true_values)):
            if i > j:
                ax = g.axes[i, j]
                ax.axvline(true_values[j], color="red", linestyle="--", linewidth=1)
                ax.axhline(true_values[i], color="red", linestyle="--", linewidth=1)
    plt.show()

if __name__ == "__main__":
    main()
