import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
from .mock_generator.lens_solver import solve_single_lens
from .mock_generator.lens_model import LensModel
from .utils import selection_function
from .config import SCATTER


def _solve_single_lens_mags(args):
    """并行子任务：计算单个透镜的两像放大率"""
    M_star_i, Mh_i, Re_i, beta_i, zl, zs = args
    model = LensModel(M_star_i, Mh_i, Re_i, zl, zs)
    xA, xB = solve_single_lens(model, beta_i)
    return model.mu_from_rt(xA), model.mu_from_rt(xB)


def compute_A_single(mu_DM=12.91, alpha_list=[0.1, 0.2],
                     n_samples=200000, ms_points=400, m_lim=26.5,
                     zl=0.3, zs=2.0, seed=42, n_jobs=None):
    """
    高精度计算固定 mu_DM 下给定 alpha 列表的 A 值（支持并行）
    """
    rng = np.random.default_rng(seed)

    # 1. 生成样本
    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    beta = rng.random(n_samples) ** 0.5
    logMh_min, logMh_max = 11.0, 15.0
    logMh = rng.uniform(logMh_min, logMh_max, n_samples)

    # 2. 准备 ms 分布
    ms_grid = np.linspace(20.0, 30.0, ms_points)
    L = 10 ** (-0.4 * (ms_grid - 24.5))
    pdf_ms = L ** (-0.3) * np.exp(-L)  # alpha_s=-1.3
    pdf_ms /= np.trapz(pdf_ms, ms_grid)

    MODEL_P = MODEL_PARAMS["deVauc"]
    beta_DM = MODEL_P["beta_h"]
    sigma_DM = MODEL_P["sigma_h"]

    results = {}
    for alpha in alpha_list:
        print(f"\n=== 计算 alpha = {alpha:.3f} ===")
        logM_star = logM_star_sps + alpha

        # 3. 并行计算两像放大率
        args_iter = zip(logM_star, logMh, logRe, beta, repeat(zl), repeat(zs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            mags = list(tqdm(pool.map(_solve_single_lens_mags, args_iter),
                             total=n_samples, desc="solving lenses"))

        mu1, mu2 = map(np.array, zip(*mags))

        # 4. 选择函数积分
        sel1 = selection_function(mu1[:, None], m_lim, ms_grid[None, :], SCATTER.mag)
        sel2 = selection_function(mu2[:, None], m_lim, ms_grid[None, :], SCATTER.mag)
        p_det = sel1 * sel2
        w_ms = np.trapz(p_det * pdf_ms[None, :], ms_grid, axis=1)

        # 5. halo mass PDF 加权
        A_sum = 0.0
        for logM_sps_i, logMh_i, w_i in zip(logM_star_sps, logMh, w_ms):
            mean = mu_DM + beta_DM * (logM_sps_i - 11.4)
            p_Mh = norm.pdf(logMh_i, loc=mean, scale=sigma_DM)
            A_sum += w_i * p_Mh

        Mh_range = logMh_max - logMh_min
        A_val = Mh_range * A_sum / n_samples
        results[alpha] = A_val
        print(f"→ A({alpha:.3f}) = {A_val:.10f}")

    return results


if __name__ == "__main__":
    vals = compute_A_single(n_jobs=8)  # n_jobs 根据 CPU 核数设置
    print("\n最终结果：", vals)
