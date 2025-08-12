# compute_A_one_image.py
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.stats import truncnorm

from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
from .mock_generator.lens_solver import solve_single_lens
from .mock_generator.lens_model import LensModel
from .utils import selection_function
from .config import SCATTER

# compute_A_one_image_compare.py
# -*- coding: utf-8 -*-
"""
对比：锚定在 Msps vs 锚定在 Mstar(含alpha) 的 logMh 条件均值
场景：只卡 A 像（B 像放开），直接从目标分布采样 logMh（截断正态，无权重）

运行：
  python -m sl_inference_only_muDMalpha.compute_A_one_image_compare

输出：
  - 控制台打印两套 A(alpha) 数值
  - 导出 CSV: A_one_image_compare.csv
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.stats import truncnorm




def _solve_single_lens_mags(args):
    """子任务：给定参数，解两像并返回 μA, μB。"""
    M_star_i, Mh_i, Re_i, beta_i, zl, zs = args
    model = LensModel(M_star_i, Mh_i, Re_i, zl, zs)
    xA, xB = solve_single_lens(model, beta_i)
    return model.mu_from_rt(xA), model.mu_from_rt(xB)


def _ms_pdf(ms_grid, alpha_s=-1.3, ms_star=24.5):
    """源星等分布 PDF（归一化）。"""
    L = 10.0 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


def _sample_logMh_truncnorm(mean, sigma, low, high, rng):
    """从截断正态 TruncN(mean, sigma; [low, high]) 采一个值。"""
    a = (low - mean) / sigma
    b = (high - mean) / sigma
    return truncnorm.rvs(a, b, loc=mean, scale=sigma, size=1, random_state=rng)[0]


def compute_A_one_image_direct_compare(
    mu_DM=12.91,
    alpha_list=(0.1, 0.2),
    n_samples=200_000,
    ms_points=400,
    m_lim=26.5,
    zl=0.3,
    zs=2.0,
    seed=42,
    n_jobs=None,
    logMh_min=11.0,
    logMh_max=15.0,
):
    """
    计算两条 A(α) 曲线：
      - anchor='sps' : E[logMh | mu_DM, logM_sps]
      - anchor='star': E[logMh | mu_DM, logM_star=logM_sps+alpha]
    只卡 A 像，B 像放开；logMh 直接目标分布采样；无权重。
    """
    rng = np.random.default_rng(seed)

    # 1) 人群样本
    pop = generate_samples(n_samples)
    logM_sps = pop["logM_star_sps"]
    logRe = pop["logRe"]
    beta = rng.random(n_samples) ** 0.5

    # 2) 源星等积分网格
    ms_grid = np.linspace(20.0, 30.0, ms_points)
    pdf_ms = _ms_pdf(ms_grid, alpha_s=-1.3, ms_star=24.5)

    # 3) 模型参数
    P = MODEL_PARAMS["deVauc"]
    sigma_DM = P["sigma_h"]
    beta_DM = P["beta_h"]

    out_sps = {}
    out_star = {}

    for alpha in alpha_list:
        print(f"\n=== alpha = {alpha:.3f} | 只卡 A 像；logMh 直采目标分布 ===")
        logM_star = logM_sps + alpha

        # 4) 两种锚定下的 logMh 直采
        # 4a) Msps 锚定：mean = mu_DM + beta_h*(logM_sps - 11.4)
        mean_sps = mu_DM + beta_DM * (logM_sps - 11.4)
        logMh_sps = np.array(
            [_sample_logMh_truncnorm(m, sigma_DM, logMh_min, logMh_max, rng) for m in mean_sps]
        )

        # 4b) Mstar 锚定（含 alpha）：mean = mu_DM + beta_h*((logM_sps+alpha) - 11.4)
        mean_star = mu_DM + beta_DM * ((logM_sps + alpha) - 11.4)
        logMh_star = np.array(
            [_sample_logMh_truncnorm(m, sigma_DM, logMh_min, logMh_max, rng) for m in mean_star]
        )

        # 5) 并行算 μA, μB（同样的 logM_star, logRe, beta；不同的是 logMh）
        # 5a) Msps 锚定
        args_iter_sps = zip(logM_star, logMh_sps, logRe, beta, repeat(zl), repeat(zs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            mags = list(
                tqdm(
                    pool.map(_solve_single_lens_mags, args_iter_sps),
                    total=n_samples,
                    desc="solving lenses [anchor=Msps]",
                )
            )
        muA_sps, muB_sps = map(np.array, zip(*mags))

        # 5b) Mstar 锚定
        args_iter_star = zip(logM_star, logMh_star, logRe, beta, repeat(zl), repeat(zs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            mags = list(
                tqdm(
                    pool.map(_solve_single_lens_mags, args_iter_star),
                    total=n_samples,
                    desc="solving lenses [anchor=Mstar]",
                )
            )
        muA_star, muB_star = map(np.array, zip(*mags))

        # 6) 只卡 A 像，源星等积分
        selA_sps = selection_function(muA_sps[:, None], m_lim, ms_grid[None, :], SCATTER.mag)
        w_ms_sps = np.trapz(selA_sps * pdf_ms[None, :], ms_grid, axis=1)
        A_sps = float(np.mean(w_ms_sps))
        out_sps[alpha] = A_sps

        selA_star = selection_function(muA_star[:, None], m_lim, ms_grid[None, :], SCATTER.mag)
        w_ms_star = np.trapz(selA_star * pdf_ms[None, :], ms_grid, axis=1)
        A_star = float(np.mean(w_ms_star))
        out_star[alpha] = A_star

        print(f"→ A_one_img [anchor=Msps] ({alpha:.3f}) = {A_sps:.10f}")
        print(f"→ A_one_img [anchor=Mstar]({alpha:.3f}) = {A_star:.10f}")

    # 汇总成表
    df = pd.DataFrame({
        "alpha": alpha_list,
        "A_one_img_anchor_Msps": [out_sps[a] for a in alpha_list],
        "A_one_img_anchor_Mstar": [out_star[a] for a in alpha_list],
    })
    return df


if __name__ == "__main__":
    df = compute_A_one_image_direct_compare(n_jobs=8)
    print("\n结果：")
    print(df.to_string(index=False))
    df.to_csv("A_one_image_compare.csv", index=False)
    print("\n已写出：A_one_image_compare.csv")
