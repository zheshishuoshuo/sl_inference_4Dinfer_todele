import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm, norm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
from .mock_generator.lens_solver import solve_single_lens
from .mock_generator.lens_model import LensModel
from .utils import selection_function
from .config import SCATTER

def _try_two_image(args):
    logM_star, logRe, logMh, beta, zl, zs = args
    try:
        model = LensModel(
            logM_star=logM_star,
            logM_halo=logMh,
            logRe=logRe,
            zl=zl,
            zs=zs,
        )
        xA, xB = solve_single_lens(model, beta)
        mu1 = model.mu_from_rt(xA)
        mu2 = model.mu_from_rt(xB)
        ok = np.isfinite(mu1) and np.isfinite(mu2)
        return 1 if ok else 0
    except Exception:
        return 0

def _sample_logMh_from_target(mu_DM, logM_sps, sigma_DM, low=11.0, high=15.0, rng=None):
    """从截断正态 p(logMh | mu_DM, logM_sps) 直接采样，避免 IS 权重方差。"""
    if rng is None:
        rng = np.random.default_rng()
    mean = mu_DM + MODEL_PARAMS["deVauc"]["beta_h"] * (logM_sps - 11.4)
    a = (low - mean) / sigma_DM
    b = (high - mean) / sigma_DM
    return truncnorm.rvs(a, b, loc=mean, scale=sigma_DM, size=1, random_state=rng)[0]

def compute_A_geometric(
    mu_DM=12.91,
    alpha_list=(0.1, 0.2),
    n_samples=200_000,
    zl=0.3,
    zs=2.0,
    seed=42,
    n_jobs=None,
    mh_low=11.0,
    mh_high=15.0,
    mode="direct",  # "direct" or "importance"
):
    """
    计算“几何截面版”的 A(α)：只看是否形成双像（不含选择函数、星等极限等）。
    mode="direct":   直接从目标 p(logMh|μDM, logM_sps) 采样（推荐）
    mode="importance":沿用 Uniform(11,15) 重要性采样（对照）
    """
    rng = np.random.default_rng(seed)

    # 生成恒星/尺寸/源位置样本（与原脚本一致）
    data = generate_samples(n_samples)
    logM_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    beta = rng.random(n_samples) ** 0.5

    MODEL_P = MODEL_PARAMS["deVauc"]
    sigma_DM = MODEL_P["sigma_h"]
    beta_DM = MODEL_P["beta_h"]

    results = {}
    for alpha in alpha_list:
        # 实际参与透镜的恒星质量
        logM_star = logM_sps + alpha

        # 采样 logMh
        if mode == "direct":
            logMh = np.array(
                [_sample_logMh_from_target(mu_DM, m_sps, sigma_DM, mh_low, mh_high, rng)
                 for m_sps in logM_sps]
            )
            weights = np.ones(n_samples)  # 直接采样时不需要权重
        else:
            # 重要性采样（和你原来一样）
            logMh = rng.uniform(mh_low, mh_high, n_samples)
            mean = mu_DM + beta_DM * (logM_sps - 11.4)
            p_tgt = norm.pdf(logMh, loc=mean, scale=sigma_DM)
            q = 1.0 / (mh_high - mh_low)
            weights = p_tgt / q

        # 并行判定是否形成双像
        args_iter = zip(logM_star, logRe, logMh, beta, repeat(zl), repeat(zs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            flags = list(
                tqdm(
                    pool.map(_try_two_image, args_iter),
                    total=n_samples,
                    desc=f"two-image? alpha={alpha:.3f} [{mode}]",
                )
            )
        flags = np.array(flags, dtype=float)

        if mode == "direct":
            A = flags.mean()  # 直接采样时，几何概率的无偏估计就是平均
        else:
            # 重要性采样的无偏估计：E_q[ 1_{two-image} * p/q ]
            A = np.sum(flags * weights) / n_samples

        results[alpha] = A

    return results

if __name__ == "__main__":
    out1 = compute_A_geometric(mode="direct", n_samples=200_000, n_jobs=8)
    print("[direct p(logMh|...)]", out1)
    out2 = compute_A_geometric(mode="importance", n_samples=200_000, n_jobs=8)
    print("[importance (Uniform)]", out2)
