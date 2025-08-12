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



def _solve_single_lens_mags(args):
    """子任务：给定一组参数，解两像并返回两像的放大率 μA, μB。"""
    M_star_i, Mh_i, Re_i, beta_i, zl, zs = args
    model = LensModel(M_star_i, Mh_i, Re_i, zl, zs)
    xA, xB = solve_single_lens(model, beta_i)
    return model.mu_from_rt(xA), model.mu_from_rt(xB)


def _sample_logMh_from_target(mu_DM, logM_sps, sigma_DM, low=11.0, high=15.0, rng=None):
    """
    从截断正态 p(logMh | mu_DM + beta_h*(logM_sps-11.4), sigma_DM) 直接采样一个 logMh。
    这样避免 Uniform+权重带来的方差与错配。
    """
    if rng is None:
        rng = np.random.default_rng()
    # mean = mu_DM + MODEL_PARAMS["deVauc"]["beta_h"] * (logM_sps - 11.4)
    mean = mu_DM + MODEL_PARAMS["deVauc"]["beta_h"] * ((logM_sps + alpha) - 11.4)
    a = (low - mean) / sigma_DM
    b = (high - mean) / sigma_DM
    # 注意：scipy 的 truncnorm 用标准化区间 [a,b]，loc/scale 再平移缩放
    return truncnorm.rvs(a, b, loc=mean, scale=sigma_DM, size=1, random_state=rng)[0]


def _ms_pdf(ms_grid, alpha_s=-1.3, ms_star=24.5):
    """源星等分布 PDF（归一化）。"""
    L = 10.0 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


def compute_A_one_image_direct(
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
    计算“只卡 A 像阈值”的 A(α)，logMh 直接来自目标分布（无权重）。

    返回：dict {alpha: A_value}
    """
    rng = np.random.default_rng(seed)

    # 1) 生成恒星/尺寸/源位置样本
    pop = generate_samples(n_samples)
    logM_sps = pop["logM_star_sps"]
    logRe = pop["logRe"]
    beta = rng.random(n_samples) ** 0.5

    # 2) 准备源星等积分
    ms_grid = np.linspace(20.0, 30.0, ms_points)
    pdf_ms = _ms_pdf(ms_grid, alpha_s=-1.3, ms_star=24.5)

    # 3) 模型常数
    MODEL_P = MODEL_PARAMS["deVauc"]
    sigma_DM = MODEL_P["sigma_h"]

    out = {}
    for alpha in alpha_list:
        print(f"\n=== alpha = {alpha:.3f} | 只卡 A 像，B 像放开；logMh 直采目标分布 ===")
        logM_star = logM_sps + alpha

        # 4) 逐样本从目标分布直采 logMh（避免 IS 权重错配）
        logMh = np.array(
            [
                _sample_logMh_from_target(
                    mu_DM, msps, sigma_DM, logMh_min, logMh_max, rng
                )
                for msps in logM_sps
            ]
        )

        # 5) 并行解透镜，取两像放大率
        args_iter = zip(logM_star, logMh, logRe, beta, repeat(zl), repeat(zs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            mags = list(
                tqdm(
                    pool.map(_solve_single_lens_mags, args_iter),
                    total=n_samples,
                    desc="solving lenses",
                )
            )
        muA, muB = map(np.array, zip(*mags))

        # 6) 只卡 A 像的选择函数，并对 ms 积分
        selA = selection_function(muA[:, None], m_lim, ms_grid[None, :], SCATTER.mag)
        w_ms = np.trapz(selA * pdf_ms[None, :], ms_grid, axis=1)  # shape (n_samples,)

        # 7) 直采 → 无需权重，几何/选择的平均通过率就是 A
        A_val = float(np.mean(w_ms))
        out[alpha] = A_val
        print(f"→ A_one_img_direct({alpha:.3f}) = {A_val:.10f}")

    return out


if __name__ == "__main__":
    vals = compute_A_one_image_direct(n_jobs=8)
    print("\n结果：", vals)
