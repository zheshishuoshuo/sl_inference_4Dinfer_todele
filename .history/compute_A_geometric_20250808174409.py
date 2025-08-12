# compute_A_one_image.py
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from compute_norm_acc.mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
from compute_norm_acc.mock_generator.lens_solver import solve_single_lens
from compute_norm_acc.mock_generator.lens_model import LensModel
from compute_norm_acc.utils import selection_function
from compute_norm_acc.config import SCATTER

def _solve_single_lens_mags(args):
    M_star_i, Mh_i, Re_i, beta_i, zl, zs = args
    model = LensModel(M_star_i, Mh_i, Re_i, zl, zs)
    xA, xB = solve_single_lens(model, beta_i)
    return model.mu_from_rt(xA), model.mu_from_rt(xB)

def compute_A_one_image(mu_DM=12.91, alpha_list=(0.1, 0.2),
                        n_samples=200_000, ms_points=400, m_lim=26.5,
                        zl=0.3, zs=2.0, seed=42, n_jobs=None,
                        logMh_min=11.0, logMh_max=15.0):
    """
    只对 A 像设亮度阈值的 A(α)，B 像不设限。
    其他与 compute_A_single 相同。
    """
    rng = np.random.default_rng(seed)

    # 1) 人群样本
    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    beta = rng.random(n_samples) ** 0.5
    logMh = rng.uniform(logMh_min, logMh_max, n_samples)
    Mh_range = logMh_max - logMh_min

    # 2) 源星等分布
    ms_grid = np.linspace(20.0, 30.0, ms_points)
    L = 10 ** (-0.4 * (ms_grid - 24.5))
    pdf_ms = L ** (-0.3) * np.exp(-L)  # alpha_s = -1.3
    pdf_ms /= np.trapz(pdf_ms, ms_grid)

    MODEL_P = MODEL_PARAMS["deVauc"]
    beta_DM = MODEL_P["beta_h"]
    sigma_DM = MODEL_P["sigma_h"]

    out = {}
    for alpha in alpha_list:
        print(f"\n=== alpha = {alpha:.3f} (A像阈值, B像放开) ===")
        logM_star = logM_star_sps + alpha

        # 3) 并行解像并取放大率
        args_iter = zip(logM_star, logMh, logRe, beta, repeat(zl), repeat(zs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            mags = list(tqdm(pool.map(_solve_single_lens_mags, args_iter),
                             total=n_samples, desc="solving lenses"))
        mu1, mu2 = map(np.array, zip(*mags))

        # 4) 只卡 A 像的选择函数
        sel1 = selection_function(mu1[:, None], m_lim, ms_grid[None, :], SCATTER.mag)
        # 关键区别：不再乘 sel2
        w_ms = np.trapz(sel1 * pdf_ms[None, :], ms_grid, axis=1)

        # 5) halo 权重并累加
        A_sum = 0.0
        for logM_sps_i, logMh_i, w_i in zip(logM_star_sps, logMh, w_ms):
            mean = mu_DM + beta_DM * (logM_sps_i - 11.4)
            p_Mh = norm.pdf(logMh_i, loc=mean, scale=sigma_DM)
            A_sum += w_i * p_Mh

        A_val = Mh_range * A_sum / n_samples
        out[alpha] = A_val
        print(f"→ A_one_img({alpha:.3f}) = {A_val:.10f}")

    return out

if __name__ == "__main__":
    vals = compute_A_one_image(n_jobs=8)
    print("\n结果：", vals)
