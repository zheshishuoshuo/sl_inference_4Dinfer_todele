import numpy as np
import pandas as pd
from .lens_properties import observed_data
from tqdm import tqdm
from .mass_sampler import generate_samples

# SPS PARAMETER
# M_star = alpha_sps * M_sps
# logM_star = log_alpha_sps + logM_sps

import multiprocessing


def simulate_single_lens(i, samples, beta_samp, logalpha_sps_sample,
                        maximum_magnitude, zl, zs):
    input_df = pd.DataFrame({
        'logM_star_sps': [samples['logM_star_sps'][i]],
        'logM_star': [samples['logM_star_sps'][i] + logalpha_sps_sample[i]],
        'logM_halo': [samples['logMh'][i]],
        'logRe': [samples['logRe'][i]],
        'beta_unit': [beta_samp[i]],
        'm_s': [samples['m_s'][i]],
        'maximum_magnitude': [maximum_magnitude],
        'logalpha_sps': [logalpha_sps_sample[i]],
        'zl': [zl],
        'zs': [zs]
    })
    return observed_data(input_df, caustic=False)


def run_mock_simulation(
    n_samples,
    maximum_magnitude=26.5,
    zl=0.3,
    zs=2.0,
    if_source=False,
    process=None,
    alpha_s=-1.3,
    m_s_star=24.5,
    logalpha: float = 0.1,
    seed = None
):
    

    if seed is not None:
        np.random.seed(seed)
    beta_samp = np.random.rand(n_samples)**0.5
    logalpha_sps_sample = np.full(n_samples, logalpha)
    samples = generate_samples(n_samples, alpha_s=alpha_s, m_s_star=m_s_star, random_state=seed)

    if process is None or process == 0:
        # ===== 串行计算 =====
        lens_results = []
        for i in tqdm(range(n_samples), desc="Processing lenses"):
            input_df = pd.DataFrame({
                'logM_star_sps': [samples['logM_star_sps'][i]],
                'logM_star': [samples['logM_star_sps'][i] + logalpha_sps_sample[i]],
                'logM_halo': [samples['logMh'][i]],
                'logRe': [samples['logRe'][i]],
                'beta_unit': [beta_samp[i]],
                'm_s': [samples['m_s'][i]],
                'maximum_magnitude': [maximum_magnitude],
                'logalpha_sps': [logalpha_sps_sample[i]],
                'zl': [zl],
                'zs': [zs]
            })
            result = observed_data(input_df, caustic=False)
            lens_results.append(result)

    else:
        # ===== 并行计算 =====
        args = [
            (i, samples, beta_samp, logalpha_sps_sample,
             maximum_magnitude, zl, zs)
            for i in range(n_samples)
        ]

        with multiprocessing.get_context("spawn").Pool(process) as pool:
            lens_results = list(tqdm(
                pool.starmap(simulate_single_lens, args),
                total=n_samples, desc=f"Processing lenses (process={process})"
            ))

    df_lens = pd.DataFrame(lens_results)
    mock_lens_data = df_lens[df_lens['is_lensed']]
    mock_observed_data = mock_lens_data[[
        'xA', 'xB', 'logM_star_sps_observed', 'logRe',
        'magnitude_observedA', 'magnitude_observedB'
    ]].copy()

    if if_source:
        return df_lens, mock_lens_data, mock_observed_data
    else:
        return mock_lens_data, mock_observed_data

if __name__ == "__main__":
        # 串行
    mock_lens_data, mock_observed_data = run_mock_simulation(1000, process=0)

    # 默认行为（串行）
    mock_lens_data, mock_observed_data = run_mock_simulation(1000)

    # 并行，使用 8 核
    mock_lens_data, mock_observed_data = run_mock_simulation(1000, process=8)
