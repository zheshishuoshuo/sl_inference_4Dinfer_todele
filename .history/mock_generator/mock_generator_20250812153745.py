import numpy as np
import pandas as pd
from .lens_properties import observed_data
from .lens_model import LensModel, kpc_to_arcsec
from ..sl_cosmology import Dang
from tqdm import tqdm
from .mass_sampler import generate_samples

# SPS PARAMETER
# M_star = alpha_sps * M_sps
# logM_star = log_alpha_sps + logM_sps

import multiprocessing


def simulate_single_lens(i, samples, logalpha_sps_sample,
                        maximum_magnitude, zl, zs, nbkg):
    """Simulate all sources for a single lens.

    Returns a list of dictionaries, one for each sampled source.
    """
    logM_star_sps = samples['logM_star_sps'][i]
    logM_star = logM_star_sps + logalpha_sps_sample[i]
    logM_halo = samples['logMh'][i]
    logRe = samples['logRe'][i]
    m_s_data = samples['m_s']
    m_s = m_s_data[i] if np.ndim(m_s_data) else m_s_data

    model = LensModel(logM_star=logM_star, logM_halo=logM_halo,
                      logRe=logRe, zl=zl, zs=zs)
    ycaust_kpc = model.solve_ycaustic()
    if ycaust_kpc is None:
        return []

    lambda_i = np.pi * ycaust_kpc**2 * nbkg
    N_i = np.random.poisson(lambda_i)
    # if N_i != 0:
    #     # return []
    #     print(f"Lens {i}: ycaustic = {ycaust_kpc:.2f} kpc, N_sources = {N_i}")

    results = []
    for _ in range(N_i):
        beta_unit = np.sqrt(np.random.rand())
        input_df = pd.DataFrame({
            'logM_star_sps': [logM_star_sps],
            'logM_star': [logM_star],
            'logM_halo': [logM_halo],
            'logRe': [logRe],
            'beta_unit': [beta_unit],
            'm_s': [m_s],
            'maximum_magnitude': [maximum_magnitude],
            'logalpha_sps': [logalpha_sps_sample[i]],
            'zl': [zl],
            'zs': [zs]
        })
        result = observed_data(input_df, caustic=False)
        result['lens_id'] = i
        result['ycaustic_kpc'] = ycaust_kpc
        result['ycaustic_arcsec'] = kpc_to_arcsec(ycaust_kpc, zl, Dang)
        results.append(result)

    return results


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
    seed = None,
    nbkg: float = 1.0
):
    """Run a mock strong-lens simulation.

    Parameters
    ----------
    n_samples : int
        Number of lens galaxies to simulate.
    nbkg : float, optional
        Surface density of background sources in ``kpc^-2``.
    """

    if seed is not None:
        np.random.seed(seed)

    logalpha_sps_sample = np.full(n_samples, logalpha)
    samples = generate_samples(n_samples, alpha_s=alpha_s,
                               m_s_star=m_s_star, random_state=seed)

    

    if process is None or process == 0:
        lens_results = []
        for i in tqdm(range(n_samples), desc="Processing lenses"):
            lens_results.extend(
                simulate_single_lens(i, samples, logalpha_sps_sample,
                                      maximum_magnitude, zl, zs, nbkg)
            )
    else:
        args = [
            (i, samples, logalpha_sps_sample, maximum_magnitude, zl, zs, nbkg)
            for i in range(n_samples)
        ]
        with multiprocessing.get_context("spawn").Pool(process) as pool:
            results = list(tqdm(
                pool.starmap(simulate_single_lens, args),
                total=n_samples, desc=f"Processing lenses (process={process})"
            ))
        lens_results = [r for sub in results for r in sub]

    df_lens = pd.DataFrame(lens_results)
    if df_lens.empty:
        mock_lens_data = pd.DataFrame(columns=df_lens.columns)
        mock_observed_data = pd.DataFrame(columns=[
            'xA', 'xB', 'logM_star_sps_observed', 'logRe',
            'magnitude_observedA', 'magnitude_observedB'
        ])
    else:
        mock_lens_data = df_lens[df_lens['is_lensed']].copy()
        mock_observed_data = mock_lens_data[
            ['xA', 'xB', 'logM_star_sps_observed', 'logRe',
             'magnitude_observedA', 'magnitude_observedB']
        ].copy()

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
