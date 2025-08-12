# run_mcmc.py
from __future__ import annotations
import os
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .likelihood import log_posterior, log_likelihood

# 可选采样器
import emcee
import dynesty
try:
    import pymultinest
except ImportError:
    pymultinest = None


def run_mcmc(
    grids,
    logM_sps_obs,
    *,
    method: str = "emcee",
    nwalkers: int = 50,
    nsteps: int = 3000,
    initial_guess: np.ndarray | None = None,
    backend_file: str = "chains_eta.h5",
    parallel: bool = False,
    nproc: int | None = None,
) -> np.ndarray:
    """
    运行不同采样器（emcee / dynesty / pymultinest）返回后验样本
    """

    method = method.lower()
    ndim = 2
    if initial_guess is None:
        initial_guess = np.array([12.5, 0.1])

    if method == "emcee":
        # --- 原 emcee 实现 ---
        base_dir = Path(__file__).parent.resolve()
        chain_dir = base_dir / "chains"
        backend_path = chain_dir / backend_file
        backend_path.parent.mkdir(parents=True, exist_ok=True)

        if backend_path.exists():
            print(f"[INFO] 继续采样：读取已有文件 {backend_path}")
            backend = emcee.backends.HDFBackend(backend_path, read_only=False)
        else:
            print(f"[INFO] 新建采样：创建新文件 {backend_path}")
            backend = emcee.backends.HDFBackend(backend_path)
            backend.reset(nwalkers, ndim)

        if backend.iteration == 0:
            print("[INFO] 从头开始采样")
            p0 = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)
        else:
            print(f"[INFO] 从第 {backend.iteration} 步继续采样")
            p0 = None

        if parallel:
            if nproc is None:
                nproc = mp.cpu_count() - 2
            with mp.Pool(processes=nproc) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    log_posterior,
                    args=(grids, logM_sps_obs),
                    backend=backend,
                    pool=pool,
                )
                sampler.run_mcmc(p0, nsteps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_posterior,
                args=(grids, logM_sps_obs),
                backend=backend,
            )
            sampler.run_mcmc(p0, nsteps, progress=True)

        return sampler.get_chain(discard=nsteps - 2000, flat=True)

    elif method == "dynesty":
        # --- dynesty 全局采样 ---
        def prior_transform(u):
            muDM = 10.0 + 6.0 * u[0]
            alpha = -0.5 + 1.5 * u[1]
            return muDM, alpha

        def loglike(theta):
            muDM, alpha = theta
            return log_likelihood((muDM, alpha), grids, logM_sps_obs)

        sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=2, nlive=500)
        sampler.run_nested()
        return sampler.results.samples

    elif method == "pymultinest":
        # --- pymultinest 多峰采样 ---
        if pymultinest is None:
            raise ImportError("pymultinest is not installed.")

        def prior_transform(cube, ndim, nparams):
            cube[0] = 10.0 + 6.0 * cube[0]
            cube[1] = -0.5 + 1.5 * cube[1]

        def loglike(cube, ndim, nparams):
            muDM, alpha = cube[0], cube[1]
            return log_likelihood(muDM, alpha, grids, logM_sps_obs)

        out_dir = Path(__file__).parent / "chains_multinest"
        out_dir.mkdir(exist_ok=True)
        pymultinest.run(
            loglike, prior_transform, 2,
            outputfiles_basename=str(out_dir / "mn_"),
            n_live_points=500, resume=False, verbose=True
        )
        results = pymultinest.Analyzer(n_params=2, outputfiles_basename=str(out_dir / "mn_"))
        return results.get_equal_weighted_posterior()[:, :-1]

    else:
        raise ValueError(f"Unknown method {method}")


__all__ = ["run_mcmc"]
