import sys
from pathlib import Path
import numpy as np

# Ensure parent directory of package is on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sl_inference_4Dinfer_todele.mock_generator.mock_generator import run_mock_simulation
from sl_inference_4Dinfer_todele.make_tabulate import tabulate_likelihood_grids
from sl_inference_4Dinfer_todele.likelihood import log_likelihood


def test_log_likelihood_runs():
    # Generate a small mock sample; using a large magnitude limit ensures detection
    _, mock_obs = run_mock_simulation(2, maximum_magnitude=99, process=0, seed=0)

    logMh_grid = np.linspace(10, 12, 3)
    grids = tabulate_likelihood_grids(mock_obs, logMh_grid)
    logM_obs = mock_obs["logM_star_sps_observed"].values

    ll = log_likelihood((12.5, 2.0, 0.3, 0.1), grids, logM_obs)
    assert np.isfinite(ll)
