import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from sl_inference_only_muDMalpha.mock_generator.mock_generator import run_mock_simulation


def test_run_mock_simulation_zero_density():
    mock_lens_data, mock_observed_data = run_mock_simulation(5, seed=42, nbkg=0.0)
    assert mock_lens_data.empty
    assert mock_observed_data.empty


def test_run_mock_simulation_columns_exist():
    mock_lens_data, mock_observed_data = run_mock_simulation(1, seed=0, nbkg=1.0)
    expected_cols = [
        'xA', 'xB', 'logM_star_sps_observed', 'logRe',
        'magnitude_observedA', 'magnitude_observedB'
    ]
    assert list(mock_observed_data.columns) == expected_cols
