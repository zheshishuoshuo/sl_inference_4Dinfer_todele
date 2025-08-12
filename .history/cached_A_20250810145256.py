import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator


def load_A_interpolator_2d(filename="A_eta_table_alpha.csv"):
    df = pd.read_csv(filename)

    mu_unique = np.sort(df["mu_DM"].unique())
    alpha_unique = np.sort(df["alpha"].unique())

    shape = (len(mu_unique), len(alpha_unique))
    values = (
        df.set_index(["mu_DM", "alpha"])  # type: ignore[index]
        .sort_index()["A"]
        .values.reshape(shape)
    )

    interp = RegularGridInterpolator(
        (mu_unique, alpha_unique), values, bounds_error=False, fill_value=None
    )
    return interp


# A_interp = load_A_interpolator_2d(
#     os.path.join(os.path.dirname(__file__), "A_eta_table_alpha.csv")
# )

A_interp = load_A_interpolator_2d(
    os.path.join(os.path.dirname(__file__), "A_eta_table_alpha.csv")
)

# === A_interp wrapper ===
def cached_A_interp(mu0, alpha):
    return A_interp((mu0, alpha))
