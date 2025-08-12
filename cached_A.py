import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator


def load_A_interpolator(filename="A_eta_table_alpha.csv"):
    df = pd.read_csv(filename)

    mu_unique = np.sort(df["mu_DM"].unique())
    beta_unique = np.sort(df["beta_DM"].unique())
    sigma_unique = np.sort(df["sigma_DM"].unique())
    alpha_unique = np.sort(df["alpha"].unique())

    shape = (
        len(mu_unique),
        len(beta_unique),
        len(sigma_unique),
        len(alpha_unique),
    )
    values = (
        df.set_index(["mu_DM", "beta_DM", "sigma_DM", "alpha"])  # type: ignore[index]
        .sort_index()["A"]
        .values.reshape(shape)
    )

    interp = RegularGridInterpolator(
        (mu_unique, beta_unique, sigma_unique, alpha_unique),
        values,
        bounds_error=False,
        fill_value=None,
    )
    return interp


A_interp = load_A_interpolator(
    os.path.join(os.path.dirname(__file__), "A_eta_table_alpha.csv")
)


# === A_interp wrapper ===
def cached_A_interp(mu0, betaDM, sigmaDM, alpha):
    return A_interp((mu0, betaDM, sigmaDM, alpha))
