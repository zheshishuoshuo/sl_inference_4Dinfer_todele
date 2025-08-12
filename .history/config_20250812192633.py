"""Global configuration for simulation and inference parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScatterConfig:
    """Measurement scatter settings for observables."""

    star: float = 0.1  # Scatter on log stellar mass [dex]
    mag: float = 0.1    # Scatter on observed magnitudes [mag]


# Global scatter configuration used throughout the package
SCATTER = ScatterConfig()
