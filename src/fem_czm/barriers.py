"""Independent EXP-floor free-energy surfaces."""
from __future__ import annotations

import numpy as np

from .constants import KB_EV_PER_K
from .parameters import ActivatedTransportParameters, ExpFloorParameters

TREF_K = 481.33
FLOOR_MIN_EV = 1.0e-4
FLOOR_MAX_FRACTION = 0.95


def exp_floor_barrier_eV(stress_Pa, temperature_K: float, parameters: ExpFloorParameters) -> np.ndarray:
    stress = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    dT = float(temperature_K) - TREF_K
    G0 = max(parameters.G00_eV + parameters.gT_eV_per_K * dT, 1.0e-12)
    sigma_c = max(parameters.sigma_c0_Pa + parameters.sT_Pa_per_K * dT, 1.0)
    floor = min(FLOOR_MAX_FRACTION * G0, max(FLOOR_MIN_EV, parameters.floor_fraction * G0))
    return floor + (G0 - floor) * np.exp(-max(parameters.alpha, 0.0) * np.power(stress / sigma_c, max(parameters.n, 1.0e-12)))


def transport_exp_floor_barrier_eV(stress_Pa, temperature_K: float, mechanism: ActivatedTransportParameters, emission_surface: ExpFloorParameters) -> np.ndarray:
    """Peierls or Taylor barrier with independent H0, entropy, alpha and n."""
    stress = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    dT = float(temperature_K) - TREF_K
    G0 = max(mechanism.H0_eV - mechanism.activation_entropy_kB * KB_EV_PER_K * dT, 1.0e-12)
    sigma_c = max(emission_surface.sigma_c0_Pa + emission_surface.sT_Pa_per_K * dT, 1.0)
    scaled_min = FLOOR_MIN_EV * mechanism.H0_eV / max(emission_surface.G00_eV, 1.0e-12)
    floor = min(FLOOR_MAX_FRACTION * G0, max(scaled_min, emission_surface.floor_fraction * G0))
    return floor + (G0 - floor) * np.exp(-max(mechanism.alpha, 0.0) * np.power(stress / sigma_c, max(mechanism.n, 1.0e-12)))
