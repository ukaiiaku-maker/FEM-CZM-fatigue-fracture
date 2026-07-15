"""Interfaces between the constitutive core and a 2-D FEM/CZM backend."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .parameters import MaterialParameterization
from .peierls_taylor import PTRates, evaluate_pt_rates


@dataclass(frozen=True)
class CrackTipDriver:
    """Mesh-objective crack-tip input from the 2-D backend.

    K_J already contains shielding caused by resolved bulk plastic strain and the
    associated FEM stress redistribution. The front-local MPZ subtracts only its
    unresolved retained-line shielding, once.
    """
    K_J_Pa_sqrt_m: float
    cleavage_direction_factor: float = 1.0
    emission_direction_factor: float = 1.0

    @property
    def K_cleave(self) -> float:
        return max(self.K_J_Pa_sqrt_m * self.cleavage_direction_factor, 0.0)

    @property
    def K_emit(self) -> float:
        return max(self.K_J_Pa_sqrt_m * self.emission_direction_factor, 0.0)


def bulk_plastic_rates(equivalent_stress_Pa: np.ndarray, forest_density_m2: np.ndarray, mobile_density_m2: np.ndarray, temperature_K: float, burgers_vector_m: float, material: MaterialParameterization) -> PTRates:
    return evaluate_pt_rates(equivalent_stress_Pa, forest_density_m2, mobile_density_m2, temperature_K, burgers_vector_m, material)
