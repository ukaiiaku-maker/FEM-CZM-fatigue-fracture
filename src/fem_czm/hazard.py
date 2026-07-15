"""Cleavage and emission hazard utilities."""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc

from .barriers import exp_floor_barrier_eV
from .constants import KB_EV_PER_K
from .parameters import ExpFloorParameters


def arrhenius_rate_s(stress_Pa, temperature_K, barrier: ExpFloorParameters, attempt_frequency_s: float):
    G = exp_floor_barrier_eV(stress_Pa, temperature_K, barrier)
    rate = attempt_frequency_s * np.exp(np.clip(-G / max(KB_EV_PER_K * temperature_K, 1.0e-300), -700.0, 0.0))
    return np.asarray(rate, dtype=float), np.asarray(G, dtype=float)


def cooperative_cleavage_rate_s(raw_rate_s, multiplicity: float = 3.0, renewal_time_s: float = 1.0e-6):
    raw = np.maximum(np.asarray(raw_rate_s, dtype=float), 0.0)
    m = max(float(multiplicity), 1.0)
    if m <= 1.0 + 1.0e-12:
        return raw
    tau = max(float(renewal_time_s), 1.0e-300)
    return gammainc(m, np.minimum(raw * tau, 1.0e12)) / tau
