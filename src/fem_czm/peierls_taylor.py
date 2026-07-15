"""Uncapped, detailed-balance Peierls--Taylor kinetics."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .barriers import transport_exp_floor_barrier_eV
from .constants import KB_EV_PER_K
from .parameters import MaterialParameterization


@dataclass(frozen=True)
class PTRates:
    peierls_net_s: np.ndarray
    peierls_forward_s: np.ndarray
    peierls_reverse_s: np.ndarray
    taylor_net_s: np.ndarray
    taylor_forward_s: np.ndarray
    taylor_reverse_s: np.ndarray
    series_s: np.ndarray
    hit_order: np.ndarray
    forest_spacing_m: np.ndarray
    jump_length_m: np.ndarray
    equivalent_plastic_rate_s: np.ndarray


def _arrhenius(barrier_eV: np.ndarray, temperature_K: float, nu0_s: float) -> np.ndarray:
    exponent = -np.asarray(barrier_eV, dtype=float) / max(KB_EV_PER_K * temperature_K, 1.0e-300)
    return max(float(nu0_s), 0.0) * np.exp(np.clip(exponent, -700.0, 0.0))


def _net_rates(stress_Pa, T_K, mechanism, emission):
    Gf = transport_exp_floor_barrier_eV(stress_Pa, T_K, mechanism, emission)
    G0 = transport_exp_floor_barrier_eV(0.0, T_K, mechanism, emission)
    forward = _arrhenius(Gf, T_K, mechanism.attempt_frequency_s)
    reverse = np.broadcast_to(_arrhenius(np.asarray(G0), T_K, mechanism.attempt_frequency_s), np.shape(forward))
    return np.maximum(forward - reverse, 0.0), forward, reverse


def evaluate_pt_rates(equivalent_stress_Pa, forest_density_m2, mobile_density_m2, temperature_K: float, burgers_vector_m: float, material: MaterialParameterization) -> PTRates:
    sigma = np.maximum(np.asarray(equivalent_stress_Pa, dtype=float), 0.0)
    rho = np.maximum(np.asarray(forest_density_m2, dtype=float), 1.0e-300)
    rho_m = np.maximum(np.asarray(mobile_density_m2, dtype=float), 0.0)
    sigma, rho, rho_m = np.broadcast_arrays(sigma, rho, rho_m)
    b = max(abs(float(burgers_vector_m)), 1.0e-300)
    resolved = sigma / math.sqrt(3.0)
    pnet, pfwd, prev = _net_rates(resolved, temperature_K, material.peierls, material.emission)
    spacing = 1.0 / (2.0 * np.sqrt(rho))
    local_taylor_stress = resolved * (spacing / b)
    t1net, t1fwd, t1rev = _net_rates(local_taylor_stress, temperature_K, material.taylor, material.emission)
    corr = material.taylor_correlation
    Lcorr = corr.m_scale / (2.0 * math.sqrt(max(corr.rho_c_m2, 1.0e-300)))
    hit_order = 1.0 + np.power(np.maximum(2.0 * Lcorr * np.sqrt(rho), 0.0), max(corr.exponent, 1.0e-12))
    tfwd = t1fwd / hit_order
    trev = t1rev / hit_order
    tnet = np.maximum(tfwd - trev, 0.0)
    series = np.zeros_like(pnet)
    active = (pnet > 0.0) & (tnet > 0.0)
    series[active] = pnet[active] * tnet[active] / (pnet[active] + tnet[active])
    jump = spacing
    eq_rate = (1.0 / math.sqrt(3.0)) * rho_m * b * jump * series
    return PTRates(pnet, pfwd, prev, tnet, tfwd, trev, series, hit_order, spacing, jump, eq_rate)
