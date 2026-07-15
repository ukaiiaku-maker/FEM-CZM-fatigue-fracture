"""Exact reduced-spatial v9.10.2/v9.10.3 compatibility solver.

This module is an audit oracle, not a proposed production architecture.  It
reproduces the equations and adaptive load integration used to select the three
published candidate rows.  It intentionally does not reuse the newer process
zone implementation, because the purpose is to detect implementation drift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammainc

from .barriers import exp_floor_barrier_eV, transport_exp_floor_barrier_eV
from .constants import KB_EV_PER_K
from .parameters import MaterialParameterization, get_material


@dataclass(frozen=True)
class V9102ParityConfig:
    length_m: float = 100.0e-6
    n_bins: int = 200
    n_systems: int = 2
    source_bin_count: int = 2
    shear_modulus_Pa: float = 160.0e9
    poisson_ratio: float = 0.28
    burgers_vector_m: float = 2.74e-10
    r0_m: float = 1.0e-6
    background_forest_density_m2: float = 5.0e12
    blunting_length_m: float = 0.5e-6
    shielding_core_m: float = 2.5e-10
    cleavage_attempt_frequency_s: float = 1.0e12
    emission_attempt_frequency_s: float = 1.0e11
    cleavage_hit_order: float = 3.0
    cleavage_correlation_time_s: float = 1.0e-6
    target_clock_increment: float = 0.25
    target_emission_hazard: float = 1.0
    source_active_fraction_min: float = 1.0e-4
    min_substep_fraction: float = 1.0e-8
    max_substeps: int = 2_000_000


@dataclass(frozen=True)
class V9102RunConfig:
    temperature_K: float = 700.0
    target_extension_um: float = 1000.0
    dK_MPa_sqrt_m: float = 0.25
    Kdot_MPa_sqrt_m_per_s: float = 0.005
    Kmax_MPa_sqrt_m: float = 80.0
    advance_increment_um: float = 5.0


@dataclass
class V9102ParityResult:
    material: str
    candidate_id: str
    run: dict[str, float]
    metrics: dict[str, float | int | bool | str]
    events: list[dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "candidate_id": self.candidate_id,
            "run": self.run,
            "metrics": self.metrics,
            "events": self.events,
        }


def _arrhenius_rate(barrier_eV, temperature_K: float, prefactor_s: float):
    exponent = -np.asarray(barrier_eV, dtype=float) / max(
        KB_EV_PER_K * float(temperature_K), 1.0e-300
    )
    return max(float(prefactor_s), 0.0) * np.exp(
        np.clip(exponent, -700.0, 0.0)
    )


def _transport_net_rates(stress_Pa, temperature_K, mechanism, emission):
    forward_barrier = transport_exp_floor_barrier_eV(
        stress_Pa, temperature_K, mechanism, emission
    )
    reverse_barrier = transport_exp_floor_barrier_eV(
        np.zeros_like(np.asarray(stress_Pa, dtype=float)),
        temperature_K,
        mechanism,
        emission,
    )
    forward = _arrhenius_rate(
        forward_barrier, temperature_K, mechanism.attempt_frequency_s
    )
    reverse = _arrhenius_rate(
        reverse_barrier, temperature_K, mechanism.attempt_frequency_s
    )
    reverse = np.broadcast_to(reverse, np.shape(forward))
    return np.maximum(forward - reverse, 0.0), forward, reverse


def _cooperative_cleavage_rate(
    raw_rate_s: float,
    hit_order: float,
    correlation_time_s: float,
) -> float:
    order = max(float(hit_order), 1.0)
    raw = max(float(raw_rate_s), 0.0)
    if order <= 1.0 + 1.0e-12:
        return raw
    tau = max(float(correlation_time_s), 1.0e-300)
    return float(gammainc(order, min(raw * tau, 1.0e12)) / tau)


def _logarithmic_mean_rate(previous: float | None, current: float) -> float:
    if previous is None:
        return max(float(current), 0.0)
    low, high = sorted((max(float(previous), 0.0), max(float(current), 0.0)))
    if low <= 0.0:
        return 0.5 * high
    if abs(high - low) <= 1.0e-12 * max(high, 1.0e-300):
        return high
    return (high - low) / math.log(high / low)


class V9102SpatialState:
    """Finite-source spatial MPZ used by the selected calibration rows."""

    def __init__(
        self,
        material: MaterialParameterization,
        config: V9102ParityConfig | None = None,
    ):
        self.material = material
        self.cfg = config or V9102ParityConfig()
        self.dx = self.cfg.length_m / self.cfg.n_bins
        self.x = (np.arange(self.cfg.n_bins, dtype=float) + 0.5) * self.dx
        self.blunting_length = max(
            self.cfg.blunting_length_m,
            0.5 * self.cfg.r0_m,
            self.dx,
        )
        shape = (self.cfg.n_systems, self.cfg.n_bins)
        self.mobile = np.zeros(shape, dtype=float)
        self.retained = np.zeros(shape, dtype=float)
        self.accumulated_slip = np.zeros(shape, dtype=float)
        capacity = max(
            float(material.state.legacy_source_sites_per_system), 0.0
        )
        self.site_capacity = np.full(self.cfg.n_systems, capacity, dtype=float)
        self.available_sites = self.site_capacity.copy()
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.recovered_total = 0.0
        self.advance_total_m = 0.0

    @property
    def available_site_fraction(self) -> float:
        capacity = float(np.sum(self.site_capacity))
        return (
            float(np.sum(self.available_sites) / capacity)
            if capacity > 0.0
            else 0.0
        )

    @property
    def mobile_count(self) -> float:
        return float(np.sum(self.mobile))

    @property
    def retained_count(self) -> float:
        return float(np.sum(self.retained))

    def local_slip_count(self) -> float:
        weights = np.exp(-self.x / self.blunting_length)
        return float(np.sum(self.accumulated_slip * weights[None, :]))

    def radius_m(self) -> float:
        return float(
            self.cfg.r0_m
            + max(float(self.material.state.c_blunt), 0.0)
            * self.cfg.burgers_vector_m
            * self.local_slip_count()
        )

    def shielding_K_Pa_sqrt_m(self) -> float:
        core = max(
            self.cfg.shielding_core_m,
            0.25 * abs(self.cfg.burgers_vector_m),
            1.0e-12,
        )
        kernel = (
            self.cfg.shear_modulus_Pa
            * self.cfg.burgers_vector_m
            / max(1.0 - self.cfg.poisson_ratio, 1.0e-6)
            / np.sqrt(2.0 * math.pi * np.maximum(self.x, core))
        )
        return float(max(np.sum(self.retained * kernel[None, :]), 0.0))

    def tip_stress_Pa(self, K_drive_Pa_sqrt_m: float) -> float:
        K_eff = max(
            float(K_drive_Pa_sqrt_m) - self.shielding_K_Pa_sqrt_m(), 0.0
        )
        return float(
            K_eff
            / math.sqrt(2.0 * math.pi * max(self.radius_m(), 1.0e-300))
        )

    def local_forest_density_m2(self) -> np.ndarray:
        local_count = np.sum(np.maximum(self.retained, 0.0), axis=0)
        return (
            self.cfg.background_forest_density_m2
            + local_count
            / max(self.dx * self.blunting_length, 1.0e-300)
        )

    def local_stress_profile_Pa(self, tip_stress_Pa: float) -> np.ndarray:
        ref = self.blunting_length
        return max(float(tip_stress_Pa), 0.0) * np.sqrt(
            ref / np.maximum(ref + self.x, ref)
        )

    def source_rate_s(
        self, K_drive_Pa_sqrt_m: float, temperature_K: float
    ) -> tuple[float, float, float]:
        # The calibrated emission barrier uses the full effective tip stress.
        sigma_tip = self.tip_stress_Pa(K_drive_Pa_sqrt_m)
        barrier = float(
            exp_floor_barrier_eV(
                sigma_tip, temperature_K, self.material.emission
            )
        )
        rate = float(
            _arrhenius_rate(
                barrier,
                temperature_K,
                self.cfg.emission_attempt_frequency_s,
            )
        )
        return rate, sigma_tip, barrier

    @staticmethod
    def _exchange_exact(mobile, retained, encounter, release, dt_s):
        total = np.maximum(mobile, 0.0) + np.maximum(retained, 0.0)
        rate = encounter + release
        retained_equilibrium = np.divide(
            encounter,
            rate,
            out=np.zeros_like(rate),
            where=rate > 0.0,
        ) * total
        decay = np.exp(-np.minimum(rate * dt_s, 700.0))
        retained_new = retained_equilibrium + (
            np.maximum(retained, 0.0) - retained_equilibrium
        ) * decay
        retained_new = np.clip(retained_new, 0.0, total)
        return total - retained_new, retained_new

    def _remap_forward(self, field: np.ndarray, distance_m: float):
        distance = max(float(distance_m), 0.0)
        if distance <= 0.0:
            return field.copy(), 0.0
        out = np.zeros_like(field)
        lost = 0.0
        for i in range(self.cfg.n_bins):
            left = i * self.dx + distance
            right = (i + 1) * self.dx + distance
            mass = field[:, i]
            if left >= self.cfg.length_m:
                lost += float(np.sum(mass))
                continue
            inside_left = max(left, 0.0)
            inside_right = min(right, self.cfg.length_m)
            inside_length = max(inside_right - inside_left, 0.0)
            if inside_length < self.dx:
                lost += float(np.sum(mass) * (1.0 - inside_length / self.dx))
            if inside_length <= 0.0:
                continue
            j0 = max(int(math.floor(inside_left / self.dx)), 0)
            j1 = min(
                int(
                    math.floor(
                        (inside_right - 1.0e-15 * self.dx) / self.dx
                    )
                ),
                self.cfg.n_bins - 1,
            )
            for j in range(j0, j1 + 1):
                overlap_left = max(inside_left, j * self.dx)
                overlap_right = min(inside_right, (j + 1) * self.dx)
                fraction = max(overlap_right - overlap_left, 0.0) / self.dx
                if fraction > 0.0:
                    out[:, j] += mass * fraction
        return out, float(lost)

    def _remap_tip_frame(self, field: np.ndarray, distance_m: float):
        distance = max(float(distance_m), 0.0)
        if distance <= 0.0:
            return field.copy(), 0.0
        out = np.zeros_like(field)
        lost = 0.0
        for i in range(self.cfg.n_bins):
            left = i * self.dx - distance
            right = (i + 1) * self.dx - distance
            mass = field[:, i]
            if right <= 0.0 or left >= self.cfg.length_m:
                lost += float(np.sum(mass))
                continue
            inside_left = max(left, 0.0)
            inside_right = min(right, self.cfg.length_m)
            inside_length = max(inside_right - inside_left, 0.0)
            if inside_length < self.dx:
                lost += float(np.sum(mass) * (1.0 - inside_length / self.dx))
            if inside_length <= 0.0:
                continue
            j0 = max(int(math.floor(inside_left / self.dx)), 0)
            j1 = min(
                int(
                    math.floor(
                        (inside_right - 1.0e-15 * self.dx) / self.dx
                    )
                ),
                self.cfg.n_bins - 1,
            )
            for j in range(j0, j1 + 1):
                overlap_left = max(inside_left, j * self.dx)
                overlap_right = min(inside_right, (j + 1) * self.dx)
                fraction = max(overlap_right - overlap_left, 0.0) / self.dx
                if fraction > 0.0:
                    out[:, j] += mass * fraction
        return out, float(lost)

    def evolve(
        self,
        dt_s: float,
        temperature_K: float,
        K_drive_Pa_sqrt_m: float,
    ) -> dict[str, float]:
        dt = max(float(dt_s), 0.0)
        source_rate, sigma_tip, emission_barrier = self.source_rate_s(
            K_drive_Pa_sqrt_m, temperature_K
        )
        hazard = max(source_rate * dt, 0.0)
        emission_probability = 1.0 - math.exp(-min(hazard, 700.0))
        emitted = self.available_sites * emission_probability
        self.available_sites = np.maximum(self.available_sites - emitted, 0.0)

        source_bins = max(
            min(int(self.cfg.source_bin_count), self.cfg.n_bins), 1
        )
        source_increment = emitted[:, None] / source_bins
        self.mobile[:, :source_bins] += source_increment
        # Exact calibrated blunting ledger: source slip is committed once when a
        # site emits.  Transport remaps this ledger but never adds another slip.
        self.accumulated_slip[:, :source_bins] += source_increment
        self.emitted_total += float(np.sum(emitted))

        stress_profile = self.local_stress_profile_Pa(sigma_tip)
        density = self.local_forest_density_m2()
        resolved = stress_profile / math.sqrt(3.0)
        peierls_net, _, _ = _transport_net_rates(
            resolved,
            temperature_K,
            self.material.peierls,
            self.material.emission,
        )
        spacing = 1.0 / (2.0 * np.sqrt(density))
        local_taylor_stress = resolved * spacing / self.cfg.burgers_vector_m
        _, taylor_forward, taylor_reverse = _transport_net_rates(
            local_taylor_stress,
            temperature_K,
            self.material.taylor,
            self.material.emission,
        )
        corr = self.material.taylor_correlation
        correlation_length = corr.m_scale / (
            2.0 * math.sqrt(max(corr.rho_c_m2, 1.0e-300))
        )
        hit_order = 1.0 + np.power(
            np.maximum(2.0 * correlation_length * np.sqrt(density), 0.0),
            max(corr.exponent, 1.0e-12),
        )
        taylor_release = np.maximum(
            taylor_forward / hit_order - taylor_reverse / hit_order,
            0.0,
        )

        # Peierls motion alone transports carriers and controls encounters.
        velocity_profile = spacing * peierls_net
        encounter = (
            self.material.state.encounter_efficiency
            * velocity_profile
            * np.sqrt(density)
        )
        for system in range(self.cfg.n_systems):
            self.mobile[system], self.retained[system] = self._exchange_exact(
                self.mobile[system],
                self.retained[system],
                encounter,
                taylor_release,
                dt,
            )

        retained_recovery = max(
            float(self.material.state.retained_recovery_rate_s), 0.0
        )
        if retained_recovery > 0.0 and dt > 0.0:
            self.retained *= math.exp(-min(retained_recovery * dt, 700.0))

        mobile_by_bin = np.sum(np.maximum(self.mobile, 0.0), axis=0)
        if float(np.sum(mobile_by_bin)) > 0.0:
            velocity = float(
                np.sum(velocity_profile * mobile_by_bin)
                / np.sum(mobile_by_bin)
            )
        else:
            velocity = float(np.mean(velocity_profile[:source_bins]))
        self.mobile, escaped = self._remap_forward(
            self.mobile, max(velocity, 0.0) * dt
        )
        self.escaped_total += escaped

        return {
            "source_rate_s": source_rate,
            "source_hazard": hazard,
            "source_probability": emission_probability,
            "dN_emit": float(np.sum(emitted)),
            "sigma_tip_Pa": sigma_tip,
            "G_emit_eV": emission_barrier,
            "peierls_rate_max_s": float(np.max(peierls_net)),
            "taylor_release_rate_max_s": float(np.max(taylor_release)),
            "glide_velocity_m_s": velocity,
            "encounter_rate_max_s": float(np.max(encounter)),
            "available_site_fraction": self.available_site_fraction,
        }

    def advance(self, distance_m: float) -> dict[str, float]:
        distance = max(float(distance_m), 0.0)
        self.mobile, wake_mobile = self._remap_tip_frame(self.mobile, distance)
        self.retained, wake_retained = self._remap_tip_frame(
            self.retained, distance
        )
        self.accumulated_slip, wake_slip = self._remap_tip_frame(
            self.accumulated_slip, distance
        )
        refresh_length = max(
            self.material.state.legacy_source_refresh_length_um * 1.0e-6,
            self.dx,
        )
        fresh_fraction = min(distance / refresh_length, 1.0)
        refreshed = (
            self.site_capacity - self.available_sites
        ) * fresh_fraction
        self.available_sites += refreshed
        self.advance_total_m += distance
        return {
            "wake_mobile": wake_mobile,
            "wake_retained": wake_retained,
            "wake_slip": wake_slip,
            "source_sites_refreshed": float(np.sum(refreshed)),
        }


def _cleavage_rate(
    state: V9102SpatialState,
    K_drive_Pa_sqrt_m: float,
    temperature_K: float,
):
    sigma_tip = state.tip_stress_Pa(K_drive_Pa_sqrt_m)
    barrier = float(
        exp_floor_barrier_eV(
            sigma_tip, temperature_K, state.material.cleavage
        )
    )
    raw = float(
        _arrhenius_rate(
            barrier,
            temperature_K,
            state.cfg.cleavage_attempt_frequency_s,
        )
    )
    effective = _cooperative_cleavage_rate(
        raw,
        state.cfg.cleavage_hit_order,
        state.cfg.cleavage_correlation_time_s,
    )
    return effective, raw, sigma_tip, barrier


def run_v9102_parity(
    material: str | MaterialParameterization,
    run: V9102RunConfig | None = None,
    parity: V9102ParityConfig | None = None,
) -> V9102ParityResult:
    material_object = get_material(material) if isinstance(material, str) else material
    run_cfg = run or V9102RunConfig()
    state = V9102SpatialState(material_object, parity)
    pz_cfg = state.cfg

    if run_cfg.Kdot_MPa_sqrt_m_per_s <= 0.0:
        raise ValueError("Kdot must be positive")
    if run_cfg.dK_MPa_sqrt_m <= 0.0:
        raise ValueError("dK must be positive")
    if run_cfg.advance_increment_um <= 0.0:
        raise ValueError("advance increment must be positive")

    target_events = int(
        round(run_cfg.target_extension_um / run_cfg.advance_increment_um)
    ) + 1
    advance_m = run_cfg.advance_increment_um * 1.0e-6
    clock = 0.0
    previous_cleavage_rate: float | None = None
    current_K = 0.0
    n_substeps = 0
    events: list[dict[str, float]] = []
    max_radius_ratio = 1.0
    max_shield = 0.0
    min_available_fraction = 1.0
    integration_stalled = False
    tolerance_K = max(
        1.0e-12, 1.0e-12 * run_cfg.Kmax_MPa_sqrt_m
    )

    while (
        current_K < run_cfg.Kmax_MPa_sqrt_m - tolerance_K
        and len(events) < target_events
    ):
        nominal_end = min(
            current_K + run_cfg.dK_MPa_sqrt_m,
            run_cfg.Kmax_MPa_sqrt_m,
        )
        while (
            current_K < nominal_end - tolerance_K
            and len(events) < target_events
        ):
            if n_substeps >= pz_cfg.max_substeps:
                integration_stalled = True
                break

            dK_proposed = nominal_end - current_K
            dt_proposed = (
                dK_proposed / run_cfg.Kdot_MPa_sqrt_m_per_s
            )

            def acceptable(fraction_trial: float) -> bool:
                dK_trial = dK_proposed * fraction_trial
                dt_trial = dt_proposed * fraction_trial
                K_trial = current_K + dK_trial
                cleavage_rate, _, _, _ = _cleavage_rate(
                    state, K_trial * 1.0e6, run_cfg.temperature_K
                )
                dclock = _logarithmic_mean_rate(
                    previous_cleavage_rate, cleavage_rate
                ) * dt_trial
                if dclock > pz_cfg.target_clock_increment:
                    return False
                if (
                    state.available_site_fraction
                    > pz_cfg.source_active_fraction_min
                ):
                    emission_rate, _, _ = state.source_rate_s(
                        K_trial * 1.0e6, run_cfg.temperature_K
                    )
                    if (
                        emission_rate * dt_trial
                        > pz_cfg.target_emission_hazard
                    ):
                        return False
                return True

            if acceptable(1.0):
                fraction = 1.0
            else:
                low = 0.0
                high = 1.0
                for _ in range(36):
                    midpoint = 0.5 * (low + high)
                    if acceptable(midpoint):
                        low = midpoint
                    else:
                        high = midpoint
                fraction = max(low, pz_cfg.min_substep_fraction)

            fraction = float(
                np.clip(fraction, pz_cfg.min_substep_fraction, 1.0)
            )
            dK_step = max(dK_proposed * fraction, tolerance_K)
            dK_step = min(dK_step, dK_proposed)
            dt_step = dK_step / run_cfg.Kdot_MPa_sqrt_m_per_s
            current_K += dK_step

            state.evolve(
                dt_step,
                run_cfg.temperature_K,
                current_K * 1.0e6,
            )
            cleavage_rate, raw_rate, sigma_tip, cleavage_barrier = _cleavage_rate(
                state,
                current_K * 1.0e6,
                run_cfg.temperature_K,
            )
            clock += _logarithmic_mean_rate(
                previous_cleavage_rate, cleavage_rate
            ) * dt_step
            previous_cleavage_rate = cleavage_rate
            n_substeps += 1

            radius_ratio = state.radius_m() / pz_cfg.r0_m
            shield = state.shielding_K_Pa_sqrt_m() / 1.0e6
            max_radius_ratio = max(max_radius_ratio, radius_ratio)
            max_shield = max(max_shield, shield)
            min_available_fraction = min(
                min_available_fraction, state.available_site_fraction
            )

            if clock >= 1.0:
                # The calibrated driver retained the residual renewal clock.
                clock -= 1.0
                event_index = len(events)
                events.append(
                    {
                        "event_index": float(event_index),
                        "a_um": float(
                            event_index * run_cfg.advance_increment_um
                        ),
                        "K_MPa_sqrt_m": float(current_K),
                        "sigma_tip_GPa": float(sigma_tip / 1.0e9),
                        "cleavage_raw_rate_s": float(raw_rate),
                        "cleavage_effective_rate_s": float(cleavage_rate),
                        "G_cleave_eV": float(cleavage_barrier),
                        "cleavage_clock_residual": float(clock),
                        "K_shield_MPa_sqrt_m": float(shield),
                        "tip_radius_ratio": float(radius_ratio),
                        "tip_radius_um": float(state.radius_m() * 1.0e6),
                        "available_site_fraction": float(
                            state.available_site_fraction
                        ),
                        "emitted_total": float(state.emitted_total),
                        "mobile_count": float(state.mobile_count),
                        "retained_count": float(state.retained_count),
                        "local_slip_count": float(state.local_slip_count()),
                    }
                )
                state.advance(advance_m)

        if integration_stalled:
            break

    event_K = np.asarray(
        [event["K_MPa_sqrt_m"] for event in events], dtype=float
    )
    event_a = np.asarray([event["a_um"] for event in events], dtype=float)
    K_init = float(event_K[0]) if event_K.size else float("nan")
    plateau_low = 0.70 * run_cfg.target_extension_um
    plateau_use = (
        (event_a >= plateau_low)
        & (event_a <= run_cfg.target_extension_um)
    )
    K_plateau = (
        float(np.median(event_K[plateau_use]))
        if np.any(plateau_use)
        else float("nan")
    )
    completed = len(events) >= target_events
    metrics: dict[str, float | int | bool | str] = {
        "temperature_K": float(run_cfg.temperature_K),
        "target_extension_um": float(run_cfg.target_extension_um),
        "n_events": int(len(events)),
        "target_events": int(target_events),
        "completed": bool(completed),
        "integration_stalled": bool(integration_stalled),
        "right_censored_at_Kmax": bool(
            not completed
            and current_K >= run_cfg.Kmax_MPa_sqrt_m - tolerance_K
        ),
        "termination_reason": (
            "target_extension"
            if completed
            else "integration_stalled"
            if integration_stalled
            else "Kmax_right_censored"
        ),
        "K_init_MPa_sqrt_m": K_init,
        "K_plateau_MPa_sqrt_m": K_plateau,
        "delta_KR_MPa_sqrt_m": (
            K_plateau - K_init
            if np.isfinite(K_init) and np.isfinite(K_plateau)
            else float("nan")
        ),
        "K_final_MPa_sqrt_m": float(current_K),
        "n_substeps": int(n_substeps),
        "max_tip_radius_ratio": float(max_radius_ratio),
        "max_K_shield_MPa_sqrt_m": float(max_shield),
        "min_available_site_fraction": float(min_available_fraction),
        "final_available_site_fraction": float(
            state.available_site_fraction
        ),
        "final_emitted_total": float(state.emitted_total),
        "final_mobile_count": float(state.mobile_count),
        "final_retained_count": float(state.retained_count),
        "final_tip_radius_ratio": float(
            state.radius_m() / pz_cfg.r0_m
        ),
        "parity_model": "v9.10.2_independent_shape_reduced_spatial",
    }
    return V9102ParityResult(
        material=material_object.name,
        candidate_id=material_object.candidate_id,
        run=asdict(run_cfg),
        metrics=metrics,
        events=events,
    )


def _write_event_csv(events: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact v9.10.2 reduced-spatial parity oracle"
    )
    parser.add_argument(
        "--material", choices=["ceramic", "weakT", "DBTT"], required=True
    )
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--target-extension-um", type=float, default=1000.0)
    parser.add_argument("--dK", type=float, default=0.25)
    parser.add_argument("--Kdot", type=float, default=0.005)
    parser.add_argument("--Kmax", type=float, default=80.0)
    parser.add_argument("--da-um", type=float, default=5.0)
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run_v9102_parity(
        args.material,
        V9102RunConfig(
            temperature_K=args.temperature,
            target_extension_um=args.target_extension_um,
            dK_MPa_sqrt_m=args.dK,
            Kdot_MPa_sqrt_m_per_s=args.Kdot,
            Kmax_MPa_sqrt_m=args.Kmax,
            advance_increment_um=args.da_um,
        ),
    )
    _write_event_csv(result.events, args.out_events)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(result.as_dict(), indent=2, allow_nan=True)
    )
    print(json.dumps(result.metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
