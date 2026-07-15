"""Reusable-source moving process zone with emergent self-limitation.

There is no available-site state, source capacity, one-shot source rule, or
crack-advance source refresh. Emission is throttled by retained-line backstress,
direct crack shielding, transport, Taylor release and recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .hazard import arrhenius_rate_s
from .parameters import MaterialParameterization
from .peierls_taylor import evaluate_pt_rates


@dataclass(frozen=True)
class ProcessZoneConfig:
    length_m: float = 100.0e-6
    n_bins: int = 200
    n_systems: int = 2
    shear_modulus_Pa: float = 160.0e9
    poisson_ratio: float = 0.28
    burgers_vector_m: float = 2.74e-10
    r0_m: float = 1.0e-6
    source_strength_per_system: float = 1.0
    resolved_emission_fraction: float = 1.0 / math.sqrt(3.0)
    backstress_geometry_factor: float = 1.0
    shielding_geometry_factor: float = 1.0
    core_radius_m: float = 2.74e-10
    mobile_recovery_rate_s: float = 0.0
    max_substep_rate_dt: float = 0.15
    max_advection_cfl: float = 0.25
    max_substeps: int = 256


class ReusableSourceProcessZone:
    """Expected line counts per unit out-of-plane thickness in moving bins."""

    def __init__(self, material: MaterialParameterization, config: ProcessZoneConfig | None = None):
        self.material = material
        self.cfg = config or ProcessZoneConfig()
        if self.cfg.n_bins < 4 or self.cfg.n_systems < 1:
            raise ValueError("process zone requires at least four bins and one slip system")
        self.dx = self.cfg.length_m / self.cfg.n_bins
        self.x = (np.arange(self.cfg.n_bins, dtype=float) + 0.5) * self.dx
        self.mobile = np.zeros((self.cfg.n_systems, self.cfg.n_bins), dtype=float)
        self.retained = np.zeros_like(self.mobile)
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.time_s = 0.0

    def copy(self):
        other = ReusableSourceProcessZone(self.material, self.cfg)
        other.mobile = self.mobile.copy()
        other.retained = self.retained.copy()
        other.emitted_total = self.emitted_total
        other.escaped_total = self.escaped_total
        other.time_s = self.time_s
        return other

    @property
    def mobile_count(self) -> float:
        return float(self.mobile.sum())

    @property
    def retained_count(self) -> float:
        return float(self.retained.sum())

    @property
    def forest_density_m2(self) -> np.ndarray:
        return np.maximum(self.retained.sum(axis=0) / max(self.dx, 1.0e-300), 1.0)

    @property
    def mobile_density_m2(self) -> np.ndarray:
        return np.maximum(self.mobile.sum(axis=0) / max(self.dx, 1.0e-300), 0.0)

    def source_backstress_Pa(self) -> np.ndarray:
        pref = self.cfg.backstress_geometry_factor * self.cfg.shear_modulus_Pa * self.cfg.burgers_vector_m / (2.0 * math.pi * (1.0 - self.cfg.poisson_ratio))
        kernel = 1.0 / np.maximum(self.x + self.cfg.core_radius_m, self.cfg.core_radius_m)
        return pref * (self.retained @ kernel)

    def shielding_K_Pa_sqrt_m(self) -> float:
        pref = self.cfg.shielding_geometry_factor * self.cfg.shear_modulus_Pa * self.cfg.burgers_vector_m / (2.0 * (1.0 - self.cfg.poisson_ratio) * math.sqrt(2.0 * math.pi))
        return float(pref * np.sum(self.retained.sum(axis=0) / np.sqrt(np.maximum(self.x, self.cfg.core_radius_m))))

    def blunted_radius_m(self) -> float:
        weights = np.exp(-self.x / max(0.2 * self.cfg.length_m, self.dx))
        local_count = float(np.sum((self.mobile + self.retained) * weights[None, :]))
        return self.cfg.r0_m + self.material.state.c_blunt * self.cfg.burgers_vector_m * local_count

    def effective_tip_stress_Pa(self, K_drive_Pa_sqrt_m: float) -> float:
        K_eff = max(float(K_drive_Pa_sqrt_m) - self.shielding_K_Pa_sqrt_m(), 0.0)
        return K_eff / math.sqrt(2.0 * math.pi * max(self.blunted_radius_m(), 1.0e-300))

    def _rates(self, K_drive_Pa_sqrt_m: float, temperature_K: float):
        sigma_tip = self.effective_tip_stress_Pa(K_drive_Pa_sqrt_m)
        tau_back = self.source_backstress_Pa()
        source_stress = np.maximum(self.cfg.resolved_emission_fraction * sigma_tip - tau_back, 0.0)
        lambda_emit, G_emit = arrhenius_rate_s(source_stress, temperature_K, self.material.emission, 1.0e11)
        lambda_emit = self.cfg.source_strength_per_system * lambda_emit
        rho_f = self.forest_density_m2
        rho_m = self.mobile_density_m2
        pt = evaluate_pt_rates(np.full_like(rho_f, sigma_tip), rho_f, rho_m, temperature_K, self.cfg.burgers_vector_m, self.material)
        velocity = pt.jump_length_m * pt.series_s
        encounter = self.material.state.encounter_efficiency * velocity * np.sqrt(rho_f)
        return sigma_tip, source_stress, lambda_emit, G_emit, pt, velocity, encounter

    def evolve(self, dt_s: float, temperature_K: float, K_drive_Pa_sqrt_m: float) -> dict[str, float]:
        dt = max(float(dt_s), 0.0)
        if dt == 0.0:
            sigma_tip, source_stress, lam, G, pt, velocity, encounter = self._rates(K_drive_Pa_sqrt_m, temperature_K)
            return self._diagnostics(sigma_tip, source_stress, lam, G, pt, velocity, encounter, 0.0)
        sigma_tip, source_stress, lam, G, pt, velocity, encounter = self._rates(K_drive_Pa_sqrt_m, temperature_K)
        max_rate = max(float(np.max(encounter)), float(np.max(pt.taylor_net_s)), self.material.state.retained_recovery_rate_s, self.cfg.mobile_recovery_rate_s, 1.0e-30)
        max_velocity = max(float(np.max(velocity)), 0.0)
        n_rate = int(math.ceil(dt * max_rate / self.cfg.max_substep_rate_dt))
        n_cfl = int(math.ceil(dt * max_velocity / max(self.dx * self.cfg.max_advection_cfl, 1.0e-300)))
        nsub = max(1, min(max(n_rate, n_cfl), int(self.cfg.max_substeps)))
        h = dt / nsub
        emitted_before = self.emitted_total
        escaped_before = self.escaped_total
        for _ in range(nsub):
            sigma_tip, source_stress, lam, G, pt, velocity, encounter = self._rates(K_drive_Pa_sqrt_m, temperature_K)
            self.mobile[:, 0] += lam * h
            self.emitted_total += float(np.sum(lam) * h)
            release = np.asarray(pt.taylor_net_s, dtype=float)
            for s in range(self.cfg.n_systems):
                m = self.mobile[s]
                r = self.retained[s]
                dm = (-encounter * m + release * r - self.cfg.mobile_recovery_rate_s * m) * h
                dr = (encounter * m - release * r - self.material.state.retained_recovery_rate_s * r) * h
                self.mobile[s] = np.maximum(m + dm, 0.0)
                self.retained[s] = np.maximum(r + dr, 0.0)
                cfl = np.clip(velocity * h / max(self.dx, 1.0e-300), 0.0, 1.0)
                flux = cfl * self.mobile[s]
                escaped = float(flux[-1])
                self.mobile[s, 1:] += flux[:-1]
                self.mobile[s] -= flux
                self.escaped_total += escaped
        self.time_s += dt
        return self._diagnostics(sigma_tip, source_stress, lam, G, pt, velocity, encounter, self.emitted_total - emitted_before, self.escaped_total - escaped_before)

    def _diagnostics(self, sigma_tip, source_stress, lam, G, pt, velocity, encounter, d_emit, d_escape=0.0):
        return {
            "sigma_tip_Pa": float(sigma_tip),
            "source_stress_min_Pa": float(np.min(source_stress)),
            "source_stress_max_Pa": float(np.max(source_stress)),
            "source_backstress_max_Pa": float(np.max(self.source_backstress_Pa())),
            "lambda_emit_total_s": float(np.sum(lam)),
            "G_emit_min_eV": float(np.min(G)),
            "dN_emit": float(d_emit),
            "dN_escape": float(d_escape),
            "mobile_count": self.mobile_count,
            "retained_count": self.retained_count,
            "K_shield_Pa_sqrt_m": self.shielding_K_Pa_sqrt_m(),
            "r_eff_m": self.blunted_radius_m(),
            "peierls_rate_max_s": float(np.max(pt.peierls_net_s)),
            "taylor_rate_max_s": float(np.max(pt.taylor_net_s)),
            "series_rate_max_s": float(np.max(pt.series_s)),
            "glide_velocity_max_m_s": float(np.max(velocity)),
            "encounter_rate_max_s": float(np.max(encounter)),
            "source_inventory_active": 0.0,
            "source_refresh_active": 0.0,
        }

    def advance(self, distance_m: float) -> dict[str, float]:
        distance = max(float(distance_m), 0.0)
        if distance == 0.0:
            return {"wake_mobile": 0.0, "wake_retained": 0.0}
        old_m = self.mobile.copy()
        old_r = self.retained.copy()
        sample_x = self.x + distance
        for s in range(self.cfg.n_systems):
            self.mobile[s] = np.interp(sample_x, self.x, old_m[s], left=0.0, right=0.0)
            self.retained[s] = np.interp(sample_x, self.x, old_r[s], left=0.0, right=0.0)
        return {"wake_mobile": float(old_m.sum() - self.mobile.sum()), "wake_retained": float(old_r.sum() - self.retained.sum())}
