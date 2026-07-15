"""Reusable-source moving process zone with emergent self-limitation.

No finite source inventory is present. Source population follows crack-tip
geometry; repeat emission is limited by nucleation, reload, near-tip backstress,
Peierls transport, Taylor release, recovery, escape and crack advance. Blunting
is driven by accumulated glide rather than stationary mobile population.
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
    background_forest_density_m2: float = 5.0e12
    process_zone_width_factor: float = 2.0

    # N_source/system = theta_active * r_eff / source_spacing.
    source_spacing_m: float = 1.0e-6
    source_active_angle_rad: float = math.pi
    source_reload_time_s: float = 1.0e-3
    resolved_emission_fraction: float = 1.0 / math.sqrt(3.0)

    mobile_source_backstress_fraction: float = 1.0
    backstress_geometry_factor: float = 1.0
    shielding_geometry_factor: float = 1.0
    core_radius_m: float = 2.74e-10

    mobile_recovery_rate_s: float = 0.0
    max_advection_cfl: float = 0.25
    # Numerical integration tolerance, not a physical emission cap.
    max_emit_increment_per_substep: float = 5.0
    population_activity_floor: float = 1.0e-12
    max_substeps: int = 100_000


class ReusableSourceProcessZone:
    """Expected dislocation-line counts per unit out-of-plane thickness."""

    def __init__(
        self,
        material: MaterialParameterization,
        config: ProcessZoneConfig | None = None,
    ):
        self.material = material
        self.cfg = config or ProcessZoneConfig()
        if self.cfg.n_bins < 4 or self.cfg.n_systems < 1:
            raise ValueError("process zone requires at least four bins and one slip system")
        if self.cfg.source_spacing_m <= 0.0:
            raise ValueError("source spacing must be positive")
        if self.cfg.source_reload_time_s < 0.0:
            raise ValueError("source reload time cannot be negative")
        if self.cfg.background_forest_density_m2 < 0.0:
            raise ValueError("background forest density cannot be negative")
        self.dx = self.cfg.length_m / self.cfg.n_bins
        self.x = (np.arange(self.cfg.n_bins, dtype=float) + 0.5) * self.dx
        self.mobile = np.zeros((self.cfg.n_systems, self.cfg.n_bins), dtype=float)
        self.retained = np.zeros_like(self.mobile)
        self.slip_count = np.zeros_like(self.mobile)
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.time_s = 0.0

    def copy(self):
        other = ReusableSourceProcessZone(self.material, self.cfg)
        other.mobile = self.mobile.copy()
        other.retained = self.retained.copy()
        other.slip_count = self.slip_count.copy()
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
    def local_slip_count(self) -> float:
        weights = np.exp(-self.x / max(0.2 * self.cfg.length_m, self.dx))
        return float(np.sum(self.slip_count * weights[None, :]))

    def blunted_radius_m(self) -> float:
        return (
            self.cfg.r0_m
            + self.material.state.c_blunt
            * self.cfg.burgers_vector_m
            * self.local_slip_count
        )

    def process_zone_width_m(self) -> float:
        width = self.cfg.process_zone_width_factor * max(
            self.blunted_radius_m(), self.cfg.r0_m
        )
        return min(max(width, self.cfg.r0_m), self.cfg.length_m)

    @property
    def forest_density_m2(self) -> np.ndarray:
        area = max(self.dx * self.process_zone_width_m(), 1.0e-300)
        return (
            self.cfg.background_forest_density_m2
            + np.maximum(self.retained.sum(axis=0), 0.0) / area
        )

    @property
    def mobile_density_m2(self) -> np.ndarray:
        area = max(self.dx * self.process_zone_width_m(), 1.0e-300)
        return np.maximum(self.mobile.sum(axis=0), 0.0) / area

    def geometric_source_count_per_system(self) -> float:
        radius = min(
            max(self.blunted_radius_m(), self.cfg.r0_m),
            self.cfg.length_m,
        )
        arc_length = max(self.cfg.source_active_angle_rad, 0.0) * radius
        return float(arc_length / self.cfg.source_spacing_m)

    def source_backstress_Pa(self) -> np.ndarray:
        pref = (
            self.cfg.backstress_geometry_factor
            * self.cfg.shear_modulus_Pa
            * self.cfg.burgers_vector_m
            / (2.0 * math.pi * (1.0 - self.cfg.poisson_ratio))
        )
        kernel = 1.0 / np.maximum(
            self.x + self.cfg.core_radius_m,
            self.cfg.core_radius_m,
        )
        resisting = (
            self.retained
            + self.cfg.mobile_source_backstress_fraction * self.mobile
        )
        return pref * (resisting @ kernel)

    def shielding_K_Pa_sqrt_m(self) -> float:
        pref = (
            self.cfg.shielding_geometry_factor
            * self.cfg.shear_modulus_Pa
            * self.cfg.burgers_vector_m
            / (
                2.0
                * (1.0 - self.cfg.poisson_ratio)
                * math.sqrt(2.0 * math.pi)
            )
        )
        return float(
            pref
            * np.sum(
                self.retained.sum(axis=0)
                / np.sqrt(np.maximum(self.x, self.cfg.core_radius_m))
            )
        )

    def effective_tip_stress_Pa(self, K_drive_Pa_sqrt_m: float) -> float:
        K_eff = max(
            float(K_drive_Pa_sqrt_m) - self.shielding_K_Pa_sqrt_m(),
            0.0,
        )
        return K_eff / math.sqrt(
            2.0 * math.pi * max(self.blunted_radius_m(), 1.0e-300)
        )

    def _rates(self, K_drive_Pa_sqrt_m: float, temperature_K: float):
        sigma_tip = self.effective_tip_stress_Pa(K_drive_Pa_sqrt_m)
        source_stress = np.maximum(
            self.cfg.resolved_emission_fraction * sigma_tip
            - self.source_backstress_Pa(),
            0.0,
        )
        lambda_site, G_emit = arrhenius_rate_s(
            source_stress,
            temperature_K,
            self.material.emission,
            1.0e11,
        )
        tau_reload = self.cfg.source_reload_time_s
        lambda_reusable = (
            lambda_site / (1.0 + lambda_site * tau_reload)
            if tau_reload > 0.0
            else lambda_site
        )
        n_source = self.geometric_source_count_per_system()
        lambda_emit = n_source * lambda_reusable

        rho_f = self.forest_density_m2
        rho_m = self.mobile_density_m2
        pt = evaluate_pt_rates(
            np.full_like(rho_f, sigma_tip),
            rho_f,
            rho_m,
            temperature_K,
            self.cfg.burgers_vector_m,
            self.material,
        )
        velocity = pt.jump_length_m * pt.series_s
        encounter = (
            self.material.state.encounter_efficiency
            * velocity
            * np.sqrt(rho_f)
        )
        return (
            sigma_tip,
            source_stress,
            lambda_emit,
            lambda_site,
            G_emit,
            n_source,
            pt,
            velocity,
            encounter,
        )

    @staticmethod
    def _exchange_exact(m, r, capture_rate, release_rate, h):
        """Exact mobile<->retained exchange for fixed first-order rates."""
        total = m + r
        q = capture_rate + release_rate
        equilibrium_mobile = np.where(
            q > 0.0,
            total * release_rate / np.maximum(q, 1.0e-300),
            m,
        )
        decay = np.exp(-np.clip(q * h, 0.0, 700.0))
        m_new = equilibrium_mobile + (m - equilibrium_mobile) * decay
        m_new = np.minimum(np.maximum(m_new, 0.0), total)
        return m_new, np.maximum(total - m_new, 0.0)

    def evolve(
        self,
        dt_s: float,
        temperature_K: float,
        K_drive_Pa_sqrt_m: float,
    ) -> dict[str, float]:
        dt = max(float(dt_s), 0.0)
        rates = self._rates(K_drive_Pa_sqrt_m, temperature_K)
        if dt == 0.0:
            return self._diagnostics(
                *rates, d_emit=0.0, d_escape=0.0, n_substeps=0
            )

        emitted_before = self.emitted_total
        escaped_before = self.escaped_total
        remaining = dt
        n_substeps = 0
        tiny = max(1.0e-15 * dt, 1.0e-30)

        while remaining > tiny:
            rates = self._rates(K_drive_Pa_sqrt_m, temperature_K)
            (
                sigma_tip,
                source_stress,
                lam,
                lambda_site,
                G,
                n_source,
                pt,
                velocity,
                encounter,
            ) = rates

            total_emit_rate = max(float(np.sum(lam)), 0.0)
            mobile_by_bin = self.mobile.sum(axis=0)
            active = mobile_by_bin > self.cfg.population_activity_floor
            active_velocity = (
                float(np.max(velocity[active])) if np.any(active) else 0.0
            )

            h = remaining
            if active_velocity > 0.0:
                h = min(
                    h,
                    self.dx * self.cfg.max_advection_cfl / active_velocity,
                )
            if total_emit_rate > 0.0:
                h = min(
                    h,
                    self.cfg.max_emit_increment_per_substep / total_emit_rate,
                )
            h = max(min(h, remaining), tiny)

            n_substeps += 1
            if n_substeps > int(self.cfg.max_substeps):
                raise RuntimeError(
                    "process-zone adaptive integration exceeded max_substeps; "
                    "reduce the outer timestep or audit source/advection kinetics"
                )

            emitted = np.asarray(lam, dtype=float) * h
            self.mobile[:, 0] += emitted
            self.emitted_total += float(np.sum(emitted))

            release = np.asarray(pt.taylor_net_s, dtype=float)
            for s in range(self.cfg.n_systems):
                m, r = self._exchange_exact(
                    self.mobile[s],
                    self.retained[s],
                    encounter,
                    release,
                    h,
                )
                if self.cfg.mobile_recovery_rate_s > 0.0:
                    m *= math.exp(
                        -min(self.cfg.mobile_recovery_rate_s * h, 700.0)
                    )
                retained_recovery = self.material.state.retained_recovery_rate_s
                if retained_recovery > 0.0:
                    r *= math.exp(-min(retained_recovery * h, 700.0))

                cfl = np.clip(
                    velocity * h / max(self.dx, 1.0e-300),
                    0.0,
                    1.0,
                )
                flux = cfl * m
                self.slip_count[s] += flux
                escaped = float(flux[-1])
                m[1:] += flux[:-1]
                m -= flux
                self.mobile[s] = np.maximum(m, 0.0)
                self.retained[s] = np.maximum(r, 0.0)
                self.escaped_total += escaped

            remaining -= h

        self.time_s += dt
        rates = self._rates(K_drive_Pa_sqrt_m, temperature_K)
        return self._diagnostics(
            *rates,
            d_emit=self.emitted_total - emitted_before,
            d_escape=self.escaped_total - escaped_before,
            n_substeps=n_substeps,
        )

    def _diagnostics(
        self,
        sigma_tip,
        source_stress,
        lam,
        lambda_site,
        G,
        n_source,
        pt,
        velocity,
        encounter,
        d_emit,
        d_escape=0.0,
        n_substeps=0,
    ):
        reload_limit = (
            1.0 / self.cfg.source_reload_time_s
            if self.cfg.source_reload_time_s > 0.0
            else float("inf")
        )
        radius = self.blunted_radius_m()
        return {
            "sigma_tip_Pa": float(sigma_tip),
            "source_stress_min_Pa": float(np.min(source_stress)),
            "source_stress_max_Pa": float(np.max(source_stress)),
            "source_backstress_max_Pa": float(
                np.max(self.source_backstress_Pa())
            ),
            "geometric_source_count_per_system": float(n_source),
            "source_spacing_m": float(self.cfg.source_spacing_m),
            "source_reload_time_s": float(self.cfg.source_reload_time_s),
            "source_reload_rate_limit_s": float(reload_limit),
            "lambda_site_max_s": float(np.max(lambda_site)),
            "lambda_emit_total_s": float(np.sum(lam)),
            "G_emit_min_eV": float(np.min(G)),
            "dN_emit": float(d_emit),
            "dN_escape": float(d_escape),
            "mobile_count": self.mobile_count,
            "retained_count": self.retained_count,
            "local_slip_count": self.local_slip_count,
            "background_forest_density_m2": float(
                self.cfg.background_forest_density_m2
            ),
            "process_zone_width_m": self.process_zone_width_m(),
            "forest_density_max_m2": float(np.max(self.forest_density_m2)),
            "mobile_density_max_m2": float(np.max(self.mobile_density_m2)),
            "K_shield_Pa_sqrt_m": self.shielding_K_Pa_sqrt_m(),
            "r_eff_m": radius,
            "tip_radius_over_r0": radius / self.cfg.r0_m,
            "tip_radius_exceeds_process_zone": float(
                radius > self.cfg.length_m
            ),
            "peierls_rate_max_s": float(np.max(pt.peierls_net_s)),
            "taylor_rate_max_s": float(np.max(pt.taylor_net_s)),
            "series_rate_max_s": float(np.max(pt.series_s)),
            "glide_velocity_max_m_s": float(np.max(velocity)),
            "encounter_rate_max_s": float(np.max(encounter)),
            "process_zone_substeps": int(n_substeps),
            "stiff_exchange_integrator_active": 1.0,
            "density_area_normalization_active": 1.0,
            "source_inventory_active": 0.0,
            "source_refresh_active": 0.0,
            "source_geometry_active": 1.0,
            "mobile_source_backstress_active": float(
                self.cfg.mobile_source_backstress_fraction > 0.0
            ),
            "mobile_direct_K_shielding_active": 0.0,
        }

    def advance(self, distance_m: float) -> dict[str, float]:
        distance = max(float(distance_m), 0.0)
        if distance == 0.0:
            return {
                "wake_mobile": 0.0,
                "wake_retained": 0.0,
                "wake_slip": 0.0,
            }
        old_m = self.mobile.copy()
        old_r = self.retained.copy()
        old_g = self.slip_count.copy()
        sample_x = self.x + distance
        for s in range(self.cfg.n_systems):
            self.mobile[s] = np.interp(
                sample_x, self.x, old_m[s], left=0.0, right=0.0
            )
            self.retained[s] = np.interp(
                sample_x, self.x, old_r[s], left=0.0, right=0.0
            )
            self.slip_count[s] = np.interp(
                sample_x, self.x, old_g[s], left=0.0, right=0.0
            )
        return {
            "wake_mobile": float(old_m.sum() - self.mobile.sum()),
            "wake_retained": float(old_r.sum() - self.retained.sum()),
            "wake_slip": float(old_g.sum() - self.slip_count.sum()),
        }
