"""Original finite-site source closure retained as a diagnostic ablation.

This module changes only source production. Transport, Peierls--Taylor kinetics,
density normalization, glide-based blunting, shielding and crack-front event
handling are inherited from the reusable-source process zone.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .hazard import arrhenius_rate_s
from .parameters import MaterialParameterization
from .process_zone import ProcessZoneConfig, ReusableSourceProcessZone


@dataclass(frozen=True)
class LegacyFiniteSiteConfig(ProcessZoneConfig):
    source_sites_per_system: float = 200.0
    source_recovery_rate_s: float = 0.0
    source_refresh_length_m: float = 2.5e-7
    source_bin_count: int = 2


class LegacyFiniteSiteProcessZone(ReusableSourceProcessZone):
    """One-shot source sites with optional time recovery and advance refresh."""

    def __init__(
        self,
        material: MaterialParameterization,
        config: LegacyFiniteSiteConfig,
    ):
        super().__init__(material, config)
        self.cfg = config
        capacity = max(float(config.source_sites_per_system), 0.0)
        self.site_capacity = np.full(self.cfg.n_systems, capacity, dtype=float)
        self.available_sites = self.site_capacity.copy()
        self.source_recovered_total = 0.0
        self.source_refreshed_total = 0.0

    @property
    def available_site_fraction(self) -> float:
        total = float(np.sum(self.site_capacity))
        return float(np.sum(self.available_sites) / total) if total > 0.0 else 0.0

    def copy(self):
        other = LegacyFiniteSiteProcessZone(self.material, self.cfg)
        other.mobile = self.mobile.copy()
        other.retained = self.retained.copy()
        other.slip_count = self.slip_count.copy()
        other.emitted_total = self.emitted_total
        other.escaped_total = self.escaped_total
        other.time_s = self.time_s
        other.site_capacity = self.site_capacity.copy()
        other.available_sites = self.available_sites.copy()
        other.source_recovered_total = self.source_recovered_total
        other.source_refreshed_total = self.source_refreshed_total
        return other

    def _raw_source_rate(self, K_drive_Pa_sqrt_m: float, temperature_K: float):
        sigma_tip = self.effective_tip_stress_Pa(K_drive_Pa_sqrt_m)
        # The original finite-site closure used the crack-tip emission hazard and
        # did not apply the new front-local source-backstress subtraction.
        source_stress = np.full(
            self.cfg.n_systems,
            max(self.cfg.resolved_emission_fraction * sigma_tip, 0.0),
            dtype=float,
        )
        rate, barrier = arrhenius_rate_s(
            source_stress,
            temperature_K,
            self.material.emission,
            1.0e11,
        )
        return sigma_tip, source_stress, rate, barrier

    def _rates(self, K_drive_Pa_sqrt_m: float, temperature_K: float):
        """Return zero continuous emission while preserving base transport rates."""
        base = super()._rates(K_drive_Pa_sqrt_m, temperature_K)
        sigma_tip, source_stress, raw_rate, barrier = self._raw_source_rate(
            K_drive_Pa_sqrt_m,
            temperature_K,
        )
        zeros = np.zeros_like(raw_rate)
        return (
            sigma_tip,
            source_stress,
            zeros,
            raw_rate,
            zeros,
            zeros,
            barrier,
            float(np.mean(self.available_sites)),
            base[8],
            base[9],
            base[10],
        )

    def _recover_sites(self, dt_s: float) -> float:
        rate = max(float(self.cfg.source_recovery_rate_s), 0.0)
        if rate <= 0.0 or dt_s <= 0.0:
            return 0.0
        fraction = 1.0 - math.exp(-min(rate * dt_s, 700.0))
        recovered = (self.site_capacity - self.available_sites) * fraction
        self.available_sites += recovered
        amount = float(np.sum(recovered))
        self.source_recovered_total += amount
        return amount

    def evolve(
        self,
        dt_s: float,
        temperature_K: float,
        K_drive_Pa_sqrt_m: float,
    ) -> dict[str, float]:
        dt = max(float(dt_s), 0.0)
        sigma_tip, source_stress, raw_rate, barrier = self._raw_source_rate(
            K_drive_Pa_sqrt_m,
            temperature_K,
        )
        hazard = np.maximum(raw_rate, 0.0) * dt
        probability = 1.0 - np.exp(-np.minimum(hazard, 700.0))
        emitted = self.available_sites * probability
        self.available_sites = np.maximum(self.available_sites - emitted, 0.0)

        source_bins = max(min(int(self.cfg.source_bin_count), self.cfg.n_bins), 1)
        self.mobile[:, :source_bins] += emitted[:, None] / source_bins
        self.emitted_total += float(np.sum(emitted))
        recovered = self._recover_sites(dt)

        out = super().evolve(dt, temperature_K, K_drive_Pa_sqrt_m)
        emitted_total = float(np.sum(emitted))
        effective_rate = emitted_total / dt if dt > 0.0 else 0.0
        out.update(
            {
                "source_mode": "legacy_finite_site",
                "source_inventory_active": 1.0,
                "source_refresh_active": 1.0,
                "source_geometry_active": 0.0,
                "mobile_source_backstress_active": 0.0,
                "legacy_source_sites_per_system": float(self.site_capacity[0]),
                "legacy_available_sites_total": float(np.sum(self.available_sites)),
                "legacy_available_site_fraction": self.available_site_fraction,
                "legacy_source_refresh_length_m": float(
                    self.cfg.source_refresh_length_m
                ),
                "legacy_source_recovery_rate_s": float(
                    self.cfg.source_recovery_rate_s
                ),
                "legacy_source_hazard_max": float(np.max(hazard)),
                "legacy_source_probability_max": float(np.max(probability)),
                "legacy_raw_site_rate_max_s": float(np.max(raw_rate)),
                "lambda_site_max_s": float(np.max(raw_rate)),
                "lambda_site_forward_max_s": float(np.max(raw_rate)),
                "lambda_site_reverse_max_s": 0.0,
                "lambda_site_net_max_s": float(np.max(raw_rate)),
                "lambda_emit_total_s": effective_rate,
                "zero_stress_net_emission_active": 0.0,
                "dN_emit": emitted_total,
                "dN_source_recovered": recovered,
                "G_emit_min_eV": float(np.min(barrier)),
                "source_stress_min_Pa": float(np.min(source_stress)),
                "source_stress_max_Pa": float(np.max(source_stress)),
                "sigma_tip_Pa": float(sigma_tip),
            }
        )
        return out

    def advance(self, distance_m: float) -> dict[str, float]:
        distance = max(float(distance_m), 0.0)
        out = super().advance(distance)
        refresh_length = max(float(self.cfg.source_refresh_length_m), self.dx)
        fresh_fraction = min(distance / refresh_length, 1.0)
        refreshed = (self.site_capacity - self.available_sites) * fresh_fraction
        self.available_sites += refreshed
        amount = float(np.sum(refreshed))
        self.source_refreshed_total += amount
        out.update(
            {
                "source_sites_refreshed": amount,
                "legacy_available_sites_total": float(np.sum(self.available_sites)),
                "legacy_available_site_fraction": self.available_site_fraction,
            }
        )
        return out
