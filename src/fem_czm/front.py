"""Shared monotonic and cyclic crack-front state machine."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .hazard import arrhenius_rate_s, cooperative_cleavage_rate_s
from .parameters import MaterialParameterization
from .process_zone import ProcessZoneConfig, ReusableSourceProcessZone


@dataclass(frozen=True)
class FrontConfig:
    advance_increment_m: float = 5.0e-6
    cleavage_attempt_frequency_s: float = 1.0e12
    cleavage_multiplicity: float = 3.0
    cleavage_renewal_time_s: float = 1.0e-6
    cleavage_clock_decay_time_s: float = 0.0


class CrackFront:
    def __init__(self, material: MaterialParameterization, front_config: FrontConfig | None = None, process_zone_config: ProcessZoneConfig | None = None):
        self.material = material
        self.cfg = front_config or FrontConfig()
        self.process_zone = ReusableSourceProcessZone(material, process_zone_config)
        self.cleavage_clock = 0.0
        self.crack_extension_m = 0.0
        self.n_advances = 0
        self.time_s = 0.0

    def step(self, K_drive_Pa_sqrt_m: float, temperature_K: float, dt_s: float) -> dict[str, float]:
        dt = max(float(dt_s), 0.0)
        pz = self.process_zone.evolve(dt, temperature_K, K_drive_Pa_sqrt_m)
        sigma = self.process_zone.effective_tip_stress_Pa(K_drive_Pa_sqrt_m)
        raw, Gc = arrhenius_rate_s(sigma, temperature_K, self.material.cleavage, self.cfg.cleavage_attempt_frequency_s)
        effective = cooperative_cleavage_rate_s(raw, self.cfg.cleavage_multiplicity, self.cfg.cleavage_renewal_time_s)
        if self.cfg.cleavage_clock_decay_time_s > 0.0 and dt > 0.0:
            self.cleavage_clock *= math.exp(-min(dt / self.cfg.cleavage_clock_decay_time_s, 80.0))
        self.cleavage_clock += float(effective) * dt
        nfire = int(math.floor(max(self.cleavage_clock, 0.0)))
        wake = {"wake_mobile": 0.0, "wake_retained": 0.0}
        if nfire:
            self.cleavage_clock -= nfire
            distance = nfire * self.cfg.advance_increment_m
            wake = self.process_zone.advance(distance)
            self.crack_extension_m += distance
            self.n_advances += nfire
        self.time_s += dt
        return {**pz, **wake, "K_drive_Pa_sqrt_m": float(K_drive_Pa_sqrt_m), "cleavage_raw_rate_s": float(raw), "cleavage_effective_rate_s": float(effective), "G_cleave_eV": float(Gc), "cleavage_clock": self.cleavage_clock, "n_fire": nfire, "crack_extension_m": self.crack_extension_m, "time_s": self.time_s}
