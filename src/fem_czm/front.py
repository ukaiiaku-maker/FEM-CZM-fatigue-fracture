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
    max_clock_increment_per_substep: float = 0.10
    max_constitutive_substep_s: float = 50.0
    max_substeps: int = 100_000


class CrackFront:
    def __init__(
        self,
        material: MaterialParameterization,
        front_config: FrontConfig | None = None,
        process_zone_config: ProcessZoneConfig | None = None,
    ):
        self.material = material
        self.cfg = front_config or FrontConfig()
        self.process_zone = ReusableSourceProcessZone(material, process_zone_config)
        self.cleavage_clock = 0.0
        self.crack_extension_m = 0.0
        self.n_advances = 0
        self.time_s = 0.0

    def _cleavage_rates(self, K_drive_Pa_sqrt_m: float, temperature_K: float):
        sigma = self.process_zone.effective_tip_stress_Pa(K_drive_Pa_sqrt_m)
        raw, Gc = arrhenius_rate_s(
            sigma,
            temperature_K,
            self.material.cleavage,
            self.cfg.cleavage_attempt_frequency_s,
        )
        effective = cooperative_cleavage_rate_s(
            raw,
            self.cfg.cleavage_multiplicity,
            self.cfg.cleavage_renewal_time_s,
        )
        return float(sigma), float(raw), float(effective), float(Gc)

    def _fire_once(self) -> dict[str, float]:
        wake = self.process_zone.advance(self.cfg.advance_increment_m)
        self.crack_extension_m += self.cfg.advance_increment_m
        self.n_advances += 1
        # Hazard accumulated beyond the first crossing belongs to a geometry that
        # no longer exists.  It is not carried into the renewed crack-tip state.
        self.cleavage_clock = 0.0
        return wake

    def step(
        self,
        K_drive_Pa_sqrt_m: float,
        temperature_K: float,
        dt_s: float,
    ) -> dict[str, float]:
        """Advance the constitutive state until ``dt_s`` or the first crack event.

        At most one crack increment is committed.  If the event occurs before the
        requested interval ends, ``unused_dt_s`` is returned so the caller can
        recompute the post-advance state at the same applied load.  This prevents
        thousands of crack increments from being executed using one obsolete tip
        geometry and one obsolete process-zone state.
        """
        dt_total = max(float(dt_s), 0.0)
        B_start = float(self.cleavage_clock)
        emitted_step = 0.0
        escaped_step = 0.0
        consumed = 0.0
        n_substeps = 0
        trial_overflow = 0.0

        # Defensive handling for a restored/checkpointed state already at or above
        # threshold.  Commit one event and return the full interval as unused.
        if self.cleavage_clock >= 1.0:
            pz = self.process_zone.evolve(0.0, temperature_K, K_drive_Pa_sqrt_m)
            sigma, raw, effective, Gc = self._cleavage_rates(
                K_drive_Pa_sqrt_m,
                temperature_K,
            )
            overflow = max(self.cleavage_clock - 1.0, 0.0)
            wake = self._fire_once()
            return {
                **pz,
                **wake,
                "K_drive_Pa_sqrt_m": float(K_drive_Pa_sqrt_m),
                "cleavage_raw_rate_s": raw,
                "cleavage_effective_rate_s": effective,
                "G_cleave_eV": Gc,
                "cleavage_clock": self.cleavage_clock,
                "cleavage_clock_start": B_start,
                "cleavage_clock_overflow_discarded": overflow,
                "n_fire": 1,
                "crack_extension_m": self.crack_extension_m,
                "time_s": self.time_s,
                "step_requested_s": dt_total,
                "step_consumed_s": 0.0,
                "unused_dt_s": dt_total,
                "front_substeps": 0,
                "event_limited": 1.0,
            }

        if dt_total == 0.0:
            pz = self.process_zone.evolve(0.0, temperature_K, K_drive_Pa_sqrt_m)
            sigma, raw, effective, Gc = self._cleavage_rates(
                K_drive_Pa_sqrt_m,
                temperature_K,
            )
            return {
                **pz,
                "wake_mobile": 0.0,
                "wake_retained": 0.0,
                "wake_slip": 0.0,
                "K_drive_Pa_sqrt_m": float(K_drive_Pa_sqrt_m),
                "cleavage_raw_rate_s": raw,
                "cleavage_effective_rate_s": effective,
                "G_cleave_eV": Gc,
                "cleavage_clock": self.cleavage_clock,
                "cleavage_clock_start": B_start,
                "cleavage_clock_overflow_discarded": 0.0,
                "n_fire": 0,
                "crack_extension_m": self.crack_extension_m,
                "time_s": self.time_s,
                "step_requested_s": 0.0,
                "step_consumed_s": 0.0,
                "unused_dt_s": 0.0,
                "front_substeps": 0,
                "event_limited": 1.0,
            }

        remaining = dt_total
        tiny = max(1.0e-15 * dt_total, 1.0e-30)
        last_pz = self.process_zone.evolve(
            0.0,
            temperature_K,
            K_drive_Pa_sqrt_m,
        )
        sigma, raw, effective, Gc = self._cleavage_rates(
            K_drive_Pa_sqrt_m,
            temperature_K,
        )

        while remaining > tiny:
            n_substeps += 1
            if n_substeps > int(self.cfg.max_substeps):
                raise RuntimeError(
                    "front integration exceeded max_substeps; reduce the outer "
                    "time/load increment or audit the cleavage clock"
                )

            sigma0, raw0, effective0, Gc0 = self._cleavage_rates(
                K_drive_Pa_sqrt_m,
                temperature_K,
            )
            h = min(remaining, max(self.cfg.max_constitutive_substep_s, tiny))
            if effective0 > 0.0:
                h = min(
                    h,
                    self.cfg.max_clock_increment_per_substep / effective0,
                )
            h = max(min(h, remaining), tiny)

            snapshot = self.process_zone.copy()
            pz_trial = self.process_zone.evolve(
                h,
                temperature_K,
                K_drive_Pa_sqrt_m,
            )
            sigma1, raw1, effective1, Gc1 = self._cleavage_rates(
                K_drive_Pa_sqrt_m,
                temperature_K,
            )
            effective_mean = 0.5 * (effective0 + effective1)
            dB_trial = max(effective_mean * h, 0.0)

            if self.cleavage_clock + dB_trial >= 1.0 and dB_trial > 0.0:
                remaining_clock = max(1.0 - self.cleavage_clock, 0.0)
                fraction = min(max(remaining_clock / dB_trial, 0.0), 1.0)
                h_event = h * fraction
                trial_overflow = max(
                    self.cleavage_clock + dB_trial - 1.0,
                    0.0,
                )

                # Restore the pre-trial state and integrate only to the crossing.
                self.process_zone = snapshot
                last_pz = self.process_zone.evolve(
                    h_event,
                    temperature_K,
                    K_drive_Pa_sqrt_m,
                )
                emitted_step += float(last_pz.get("dN_emit", 0.0))
                escaped_step += float(last_pz.get("dN_escape", 0.0))
                consumed += h_event
                self.time_s += h_event
                sigma, raw, effective, Gc = self._cleavage_rates(
                    K_drive_Pa_sqrt_m,
                    temperature_K,
                )
                wake = self._fire_once()
                last_pz = dict(last_pz)
                last_pz["dN_emit"] = emitted_step
                last_pz["dN_escape"] = escaped_step
                return {
                    **last_pz,
                    **wake,
                    "K_drive_Pa_sqrt_m": float(K_drive_Pa_sqrt_m),
                    "cleavage_raw_rate_s": raw,
                    "cleavage_effective_rate_s": effective,
                    "G_cleave_eV": Gc,
                    "cleavage_clock": self.cleavage_clock,
                    "cleavage_clock_start": B_start,
                    "cleavage_clock_overflow_discarded": trial_overflow,
                    "n_fire": 1,
                    "crack_extension_m": self.crack_extension_m,
                    "time_s": self.time_s,
                    "step_requested_s": dt_total,
                    "step_consumed_s": consumed,
                    "unused_dt_s": max(dt_total - consumed, 0.0),
                    "front_substeps": n_substeps,
                    "event_limited": 1.0,
                }

            # Accept the complete trial substep.
            last_pz = pz_trial
            emitted_step += float(last_pz.get("dN_emit", 0.0))
            escaped_step += float(last_pz.get("dN_escape", 0.0))
            self.cleavage_clock += dB_trial
            consumed += h
            remaining -= h
            self.time_s += h
            sigma, raw, effective, Gc = sigma1, raw1, effective1, Gc1

            if self.cfg.cleavage_clock_decay_time_s > 0.0 and h > 0.0:
                self.cleavage_clock *= math.exp(
                    -min(h / self.cfg.cleavage_clock_decay_time_s, 80.0)
                )

        last_pz = dict(last_pz)
        last_pz["dN_emit"] = emitted_step
        last_pz["dN_escape"] = escaped_step
        return {
            **last_pz,
            "wake_mobile": 0.0,
            "wake_retained": 0.0,
            "wake_slip": 0.0,
            "K_drive_Pa_sqrt_m": float(K_drive_Pa_sqrt_m),
            "cleavage_raw_rate_s": raw,
            "cleavage_effective_rate_s": effective,
            "G_cleave_eV": Gc,
            "cleavage_clock": self.cleavage_clock,
            "cleavage_clock_start": B_start,
            "cleavage_clock_overflow_discarded": trial_overflow,
            "n_fire": 0,
            "crack_extension_m": self.crack_extension_m,
            "time_s": self.time_s,
            "step_requested_s": dt_total,
            "step_consumed_s": consumed,
            "unused_dt_s": 0.0,
            "front_substeps": n_substeps,
            "event_limited": 1.0,
        }
