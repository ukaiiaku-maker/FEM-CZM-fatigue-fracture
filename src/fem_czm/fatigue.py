"""Fatigue integration using the same front state and barriers as monotonic loading."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .front import CrackFront


@dataclass(frozen=True)
class FatigueConfig:
    load_ratio_R: float = 0.1
    frequency_Hz: float = 1000.0
    phase_points: int = 32
    max_cycles_per_chunk: float = 1.0
    closure_clip: bool = True


class FatigueIntegrator:
    def __init__(self, front: CrackFront, config: FatigueConfig | None = None):
        self.front = front
        self.cfg = config or FatigueConfig()
        if self.cfg.frequency_Hz <= 0.0:
            raise ValueError("frequency must be positive")
        if self.cfg.phase_points < 4:
            raise ValueError("at least four phase points are required")
        if self.cfg.max_cycles_per_chunk <= 0.0:
            raise ValueError("max_cycles_per_chunk must be positive")
        self.cycles = 0.0

    def waveform(self, Kmax_Pa_sqrt_m: float) -> np.ndarray:
        phase = (
            np.arange(self.cfg.phase_points, dtype=float) + 0.5
        ) / self.cfg.phase_points
        q = (
            0.5 * (1.0 + self.cfg.load_ratio_R)
            + 0.5
            * (1.0 - self.cfg.load_ratio_R)
            * np.cos(2.0 * math.pi * phase)
        )
        if self.cfg.closure_clip:
            q = np.maximum(q, 0.0)
        return float(Kmax_Pa_sqrt_m) * q

    def advance_cycles(
        self,
        Kmax_Pa_sqrt_m: float,
        temperature_K: float,
        cycles: float,
    ) -> dict[str, float]:
        remaining_cycles = max(float(cycles), 0.0)
        total_fire = 0
        emitted0 = self.front.process_zone.emitted_total
        escaped0 = self.front.process_zone.escaped_total
        ext0 = self.front.crack_extension_m
        last: dict[str, float] = {}
        n_ordered_sequences = 0

        while remaining_cycles > 0.0:
            chunk = min(remaining_cycles, self.cfg.max_cycles_per_chunk)
            dt_phase = chunk / self.cfg.frequency_Hz / self.cfg.phase_points
            for K in self.waveform(Kmax_Pa_sqrt_m):
                phase_remaining = dt_phase
                tiny = max(1.0e-15 * dt_phase, 1.0e-30)
                while phase_remaining > tiny:
                    last = self.front.step(
                        float(K),
                        temperature_K,
                        phase_remaining,
                    )
                    total_fire += int(last.get("n_fire", 0))
                    new_remaining = float(last.get("unused_dt_s", 0.0))
                    if int(last.get("n_fire", 0)) == 0:
                        phase_remaining = 0.0
                    elif new_remaining >= phase_remaining * (1.0 - 1.0e-14):
                        raise RuntimeError("event-limited fatigue phase made no time progress")
                    else:
                        phase_remaining = new_remaining
            self.cycles += chunk
            remaining_cycles -= chunk
            n_ordered_sequences += 1

        da = self.front.crack_extension_m - ext0
        return {
            **last,
            "cycles_total": self.cycles,
            "cycles_advanced": float(cycles),
            "n_fire_block": total_fire,
            "dN_emit_block": self.front.process_zone.emitted_total - emitted0,
            "dN_escape_block": self.front.process_zone.escaped_total - escaped0,
            "da_block_m": da,
            "da_dN_m_per_cycle": da / max(float(cycles), 1.0e-300),
            "ordered_cycle_sequences": n_ordered_sequences,
            "max_cycles_per_ordered_sequence": float(
                self.cfg.max_cycles_per_chunk
            ),
            "fatigue_uses_shared_front_state": 1.0,
            "phenomenological_paris_law_active": 0.0,
        }
