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
    max_cycles_per_chunk: float = 1.0e4
    closure_clip: bool = True


class FatigueIntegrator:
    def __init__(self, front: CrackFront, config: FatigueConfig | None = None):
        self.front = front
        self.cfg = config or FatigueConfig()
        if self.cfg.frequency_Hz <= 0.0:
            raise ValueError("frequency must be positive")
        if self.cfg.phase_points < 4:
            raise ValueError("at least four phase points are required")
        self.cycles = 0.0

    def waveform(self, Kmax_Pa_sqrt_m: float) -> np.ndarray:
        phase = (np.arange(self.cfg.phase_points) + 0.5) / self.cfg.phase_points
        q = 0.5 * (1.0 + self.cfg.load_ratio_R) + 0.5 * (1.0 - self.cfg.load_ratio_R) * np.cos(2.0 * math.pi * phase)
        if self.cfg.closure_clip:
            q = np.maximum(q, 0.0)
        return float(Kmax_Pa_sqrt_m) * q

    def advance_cycles(self, Kmax_Pa_sqrt_m: float, temperature_K: float, cycles: float) -> dict[str, float]:
        remaining = max(float(cycles), 0.0)
        total_fire = 0
        emitted0 = self.front.process_zone.emitted_total
        ext0 = self.front.crack_extension_m
        last = {}
        while remaining > 0.0:
            chunk = min(remaining, self.cfg.max_cycles_per_chunk)
            dt_phase = chunk / self.cfg.frequency_Hz / self.cfg.phase_points
            for K in self.waveform(Kmax_Pa_sqrt_m):
                last = self.front.step(float(K), temperature_K, dt_phase)
                total_fire += int(last["n_fire"])
            self.cycles += chunk
            remaining -= chunk
        return {**last, "cycles_total": self.cycles, "cycles_advanced": float(cycles), "n_fire_block": total_fire, "dN_emit_block": self.front.process_zone.emitted_total - emitted0, "da_block_m": self.front.crack_extension_m - ext0, "da_dN_m_per_cycle": (self.front.crack_extension_m - ext0) / max(float(cycles), 1.0e-300)}
