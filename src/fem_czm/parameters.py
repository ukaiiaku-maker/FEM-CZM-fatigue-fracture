"""The three selected v9.10.2/v9.10.3 parameterizations.

The finite source-site count and crack-advance source-refresh length are kept
only as provenance.  They are intentionally not consumed by the constitutive
code in this repository.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class ExpFloorParameters:
    G00_eV: float
    gT_eV_per_K: float
    sigma_c0_Pa: float
    sT_Pa_per_K: float
    alpha: float
    n: float
    floor_fraction: float


@dataclass(frozen=True)
class ActivatedTransportParameters:
    H0_eV: float
    activation_entropy_kB: float
    alpha: float
    n: float
    attempt_frequency_s: float


@dataclass(frozen=True)
class TaylorCorrelationParameters:
    rho_c_m2: float
    m_scale: float
    exponent: float = 1.0


@dataclass(frozen=True)
class StateParameters:
    encounter_efficiency: float
    retained_recovery_rate_s: float
    c_blunt: float
    legacy_source_sites_per_system: float
    legacy_source_refresh_length_um: float


@dataclass(frozen=True)
class MaterialParameterization:
    name: str
    candidate_id: str
    cleavage: ExpFloorParameters
    emission: ExpFloorParameters
    peierls: ActivatedTransportParameters
    taylor: ActivatedTransportParameters
    taylor_correlation: TaylorCorrelationParameters
    state: StateParameters

    def as_dict(self) -> dict:
        return asdict(self)


def _ef(G00, gT_meV, sig_GPa, sT_MPa, alpha, n, floor):
    return ExpFloorParameters(
        G00_eV=G00,
        gT_eV_per_K=gT_meV * 1.0e-3,
        sigma_c0_Pa=sig_GPa * 1.0e9,
        sT_Pa_per_K=sT_MPa * 1.0e6,
        alpha=alpha,
        n=n,
        floor_fraction=floor,
    )


MATERIALS: Mapping[str, MaterialParameterization] = {
    "ceramic": MaterialParameterization(
        name="ceramic",
        candidate_id="ceramic_restart02_candidate00",
        cleavage=_ef(4.67114, 3.11629, 5.82338, -4.03232, 1.81496, 2.26503, 0.00779574),
        emission=_ef(2.72111, 3.67522, 5.45915, -0.395077, 0.630857, 1.53995, 0.130553),
        peierls=ActivatedTransportParameters(5.78888, 16.1086, 1.51737, 1.32239, 1.0e12),
        taylor=ActivatedTransportParameters(11.6782, 4.58349, 0.458067, 1.45685, 1.0e11),
        taylor_correlation=TaylorCorrelationParameters(5.8335e12, 0.577581),
        state=StateParameters(2.0256, 3.0677e-4, 1.30653, 12.7187, 38.3812),
    ),
    "weakT": MaterialParameterization(
        name="weakT",
        candidate_id="weakT_restart00_candidate00",
        cleavage=_ef(2.50555, 4.96706, 5.42818, 0.730602, 1.26849, 0.60091, 0.0931368),
        emission=_ef(3.32089, 3.85417, 0.91416, 3.96287, 0.418531, 2.38294, 0.118463),
        peierls=ActivatedTransportParameters(0.428613, -4.95754, 1.85782, 1.22264, 1.0e12),
        taylor=ActivatedTransportParameters(9.41445, -17.9565, 0.830866, 2.18379, 1.0e11),
        taylor_correlation=TaylorCorrelationParameters(3.0928e12, 1.86172),
        state=StateParameters(0.565808, 6.94747e-3, 2.72688, 2.43878, 28.2448),
    ),
    "DBTT": MaterialParameterization(
        name="DBTT",
        candidate_id="DBTT_restart01_candidate05",
        cleavage=_ef(3.1188, 9.19781, 6.11734, 4.69187, 1.38442, 1.66645, 0.182817),
        emission=_ef(2.15785, -2.36337, 4.19637, 4.53757, 1.29452, 0.877242, 0.0985189),
        peierls=ActivatedTransportParameters(3.35812, -11.268, 1.17387, 1.1761, 1.0e12),
        taylor=ActivatedTransportParameters(10.4361, -16.7822, 1.31454, 2.08569, 1.0e11),
        taylor_correlation=TaylorCorrelationParameters(8.7400e14, 0.561537),
        state=StateParameters(0.50797, 2.3672e-8, 2.95208, 14.0087, 54.7736),
    ),
}


def get_material(name: str) -> MaterialParameterization:
    key = str(name)
    aliases = {"dbtt": "DBTT", "weakt": "weakT", "weak-t": "weakT", "fcc": "weakT"}
    key = aliases.get(key.lower(), key)
    try:
        return MATERIALS[key]
    except KeyError as exc:
        raise KeyError(f"unknown material class {name!r}; choose {sorted(MATERIALS)}") from exc
