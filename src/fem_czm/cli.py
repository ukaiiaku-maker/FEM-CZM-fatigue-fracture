"""Reduced-front command line driver used for constitutive regression tests."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .fatigue import FatigueConfig, FatigueIntegrator
from .finite_site_process_zone import (
    LegacyFiniteSiteConfig,
    LegacyFiniteSiteProcessZone,
)
from .front import CrackFront, FrontConfig
from .parameters import get_material
from .process_zone import ProcessZoneConfig


def _write(rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=sorted({k for r in rows for k in r}),
            )
            writer.writeheader()
            writer.writerows(rows)


def _common_process_zone_kwargs(args) -> dict:
    return {
        "source_spacing_m": args.source_spacing_um * 1.0e-6,
        "source_active_angle_rad": math.radians(args.source_active_angle_deg),
        "source_reload_time_s": args.source_reload_time_s,
        "mobile_source_backstress_fraction": args.mobile_source_backstress_fraction,
        "backstress_geometry_factor": args.backstress_geometry_factor,
        "shielding_geometry_factor": args.shielding_geometry_factor,
    }


def _build_front(args) -> CrackFront:
    material = get_material(args.material)
    front_config = FrontConfig(advance_increment_m=args.da_um * 1.0e-6)

    if args.source_mode == "legacy_finite_site":
        sites = (
            material.state.legacy_source_sites_per_system
            if args.legacy_source_sites_per_system is None
            else args.legacy_source_sites_per_system
        )
        refresh_um = (
            material.state.legacy_source_refresh_length_um
            if args.legacy_source_refresh_length_um is None
            else args.legacy_source_refresh_length_um
        )
        config = LegacyFiniteSiteConfig(
            **_common_process_zone_kwargs(args),
            source_sites_per_system=sites,
            source_recovery_rate_s=args.legacy_source_recovery_rate_s,
            source_refresh_length_m=refresh_um * 1.0e-6,
            source_bin_count=args.legacy_source_bin_count,
        )
        front = CrackFront(material, front_config, ProcessZoneConfig())
        front.process_zone = LegacyFiniteSiteProcessZone(material, config)
        return front

    return CrackFront(
        material,
        front_config,
        ProcessZoneConfig(**_common_process_zone_kwargs(args)),
    )


def monotonic(args):
    front = _build_front(args)
    rows = []
    target = args.target_extension_um * 1.0e-6

    # Record the unloaded initial state without imposing an artificial dwell at K=0.
    initial = front.step(0.0, args.temperature, 0.0)
    initial["K_MPa_sqrt_m"] = 0.0
    initial["source_mode"] = args.source_mode
    rows.append(initial)

    K = args.dK
    dt_interval = args.dK / args.Kdot
    last_K = 0.0
    while K <= args.Kmax + 1.0e-12 and front.crack_extension_m < target:
        remaining = dt_interval
        while (
            remaining > max(1.0e-15 * dt_interval, 1.0e-30)
            and front.crack_extension_m < target
        ):
            row = front.step(K * 1.0e6, args.temperature, remaining)
            row["K_MPa_sqrt_m"] = K
            row["source_mode"] = args.source_mode
            row["load_interval_remaining_s"] = float(
                row.get("unused_dt_s", 0.0)
            )
            rows.append(row)
            new_remaining = float(row.get("unused_dt_s", 0.0))
            if int(row.get("n_fire", 0)) == 0:
                remaining = 0.0
            elif new_remaining >= remaining * (1.0 - 1.0e-14):
                raise RuntimeError("event-limited front made no time progress")
            else:
                remaining = new_remaining
        last_K = K
        K += args.dK

    completed = front.crack_extension_m >= target
    if rows:
        rows[-1]["completed_target_extension"] = float(completed)
        rows[-1]["right_censored_at_Kmax"] = float(
            (not completed) and last_K >= args.Kmax - 1.0e-12
        )
        rows[-1]["termination_reason"] = (
            "target_extension" if completed else "Kmax_right_censored"
        )
    _write(rows, Path(args.out))
    return rows


def fatigue(args):
    front = _build_front(args)
    integ = FatigueIntegrator(
        front,
        FatigueConfig(
            load_ratio_R=args.R,
            frequency_Hz=args.frequency,
            phase_points=args.phase_points,
            max_cycles_per_chunk=args.explicit_cycle_chunk,
        ),
    )
    rows = []
    target = args.target_extension_um * 1.0e-6
    while integ.cycles < args.cycles_max and front.crack_extension_m < target:
        row = integ.advance_cycles(
            args.Kmax * 1.0e6,
            args.temperature,
            min(args.block_cycles, args.cycles_max - integ.cycles),
        )
        row["source_mode"] = args.source_mode
        rows.append(row)
    completed = front.crack_extension_m >= target
    if rows:
        rows[-1]["completed_target_extension"] = float(completed)
        rows[-1]["right_censored_at_cycles_max"] = float(not completed)
        rows[-1]["termination_reason"] = (
            "target_extension" if completed else "cycles_max_right_censored"
        )
    _write(rows, Path(args.out))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--material",
        choices=["ceramic", "weakT", "DBTT"],
        required=True,
    )
    common.add_argument("--temperature", type=float, required=True)
    common.add_argument("--da-um", type=float, default=5.0)
    common.add_argument("--target-extension-um", type=float, default=100.0)
    common.add_argument(
        "--source-mode",
        choices=["reusable_emergent", "legacy_finite_site"],
        default="reusable_emergent",
    )
    common.add_argument("--source-spacing-um", type=float, default=1.0)
    common.add_argument("--source-active-angle-deg", type=float, default=180.0)
    common.add_argument("--source-reload-time-s", type=float, default=1.0e-3)
    common.add_argument(
        "--mobile-source-backstress-fraction",
        type=float,
        default=1.0,
    )
    common.add_argument("--backstress-geometry-factor", type=float, default=1.0)
    common.add_argument("--shielding-geometry-factor", type=float, default=1.0)
    common.add_argument(
        "--legacy-source-sites-per-system",
        type=float,
        default=None,
        help="defaults to the selected candidate's retained provenance value",
    )
    common.add_argument(
        "--legacy-source-refresh-length-um",
        type=float,
        default=None,
        help="defaults to the selected candidate's retained provenance value",
    )
    common.add_argument(
        "--legacy-source-recovery-rate-s",
        type=float,
        default=0.0,
    )
    common.add_argument("--legacy-source-bin-count", type=int, default=2)
    common.add_argument("--out", required=True)

    m = sub.add_parser("monotonic", parents=[common])
    m.add_argument("--Kmax", type=float, default=80.0)
    m.add_argument("--dK", type=float, default=0.25)
    m.add_argument("--Kdot", type=float, default=0.005)
    m.set_defaults(func=monotonic)

    f = sub.add_parser("fatigue", parents=[common])
    f.add_argument("--Kmax", type=float, required=True)
    f.add_argument("--R", type=float, default=0.1)
    f.add_argument("--frequency", type=float, default=1000.0)
    f.add_argument("--phase-points", type=int, default=32)
    f.add_argument("--block-cycles", type=float, default=1000.0)
    f.add_argument(
        "--explicit-cycle-chunk",
        type=float,
        default=1.0,
        help="maximum number of cycles integrated per ordered phase sequence",
    )
    f.add_argument("--cycles-max", type=float, default=1.0e6)
    f.set_defaults(func=fatigue)

    args = p.parse_args(argv)
    if getattr(args, "Kdot", 1.0) <= 0.0:
        raise SystemExit("Kdot must be positive")
    if args.source_spacing_um <= 0.0:
        raise SystemExit("source spacing must be positive")
    if (
        args.legacy_source_sites_per_system is not None
        and args.legacy_source_sites_per_system < 0.0
    ):
        raise SystemExit("legacy source sites cannot be negative")
    rows = args.func(args)
    print(json.dumps(rows[-1] if rows else {}, indent=2, default=float))


if __name__ == "__main__":
    main()
