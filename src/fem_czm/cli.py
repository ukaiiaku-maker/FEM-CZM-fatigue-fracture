"""Reduced-front command line driver used for constitutive regression tests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .fatigue import FatigueConfig, FatigueIntegrator
from .front import CrackFront, FrontConfig
from .parameters import get_material


def _write(rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)


def monotonic(args):
    front = CrackFront(get_material(args.material), FrontConfig(advance_increment_m=args.da_um * 1e-6))
    rows = []
    K = 0.0
    while K <= args.Kmax and front.crack_extension_m < args.target_extension_um * 1e-6:
        rows.append(front.step(K * 1e6, args.temperature, args.dK / args.Kdot))
        rows[-1]["K_MPa_sqrt_m"] = K
        K += args.dK
    _write(rows, Path(args.out))
    return rows


def fatigue(args):
    front = CrackFront(get_material(args.material), FrontConfig(advance_increment_m=args.da_um * 1e-6))
    integ = FatigueIntegrator(front, FatigueConfig(args.R, args.frequency, args.phase_points, args.block_cycles))
    rows = []
    while integ.cycles < args.cycles_max and front.crack_extension_m < args.target_extension_um * 1e-6:
        rows.append(integ.advance_cycles(args.Kmax * 1e6, args.temperature, min(args.block_cycles, args.cycles_max - integ.cycles)))
    _write(rows, Path(args.out))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--material", choices=["ceramic", "weakT", "DBTT"], required=True)
    common.add_argument("--temperature", type=float, required=True)
    common.add_argument("--da-um", type=float, default=5.0)
    common.add_argument("--target-extension-um", type=float, default=100.0)
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
    f.add_argument("--cycles-max", type=float, default=1.0e6)
    f.set_defaults(func=fatigue)
    args = p.parse_args(argv)
    rows = args.func(args)
    print(json.dumps(rows[-1] if rows else {}, indent=2, default=float))


if __name__ == "__main__":
    main()
