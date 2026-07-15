# FEM-CZM fatigue-fracture

A clean implementation of temperature-dependent fracture and fatigue using the
three selected ceramic, weak-temperature/FCC-like, and DBTT parameterizations.

## Design change

This repository removes the finite source inventory from the active physics.
Dislocation sources are reusable. Emission becomes self-limiting through
retained-line shielding, slip-system source backstress, Peierls transport,
Taylor release, recovery, escape and crack advance.

The previous source count and source-refresh length remain only as provenance in
`parameters.py`; neither can affect a calculation.

## Current scope

Version 0.1 contains the shared constitutive core, reduced moving process zone,
monotonic crack-front driver, fatigue cycle-block driver, 2-D coupling contract,
three parameter sets, tests and CI. The adaptive 2-D FEM/CZM backend is the next
tracked implementation stage; this branch does not claim full 2-D validation.

## Installation

```bash
conda create -n fem-czm-fatigue-fracture python=3.12 -y
conda activate fem-czm-fatigue-fracture
python -m pip install -e '.[dev]'
pytest
```

## Reduced monotonic example

```bash
fem-czm monotonic \
  --material DBTT --temperature 700 \
  --Kmax 80 --dK 0.25 --Kdot 0.005 \
  --target-extension-um 100 \
  --out runs/dbtt_700K_monotonic.csv
```

## Fatigue example

```bash
fem-czm fatigue \
  --material DBTT --temperature 700 \
  --Kmax 20 --R 0.1 --frequency 1000 \
  --block-cycles 1000 --cycles-max 1000000 \
  --target-extension-um 100 \
  --out runs/dbtt_700K_fatigue.csv
```

See `docs/PHYSICS.md` and `docs/VALIDATION_PLAN.md` before interpreting output.
