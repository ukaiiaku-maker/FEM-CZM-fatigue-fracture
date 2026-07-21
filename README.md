# FEM-CZM fatigue-fracture

Utilities for temperature-dependent fracture and fatigue calculations and for
analyzing Arrhenius FEM/CZM campaign outputs.

## Design change

This repository removes the finite source inventory from the active reduced-model
physics. Dislocation sources are reusable. Emission becomes self-limiting through
retained-line shielding, slip-system source backstress, Peierls transport, Taylor
release, recovery, escape and crack advance.

The previous source count and source-refresh length remain provenance only and
must not control a calculation.

## Current scope

The repository contains reduced-model documentation and an installable campaign
analysis package. The adaptive 2-D FEM/CZM solver and its production run outputs
can reside in a separate checkout; the analysis command accepts that campaign
output directory directly.

## Install from GitHub

Install the current analysis branch directly:

```bash
conda create -n fem-czm-analysis python=3.12 -y
conda activate fem-czm-analysis

python -m pip install \
  "git+https://github.com/ukaiiaku-maker/FEM-CZM-fatigue-fracture.git@analysis-k-vs-t-v1"
```

For an editable development checkout:

```bash
git clone \
  --branch analysis-k-vs-t-v1 \
  --single-branch \
  https://github.com/ukaiiaku-maker/FEM-CZM-fatigue-fracture.git

cd FEM-CZM-fatigue-fracture
python -m pip install -e '.[dev]'
pytest -q
```

After the analysis branch is merged, replace `analysis-k-vs-t-v1` with `main`.

## Plot initial, mean, and late K versus temperature

```bash
OUTROOT=/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v10_0_5_13_5_long_corridor_dedup/runs/v10_0_5_13_5_tip_only_4class_300_1200K_100um_macro100_v1

fem-czm-plot-k-vs-t "$OUTROOT" \
  --late-fraction 0.25 \
  --min-late-events 3 \
  --formats png pdf \
  --strict
```

The command creates:

```text
$OUTROOT/analysis_K_vs_T/
├── K_initial_vs_T.png
├── K_initial_vs_T.pdf
├── K_mean_vs_T.png
├── K_mean_vs_T.pdf
├── K_late_vs_T.png
├── K_late_vs_T.pdf
├── K_vs_T_event_summary.csv
└── K_vs_T_event_summary.json
```

### Definitions

- **Initial K:** `K_J` at the first accepted crack advance.
- **Mean K:** arithmetic mean of `K_J` over accepted crack advances, with one
  value per advance event. Solver waiting steps do not receive extra weight.
- **Late K:** mean of the final fraction of accepted crack-advance events. The
  default `--late-fraction 0.25` uses the final five events in a 20-event,
  100 micrometre run.

The reader prioritizes R-curve/advance-event CSV files. If no suitable event CSV
is present, it parses `<< ADVANCE` records from each case's `run.log`; the
campaign-level `console.log` is a secondary fallback. The summary CSV records the
chosen source file for every case.

Use `--include-incomplete` to analyze right-censored cases. Without that flag,
known failed or incomplete cases are excluded. `--strict` makes missing or
unreadable cases fail the command after the diagnostic summary is written.

## Reduced-model examples

The original reduced monotonic and fatigue model documentation remains in
`docs/PHYSICS.md` and `docs/VALIDATION_PLAN.md`. Those calculations are distinct
from the full adaptive 2-D FEM/CZM campaign analyzed above.
