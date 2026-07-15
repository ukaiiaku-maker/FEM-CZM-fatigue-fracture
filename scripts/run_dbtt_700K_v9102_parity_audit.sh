#!/usr/bin/env bash
set -euo pipefail

OUTROOT=${OUTROOT:-runs/dbtt_700K_v9102_parity_audit_v1}
mkdir -p "$OUTROOT"

python -m fem_czm.v9102_parity \
  --material DBTT \
  --temperature 700 \
  --target-extension-um 1000 \
  --dK 0.25 \
  --Kdot 0.005 \
  --Kmax 80 \
  --da-um 5 \
  --out-events "$OUTROOT/DBTT_700K_v9102_events.csv" \
  --out-summary "$OUTROOT/DBTT_700K_v9102_summary.json"

python - "$OUTROOT/DBTT_700K_v9102_summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
m = data["metrics"]
expected = {
    "K_init_MPa_sqrt_m": 28.123898,
    "K_plateau_MPa_sqrt_m": 30.963772,
    "max_tip_radius_ratio": 1.009401,
    "final_emitted_total": 539.531349,
}
tolerance = {
    "K_init_MPa_sqrt_m": 2.0e-3,
    "K_plateau_MPa_sqrt_m": 2.0e-3,
    "max_tip_radius_ratio": 2.0e-5,
    "final_emitted_total": 2.0e-2,
}
passed = bool(m["completed"] and not m["right_censored_at_Kmax"])
for key, reference in expected.items():
    value = float(m[key])
    error = value - reference
    print(f"{key}: value={value:.9g} reference={reference:.9g} error={error:+.3g}")
    passed = passed and math.isfinite(value) and abs(error) <= tolerance[key]
print(f"PARITY_GATE={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if passed else 1)
PY
