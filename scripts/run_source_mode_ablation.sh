#!/usr/bin/env bash
set -euo pipefail

OUTROOT=${OUTROOT:-runs/dbtt_700K_source_mode_ablation_v1}
MATERIAL=${MATERIAL:-DBTT}
T_K=${T_K:-700}
KMAX=${KMAX:-80}
DK=${DK:-0.25}
KDOT=${KDOT:-0.005}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}

mkdir -p "$OUTROOT"

for mode in reusable_emergent legacy_finite_site; do
  out="$OUTROOT/${MATERIAL}_${T_K}K_${mode}.csv"
  echo "=== ${MATERIAL} T=${T_K}K source_mode=${mode} ==="
  fem-czm monotonic \
    --material "$MATERIAL" \
    --temperature "$T_K" \
    --Kmax "$KMAX" \
    --dK "$DK" \
    --Kdot "$KDOT" \
    --target-extension-um "$TARGET_EXT_UM" \
    --source-mode "$mode" \
    --source-spacing-um 1.0 \
    --source-active-angle-deg 180 \
    --source-reload-time-s 1e-3 \
    --out "$out"
done

python - "$OUTROOT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.csv")):
    df = pd.read_csv(path)
    last = df.iloc[-1]
    fired = df.loc[df.get("n_fire", 0) > 0]
    rows.append({
        "file": path.name,
        "source_mode": last.get("source_mode", "unknown"),
        "K_init_MPa_sqrt_m": float(fired.iloc[0]["K_MPa_sqrt_m"]) if len(fired) else float("nan"),
        "K_final_MPa_sqrt_m": float(last.get("K_MPa_sqrt_m", float("nan"))),
        "crack_extension_um": 1e6 * float(last.get("crack_extension_m", 0.0)),
        "right_censored": float(last.get("right_censored_at_Kmax", 0.0)),
        "r_eff_um": 1e6 * float(last.get("r_eff_m", float("nan"))),
        "mobile_count": float(last.get("mobile_count", float("nan"))),
        "retained_count": float(last.get("retained_count", float("nan"))),
        "K_shield_MPa_sqrt_m": 1e-6 * float(last.get("K_shield_Pa_sqrt_m", float("nan"))),
        "available_site_fraction": float(last.get("legacy_available_site_fraction", float("nan"))),
        "emitted_total": float(df.get("dN_emit", pd.Series(dtype=float)).fillna(0.0).sum()),
        "termination_reason": last.get("termination_reason", ""),
    })
summary = pd.DataFrame(rows)
summary.to_csv(root / "source_mode_summary.csv", index=False)
print(summary.to_string(index=False))
PY
