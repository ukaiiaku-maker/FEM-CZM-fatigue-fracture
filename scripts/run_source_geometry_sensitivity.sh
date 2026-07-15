#!/usr/bin/env bash
set -euo pipefail

MATERIAL=${MATERIAL:-DBTT}
T_K=${T_K:-700}
KMAX=${KMAX:-80}
DK=${DK:-0.25}
KDOT=${KDOT:-0.005}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}
OUTROOT=${OUTROOT:-runs/source_geometry_sensitivity_v1}

SPACINGS_UM=${SPACINGS_UM:-"0.5 1.0 2.0"}
RELOAD_TIMES_S=${RELOAD_TIMES_S:-"1e-4 1e-3 1e-2"}

mkdir -p "$OUTROOT"

for spacing in $SPACINGS_UM; do
  for reload in $RELOAD_TIMES_S; do
    tag="spacing_${spacing}um_reload_${reload}s"
    echo "=== $MATERIAL T=${T_K}K $tag ==="
    fem-czm monotonic \
      --material "$MATERIAL" \
      --temperature "$T_K" \
      --Kmax "$KMAX" \
      --dK "$DK" \
      --Kdot "$KDOT" \
      --target-extension-um "$TARGET_EXT_UM" \
      --source-spacing-um "$spacing" \
      --source-reload-time-s "$reload" \
      --out "$OUTROOT/${MATERIAL}_${T_K}K_${tag}.csv"
  done
done
