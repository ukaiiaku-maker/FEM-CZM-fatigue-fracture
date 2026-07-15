#!/usr/bin/env bash
set -euo pipefail
for cls in ceramic weakT DBTT; do
  for T in 300 700 900 1200; do
    fem-czm monotonic \
      --material "$cls" --temperature "$T" \
      --Kmax 80 --dK 1 --Kdot 0.005 \
      --target-extension-um 10 \
      --out "runs/smoke/${cls}_${T}K.csv"
  done
done
