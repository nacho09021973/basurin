#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

RUN_ID="${1:-2026-01-20__exp03}"

PY="./.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] No encuentro $PY. Crea/activa la venv en la raíz del repo."
  exit 1
fi

mkdir -p "runs/${RUN_ID}/logs"
LOG="runs/${RUN_ID}/logs/exp03.log"

"$PY" experiment/exp_03_c3_metric_sensitivity/05_exp03_c3_metric_sensitivity.py --run "$RUN_ID" 2>&1 | tee "$LOG"
