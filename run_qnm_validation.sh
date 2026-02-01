#!/usr/bin/env bash
# =============================================================================
# run_qnm_validation.sh — Full QNM pipeline validation
# =============================================================================
# Ejecuta:
#   1. Bloque B (genera spectrum.h5 con M² cerrado)
#   2. EXP_RINGDOWN_QNM_00 (horizon absorber gate)
#   3. EXP_RINGDOWN_QNM_01 (closed↔open limit validation)
#
# Uso:
#   ./run_qnm_validation.sh [RUN_ID]
#
# Requiere: numpy, scipy, h5py
# =============================================================================

set -euo pipefail

RUN_ID="${1:-$(date +%Y-%m-%d)__qnm_validation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "QNM Pipeline Validation"
echo "Run ID: $RUN_ID"
echo "=============================================="

cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Executive gate: RUN_VALID must exist and PASS
# -----------------------------------------------------------------------------
RUN_VALID_PATH="runs/$RUN_ID/RUN_VALID/verdict.json"
if [[ ! -f "$RUN_VALID_PATH" ]]; then
    echo "ERROR: RUN_VALID missing: $RUN_VALID_PATH"
    exit 2
fi

VERDICT_RUN_VALID=$(jq -r '.verdict // .results.overall_verdict // empty' "$RUN_VALID_PATH")
if [[ "$VERDICT_RUN_VALID" != "PASS" ]]; then
    echo "ERROR: RUN_VALID != PASS (got: ${VERDICT_RUN_VALID:-empty})"
    exit 2
fi

# -----------------------------------------------------------------------------
# Step 1: Generate Bloque B spectrum (closed BC, Hermitian)
# -----------------------------------------------------------------------------
echo ""
echo "[1/3] Generating Bloque B spectrum (closed boundary)..."
echo "      → runs/$RUN_ID/spectrum/outputs/spectrum.h5"

python 03_sturm_liouville.py \
    --run "$RUN_ID" \
    --mode sweep_delta \
    --delta-min 1.6 \
    --delta-max 3.0 \
    --n-delta 5 \
    --n-modes 5 \
    --n-z 512

echo "      ✓ Bloque B complete"

# -----------------------------------------------------------------------------
# Step 2: EXP_RINGDOWN_QNM_00 — Horizon absorber gate
# -----------------------------------------------------------------------------
echo ""
echo "[2/3] Running EXP_RINGDOWN_QNM_00 (horizon absorber)..."
echo "      Contracts: C1 (decay), C2 (stability)"

python experiment/ringdown/exp_ringdown_qnm_00_open_bc.py \
    --run "$RUN_ID" \
    --f-hz 250.0 \
    --tau-s 0.004 \
    --seed 42

VERDICT_00=$(python -c "import json; print(json.load(open('runs/$RUN_ID/experiment/ringdown/EXP_RINGDOWN_QNM_00_open_bc/outputs/contract_verdict.json'))['verdict'])")
echo "      → Verdict: $VERDICT_00"

# -----------------------------------------------------------------------------
# Step 3: EXP_RINGDOWN_QNM_01 — Closed↔Open limit validation
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Running EXP_RINGDOWN_QNM_01 (closed↔open limit)..."
echo "      Contracts: C3 (ω_R²→M²), C4 (monotonicity)"

python experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py \
    --run "$RUN_ID" \
    --gamma-sweep "0.0,0.01,0.1,1.0,10.0,100.0,250.0" \
    --mode-indices "0,1,2" \
    --seed 42

VERDICT_01=$(python -c "import json; print(json.load(open('runs/$RUN_ID/experiment/ringdown/EXP_RINGDOWN_QNM_01_closed_open_limit/outputs/contract_verdict.json'))['verdict'])")
echo "      → Verdict: $VERDICT_01"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Run ID:           $RUN_ID"
echo "Bloque B:         runs/$RUN_ID/spectrum/outputs/spectrum.h5"
echo "EXP_QNM_00:       $VERDICT_00"
echo "EXP_QNM_01:       $VERDICT_01"
echo ""

if [[ "$VERDICT_00" == "PASS" && "$VERDICT_01" == "PASS" ]]; then
    echo "✓ ALL CONTRACTS PASS"
    echo ""
    echo "Pipeline internally consistent."
    echo "Ready to proceed to GW150914 ringdown analysis."
    exit 0
else
    echo "✗ SOME CONTRACTS FAILED"
    echo ""
    echo "Check:"
    echo "  runs/$RUN_ID/experiment/ringdown/EXP_RINGDOWN_QNM_00_open_bc/outputs/contract_verdict.json"
    echo "  runs/$RUN_ID/experiment/ringdown/EXP_RINGDOWN_QNM_01_closed_open_limit/outputs/contract_verdict.json"
    exit 1
fi
