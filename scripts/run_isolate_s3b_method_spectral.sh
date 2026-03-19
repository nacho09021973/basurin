#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Isolation experiment: s3b-method = spectral_two_pass
#
# Purpose : Determine whether the Q_221 identifiability bottleneck
#           comes from the extractor topology (hilbert_peakband) or
#           is intrinsic to the signal.
#
# Knobs held constant (baseline):
#   - band-strategy       : default_split_60_40
#   - residual-strategy   : refit_220_each_iter
#   - t0-shift-ms         : 0
#
# Knob under test:
#   - s3b-method          : spectral_two_pass  (vs baseline hilbert_peakband)
#
# Note: uses --synthetic because GWOSC does not yet serve O4b strain
#       for GW250114_082203.  Replace with --reuse-strain (no --synthetic)
#       once real data is available locally.
#
# Expected artefacts on success:
#   runs_tmp/<run-id>/s3b_multimode_estimates/stage_summary.json
#   runs_tmp/<run-id>/s3b_multimode_estimates/outputs/multimode_estimates.json
# -----------------------------------------------------------------------
set -euo pipefail

RUN_ID="mvp_GW250114_082203_default_refit220_t00_spectral_20260319A"

python -m mvp.pipeline multimode \
  --event-id GW250114_082203 \
  --run-id "${RUN_ID}" \
  --atlas-default \
  --synthetic \
  --s3b-method spectral_two_pass \
  --bootstrap-221-residual-strategy refit_220_each_iter \
  --band-strategy default_split_60_40 \
  --t0-shift-ms 0

echo ""
echo "=== Post-run validation ==="
RUN_DIR="runs_tmp/${RUN_ID}/s3b_multimode_estimates"

if [[ -f "${RUN_DIR}/outputs/multimode_estimates.json" ]]; then
    echo "PASS: multimode_estimates.json exists"
else
    echo "FAIL: multimode_estimates.json NOT found"
    exit 1
fi

if [[ -f "${RUN_DIR}/stage_summary.json" ]]; then
    echo "PASS: stage_summary.json exists"
else
    echo "FAIL: stage_summary.json NOT found"
    exit 1
fi

echo ""
echo "All artefacts present. Run ${RUN_ID} completed successfully."
