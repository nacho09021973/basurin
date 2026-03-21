#!/usr/bin/env bash
set -uo pipefail

# run_population_batch.sh
# -----------------------
# Launch the full population study (Phase 2): batch pipeline for all 55 BBH
# events from the GWTC catalog, followed by s5 aggregation.
#
# Usage:
#   ./tools/run_population_batch.sh                    # full run
#   ./tools/run_population_batch.sh --dry-run          # show what would run
#   ./tools/run_population_batch.sh --reuse-strain     # skip re-downloading strain
#
# Requirements:
#   - data/losc/<EVENT_ID>/ populated with HDF5 strain files
#   - docs/ringdown/atlas/atlas_berti_v2.json present
#   - Python deps installed (numpy, scipy, h5py, etc.)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# ── All 55 BBH events from gwtc_events_t0.json (excludes GW170817 BNS) ──
EVENTS="GW190408_181802,GW190412,GW190413_052954,GW190413_134308,\
GW190421_213856,GW190503_185404,GW190512_180714,GW190513_205428,\
GW190517_055101,GW190519_153544,GW190521,GW190521_074359,\
GW190602_175927,GW190620_030421,GW190630_185205,GW190701_203306,\
GW190706_222641,GW190707_093326,GW190708_232457,GW190720_000836,\
GW190725_174728,GW190727_060333,GW190728_064510,GW190803_022701,\
GW190805_211137,GW190814,GW190828_063405,GW190828_065509,\
GW190910_112807,GW190915_235702,GW190924_021846,GW190925_232845,\
GW190930_133541,GW191103_012549,GW191105_143521,GW191109_010717,\
GW191129_134029,GW191204_171526,GW191215_223052,GW191216_213338,\
GW191222_033537,GW191230_180458,GW200112_155838,GW200115_042309,\
GW200128_022011,GW200129_065458,GW200202_154313,GW200208_130117,\
GW200209_085452,GW200219_094415,GW200224_222234,GW200225_060421,\
GW200302_015811,GW200311_115853,GW200316_215756"

AGG_RUN_ID="population_phase2_$(date -u +%Y%m%dT%H%M%SZ)"

# ── Parse args ──
DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        *)           EXTRA_ARGS+=("$arg") ;;
    esac
done

echo "============================================================"
echo "POPULATION STUDY - PHASE 2"
echo "============================================================"
echo "  Events:    55 BBH (GWTC-1/2.1/3)"
echo "  Agg run:   ${AGG_RUN_ID}"
echo "  Mode:      batch (continue-on-failure)"
echo "  Estimator: dual"
echo "  Atlas:     docs/ringdown/atlas/atlas_berti_v2.json"
echo "  Catalog:   gwtc_events_t0.json"
echo "============================================================"

CMD=(
    python3 -m mvp.pipeline batch
    --events "$EVENTS"
    --atlas-default
    --agg-run-id "$AGG_RUN_ID"
    --min-coverage 0.5
    --estimator dual
    --offline
    --reuse-strain
    --catalog-path gwtc_events_t0.json
    "${EXTRA_ARGS[@]}"
)

if $DRY_RUN; then
    echo ""
    echo "[dry-run] Would execute:"
    echo "  ${CMD[*]}"
    echo ""
    echo "[dry-run] Output will be in: runs/${AGG_RUN_ID}/"
    exit 0
fi

echo ""
echo "Starting at $(date -u '+%Y-%m-%d %H:%M:%S UTC')..."
echo ""

"${CMD[@]}" 2>&1 | tee "runs/${AGG_RUN_ID}_log.txt"
RC=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ $RC -eq 0 ]; then
    echo "POPULATION BATCH COMPLETE"
else
    echo "POPULATION BATCH FINISHED WITH ERRORS (exit=$RC)"
fi
echo "  Agg run:  runs/${AGG_RUN_ID}/"
echo "  Log:      runs/${AGG_RUN_ID}_log.txt"
echo "  Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

exit $RC
