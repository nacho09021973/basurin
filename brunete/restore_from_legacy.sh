#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CANDIDATES_FILE="${REPO_ROOT}/brunete/move_to_legaxy_candidates.txt"
KEEP_FILE="${REPO_ROOT}/brunete/keep_set.txt"
EXPECTED_MOVE_COUNT=364

if [[ ! -f "${CANDIDATES_FILE}" ]]; then
  echo "ERROR: missing candidates file: ${CANDIDATES_FILE}" >&2
  exit 1
fi

if [[ ! -f "${KEEP_FILE}" ]]; then
  echo "ERROR: missing keep-set file: ${KEEP_FILE}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

declare -A KEEP_SET=()
declare -A AMBIGUOUS_SET=()

while IFS= read -r path || [[ -n "${path}" ]]; do
  [[ -z "${path}" ]] && continue
  KEEP_SET["${path}"]=1
done < "${KEEP_FILE}"

AMBIGUOUS_PATHS=(
  "docs/BRUNETE_INCONSISTENCIES.md"
  "docs/BRUNETE_INVENTORY.md"
  "docs/BRUNETE_S6C.md"
  "docs/BRUNETE_TEXT_FIXES.md"
  "docs/estudio_brunete.md"
  "docs/estudio_brunete_completo.md"
  "docs/metodo_brunete.md"
  "mvp/brunete/__init__.py"
  "mvp/brunete/core.py"
  "mvp/s6c_brunete_psd_curvature.py"
  "tests/test_brunete_core.py"
  "tests/test_s6c_brunete_psd_curvature_integration.py"
)

for path in "${AMBIGUOUS_PATHS[@]}"; do
  AMBIGUOUS_SET["${path}"]=1
done

prechecked_count=0
while IFS= read -r dst || [[ -n "${dst}" ]]; do
  [[ -z "${dst}" ]] && continue

  if [[ -n "${KEEP_SET["${dst}"]+x}" ]]; then
    echo "ERROR: refusing to restore keep-set path from legacy list: ${dst}" >&2
    exit 1
  fi

  if [[ -n "${AMBIGUOUS_SET["${dst}"]+x}" ]]; then
    echo "ERROR: refusing to restore ambiguous excluded path from legacy list: ${dst}" >&2
    exit 1
  fi

  src="legacy/${dst}"
  git ls-files --error-unmatch -- "${src}" >/dev/null 2>&1

  if [[ -e "${dst}" ]]; then
    echo "ERROR: restore destination already exists: ${dst}" >&2
    exit 1
  fi

  prechecked_count=$((prechecked_count + 1))
done < "${CANDIDATES_FILE}"

if [[ "${prechecked_count}" -ne "${EXPECTED_MOVE_COUNT}" ]]; then
  echo "ERROR: restore precheck count mismatch: expected ${EXPECTED_MOVE_COUNT}, got ${prechecked_count}" >&2
  exit 1
fi

restored_count=0

while IFS= read -r dst || [[ -n "${dst}" ]]; do
  [[ -z "${dst}" ]] && continue

  if [[ -n "${KEEP_SET["${dst}"]+x}" ]]; then
    echo "ERROR: refusing to restore keep-set path from legacy list: ${dst}" >&2
    exit 1
  fi

  if [[ -n "${AMBIGUOUS_SET["${dst}"]+x}" ]]; then
    echo "ERROR: refusing to restore ambiguous excluded path from legacy list: ${dst}" >&2
    exit 1
  fi

  src="legacy/${dst}"
  git ls-files --error-unmatch -- "${src}" >/dev/null 2>&1

  mkdir -p "$(dirname "${dst}")"
  git mv "${src}" "${dst}"
  restored_count=$((restored_count + 1))
done < "${CANDIDATES_FILE}"

if [[ "${restored_count}" -ne "${EXPECTED_MOVE_COUNT}" ]]; then
  echo "ERROR: restored count mismatch: expected ${EXPECTED_MOVE_COUNT}, got ${restored_count}" >&2
  exit 1
fi

echo "Expected files restored: ${restored_count}"
echo "Post-restore validation:"
echo "  git rev-parse --show-toplevel"
echo "  python - <<'PY'"
echo "import brunete.brunete_prepare_events"
echo "import brunete.brunete_list_events"
echo "import brunete.brunete_audit_cohort_authority"
echo "import brunete.brunete_run_batch"
echo "import brunete.brunete_classify_geometries"
echo "import brunete.brunete_bounded_analysis"
echo "import brunete.experiment.b5a"
echo "import brunete.experiment.b5b"
echo "import brunete.experiment.b5c"
echo "import brunete.experiment.b5e"
echo "import brunete.experiment.b5f"
echo "import brunete.experiment.b5h"
echo "import brunete.experiment.b5z"
echo "PY"
echo "  pytest tests/brunete tests/test_brunete_experiment.py -q"
echo "  BASURIN_RUNS_ROOT=\"\$(mktemp -d)\" pytest tests/brunete/test_prepare_events.py tests/brunete/test_run_batch.py tests/brunete/test_classify_geometries.py tests/test_brunete_experiment.py -q"
echo "  python -m brunete.brunete_prepare_events --help"
echo "  python -m brunete.brunete_run_batch --help"
echo "  python -m brunete.brunete_classify_geometries --help"
echo "  python -m brunete.brunete_bounded_analysis --help"
