#!/usr/bin/env bash
set -uo pipefail

# fetch_losc_batch.sh
# -------------------
# Descarga en batch archivos LOSC/GWOSC HDF5 por EVENT_ID desde un archivo.
#
# Uso:
#   ./tools/fetch_losc_batch.sh <events_file>
#
# Donde <events_file> contiene un EVENT_ID por línea. Admite líneas vacías y
# comentarios con '#'.
#
# Salida:
#   data/losc/<EVENT_ID>/*.hdf5
#   data/losc/<EVENT_ID>/SHA256SUMS.txt   (solo si hay .hdf5)
#
# Propiedades:
# - Resolución determinista de shortName sin usar event-versions?name-contains:
#   prueba <EVENT>-v2 y luego <EVENT>-v1 (HTTP 200).
# - Idempotente: nunca sobreescribe archivos existentes.
# - Tolerante a respuestas no-JSON o vacías en strain-files.
# - Continúa el batch ante eventos/detectores sin archivo.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOSC_ROOT="${LOSC_ROOT:-${REPO_ROOT}/data/losc}"
CURL_BIN="${CURL_BIN:-curl}"
DETECTORS=(H1 L1 V1)

usage() {
  cat <<'USAGE'
Uso:
  ./tools/fetch_losc_batch.sh <events_file>

Argumentos:
  events_file   Archivo con EVENT_ID por línea (permite comentarios con #).

Variables opcionales:
  LOSC_ROOT     Root destino de descargas (default: <repo>/data/losc)
  CURL_BIN      Comando curl a usar (default: curl)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -ne 1 ]]; then
  usage
  [[ "$#" -eq 1 ]] || exit 0
  exit 1
fi

EVENTS_FILE="$1"
if [[ ! -f "$EVENTS_FILE" ]]; then
  echo "[fetch_losc_batch] ERROR: no existe events_file: $EVENTS_FILE" >&2
  exit 1
fi

mkdir -p "$LOSC_ROOT"

http_code() {
  local url="$1"
  "${CURL_BIN}" -sS -L --retry 5 --retry-delay 2 --max-time 60 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true
}

fetch_json_with_retries() {
  local url="$1"
  local max_attempts="${2:-3}"
  local attempt=1
  local body=""
  local trimmed=""

  while (( attempt <= max_attempts )); do
    body="$("${CURL_BIN}" -sS -L --retry 5 --retry-delay 2 "$url" 2>/dev/null || true)"
    trimmed="${body#"${body%%[![:space:]]*}"}"
    if [[ "${trimmed:0:1}" == "{" ]]; then
      printf '%s' "$body"
      return 0
    fi
    ((attempt++))
    sleep 1
  done

  printf '%s' "$body"
  return 1
}

json_results_count() {
  python - <<'PY'
import json
import os

raw = os.environ.get("GWOSC_STRAIN_JSON", "")
try:
    payload = json.loads(raw)
except Exception:
    print(0)
    raise SystemExit(0)

count = payload.get("results_count")
if isinstance(count, int):
    print(count)
    raise SystemExit(0)

results = payload.get("results")
if isinstance(results, list):
    print(len(results))
    raise SystemExit(0)

strain_files = payload.get("strain_files")
if isinstance(strain_files, list):
    print(len(strain_files))
    raise SystemExit(0)

print(0)
PY
}

has_strain() {
  local shortname="$1"
  local url="https://gwosc.org/api/v2/event-versions/${shortname}/strain-files?detector=H1"
  local body=""

  if ! body="$(fetch_json_with_retries "$url" 3)"; then
    echo "[fetch_losc_batch] WARN ${shortname}: strain-files devolvió NOT_JSON tras reintentos" >&2
    printf '0\n'
    return 0
  fi

  GWOSC_STRAIN_JSON="$body"
  export GWOSC_STRAIN_JSON
  json_results_count
}

resolve_short_name() {
  local event_id="$1"
  local candidate
  for candidate in "${event_id}-v2" "${event_id}-v1"; do
    local url="https://gwosc.org/api/v2/event-versions/${candidate}"
    if [[ "$(http_code "$url")" == "200" ]]; then
      local strain_count
      strain_count="$(has_strain "$candidate")"
      if [[ "$strain_count" =~ ^[0-9]+$ ]] && (( strain_count > 0 )); then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

best_download_url_for_detector() {
  local json_payload="$1"
  python - <<'PY'
import json
import os
import sys

raw = os.environ.get("GWOSC_STRAIN_JSON", "")

if not raw.strip():
    print("")
    raise SystemExit(0)

try:
    payload = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)

candidates_hdf = []
candidates_gwf = []


def visit(node):
    if isinstance(node, dict):
        url = node.get("download_url")
        if isinstance(url, str) and url.strip():
            file_format = node.get("file_format")
            file_format_norm = str(file_format).strip().upper() if file_format is not None else ""
            dur = node.get("duration")
            sr = node.get("sample_rate")

            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            dur_f = to_float(dur)
            sr_f = to_float(sr)

            score = 0
            if dur_f is not None and abs(dur_f - 32.0) < 1e-9:
                score += 2
            if sr_f is not None and abs(sr_f - 4096.0) < 1e-9:
                score += 1

            is_hdf = (
                file_format_norm in {"HDF", "HDF5", "HDF5 (HDF)"}
                or url.lower().endswith(".hdf5")
                or url.lower().endswith(".h5")
            )
            is_gwf = (
                "GWF" in file_format_norm
                or url.lower().endswith(".gwf")
            )

            if is_hdf:
                candidates_hdf.append((score, url))
            elif is_gwf:
                candidates_gwf.append((score, url))

        for v in node.values():
            visit(v)
    elif isinstance(node, list):
        for item in node:
            visit(item)


visit(payload)

if candidates_hdf:
    best_idx = max(range(len(candidates_hdf)), key=lambda i: candidates_hdf[i][0])
    print(f"{candidates_hdf[best_idx][1]}\thdf")
    raise SystemExit(0)

if candidates_gwf:
    best_idx = max(range(len(candidates_gwf)), key=lambda i: candidates_gwf[i][0])
    print(f"{candidates_gwf[best_idx][1]}\tgwf")
    raise SystemExit(0)

print("")
PY
}

write_sha256sums_if_any() {
  local event_dir="$1"
  if ! find "$event_dir" -maxdepth 1 -type f -iname '*.hdf5' | read -r _; then
    return 0
  fi

  (
    cd "$event_dir" || exit 1
    find . -maxdepth 1 -type f -iname '*.hdf5' -printf '%f\n' | LC_ALL=C sort | while IFS= read -r fname; do
      sha256sum "$fname"
    done > SHA256SUMS.txt
  )
}

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -n "$line" ]] || continue

  event_id="$line"
  event_dir="${LOSC_ROOT}/${event_id}"
  mkdir -p "$event_dir"

  if ! short_name="$(resolve_short_name "$event_id")"; then
    echo "[fetch_losc_batch] WARN ${event_id}: no existe ${event_id}-v2 ni ${event_id}-v1" >&2
    continue
  fi

  echo "[fetch_losc_batch] ${event_id}: shortName=${short_name}"

  for det in "${DETECTORS[@]}"; do
    url="https://gwosc.org/api/v2/event-versions/${short_name}/strain-files?detector=${det}"
    if ! body="$(fetch_json_with_retries "$url" 3)"; then
      echo "[fetch_losc_batch] WARN ${event_id} ${det}: strain-files devolvió NOT_JSON tras reintentos" >&2
      continue
    fi
    GWOSC_STRAIN_JSON="$body"
    export GWOSC_STRAIN_JSON
    download_selection="$(best_download_url_for_detector "$body")"
    IFS=$'\t' read -r download_url selected_kind <<< "$download_selection"

    if [[ -z "$download_url" ]]; then
      echo "[fetch_losc_batch] ${event_id} ${det}: (no file)"
      continue
    fi

    if [[ "$selected_kind" == "gwf" ]]; then
      echo "[fetch_losc_batch] WARN ${event_id} ${det}: sin HDF/HDF5, usando GWF" >&2
    fi

    fname="$(basename "$download_url")"
    if [[ -z "$fname" || "$fname" == "/" || "$fname" == "." ]]; then
      echo "[fetch_losc_batch] ${event_id} ${det}: (no file)"
      continue
    fi

    out_path="${event_dir}/${fname}"
    if [[ -f "$out_path" ]]; then
      echo "[fetch_losc_batch] ${event_id} ${det}: SKIP ${fname}"
      continue
    fi

    echo "[fetch_losc_batch] ${event_id} ${det}: GET ${fname}"
    if ! "${CURL_BIN}" -fL --retry 5 --retry-delay 2 -o "$out_path" "$download_url"; then
      echo "[fetch_losc_batch] WARN ${event_id} ${det}: fallo descarga ${download_url}" >&2
      rm -f "$out_path"
      continue
    fi
  done

  write_sha256sums_if_any "$event_dir"
done < "$EVENTS_FILE"
