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

resolve_short_name() {
  local event_id="$1"
  local candidate
  for candidate in "${event_id}-v2" "${event_id}-v1"; do
    local url="https://gwosc.org/api/v2/event-versions/${candidate}"
    if [[ "$(http_code "$url")" == "200" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

best_hdf5_url_for_detector() {
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

candidates = []


def visit(node):
    if isinstance(node, dict):
        url = node.get("download_url")
        if isinstance(url, str) and url.lower().endswith(".hdf5"):
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
            candidates.append((score, url))

        for v in node.values():
            visit(v)
    elif isinstance(node, list):
        for item in node:
            visit(item)


visit(payload)

if not candidates:
    print("")
    raise SystemExit(0)

# Preferir score mayor; en empate conservar orden de aparición.
best_idx = max(range(len(candidates)), key=lambda i: candidates[i][0])
print(candidates[best_idx][1])
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
    body="$("${CURL_BIN}" -sS -L --retry 5 --retry-delay 2 "$url" 2>/dev/null || true)"
    GWOSC_STRAIN_JSON="$body"
    export GWOSC_STRAIN_JSON
    download_url="$(best_hdf5_url_for_detector "$body")"

    if [[ -z "$download_url" ]]; then
      echo "[fetch_losc_batch] ${event_id} ${det}: (no file)"
      continue
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
