#!/usr/bin/env bash
set -euo pipefail

# Descarga HDF5 de GWTC-1 en data/losc/<EVENT_ID>/ con idempotencia.
#
# Cambio clave: la resolución de shortName ya NO depende de
# /api/v2/event-versions?name-contains=... (a veces devuelve HTML/empty).
# Ahora es determinista: primero <EVENT>-v2, si no existe entonces <EVENT>-v1,
# comprobando solo HTTP status (sin parsear JSON en este paso).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOSC_ROOT="${LOSC_ROOT:-${REPO_ROOT}/data/losc}"
CURL_BIN="${CURL_BIN:-curl}"

usage() {
  cat <<'EOF'
Uso:
  tools/fetch_losc_batch.sh EVENT_ID [EVENT_ID ...]

Variables opcionales:
  LOSC_ROOT   Directorio destino (default: <repo>/data/losc)
  CURL_BIN    Binario curl a usar (útil para tests)

Ejemplo:
  tools/fetch_losc_batch.sh GW150914 GW151226
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  usage
  exit 0
fi

http_code() {
  local url="$1"
  "${CURL_BIN}" -sS -o /dev/null -w '%{http_code}' --max-time 30 "$url" || true
}

resolve_short_name() {
  local event_id="$1"
  local url_v2="https://gwosc.org/api/v2/event-versions/${event_id}-v2"
  local url_v1="https://gwosc.org/api/v2/event-versions/${event_id}-v1"

  if [[ "$(http_code "$url_v2")" == "200" ]]; then
    printf '%s/v2\n' "$event_id"
    return 0
  fi
  if [[ "$(http_code "$url_v1")" == "200" ]]; then
    printf '%s/v1\n' "$event_id"
    return 0
  fi
  return 1
}

extract_hdf5_urls() {
  python - <<'PY'
import json
import os
import urllib.parse

payload = json.loads(os.environ["GWOSC_EVENT_JSON"])
for _, event in sorted((payload.get("events") or {}).items()):
    for item in event.get("strain") or []:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        name = os.path.basename(urllib.parse.urlparse(url).path)
        if not name.lower().endswith((".hdf5", ".h5")):
            continue
        print(f"{url}\t{name}")
PY
}

mkdir -p "$LOSC_ROOT"

for event_id in "$@"; do
  event_dir="${LOSC_ROOT}/${event_id}"
  mkdir -p "$event_dir"

  if find "$event_dir" -maxdepth 1 -type f \( -iname '*.hdf5' -o -iname '*.h5' \) | read -r _; then
    echo "[fetch_losc_batch] SKIP ${event_id}: ya existen HDF5 en ${event_dir}"
    continue
  fi

  if ! short_name="$(resolve_short_name "$event_id")"; then
    echo "[fetch_losc_batch] ERROR ${event_id}: no se pudo resolver shortName (${event_id}-v2/${event_id}-v1)" >&2
    exit 1
  fi

  meta_url="https://gwosc.org/eventapi/json/GWTC-1-confident/${short_name}"
  echo "[fetch_losc_batch] ${event_id}: shortName=${short_name}"
  echo "[fetch_losc_batch] GET ${meta_url}"

  event_json="$("${CURL_BIN}" -fsSL --retry 3 --retry-delay 2 "$meta_url")"
  GWOSC_EVENT_JSON="$event_json"
  export GWOSC_EVENT_JSON

  found=0
  while IFS=$'\t' read -r url filename; do
    [[ -n "${url:-}" ]] || continue
    found=1
    out_path="${event_dir}/${filename}"
    if [[ -f "$out_path" ]]; then
      echo "[fetch_losc_batch] SKIP archivo existente: ${out_path}"
      continue
    fi
    echo "[fetch_losc_batch] DESCARGA ${url} -> ${out_path}"
    "${CURL_BIN}" -fL --retry 5 --retry-delay 2 -o "$out_path" "$url"
  done < <(extract_hdf5_urls)

  if [[ "$found" -eq 0 ]]; then
    echo "[fetch_losc_batch] ERROR ${event_id}: no se encontraron URLs HDF5 en ${meta_url}" >&2
    exit 1
  fi
done
