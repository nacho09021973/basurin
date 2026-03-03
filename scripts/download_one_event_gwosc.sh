#!/usr/bin/env bash
set -euo pipefail

# Descarga strain H1+L1 de UN evento desde GWOSC v2 a data/losc/<EVENT_ID>/
# Requisitos: curl jq sha256sum
#
# Uso:
#   ./scripts/download_one_event_gwosc.sh GW170608
#   STRICT_32=1 ./scripts/download_one_event_gwosc.sh GW170608
#   DURATION=32 FORMAT=hdf5 ./scripts/download_one_event_gwosc.sh GW170608
#
# Variables:
#   OUT_ROOT  (default: data/losc)
#   DURATION  (default: 32)
#   FORMAT    (default: hdf5)
#   STRICT_32 (default: 0)  # si 1, falla si no hay HDF5 con duration=32 para H1/L1

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Uso: download_one_event_gwosc.sh <EVENT_ID>

Variables:
  OUT_ROOT   (default: data/losc)
  DURATION   (default: 32)
  FORMAT     (default: hdf5)
  STRICT_32  (default: 0)  # si 1, exige format=hdf5 y duration=DURATION

Ejemplos:
  ./scripts/download_one_event_gwosc.sh GW170608
  STRICT_32=1 ./scripts/download_one_event_gwosc.sh GW170608
EOF
  exit 0
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] falta dependencia: $1" >&2; exit 2; }; }
need curl
need jq
need sha256sum

EVENT_IN="${1:-}"
if [[ -z "$EVENT_IN" ]]; then
  echo "Uso: $0 <EVENT_ID>" >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-data/losc}"
DURATION="${DURATION:-32}"
FORMAT="${FORMAT:-hdf5}"
STRICT_32="${STRICT_32:-0}"

normalize_event() {
  local e="$1"
  if [[ "$e" == "GW190521" ]]; then
    echo "GW190521_030229"
  else
    echo "$e"
  fi
}

fetch_json() {
  local url="$1"
  curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors "$url"
}

flatten_strain_files() {
  # Entrada: JSON del endpoint strain-files (puede ser array, o {strain_files:[...]}, o paginado {results:[...], next:...})
  # Salida: array JSON con entradas de ficheros
  local json="$1"
  # Caso paginado: devolver .results
  if jq -e "type==\"object\" and has(\"results\")" >/dev/null <<<"$json"; then
    jq -c ".results" <<<"$json"
    return 0
  fi
  # Caso antiguo: {strain_files:[...]}
  if jq -e "type==\"object\" and has(\"strain_files\")" >/dev/null <<<"$json"; then
    jq -c ".strain_files" <<<"$json"
    return 0
  fi
  # Caso array: ya OK
  echo "$json"
}

flatten_strain_files() {
  # Entrada: JSON del endpoint strain-files (puede ser array, o {strain_files:[...]}, o paginado {results:[...], next:...})
  # Salida: array JSON con entradas de ficheros
  local json="$1"
  # Caso paginado: devolver .results
  if jq -e "type==\"object\" and has(\"results\")" >/dev/null <<<"$json"; then
    jq -c ".results" <<<"$json"
    return 0
  fi
  # Caso antiguo: {strain_files:[...]}
  if jq -e "type==\"object\" and has(\"strain_files\")" >/dev/null <<<"$json"; then
    jq -c ".strain_files" <<<"$json"
    return 0
  fi
  # Caso array: ya OK
  echo "$json"
}

pick_url() {
  local det="$1"
  local dur="$2"
  local fmt="$3"
  local strict="$4"
  local json="$5"

  local u=""
  u="$(jq -r --arg det "$det" --arg fmt "$fmt" --argjson dur "$dur" '
      (type=="array") as $is_array
      | (if $is_array then . else (.strain_files // []) end) as $files
      | ($files | map(select(.detector==$det and .format==$fmt and .duration==$dur)) | .[0].download_url) // empty
    ' <<<"$json")"
  if [[ -n "$u" ]]; then echo "$u"; return 0; fi

  if [[ "$strict" == "1" ]]; then
    echo ""
    return 0
  fi

  u="$(jq -r --arg det "$det" --arg fmt "$fmt" '
      (type=="array") as $is_array
      | (if $is_array then . else (.strain_files // []) end) as $files
      | ($files | map(select(.detector==$det and .format==$fmt and (.download_url|type=="string"))) | sort_by(.duration) | .[0].download_url) // empty
    ' <<<"$json")"
  if [[ -n "$u" ]]; then echo "$u"; return 0; fi

  u="$(jq -r --arg det "$det" '
      (type=="array") as $is_array
      | (if $is_array then . else (.strain_files // []) end) as $files
      | ($files | map(select(.detector==$det and (.download_url|type=="string"))) | sort_by(.format, .duration) | .[0].download_url) // empty
    ' <<<"$json")"
  echo "$u"
}

EVENT_ID="$(normalize_event "$EVENT_IN")"
OUT_DIR="$OUT_ROOT/$EVENT_ID"
mkdir -p "$OUT_DIR"

echo "[INFO] Event=$EVENT_ID OUT_DIR=$OUT_DIR DURATION=$DURATION FORMAT=$FORMAT STRICT_32=$STRICT_32"

EVENTS_JSON="$(fetch_json "https://gwosc.org/api/v2/events/${EVENT_ID}")"
DETAIL_URL="$(jq -r '
  if (.versions and (.versions|length)>0) then .versions[-1].detail_url
  elif (.events and (.events|length)>0 and .events[0].versions and (.events[0].versions|length)>0) then .events[0].versions[-1].detail_url
  else empty end
' <<<"$EVENTS_JSON")"

if [[ -z "$DETAIL_URL" || "$DETAIL_URL" == "null" ]]; then
  echo "[ERROR] No pude resolver detail_url para $EVENT_ID desde /api/v2/events" >&2
  echo "[DEBUG] top-level keys: $(jq -r 'keys|join(",")' <<<"$EVENTS_JSON")" >&2
  exit 3
fi

DETAIL_JSON="$(fetch_json "$DETAIL_URL")"
STRAIN_FILES_URL="$(jq -r '.strain_files_url // empty' <<<"$DETAIL_JSON")"
if [[ -z "$STRAIN_FILES_URL" || "$STRAIN_FILES_URL" == "null" ]]; then
  echo "[ERROR] No hay strain_files_url en detail_url=$DETAIL_URL" >&2
  echo "[DEBUG] keys(detail): $(jq -r 'keys|join(",")' <<<"$DETAIL_JSON")" >&2
  exit 4
fi

STRAIN_FILES_JSON="$(fetch_json "$STRAIN_FILES_URL")"

H1_URL="$(pick_url "H1" "$DURATION" "$FORMAT" "$STRICT_32" "$STRAIN_FILES_JSON")"
L1_URL="$(pick_url "L1" "$DURATION" "$FORMAT" "$STRICT_32" "$STRAIN_FILES_JSON")"

if [[ -z "$H1_URL" || -z "$L1_URL" ]]; then
  echo "[ERROR] No encontré URLs H1/L1 para $EVENT_ID (duration=$DURATION format=$FORMAT strict=$STRICT_32)" >&2
  echo "[DEBUG] strain_files_url=$STRAIN_FILES_URL" >&2
  echo "[DEBUG] candidatos H1:" >&2
  jq -r '
    (type=="array") as $is_array
    | (if $is_array then . else (.strain_files // []) end) as $files
    | ($files | map(select(.detector=="H1")) | sort_by(.format,.duration) | .[] | "\(.format)\t\(.duration)\t\((.download_url//"")|split("?")[0]|split("/")[-1])")
  ' <<<"$STRAIN_FILES_JSON" | head -n 60 >&2
  echo "[DEBUG] candidatos L1:" >&2
  jq -r '
    (type=="array") as $is_array
    | (if $is_array then . else (.strain_files // []) end) as $files
    | ($files | map(select(.detector=="L1")) | sort_by(.format,.duration) | .[] | "\(.format)\t\(.duration)\t\((.download_url//"")|split("?")[0]|split("/")[-1])")
  ' <<<"$STRAIN_FILES_JSON" | head -n 60 >&2
  exit 5
fi

dl() {
  local url="$1"
  local out_dir="$2"
  local name out
  name="$(basename "${url%%\?*}")"
  out="$out_dir/$name"
  if [[ -s "$out" ]]; then
    echo "[SKIP] existe: $out"
    echo "$out"
    return 0
  fi
  echo "[GET]  $url"
  curl -fL --retry 5 --retry-delay 2 --retry-all-errors -o "$out" "$url"
  echo "$out"
}

H1_PATH="$(dl "$H1_URL" "$OUT_DIR")"
L1_PATH="$(dl "$L1_URL" "$OUT_DIR")"

echo "[HASH] $EVENT_ID"
sha256sum "$H1_PATH" "$L1_PATH"
echo "[DONE] $EVENT_ID"
