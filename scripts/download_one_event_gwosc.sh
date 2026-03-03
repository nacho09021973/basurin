#!/usr/bin/env bash
set -euo pipefail

# USO:
#   ./scripts/download_one_event_gwosc.sh GW170608
#   STRICT_32=1 ./scripts/download_one_event_gwosc.sh GW170608

usage() {
  echo "Uso: ./scripts/download_one_event_gwosc.sh <EVENT_ID>" >&2
}

log_info() { echo "[INFO] $*"; }
log_get() { echo "[GET] $*"; }
log_skip() { echo "[SKIP] $*"; }
log_hash() { echo "[HASH] $*"; }
log_done() { echo "[DONE] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    log_error "Falta dependencia requerida: $cmd"
    exit 1
  }
}

basename_no_query() {
  local url="$1"
  local clean="${url%%\?*}"
  basename "$clean"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 1 ]]; then
  usage
  exit 1
fi

require_cmd bash
require_cmd curl
require_cmd jq
require_cmd sha256sum

INPUT_EVENT_ID="$1"
EVENT_ID="$INPUT_EVENT_ID"
if [[ "$EVENT_ID" == "GW190521" ]]; then
  EVENT_ID="GW190521_030229"
  log_info "Normalización aplicada: GW190521 -> $EVENT_ID"
fi

DURATION="${DURATION:-32}"
FORMAT="${FORMAT:-hdf5}"
STRICT_32="${STRICT_32:-0}"

OUT_DIR="data/losc/${EVENT_ID}"
mkdir -p "$OUT_DIR"

CURL_UA="${CURL_UA:-basurin-gwosc-downloader/1.0}"

fetch_json() {
  local url="$1"
  log_get "$url" >&2
  curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -H "User-Agent: ${CURL_UA}" "$url"
}

EVENT_API_URL="https://gwosc.org/api/v2/events/${EVENT_ID}"
EVENT_JSON="$(fetch_json "$EVENT_API_URL")" || {
  log_error "No se pudo obtener evento: $EVENT_ID"
  exit 1
}

DETAIL_URL="$(jq -r '.versions[-1].detail_url // empty' <<<"$EVENT_JSON")"
if [[ -z "$DETAIL_URL" || "$DETAIL_URL" == "null" ]]; then
  log_error "No se encontró versions[-1].detail_url en respuesta de evento para $EVENT_ID"
  exit 1
fi
if [[ "$DETAIL_URL" =~ ^/ ]]; then
  DETAIL_URL="https://gwosc.org${DETAIL_URL}"
fi

DETAIL_JSON="$(fetch_json "$DETAIL_URL")" || {
  log_error "No se pudo obtener detail_url: $DETAIL_URL"
  exit 1
}

STRAIN_FILES_URL="$(jq -r '.strain_files_url // empty' <<<"$DETAIL_JSON")"
if [[ -z "$STRAIN_FILES_URL" || "$STRAIN_FILES_URL" == "null" ]]; then
  log_error "El detail JSON no contiene strain_files_url"
  exit 1
fi
if [[ "$STRAIN_FILES_URL" =~ ^/ ]]; then
  STRAIN_FILES_URL="https://gwosc.org${STRAIN_FILES_URL}"
fi

STRAIN_JSON="$(fetch_json "$STRAIN_FILES_URL")" || {
  log_error "No se pudo obtener strain_files_url: $STRAIN_FILES_URL"
  exit 1
}

CANDIDATES_JSON="$(jq -c 'if type=="array" then . elif type=="object" then (.strain_files // []) else [] end' <<<"$STRAIN_JSON")"

pick_url_for_detector() {
  local detector="$1"
  jq -r --arg det "$detector" --arg fmt "$FORMAT" --argjson dur "$DURATION" --argjson strict "$STRICT_32" '
    def det_items: .[] | select((.detector // "") == $det);
    def has_url: select((.download_url // "") != "");
    def dur_num: (.duration | tonumber?);

    [det_items | has_url] as $all
    | [ $all[] | select(((.format // "") | ascii_downcase) == ($fmt | ascii_downcase) and (dur_num == $dur)) ] as $exact
    | if ($exact | length) > 0 then
        $exact[0].download_url
      elif $strict == 1 then
        ""
      else
        [ $all[] | select(((.format // "") | ascii_downcase) == ($fmt | ascii_downcase)) | . + {__dur: (dur_num // 1e18)} ]
        | sort_by(.__dur)
        | if length > 0 then .[0].download_url else "" end
      end
    | if . == "" and $strict == 0 then
        (if ($all | length) > 0 then $all[0].download_url else "" end)
      else
        .
      end
  ' <<<"$CANDIDATES_JSON"
}

print_debug_candidates() {
  local detector="$1"
  echo "[INFO] Candidatos ${detector} (format<TAB>duration<TAB>download_url_basename):" >&2
  jq -r --arg det "$detector" '
    .[]
    | select((.detector // "") == $det and (.download_url // "") != "")
    | [(.format // ""), ((.duration // "")|tostring), ((.download_url // "") | sub("\\?.*$"; "") | split("/") | .[-1])]
    | @tsv
  ' <<<"$CANDIDATES_JSON" >&2 || true
}

H1_URL="$(pick_url_for_detector "H1")"
L1_URL="$(pick_url_for_detector "L1")"

if [[ -z "$H1_URL" || -z "$L1_URL" ]]; then
  log_error "No se pudieron resolver URLs para H1/L1 con FORMAT=${FORMAT}, DURATION=${DURATION}, STRICT_32=${STRICT_32}"
  print_debug_candidates "H1"
  print_debug_candidates "L1"
  if [[ "$STRICT_32" == "1" ]]; then
    log_error "STRICT_32=1 exige coincidencia exacta format=${FORMAT}, duration=${DURATION}"
  fi
  exit 1
fi

download_one() {
  local detector="$1"
  local url="$2"
  local filename
  filename="$(basename_no_query "$url")"
  local dest="$OUT_DIR/$filename"

  if [[ -s "$dest" ]]; then
    log_skip "${detector}: ya existe y no vacío -> $dest"
  else
    log_get "Descargando ${detector}: $url"
    curl -fL --retry 5 --retry-delay 2 --retry-all-errors -H "User-Agent: ${CURL_UA}" -o "$dest" "$url"
  fi

  if [[ ! -s "$dest" ]]; then
    log_error "Archivo descargado inválido o vacío: $dest"
    exit 1
  fi

  echo "$dest"
}

log_info "EVENT_ID=${EVENT_ID}"
log_info "OUT_DIR=${OUT_DIR}"
log_info "FORMAT=${FORMAT} DURATION=${DURATION} STRICT_32=${STRICT_32}"

H1_FILE="$(download_one "H1" "$H1_URL")"
L1_FILE="$(download_one "L1" "$L1_URL")"

log_hash "sha256sum ${H1_FILE}"
sha256sum "$H1_FILE"
log_hash "sha256sum ${L1_FILE}"
sha256sum "$L1_FILE"

log_done "Descarga completada para ${EVENT_ID} en ${OUT_DIR}"
