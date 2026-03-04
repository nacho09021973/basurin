# readme_experiment_4.md — Experimento 4: Batch con t0 + mode_filter + calibración epsilon (`mahalanobis_log`)

## A) Objetivo científico-operativo

Este experimento define un flujo **offline-first** para ejecutar un batch de eventos con `t0` conocido y filtrado por modo (`mode_filter`) para `s4_geometry_filter` con `metric=mahalanobis_log`.

Objetivos concretos:

1. Correr un conjunto de eventos “con t0” para un modo dado (`(2,2,0)` o `(2,2,1)`).
2. Producir por evento `s4_geometry_filter/outputs/compatible_set.json` con geometrías compatibles (o `n_compatible=0` si no hay).
3. Agregar resultados en `results.csv` con trazas mínimas por evento:
   - `event_id`
   - `run_id`
   - `status`
   - `len_compatible`
   - `epsilon_used`
   - `mode_filter`
   - campos de error (si aplica)
4. Dejar preparado el análisis de **geometrías en común** entre corridas 220 vs 221.

> Referencia reproducible conocida: `batch_with_t0_220_eps2500_fixlen_20260304T160054Z` (44 eventos, `threshold_d2=2500.0`, `Counter(len_compatible)={156:42, 0:2}`).

---

## B) Inputs / artefactos fuente de verdad

### 1) Catálogo t0 GWOSC v2
Ruta canónica bajo run de auditoría:

- `runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json`

Contrato de ubicación general:

- `runs/<audit_run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json`

### 2) Lista de eventos con t0
Ruta canónica bajo run de preparación:

- `runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt`

Contrato de ubicación general:

- `runs/<prep_run_id>/external_inputs/events_with_t0.txt`

Esta lista corresponde a la intersección entre `data/losc/*` y `keys(t0_catalog_gwosc_v2.json)`.

### 3) Atlas de geometrías
Atlas por defecto usado por `s4` (si no se sobreescribe explícitamente):

- `docs/ringdown/atlas/atlas_berti_v2.json`

### 4) Nota de gobernanza (obligatoria)

- IO determinista: escribir solo bajo `runs/<run_id>/...` (o `BASURIN_RUNS_ROOT` efectivo).
- No escribir outputs fuera de `runs/...`.
- Gating: no usar artefactos downstream si `RUN_VALID != PASS`.

---

## C) Comandos exactos (220 y 221)

### Variables base (rutas exactas reproducibles)

```bash
T0_CATALOG="runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json"
EVENTS_FILE="runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt"
```

### Batch modo 220 (`mode_filter="(2,2,0)"`)

```bash
T0_CATALOG="runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json"
EVENTS_FILE="runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt"
BATCH_RUN_ID="batch_with_t0_220_eps2500_$(date -u +%Y%m%dT%H%M%SZ)"

python -m mvp.experiment_offline_batch \
  --batch-run-id "$BATCH_RUN_ID" \
  --events-file "$EVENTS_FILE" \
  --t0-catalog "$T0_CATALOG" \
  --mode-filter "(2,2,0)" \
  --epsilon-default 2500 \
  --epsilon-fallback 2500
```

### Batch modo 221 (`mode_filter="(2,2,1)"`)

```bash
T0_CATALOG="runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json"
EVENTS_FILE="runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt"
BATCH_RUN_ID="batch_with_t0_221_eps2500_$(date -u +%Y%m%dT%H%M%SZ)"

python -m mvp.experiment_offline_batch \
  --batch-run-id "$BATCH_RUN_ID" \
  --events-file "$EVENTS_FILE" \
  --t0-catalog "$T0_CATALOG" \
  --mode-filter "(2,2,1)" \
  --epsilon-default 2500 \
  --epsilon-fallback 2500
```

---

## D) Auditoría / checks (contract-first)

### 1) Verificar PASS del experimento batch

```bash
python - <<'PY'
import json, os
batch_run_id = os.environ["BATCH_RUN_ID"]
p = f"runs/{batch_run_id}/experiment/offline_batch/stage_summary.json"
d = json.load(open(p, "r", encoding="utf-8"))
print("stage_summary:", p)
print("verdict:", d.get("verdict"))
PY
```

Se espera `verdict: PASS`.

### 2) Verificar `results.csv` y cantidad de filas

```bash
python - <<'PY'
import csv, os
batch_run_id = os.environ["BATCH_RUN_ID"]
p = f"runs/{batch_run_id}/experiment/offline_batch/outputs/results.csv"
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("results_csv:", p)
print("rows:", len(rows))
PY
```

### 3) Estadística de `len_compatible` y eventos con 0

`len_compatible` es el conteo de geometrías compatibles por evento. En términos de `s4`, equivale a `n_compatible` (o `len(compatible_geometries)` en `compatible_set.json`).

```bash
python - <<'PY'
import csv, os
from collections import Counter

batch_run_id = os.environ["BATCH_RUN_ID"]
p = f"runs/{batch_run_id}/experiment/offline_batch/outputs/results.csv"
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def parse_int(v):
    try:
        return int(v)
    except Exception:
        return None

lens = [parse_int(r.get("len_compatible")) for r in rows]
c = Counter([x for x in lens if x is not None])
zeros = [r.get("event_id") for r in rows if parse_int(r.get("len_compatible")) == 0]
print("Counter(len_compatible)=", c)
print("events_len_compatible_0=", zeros)
PY
```

### 4) Auditoría `s4` por evento (subrun)

Para cada `event_run_id` reportado en `results.csv`, revisar:

- `runs/<event_run_id>/s4_geometry_filter/stage_summary.json`
  - `results.d2_min`
  - `threshold_d2`
  - `n_compatible`
  - `n_atlas`
- `runs/<event_run_id>/s4_geometry_filter/outputs/compatible_set.json`
  - `n_compatible`
  - `compatible_geometries`
  - `mode_filter`

Chequeo puntual:

```bash
python - <<'PY'
import json, os
event_run_id = os.environ["EVENT_RUN_ID"]
ss = f"runs/{event_run_id}/s4_geometry_filter/stage_summary.json"
cs = f"runs/{event_run_id}/s4_geometry_filter/outputs/compatible_set.json"
d1 = json.load(open(ss, "r", encoding="utf-8"))
d2 = json.load(open(cs, "r", encoding="utf-8"))
print("stage_summary:", ss)
print("d2_min:", ((d1.get("results") or {}).get("d2_min")))
print("threshold_d2:", d1.get("threshold_d2"))
print("n_compatible(stage_summary):", d1.get("n_compatible"))
print("n_atlas:", d1.get("n_atlas"))
print("compatible_set:", cs)
print("n_compatible(compatible_set):", d2.get("n_compatible"))
print("len(compatible_geometries):", len(d2.get("compatible_geometries") or []))
print("mode_filter:", d2.get("mode_filter"))
PY
```

### 5) Interpretación de `n_compatible=0`

Interpretación contract-first:

- Si `n_compatible=0`, no pasó ninguna geometría el umbral de compatibilidad.
- Con `metric=mahalanobis_log`, esto típicamente significa `d2_min > threshold_d2`.

Ejemplos reproducibles para `epsilon=2500` (batch 220 de referencia):

- `GW190521`: `d2_min=2944.80232 > 2500`
- `GW191113_071753`: `d2_min=5577.32032 > 2500`

---

## E) Comparación 220 vs 221: geometrías en común

### Plan de comparación (sin stage canónico nuevo)

1. Para cada evento compartido entre ambos batches, cargar:
   - `compatible_set.json` de 220
   - `compatible_set.json` de 221
2. Definir identificador estable de geometría usando la representación existente en cada entrada de `compatible_geometries` (string canónico JSON con claves ordenadas).
3. Calcular intersección por evento y/o global.
4. Reportar conteos:
   - `|G220|`
   - `|G221|`
   - `|G220 ∩ G221|`

### Snippet one-shot (análisis local bajo `runs/<analysis_run_id>/...`)

> Este script es para análisis ad-hoc; **no** reemplaza stages canónicos.

```bash
BATCH_220="batch_with_t0_220_eps2500_fixlen_20260304T160054Z"
BATCH_221="batch_with_t0_221_eps2500_<REEMPLAZAR_POR_RUN_REAL>"
ANALYSIS_RUN_ID="analysis_common_geometries_$(date -u +%Y%m%dT%H%M%SZ)"

python - <<'PY'
import csv
import json
import os
from pathlib import Path

batch_220 = os.environ["BATCH_220"]
batch_221 = os.environ["BATCH_221"]
analysis_run_id = os.environ["ANALYSIS_RUN_ID"]

out_root = Path("runs") / analysis_run_id / "experiment" / "common_geometries_220_221"
out_outputs = out_root / "outputs"
out_outputs.mkdir(parents=True, exist_ok=True)

r220 = Path("runs") / batch_220 / "experiment" / "offline_batch" / "outputs" / "results.csv"
r221 = Path("runs") / batch_221 / "experiment" / "offline_batch" / "outputs" / "results.csv"

def load_results(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_event = {}
    for r in rows:
        ev = r.get("event_id")
        rid = r.get("run_id")
        if ev and rid:
            by_event[ev] = rid
    return by_event

def canonical_geom_id(geom):
    return json.dumps(geom, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def load_geom_set(event_run_id):
    p = Path("runs") / event_run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    geoms = d.get("compatible_geometries") or []
    return {canonical_geom_id(g) for g in geoms}

m220 = load_results(r220)
m221 = load_results(r221)
common_events = sorted(set(m220) & set(m221))

rows = []
g220_global = set()
g221_global = set()

for ev in common_events:
    g220 = load_geom_set(m220[ev])
    g221 = load_geom_set(m221[ev])
    inter = g220 & g221
    g220_global |= g220
    g221_global |= g221
    rows.append({
        "event_id": ev,
        "run_id_220": m220[ev],
        "run_id_221": m221[ev],
        "G220": len(g220),
        "G221": len(g221),
        "G220_inter_G221": len(inter),
    })

global_inter = g220_global & g221_global

per_event_csv = out_outputs / "common_geometries_per_event.csv"
with per_event_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["event_id", "run_id_220", "run_id_221", "G220", "G221", "G220_inter_G221"])
    w.writeheader()
    w.writerows(rows)

summary = {
    "batch_220": batch_220,
    "batch_221": batch_221,
    "n_common_events": len(common_events),
    "G220_global": len(g220_global),
    "G221_global": len(g221_global),
    "G220_inter_G221_global": len(global_inter),
    "outputs": {
        "per_event_csv": str(per_event_csv),
    },
}
(out_outputs / "summary_common_geometries.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("ANALYSIS_ROOT=", out_root)
print("OUTPUTS_DIR=", out_outputs)
print("PER_EVENT_CSV=", per_event_csv)
print("SUMMARY_JSON=", out_outputs / "summary_common_geometries.json")
PY
```

---

## Checklist de aceptación rápida

- Comandos 220 y 221 ejecutables, con rutas exactas para catálogo t0 y lista de eventos.
- Auditoría mínima del batch (`stage_summary.json` + `results.csv`).
- Interpretación explícita de `len_compatible`/`n_compatible`.
- Casos `n_compatible=0` documentados por `d2_min > threshold_d2`.
- Plan + snippet para comparación de geometrías en común 220 vs 221.
