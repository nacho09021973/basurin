# Fase 2 — Poblaciones offline batch (220 / 221)

## Objetivo

Ejecutar el batch poblacional offline-first para los modos de ringdown relevantes y materializar un conjunto de runs por evento con trazabilidad completa.

En el estado actual del proyecto, esta fase se ejecuta por separado para:

- modo `(2,2,0)`
- modo `(2,2,1)`

---

## Entradas canónicas

- `runs/<audit_run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/<prep_run_id>/external_inputs/events_with_t0.txt`
- atlas efectivo del experimento batch

---

## Salidas canónicas

Bajo:

- `runs/<batch_run_id>/experiment/offline_batch/`

Artefactos principales:

- `stage_summary.json`
- `manifest.json`
- `outputs/results.csv`

Además, el batch materializa runs por evento del tipo:

- `runs/mvp_<EVENT_ID>_real_offline_<timestamp>/...`

con stages como:

- `s4_geometry_filter/outputs/compatible_set.json`
- `s6_information_geometry/...`
- `s6b_information_geometry_ranked/...`

---

## Gate de salida

Cada batch solo es válido si:

- `runs/<batch_run_id>/experiment/offline_batch/stage_summary.json` existe
- `verdict == "PASS"`

Y el downstream solo debe usar batches explícitamente declarados como canónicos.

---

## Runs canónicos actuales (2026-03-16)

- `batch_with_t0_220_eps2500_20260316T081911Z`
- `batch_with_t0_221_eps2500_20260316T082550Z`

Estos sustituyen a los batches antiguos puestos en cuarentena bajo:

- `runs/_quarantine_phase2_20260316/`

---

## Comandos canónicos

### 1) Batch 220

```bash
T0_CATALOG="runs/<audit_run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json"
EVENTS_FILE="runs/<prep_run_id>/external_inputs/events_with_t0.txt"
BATCH_RUN_ID="batch_with_t0_220_eps2500_$(date -u +%Y%m%dT%H%M%SZ)"

python -m mvp.experiment_offline_batch \
  --batch-run-id "$BATCH_RUN_ID" \
  --events-file "$EVENTS_FILE" \
  --t0-catalog "$T0_CATALOG" \
  --mode-filter "(2,2,0)" \
  --epsilon-default 2500 \
  --epsilon-fallback 2500
```

### 2) Batch 221

```bash
T0_CATALOG="runs/<audit_run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json"
EVENTS_FILE="runs/<prep_run_id>/external_inputs/events_with_t0.txt"
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

## Validación mínima

```bash
python - <<'PY'
import json
p = 'runs/<batch_run_id>/experiment/offline_batch/stage_summary.json'
d = json.load(open(p, 'r', encoding='utf-8'))
print(d.get('verdict'))
PY
```

Esperado:

- `PASS`

---

## Fallos típicos

- batch en `PASS` global con filas de `results.csv` en `FAIL` a nivel evento
- suponer que todos los `run_id` de `results.csv` tienen `compatible_set.json`
- reutilizar batches antiguos corregidos posteriormente
- usar runs en cuarentena como upstream de fases 3/4

---

## Estado operativo que debe dejar documentado el equipo

Registrar siempre:

- `batch_run_id` 220 canónico
- `batch_run_id` 221 canónico
- `verdict` de ambos
- número de eventos válidos por batch
- ubicación de cualquier cuarentena de runs obsoletos

---

## Dependencia hacia la fase siguiente

La fase 3 solo puede arrancar contra los **batch runs canónicos** de esta fase.
