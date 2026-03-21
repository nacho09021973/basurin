# Fase 3 — Intersección 220 vs 221 y soporte común

## Objetivo

Determinar qué soporte común existe entre los batches 220 y 221 y con qué definición exacta de intersección.

Esta fase tiene dos niveles conceptuales distintos y no deben confundirse:

1. **Intersección exacta por `geometry_id`**
2. **Intersección física por `phys_key`**

---

## Estado actual (actualizado 2026-03-20)

La fase 3 ya no está solo en diagnóstico ad-hoc: existe un **experimento formal reproducible**, ejecutado sobre un **host-run con `RUN_VALID=PASS`**, con artefactos contractuales bajo `runs/<run_id>/experiment/phase3_physkey_common/...`.

### Host-run canónico usado para la validación real

- `mvp_GW150914_221info_20260320T105521Z`

### Batches validados

- `batch_with_t0_220_eps2500_20260320T113433Z`
- `batch_with_t0_221_eps2500_20260320T114207Z`

### Entrypoint formal

```bash
python -m mvp.experiment_phase3_physkey_common \
  --run-id mvp_GW150914_221info_20260320T105521Z \
  --batch-220 batch_with_t0_220_eps2500_20260320T113433Z \
  --batch-221 batch_with_t0_221_eps2500_20260320T114207Z
```

---

## 3.A Intersección exacta por `geometry_id`

Resultado diagnóstico observado:

- intersección global nula
- intersección por evento nula

Interpretación:

- **no es criterio válido para decidir soporte físico**, porque `geometry_id` incorpora el modo y por construcción 220 y 221 no intersectan literalmente.

Este cálculo solo sirve como diagnóstico estructural de naming/identidad exacta y **no debe usarse como definición física de soporte común**.

---

## 3.B Intersección física por `phys_key`

### Definición exacta usada por el experimento formal

Definición normalizada válida:

- `phys_key = (family, provenance, M_solar, chi)`
- `family = lower(strip(metadata.family or row.family))`
- `provenance = metadata.source if existe; si no, metadata.ref; con fallback a row.source / row.ref`
- `M_solar = round(float(metadata.M_solar or row.M_solar), 6)`
- `chi = round(float(metadata.chi or row.chi), 6)`

### Serialización estable

El experimento formal serializa cada `phys_key` como array JSON determinista, no como string libre:

```text
["family","provenance",M_solar,chi]
```

Esto evita ambigüedad de parseo.

---

## Resultado formal validado en real

### Trazabilidad de entrada

- `n_rows_total_220 = 44`
- `n_rows_total_221 = 44`
- `n_rows_skipped_status_220 = 7`
- `n_rows_skipped_status_221 = 7`
- `n_rows_skipped_missing_compatible_220 = 0`
- `n_rows_skipped_missing_compatible_221 = 0`

### Población válida usada por el análisis

- `n_events_valid_220 = 38`
- `n_events_valid_221 = 38`
- `n_common_events = 38`

### Soporte físico global

- `K220 = 851`
- `K221 = 800`
- `K220_inter_K221 = 800`

### Diagnóstico por evento

- `n_empty_intersection_events = 0`
- `empty_intersection_events = []`
- `n_non_subset_cases = 37`

### Interpretación

- sí existe soporte físico común
- no hay ningún evento válido con intersección física vacía
- la hipótesis histórica `K220 ⊂ K221` **ya no vale**
- el patrón observado es compatible con `K221 ⊂ K220`

---

## Entradas canónicas

- `runs/<batch_220>/experiment/offline_batch/outputs/results.csv`
- `runs/<batch_221>/experiment/offline_batch/outputs/results.csv`
- `runs/<event_run>/s4_geometry_filter/outputs/compatible_set.json`

---

## Regla contractual crítica

No se debe asumir que toda fila de `results.csv` es utilizable.

Solo deben entrar en el análisis las filas con:

- `status == PASS`
- `runs/<run_id>/s4_geometry_filter/outputs/compatible_set.json` existente

Además, para los **batches** de entrada, el contrato real observado no es `RUN_VALID`, sino:

- `runs/<batch_run_id>/experiment/offline_batch/stage_summary.json`
- `verdict == "PASS"`
- `runs/<batch_run_id>/experiment/offline_batch/outputs/results.csv` existente

El **host-run** del experimento sí debe cumplir:

- `runs/<run_id>/RUN_VALID/verdict.json`
- `verdict == "PASS"`

---

## Contrato observado de `compatible_set.json`

Cada entrada de `compatible_geometries` expone la semántica física en:

- `metadata.family`
- `metadata.M_solar`
- `metadata.chi`
- `metadata.source` **o** `metadata.ref`

Con fallback permitido a top-level solo si el campo falta en `metadata`.

No debe asumirse que:

- `metadata.source` siempre existe
- `M_solar` y `chi` siempre están en top-level
- toda geometría compatible trae una `provenance` uniforme sin normalización

---

## Artefactos canónicos actuales de fase 3

### Artefacto formal reproducible

Bajo el host-run validado (corrida canónica 2026-03-20):

- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase3_physkey_common/outputs/summary_physkey_common.json`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase3_physkey_common/outputs/per_event_physkey_intersection.csv`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase3_physkey_common/stage_summary.json`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase3_physkey_common/manifest.json`

### Diagnósticos históricos previos (no canónicos)

#### Corrida canónica anterior (2026-03-16, reemplazada por la de 2026-03-20)

- `runs/host_phase3_physkey_20260316T001000Z/experiment/phase3_physkey_common/outputs/summary_physkey_common.json`

#### Diagnóstico exacto por `geometry_id`

- `runs/analysis_common_220_221_20260316T090350Z/experiment/common_geometries_220_221/...`

#### Diagnóstico físico previo por `phys_key`

- `runs/analysis_physkey_common_220_221_20260316T092822Z/experiment/physkey_common_220_221/outputs/summary_physkey_common.json`
- `runs/analysis_physkey_common_220_221_20260316T092822Z/experiment/physkey_common_220_221/outputs/per_event_physkey_intersection.csv`

Estos diagnósticos previos fueron útiles para descubrir la semántica correcta, pero el cierre operativo actual es el experimento formal `mvp.experiment_phase3_physkey_common`.

---

## Gate de salida

La fase 3 debe considerarse **cerrada contractualmente** porque ya existe un experimento formal que:

- filtra `results.csv` por `status == PASS`
- exige `compatible_set.json` existente para cada subrun usado
- normaliza `source/ref` a una `provenance` estable
- usa `metadata.*` con fallback controlado
- emite artefactos reproducibles
- deja `manifest.json` y `stage_summary.json`
- tiene tests de regresión
- ha sido ejecutado en real sobre batches corregidos

---

## Tests y cierre de implementación

El experimento formal quedó acompañado por tests específicos:

- `tests/test_experiment_phase3_physkey_common.py`

Cobertura validada:

- caso base
- filtrado de filas `FAIL`
- normalización `metadata.source` vs `metadata.ref`
- detección de `non_subset_cases`
- abort por falta de `provenance`
- abort por falta de `M_solar`
- gating correcto de batches por `offline_batch/stage_summary.json`

---

## Fallos típicos

- usar intersección literal por `geometry_id` para decidir soporte físico
- suponer `compatible_set.json` para filas `FAIL`
- asumir `metadata.source` siempre presente
- asumir la antigua relación `K220 ⊂ K221`
- exigir `RUN_VALID` a los batch runs de `offline_batch` cuando su contrato real es `stage_summary.json` con `verdict == PASS`
- usar scripts ad-hoc como si fueran artefacto canónico de fase 3

---

## Estado operativo que debe dejar documentado el equipo

Registrar explícitamente:

- host-run canónico usado
- batchs 220/221 usados
- definición exacta de `phys_key`
- regla exacta de `provenance`
- redondeo aplicado a `M_solar` y `chi`
- `K220`, `K221`, `K220_inter_K221`
- `n_empty_intersection_events`
- `n_non_subset_cases`
- artefactos exactos emitidos
- si el análisis es canónico o solo diagnóstico

A fecha 2026-03-20, **la fase 3 es canónica y operativa** (corrida canónica reemplazada por los batches `eps2500` del 2026-03-20; la corrida anterior de 2026-03-16 queda como referencia histórica).

---

## Dependencia hacia la fase siguiente

La fase 4 solo puede interpretarse correctamente usando la **intersección física por `phys_key`**, no la intersección exacta por `geometry_id`.

En particular:

- fase 3 define el soporte común multimodo físicamente interpretable
- fase 4 debe operar downstream sobre ese soporte común
- cualquier filtro adicional por área de Hawking debe aplicarse **después** de esta intersección física
