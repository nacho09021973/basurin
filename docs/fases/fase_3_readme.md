# Fase 3 — Intersección 220 vs 221 y soporte común

## Objetivo

Determinar qué soporte común existe entre los batches 220 y 221 y con qué definición exacta de intersección.

Esta fase tiene dos niveles conceptuales distintos y no deben confundirse:

1. **Intersección exacta por `geometry_id`**
2. **Intersección física por `phys_key`**

---

## Estado actual (2026-03-16)

### 3.A Intersección exacta por `geometry_id`

Resultado diagnóstico observado:

- intersección global nula
- intersección por evento nula

Interpretación:

- **no es criterio válido para decidir E5**, porque `geometry_id` incorpora el modo y por construcción 220 y 221 no intersectan literalmente.

Este cálculo se hizo como diagnóstico ad-hoc y **no es artefacto canónico**.

### 3.B Intersección física por `phys_key`

Definición normalizada válida:

- `phys_key = (family, provenance, M_solar, chi)`
- `provenance = metadata.source if existe; si no, metadata.ref`
- `M_solar = round(metadata.M_solar, 6)`
- `chi = round(metadata.chi, 6)`

Resultado diagnóstico observado sobre batches corregidos:

- `n_common_events = 37`
- `K220 = 851`
- `K221 = 800`
- `K220_inter_K221 = 800`
- `n_empty_intersection_events = 0`
- `n_non_subset_cases = 37`

Interpretación:

- sí existe soporte físico común
- la hipótesis histórica `K220 ⊂ K221` ya no vale
- globalmente parece cumplirse `K221 ⊂ K220`

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

---

## Contrato observado de `compatible_set.json`

Cada entrada de `compatible_geometries` expone la semántica física en:

- `metadata.family`
- `metadata.M_solar`
- `metadata.chi`
- `metadata.source` **o** `metadata.ref`

No debe buscarse `M_solar` ni `chi` en top-level.

---

## Artefactos diagnósticos actuales

### Diagnóstico exacto por `geometry_id` (no canónico)

- `runs/analysis_common_220_221_20260316T090350Z/experiment/common_geometries_220_221/...`

### Diagnóstico físico por `phys_key` (no canónico, pero semánticamente correcto)

- `runs/analysis_physkey_common_220_221_20260316T092822Z/experiment/physkey_common_220_221/outputs/summary_physkey_common.json`
- `runs/analysis_physkey_common_220_221_20260316T092822Z/experiment/physkey_common_220_221/outputs/per_event_physkey_intersection.csv`

---

## Gate de salida

La fase 3 solo puede considerarse cerrada contractualmente cuando exista un script/experimento formal que:

- lea `results.csv` filtrando `status == PASS`
- lea `compatible_set.json` usando `metadata.*`
- normalice `source/ref` a una `provenance` estable
- emita artefactos reproducibles con tests

A día de hoy, el diagnóstico existe, pero **todavía no hay cierre canónico**.

---

## Fallos típicos

- usar intersección literal por `geometry_id` para decidir soporte físico
- suponer `compatible_set.json` para filas `FAIL`
- asumir `metadata.source` siempre presente
- asumir la antigua relación `K220 ⊂ K221`

---

## Estado operativo que debe dejar documentado el equipo

Registrar explícitamente:

- batchs 220/221 usados
- definición exacta de `phys_key`
- si el análisis es canónico o diagnóstico
- `K220`, `K221`, `K220_inter_K221`
- `n_empty_intersection_events`
- `n_non_subset_cases`

---

## Dependencia hacia la fase siguiente

La fase 4 solo puede interpretarse correctamente usando la **intersección física**, no la intersección exacta por `geometry_id`.
