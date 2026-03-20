# Informe de auditoría BASURIN — 220/221, Fase 3/Fase 4

**Fecha:** 2026-03-20  
**Contexto:** auditoría operativa y científica del carril multimodo `220/221`, validación de batches canónicos rehechos y evaluación del papel efectivo de `221` y del filtro de Hawking.  
**Estado general:** vamos por buen camino.

---

## 1. Resumen ejecutivo

Hoy se ha cerrado una cadena completa y coherente de validación:

1. Se saneó el frente de CLI / tests del pipeline en `main`.
2. Se confirmó que el path `single` **no** era el carril correcto para evaluar el `221`.
3. Se rehizo la cohorte canónica de Fase 2 desde cero, por separado para `220` y `221`.
4. Se auditó la cohorte efectiva y los fallos por datos offline.
5. Se ejecutó la Fase 3 (`phys_key_common`) sobre los nuevos batches canónicos.
6. Se ejecutó la Fase 4 / E5 (`hawking_area_common_support`) sobre ese soporte común.
7. Se concluyó que:
   - el `221` **sí aporta información útil** y **sí discrimina** respecto a `220`;
   - el filtro de Hawking, en la cohorte actual rehecha, **no discrimina adicionalmente**;
   - Hawking **valida compatibilidad física**, pero no añade recorte efectivo del soporte.

La conclusión técnica fuerte del día es:

```text
K_common = K220 ∩ K221 = K221
K_hawking = K_common
```

para la cohorte canónica rehecha actual.

---

## 2. Hallazgos principales del día

### 2.1. Estado de CLI / parser / tests

Se confirmó que el frente de bloqueos pre-E2E ya estaba resuelto en `main`.

Evidencia:

- `pytest -q tests/test_pipeline_orchestrator.py tests/unit/test_cli_parity.py` → **33 passed**
- `pytest -q -o "addopts=" tests/unit/ -v` → **48 passed**

Trazabilidad de commits relevantes:

- `6a4fe0a` — `Fix single pipeline delta-lnL compatibility`
- `b1be03d` — `Fix argparse threshold mode conflict`

Inspección fina de `git show` dejó claro que el segundo commit, pese a su mensaje, corrigió realmente el reenvío de `informative_threshold` en el dispatch `single`.

### 2.2. Corrección conceptual importante: `single` no era el carril correcto para 221

Se verificó empíricamente que el path `single` produce:

- `s1_fetch_strain`
- `s2_ringdown_window`
- `s3_ringdown_estimates`
- `s4_geometry_filter`
- `s6_information_geometry`
- `s6b_information_geometry_ranked`

pero **no** materializa el carril explícito multimodo del `221` (`s3b_multimode_estimates`, `s4g`, `s4h`, `s4i`, `s4j`, `s4k`).

Conclusión: intentar responder “si sale el 221” con `single` era metodológicamente incorrecto.

### 2.3. Promoción falsa del 221 en el routing multimodo

Se validó el arreglo conceptual del pipeline multimodo:

- La rama explícita `s4g -> s4h -> s4i -> s4f -> s4j -> s4k` ya existía.
- El problema era que `_build_mode221_obs_payload()` infería usabilidad de `221` con heurísticas sobre strings de `reason`.
- El gate correcto debía ser el SSOT:
  - `s3b_multimode_estimates/stage_summary.json -> multimode_viability.class`
- Solo `MULTIMODE_OK` debe autorizar la promoción del `221` downstream.

Esto ya quedó integrado y validado por tests específicos.

---

## 3. Rehecho canónico de Fase 2

### 3.1. Inputs canónicos usados

Se confirmó la existencia de:

- `runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt`

### 3.2. Batch canónico 220

**Run ID:** `batch_with_t0_220_eps2500_20260320T113433Z`

Resultado contractual:

- `runs/batch_with_t0_220_eps2500_20260320T113433Z/experiment/offline_batch/stage_summary.json`
- `verdict: PASS`

### 3.3. Batch canónico 221

**Run ID:** `batch_with_t0_221_eps2500_20260320T114207Z`

Resultado contractual:

- `runs/batch_with_t0_221_eps2500_20260320T114207Z/experiment/offline_batch/stage_summary.json`
- `verdict: PASS`

### 3.4. Cohorte efectiva

Comparación de cohortes PASS:

- `PASS_220 = 38`
- `PASS_221 = 38`
- `COMMON_PASS = 38`
- `ONLY_220 = []`
- `ONLY_221 = []`

Conclusión:

```text
La cohorte efectiva válida de 220 y 221 coincide exactamente en 38 eventos.
```

---

## 4. Auditoría de eventos fallidos por datos offline

Tanto en 220 como en 221 aparecieron **6 FAIL**, idénticos en ambos batches:

- `GW170817`
- `GW190426_152155`
- `GW190531_023648`
- `GW191126_115259`
- `GW200201_203549`
- `GW200311_103121`

### 4.1. Clasificación final de los 6 FAIL

#### A) Datos corruptos

- **GW170817**
  - `RUN_VALID = FAIL`
  - `stage = s1_fetch_strain`
  - razón: HDF5 truncado/corrupto

#### B) Ausencia de inputs offline válidos

- **GW190426_152155**
  - `s0_oracle_mvp FAIL`
  - razón: `Offline policy active (--require-offline) but no valid --local-hdf5 inputs were provided`

- **GW190531_023648**
  - `data/losc/GW190531_023648` ausente

- **GW200201_203549**
  - `data/losc/GW200201_203549` ausente

- **GW200311_103121**
  - `data/losc/GW200311_103121` ausente

- **GW191126_115259**
  - `s0_oracle_mvp FAIL`
  - `parameters.local_hdf5 = []`
  - no entró con inputs offline válidos en el run

### 4.2. Conclusión de datos offline

De los 6 FAIL:

- **1** por archivo corrupto (`GW170817`)
- **5** por ausencia de inputs offline válidos o completos

No se abordó hoy la reparación de esos inputs. Se decidió continuar la auditoría con la cohorte efectiva de 38 eventos válidos.

---

## 5. Fase 3 — soporte físico común por `phys_key`

### 5.1. Ejecución

**Host run:** `mvp_GW150914_221info_20260320T105521Z`

Comando ejecutado:

```bash
python -m mvp.experiment_phase3_physkey_common \
  --run-id mvp_GW150914_221info_20260320T105521Z \
  --batch-220 batch_with_t0_220_eps2500_20260320T113433Z \
  --batch-221 batch_with_t0_221_eps2500_20260320T114207Z
```

Artefactos principales:

- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase3_physkey_common/stage_summary.json`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase3_physkey_common/outputs/summary_physkey_common.json`

### 5.2. Resultado

Hallazgos clave:

- `n_events_valid_220 = 38`
- `n_events_valid_221 = 38`
- `n_common_events = 38`
- `n_empty_intersection_events = 0`
- `empty_intersection_events = []`
- `n_non_subset_cases = 38`

La definición correcta de intersección usada fue:

```text
phys_key = (family, provenance, M_solar, chi)
```

no `geometry_id` literal.

### 5.3. Lectura científica

- Existe soporte físico común `220∩221` en toda la cohorte válida.
- No se vacía ningún evento en la intersección.
- La semántica histórica `K220 ⊂ K221` ya no debe asumirse; la cohorte nueva se comporta al revés.

---

## 6. Fase 4 / E5 — Hawking sobre soporte común

### 6.1. Ejecución

Comando ejecutado:

```bash
python -m mvp.experiment_phase4_hawking_area_common_support \
  --host-run mvp_GW150914_221info_20260320T105521Z \
  --batch-220 batch_with_t0_220_eps2500_20260320T113433Z \
  --batch-221 batch_with_t0_221_eps2500_20260320T114207Z
```

Artefactos principales:

- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase4_hawking_area_common_support/stage_summary.json`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase4_hawking_area_common_support/outputs/hawking_area_summary.json`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase4_hawking_area_common_support/outputs/per_event_common_support.csv`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase4_hawking_area_common_support/outputs/per_event_hawking_area.csv`

### 6.2. Resultado contractual

- `stage_summary.json -> verdict: PASS`

### 6.3. Resultado científico agregado

De `hawking_area_summary.json`:

- `n_events_with_nonempty_hawking = 38`
- `n_rows_input_common = 30400`
- `n_rows_hawking_pass = 30400`
- `n_rows_hawking_fail = 0`

### 6.4. Resultado per-evento

De `per_event_common_support.csv`:

- `n_k220 = 851`
- `n_k221 = 800`
- `n_k_common = 800`
- `n_k_hawking = 800`

Verificación completa sobre los 38 eventos:

- `n_k221 != n_k_common: 0`
- `n_k_common != n_k_hawking: 0`

### 6.5. Lectura correcta

Esto demuestra que:

```text
K_common = K220 ∩ K221 = K221
K_hawking = K_common
```

para la cohorte rehecha actual.

Por tanto:

- **el 221 sí discrimina** respecto a 220;
- **Hawking no discrimina adicionalmente** en esta corrida;
- Hawking **valida compatibilidad física**, pero no añade recorte efectivo del soporte.

---

## 7. Conclusión científica y de ingeniería del día

### 7.1. Conclusión científica

La cohorte canónica rehecha muestra que:

1. El `221` no es ruido inútil a nivel poblacional.
2. El `221` sí reduce el soporte geométrico respecto a `220`.
3. El soporte común `220∩221` no entra en conflicto con la restricción de Hawking.
4. En esta cohorte actual, Hawking no discrimina adicionalmente.

### 7.2. Conclusión de ingeniería

Esto justifica cambiar el pipeline en la forma conservadora correcta:

- **si** `s3b_multimode_estimates.multimode_viability.class == MULTIMODE_OK`
  - promover la rama explícita:
    - `s4g_mode220_geometry_filter`
    - `s4h_mode221_geometry_filter`
    - `s4i_common_geometry_intersection`
    - `s4f_area_observation`
    - `s4j_hawking_area_filter`
    - `s4k_event_support_region`
- **si no**
  - mantener el fallback actual monomodo / conservador

Matiz importante que debe quedar escrito en el cambio:

```text
En el estado actual, la reducción efectiva del soporte proviene de 221;
s4j_hawking_area_filter valida compatibilidad física, pero no recorta la región.
```

---

## 8. Artefactos canónicos fijados hoy

### 8.1. Batches canónicos

- `batch_with_t0_220_eps2500_20260320T113433Z`
- `batch_with_t0_221_eps2500_20260320T114207Z`

### 8.2. Host run de análisis

- `mvp_GW150914_221info_20260320T105521Z`

### 8.3. Experimentos ejecutados en el host run

- `experiment/phase3_physkey_common/`
- `experiment/phase4_hawking_area_common_support/`

---

## 9. Qué queda pendiente

### 9.1. Reparación de inputs offline

No se ha resuelto aún la disponibilidad/corrupción de datos de estos eventos:

- `GW170817`
- `GW190426_152155`
- `GW190531_023648`
- `GW191126_115259`
- `GW200201_203549`
- `GW200311_103121`

Esto no bloqueó la auditoría de la cohorte efectiva, pero sí deja pendiente una futura rehechura de cohorte completa si se quiere incorporar esos seis casos.

### 9.2. Formalizar el cambio de pipeline

Pendiente redactar y aplicar el parche de orquestación para promover condicionalmente la rama multimodo cuando `MULTIMODE_OK`.

---

## 10. Plan previsto para el próximo día

### Objetivo principal

Formalizar el cambio de pipeline usando la evidencia de hoy como criterio de aceptación.

### 10.1. Tarea 1 — parche de orquestación

Preparar un parche mínimo sobre `mvp/pipeline.py` que haga lo siguiente:

- si `multimode_viability.class == MULTIMODE_OK`, usar como artefacto preferente downstream la rama:
  - `s4g -> s4h -> s4i -> s4f -> s4j -> s4k`
- si no, mantener el fallback actual

### 10.2. Tarea 2 — tests de regresión

Añadir o reforzar tests que prueben:

- `MULTIMODE_OK` → promoción de rama explícita
- no `MULTIMODE_OK` → fallback conservador
- no se rompe `single`
- no se rompe la agregación condicionada por multimodo

### 10.3. Tarea 3 — documentar el significado físico correcto

Dejar explícito en documentación técnica / PR summary:

- `221` es el discriminante efectivo del soporte
- Hawking valida compatibilidad física, pero no recorta en la cohorte rehecha actual

### 10.4. Tarea 4 — dejar anotado backlog de datos offline

Crear nota técnica separada con los 6 eventos pendientes por inputs offline, para no mezclar ese frente con el cambio de routing multimodo.

---

## 11. Recomendación final

Sí: **vamos por buen camino**.

Lo importante de hoy no ha sido solo que “el 221 salga”, sino que ya se ha demostrado de forma auditable que:

- la cohorte rehecha 220/221 es estable;
- el `221` aporta recorte efectivo;
- el soporte común resultante es compatible con Hawking;
- y eso ya da base empírica suficiente para promover el carril multimodo en el pipeline, con fallback conservador cuando no haya `MULTIMODE_OK`.

