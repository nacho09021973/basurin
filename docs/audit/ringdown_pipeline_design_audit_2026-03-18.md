# Auditoría de diseño — Pipeline de Ringdown (220/221)

**Fecha:** 2026-03-18
**Alcance:** Pipeline s1 → s2 → s3 → s3b → experimentos downstream (qnm_221_literature_check, t0_sweep, s4d_kerr_from_multimode, s5_event_row)
**Versión:** Basada en el estado actual del repo (`mvp/contracts.py`, `mvp/s3b_multimode_estimates.py`, `mvp/experiment_qnm_221_literature_check.py`, `gwtc_events_t0.json`)

---

## 1. Resumen ejecutivo

El pipeline de ringdown de BASURIN tiene una arquitectura contract-first sólida en su
diseño de stages (init/check_inputs/finalize/abort, SHA256, IO determinista). Sin
embargo, la auditoría identifica **7 fallos de diseño** que no son bugs aislados sino
deficiencias arquitectónicas que afectan la fiabilidad del canal 221 y, por extensión,
cualquier afirmación física derivada del overtone.

Los fallos se agrupan en tres ejes:

1. **Contratos de datos frágiles**: el catálogo canónico admite formatos heterogéneos
   sin validación de schema; los artefactos de s3b no tienen schema formal validable
   por consumidores.
2. **Propagación de calidad incompleta**: los quality_flags y el verdict upstream de s3b
   no se honran obligatoriamente; un consumidor puede extraer valores numéricos de un
   modo marcado como INSUFFICIENT_DATA.
3. **Confusión semántica entre ejecución y ciencia**: `stage_summary.verdict = "PASS"`
   coexiste con `results.verdict = "REJECTED"`, y la frontera entre "hay un número" y
   "hay una medición válida" no está formalizada.

---

## 2. Hallazgos críticos

### H1 — Ausencia de gate obligatorio upstream en consumidores del 221

**Categoría:** Falta de propagación de calidad upstream
**Severidad:** CRÍTICO

**Archivo/función afectada:**
- `mvp/experiment_qnm_221_literature_check.py:103-160` — `_extract_221_from_multimode()`

**Artefacto afectado:**
- `multimode_estimates.json` → `modes[].fit.stability.lnf_p50`, `lnQ_p50`

**Evidencia concreta:**
La función `_extract_221_from_multimode()` sí comprueba `_upstream_221_block_reason()` (línea 106), que detecta `quality_flags` que empiecen por `221_` y `results.verdict != "PASS"`. Sin embargo:

1. El filtro en `_upstream_221_block_reason()` (línea 98) compara `verdict != "PASS"`, pero el verdict canónico de s3b cuando el 220 funciona pero el 221 no es `"OK"` (no `"PASS"`). Es decir, un `verdict = "OK"` con `quality_flags = ["221_cv_Q_explosive", "221_lnQ_span_explosive"]` **sí se bloquea** por la rama de flags, pero solo por coincidencia de naming convention, no por contrato formal.

2. Si un futuro flag del 221 no empieza por `"221_"` (por ejemplo, `"overtone_insufficient"` o `"mode_1_unstable"`), el filtro lo ignoraría y dejaría pasar valores numéricos de un modo inestable.

3. No existe un campo booleano canónico `mode_221_usable: bool` en el schema de `multimode_estimates_v1`. El consumidor tiene que parsear flags textuales y adivinar semántica.

**Riesgo:**
- **Falso positivo físico**: un 221 con `lnQ_span_explosive` podría pasar Gate A Kerr si sus p50 caen accidentalmente cerca de la predicción Kerr. El pipeline reportaría `KERR_COMPATIBLE` para un modo que upstream ya marcó como inestable.
- **Reproducibilidad débil**: la política de bloqueo depende de convención de naming de flags, no de contrato.

**Cambio mínimo recomendado:**
1. Añadir a `multimode_estimates_v1` un campo canónico por modo:
   ```json
   {
     "mode_221_usable": false,
     "mode_221_usable_reason": "221_lnQ_span_explosive"
   }
   ```
2. En `_upstream_221_block_reason()`, comprobar `mode_221_usable == false` en lugar de parsear flags textuales.
3. En `s3b_multimode_estimates.py`, calcular `mode_221_usable` directamente desde el resultado de `evaluate_mode()` (que ya devuelve `ok: bool`).

**Test de regresión:**
- Test que construye un `multimode_estimates.json` con `results.verdict = "OK"`, `quality_flags = []`, pero `mode_221.fit.stability.lnQ_span = 5.0` (explosivo) y `mode_221_usable = false`. Verificar que el experimento 221 devuelve `INSUFFICIENT_DATA`, no `KERR_COMPATIBLE`.

---

### H2 — Catálogo canónico con schema heterogéneo (scalar vs object)

**Categoría:** Contratos de datos frágiles / Schema drift
**Severidad:** CRÍTICO

**Archivo/función afectada:**
- `gwtc_events_t0.json:842` — entrada `"GW250114_082203": 1420878141.2`
- `mvp/s1_fetch_strain.py:136-150` — `_extract_gps_from_catalog_entry()`

**Artefacto afectado:**
- Catálogo canónico `gwtc_events_t0.json`

**Evidencia concreta:**
Todas las entradas de GWTC (GW170817 a GW200316) son objetos `dict` con campos tipados (`GPS`, `Mf`, `chirp_mass`, `detectors`, etc.). La entrada `GW250114_082203` es un escalar `float` (solo GPS). Esto significa que:

1. `_extract_gps_from_catalog_entry()` necesitó una rama especial (`elif isinstance(entry, (int, float))`) para no fallar.
2. Cualquier consumidor downstream que necesite `Mf`, `chi_eff`, o `detectors` de este evento recibirá `None` sin warning.
3. El stage `s4d_kerr_from_multimode` necesita `Mf` y `af` para predicción Kerr; si no están en el catálogo, se recurre a fallbacks con heurística de descubrimiento de remnant (`_discover_remnant_path()`).

**Riesgo:**
- **Run aparentemente válido con semántica incompleta**: s1 pasa con `verdict = "PASS"`, pero el run no tiene metadata canónica del evento. Stages que dependan del catálogo (no solo de strain) fallarán más tarde con errores crípticos.
- **Mantenimiento peligroso**: cada nuevo evento O3b/O4 podría añadirse como scalar por inercia, degradando silenciosamente el pipeline.

**Cambio mínimo recomendado:**
1. Formalizar un JSON Schema para `gwtc_events_t0.json` que requiera al mínimo `{GPS: number, Mf: number|null}`.
2. Añadir un test CI (`test_catalog_schema.py`) que valide todas las entradas del catálogo contra el schema.
3. Normalizar `GW250114_082203` a formato objeto: `{"GPS": 1420878141.2, "Mf": null, ...}`.
4. Opcionalmente, hacer que `_extract_gps_from_catalog_entry()` emita un warning cuando recibe un escalar, indicando que la entrada es incompleta.

**Test de regresión:**
- Test que carga `gwtc_events_t0.json` y verifica que toda entrada sea `dict` con key `"GPS"` numérica. Falla si alguna entrada es escalar.

---

## 3. Hallazgos altos

### H3 — Colisión semántica entre verdict de ejecución y verdict científico

**Categoría:** Semántica inconsistente de verdict/status
**Severidad:** ALTO

**Archivos afectados:**
- `mvp/experiment_qnm_221_literature_check.py:829` — `"verdict": "PASS"` en `stage_summary_payload`
- `mvp/experiment_qnm_221_literature_check.py:833` — `"verdict": verdict` en `results` (puede ser `"REJECTED"`)
- `mvp/s3b_multimode_estimates.py:1010` — `verdict = "OK" if mode_220_ok else "INSUFFICIENT_DATA"`

**Evidencia concreta:**
En `experiment_qnm_221_literature_check.py`, la línea 829 escribe:
```python
stage_summary_payload = {
    ...
    "verdict": "PASS",          # ejecución contractual del stage
    "results": {
        "verdict": verdict,     # KERR_COMPATIBLE | WEAK_EVIDENCE | REJECTED | INSUFFICIENT_DATA
        ...
    },
}
```

Un `stage_summary.verdict = "PASS"` + `results.verdict = "REJECTED"` es contractualmente correcto (el stage se ejecutó bien y determinó que el 221 no es compatible), pero:

1. Cualquier automatización que haga `if verdict == "PASS": ...` sobre `stage_summary` sin distinguir el nivel de anidación promoverá runs rechazados.
2. En s3b, el `results.verdict` es `"OK"` o `"INSUFFICIENT_DATA"`, mientras que en el experimento 221 es `"KERR_COMPATIBLE"` / `"REJECTED"` / etc. No hay enum compartido.
3. `s5_event_row.py:95` propaga `results.verdict` al event_row sin transformación, mezclando vocabularios.

**Riesgo:**
- **Automatización downstream rota**: un script que filtre por `verdict == "PASS"` incluiría runs cuyo 221 fue rechazado.
- **Auditoría difícil**: un revisor humano necesita saber que "PASS" a nivel de stage ≠ "resultado físico positivo".

**Cambio mínimo recomendado:**
1. Renombrar el campo de ejecución a `stage_execution_status: "COMPLETED" | "ABORTED"` en `stage_summary`.
2. Reservar `verdict` exclusivamente para conclusiones científicas.
3. Alternativa mínima: documentar formalmente en `contracts.py` que `stage_summary.verdict` es ejecución y `results.verdict` es ciencia, y añadir un assert en `finalize()` que impida `results.verdict = "PASS"` (para evitar la confusión inversa).

**Test de regresión:**
- Test que escanee todos los `stage_summary.json` de un run golden y verifique que `stage_summary.verdict ∈ {"PASS", "FAIL"}` y que `results.verdict ∉ {"PASS", "FAIL"}` (forzando vocabularios disjuntos).

---

### H4 — Schema implícito de `multimode_estimates_v1`: lista `modes[]` sin contrato formal

**Categoría:** Schema drift / API interna implícita
**Severidad:** ALTO

**Archivos afectados:**
- `mvp/s3b_multimode_estimates.py` — produce `modes[]`
- `mvp/experiment_qnm_221_literature_check.py:75-84` — `_extract_mode_dict_candidates()` (búsqueda recursiva)
- `mvp/s5_event_row.py:70-97` — itera `modes[]` buscando `label == "220"` / `"221"`
- `tests/fixtures/qnm_221_literature_check/multimode_estimates.real_schema.json` — fixture de referencia

**Evidencia concreta:**
No existe un JSON Schema formal para `multimode_estimates_v1`. La estructura se define implícitamente por el código de s3b y se documenta parcialmente en `docs/schemas_json_v1_multimode.md`. Consecuencias:

1. `_extract_mode_dict_candidates()` (línea 75) hace una **búsqueda recursiva** por todo el árbol JSON para encontrar dicts que parezcan modos. Esto significa que cualquier dict anidado con `label: "221"` sería candidato, incluyendo metadata de debug.
2. La fixture `multimode_estimates.real_schema.json` muestra modos con `ln_f: null`, `ln_Q: null`, `Sigma: null` — es decir, el schema permite modos completamente vacíos. Un consumidor que no compruebe `null` extraerá NaN.
3. No hay un campo `l`, `m`, `n` garantizado en cada modo; el matching depende de `label ∈ {"221", "(2,2,1)"}` (texto) **o** `l==2, m==2, n==1` (numérico). Si un productor cambia el formato de label, el consumidor no lo encontraría.

**Riesgo:**
- **Schema drift silencioso**: un cambio en s3b que renombre `label` o reestructure `modes[]` romperá consumidores sin error de compilación.
- **Falso positivo por match accidental**: la búsqueda recursiva podría matchear un dict de metadata que casualmente tenga `label: "221"`.

**Cambio mínimo recomendado:**
1. Crear `mvp/schemas/multimode_estimates_v1.schema.json` con:
   - `modes`: array de objetos con `mode: [int,int,int]` (requerido), `label: string` (requerido), `ln_f: number|null`, `ln_Q: number|null`, `Sigma: array|null`, `fit.stability.valid_fraction: number`.
   - `results.verdict: enum["OK","INSUFFICIENT_DATA"]`
   - `results.quality_flags: array<string>`
2. En `s3b`, validar el payload de salida contra el schema antes de `write_json_atomic()`.
3. En consumidores, usar lookup directo por `mode == [2,2,1]` en vez de búsqueda recursiva.

**Test de regresión:**
- Test que valide la fixture `multimode_estimates.real_schema.json` contra el nuevo JSON Schema formal. Falla si hay drift.

---

### H5 — Política de extracción `fit.stability.p50` demasiado agresiva para 221

**Categoría:** Mezcla de evidencia estadística con decisión física
**Severidad:** ALTO

**Archivos afectados:**
- `mvp/experiment_qnm_221_literature_check.py:118-125` — extracción de `lnf_p50`/`lnQ_p50`
- `mvp/s3b_multimode_estimates.py:650-661` — `can_materialize_point`

**Evidencia concreta:**
La función `_extract_221_from_multimode()` (línea 120-125) extrae `fit.stability.lnf_p50` y `lnQ_p50` y los usa para calcular `f221` y `tau221`:

```python
lnf_p50 = _safe_float(stability.get("lnf_p50"))
lnq_p50 = _safe_float(stability.get("lnQ_p50"))
if lnf_p50 is not None and lnq_p50 is not None:
    f221 = math.exp(lnf_p50)
    tau221 = math.exp(lnq_p50 - lnf_p50) / math.pi
    return f221, tau221, "mode_221.fit.stability.p50"
```

Mientras que en s3b (línea 650-658), `can_materialize_point` requiere:
- `samples.shape[0] >= min_point_samples` (default 2)
- `valid_fraction >= min_point_valid_fraction` (default 0.0)

Esto significa que con solo 2 muestras bootstrap válidas de 200, y valid_fraction = 0.01, se materializa un `lnf_p50` que será la mediana de 2 puntos. El consumidor luego lo usa como "la frecuencia del 221" para Gate A Kerr.

En el caso del 221, las distribuciones bootstrap son frecuentemente multimodales o tiene colas explosivas (como evidencian `221_lnQ_span_explosive`, `221_cv_Q_explosive`). Usar la mediana de una distribución con span > 4.0 en log-space como "medición" es estadísticamente no informativo.

**Riesgo:**
- **Confusión "hay un número" vs "hay una medición válida"**: el pipeline produce una `f221_measured` incluso cuando la posterior del 221 es inútil. Downstream, `Gate A Kerr` compara este número contra la predicción Kerr y podría dar `PASS` por coincidencia.
- **Falso positivo físico** en eventos donde el 221 no es detectable.

**Cambio mínimo recomendado:**
1. Añadir a `multimode_estimates_v1` un campo `mode_221_measurement_quality: "RELIABLE" | "MARGINAL" | "NOT_USABLE"` basado en umbrales combinados de `valid_fraction`, `lnQ_span`, `cv_Q`, y el bool `ok` de `evaluate_mode()`.
2. En `_extract_221_from_multimode()`, devolver `(None, None, "mode_221_not_usable")` si `measurement_quality == "NOT_USABLE"`, independientemente de que existan p50.
3. Subir `min_point_valid_fraction` del 221 a un valor razonable (e.g., 0.30) como barrera de materialización.

**Test de regresión:**
- Test con fixture donde `mode_221.fit.stability.valid_fraction = 0.02`, `lnQ_span = 5.0`, pero `lnf_p50` y `lnQ_p50` existen. Verificar que el consumidor devuelve `INSUFFICIENT_DATA`.

---

## 4. Hallazgos medios

### H6 — Fallbacks de descubrimiento que enmascaran errores

**Categoría:** Políticas de fallback peligrosas
**Severidad:** MEDIO

**Archivos afectados:**
- `mvp/experiment_qnm_221_literature_check.py:188-217` — `_discover_remnant_path()`: 5 candidatos JSON + HDF5 heurístico
- `mvp/s1_fetch_strain.py:153-203` — `_fetch_gps_center()`: catálogo → GWOSC API fallback
- `mvp/s3b_multimode_estimates.py:1288-1299` — fallback de PSD a Welch interno

**Evidencia concreta:**
El pipeline tiene múltiples cadenas de fallback silenciosas:

1. **Remnant discovery** (experiment_qnm_221_literature_check.py:188-217): busca en 5 rutas de JSON canónico + cualquier HDF5 en `gwtc_posteriors/raw/`. Si hay 3 archivos HDF5, toma el primero cuyo nombre contenga el event_id. Si no, toma `h5_candidates[0]`. No hay validación de que el archivo seleccionado contenga realmente posterior samples del evento correcto.

2. **GPS center** (s1_fetch_strain.py): si el catálogo local no tiene el evento, cae silenciosamente al API GWOSC. Si el API falla, sí abort. Pero el fallback al API introduce una dependencia de red no declarada en el contrato.

3. **PSD fallback** (s3b): si `--psd-path` falla o el detector no se puede inferir, usa Welch interno. La estimación espectral cambia significativamente entre PSD medida y Welch, pero la flag es solo un print, no un `quality_flag` en el artefacto.

**Riesgo:**
- **Run no reproducible**: el mismo comando con diferente estado de red o filesystem produce resultados distintos.
- **Errores enmascarados**: un remnant HDF5 incorrecto pasa sin detección.

**Cambio mínimo recomendado:**
1. Hacer que cada fallback añada un `quality_flag` al artefacto final (e.g., `"remnant_fallback_hdf5_heuristic"`, `"gps_resolved_via_api"`, `"psd_fallback_internal_welch"`).
2. En `_discover_remnant_path()`, si se usa el HDF5 heurístico, registrar el path y hash en `provenance` del stage.
3. Considerar hacer que el fallback de PSD sea un `quality_flag` en `multimode_estimates.json`, no solo un print.

**Test de regresión:**
- Test de integración que ejecuta s3b sin `--psd-path` y verifica que `quality_flags` contiene `"psd_fallback_internal_welch"`.

---

### H7 — Ausencia de distinción formal entre señal principal, overtone marginal, y medida no usable

**Categoría:** Observabilidad insuficiente / Acoplamiento excesivo
**Severidad:** MEDIO

**Archivos afectados:**
- `mvp/s3b_multimode_estimates.py:36-38` — `TARGET_MODES` trata 220 y 221 como array homogéneo
- `mvp/multimode_viability.py:12-25` — umbrales diferenciados pero no formalizados como "tier"

**Evidencia concreta:**
`TARGET_MODES = [{"mode": [2,2,0], "label": "220"}, {"mode": [2,2,1], "label": "221"}]` y ambos modos se procesan con el mismo `evaluate_mode()`. La diferenciación se hace post-hoc en umbrales (`min_valid_fraction_221` vs `min_valid_fraction_220`) y en flags, pero no hay un concepto formal de **tier de modo** que gobierne:

1. Si un modo es "señal principal gobernable" (220) vs "overtone marginal" (221).
2. Qué umbrales mínimos de materialización aplican a cada tier.
3. Si un downstream puede usar el punto estimado como "medida" vs "upper bound" vs "no informativo".

El resultado es que `multimode_viability.py` tiene 12 thresholds hardcodeados con nombres que implícitamente codifican la asimetría 220/221, pero esto no es un contrato formal.

**Riesgo:**
- Si se añade un modo 330 o 320, habría que duplicar toda la lógica de thresholds sin guía arquitectónica.
- Los umbrales del 221 son actualmente permisivos (`MIN_VALID_FRAC_221 = 0.30`, `min_point_valid_fraction = 0.0`), lo que permite materializar puntos estadísticamente vacíos.

**Cambio mínimo recomendado:**
1. Formalizar un enum de tier: `PRIMARY | OVERTONE | EXPERIMENTAL`.
2. Asociar a cada tier umbrales mínimos de materialización y gating.
3. Hacer que `evaluate_mode()` acepte un `tier: str` que determine los defaults.

**Test de regresión:**
- Test parametrizado que verifica que para tier=OVERTONE, `min_point_valid_fraction >= 0.20` y `max_lnq_span <= 2.0`.

---

## 5. Cambios mínimos recomendados (resumen consolidado)

| Prioridad | Hallazgo | Cambio mínimo | Esfuerzo |
|-----------|----------|---------------|----------|
| P0 | H1 — Gate obligatorio | Añadir `mode_221_usable: bool` a schema s3b; consumidores lo honran | Bajo |
| P0 | H2 — Catálogo heterogéneo | JSON Schema + test CI + normalizar GW250114 | Bajo |
| P1 | H3 — Colisión de verdict | Documentar formalmente en contracts.py; a medio plazo, renombrar a `stage_execution_status` | Bajo-Medio |
| P1 | H4 — Schema implícito modes[] | Crear JSON Schema formal para `multimode_estimates_v1` | Medio |
| P1 | H5 — p50 agresivo para 221 | Añadir `measurement_quality` y subir `min_point_valid_fraction` del 221 | Bajo |
| P2 | H6 — Fallbacks silenciosos | Añadir quality_flags por cada fallback activado | Bajo |
| P2 | H7 — Tiers de modo | Formalizar tier enum; no bloquea operativa actual | Medio |

---

## 6. Tests de regresión prioritarios

1. **test_catalog_schema_homogeneous**: Toda entrada de `gwtc_events_t0.json` es `dict` con `GPS: number`. (Cierra H2)

2. **test_221_unusable_blocks_downstream**: Fixture con `mode_221_usable = false` pero `lnf_p50` presente → experimento 221 devuelve `INSUFFICIENT_DATA`. (Cierra H1)

3. **test_verdict_vocabulary_disjoint**: Recorre stage_summaries de un run golden; `stage_summary.verdict ∈ {"PASS","FAIL"}`; `results.verdict ∉ {"PASS","FAIL"}`. (Cierra H3)

4. **test_multimode_estimates_schema_validation**: Valida artefactos de s3b contra `multimode_estimates_v1.schema.json`. (Cierra H4)

5. **test_221_marginal_not_materialized**: `valid_fraction = 0.02`, `lnQ_span = 5.0` → `f221 = None`. (Cierra H5)

6. **test_fallback_flags_propagated**: Ejecución de s3b sin PSD externa → `quality_flags` contiene flag de fallback. (Cierra H6)

---

## 7. Riesgos si no se corrige

### Riesgo inmediato (H1 + H5)
Un run de un evento O4 donde el 221 no es detectable podría producir:
- `s3b: verdict = "OK"` (porque el 220 sí funciona)
- `mode_221: lnf_p50 = 5.8` (mediana de 3 muestras bootstrap)
- `quality_flags: ["221_cv_Q_explosive", "221_lnQ_span_explosive"]`
- `experiment_qnm_221_literature_check: verdict = "KERR_COMPATIBLE"` (si los p50 caen por azar cerca de Kerr)

Esto sería un **falso positivo físico**: el pipeline afirma compatibilidad Kerr del overtone basándose en una "medición" estadísticamente vacía.

### Riesgo de mantenimiento (H2 + H4)
Cada nuevo evento O4 añadido como escalar al catálogo y cada cambio en la estructura de `multimode_estimates.json` son bombas de tiempo sin validación de schema.

### Riesgo de auditoría (H3)
Un revisor externo que examine `stage_summary.verdict = "PASS"` + `results.verdict = "REJECTED"` sin conocimiento del sistema asumirá inconsistencia. En un contexto de paper o revisión científica, esto erosiona la credibilidad del pipeline.

### Riesgo de reproducibilidad (H6)
Runs que dependen de fallbacks de red (GPS via API) o heurísticas de filesystem (remnant HDF5) no son completamente reproducibles. Esto viola el principio de IO determinista que BASURIN declara.
