# README — Validación post-hoc del modo 221 en BASURIN
## Estado
**Resultado final:** validación **no cerrada**.
**Conclusión operativa:** en el estado actual del flujo, **no hemos podido validar los 221** y el intento ha quedado **bloqueado por integración/contracts upstream**, no por falta de tests del experimento ni por un problema aislado del extractor 221.
La formulación rigurosa es:
- **no** hemos demostrado coherencia Kerr del 221;
- **no** hemos demostrado una detección robusta del 221;
- el pipeline/evidencia disponible nos deja, de forma repetida, en **`INSUFFICIENT_DATA`** o en carriles internos **`SINGLEMODE_ONLY` / `INCONCLUSIVE`**;
- por tanto, **se deja el intento por imposible en el estado actual del repositorio/flujo**, salvo que se repare la cadena upstream de materialización del remanente canónico.
---
## Objetivo que se intentó validar
Se implementó y ejecutó un experimento post-hoc para evaluar si el supuesto modo **221** era aceptable en el siguiente sentido estricto:
1. **Compatibilidad con Kerr** para un remanente parametrizado por `(Mf, af)`.
2. **Estabilidad frente a variación de `t0`**.
3. **Mejora de ajuste / model selection** frente a `220` solo, sin sobreafirmar detección.
El veredicto del experimento se definió como uno de:
- `KERR_COMPATIBLE`
- `WEAK_EVIDENCE`
- `REJECTED`
- `INSUFFICIENT_DATA`
---
## Qué se implementó
Se añadió el experimento:
- `mvp/experiment_qnm_221_literature_check.py`
con salida bajo:
- `runs/<run_id>/experiment/qnm_221_literature_check/`
Artefactos producidos por el experimento:
- `outputs/kerr_oracle_221.json`
- `outputs/t0_stability_221.csv`
- `outputs/model_selection_220_vs_220221.csv`
- `outputs/summary_221_validation.json`
- `manifest.json`
- `stage_summary.json`
También se añadieron tests de regresión en:
- `tests/test_experiment_qnm_221_literature_check.py`
Cobertura efectiva alcanzada en tests:
- no escritura fuera de `runs/<run_id>/experiment/qnm_221_literature_check/`
- abort si `RUN_VALID != PASS`
- presencia de campos mínimos en `summary_221_validation.json`
- dominio cerrado de `verdict`
- compatibilidad del extractor 221 con schema real de `s3b`
- extracción de `Mf/af` desde HDF5 externo
- degradación limpia si el HDF5 no contiene `final_spin`
Resultado de tests alcanzado durante la sesión:
- `7 passed in 2.05s`
---
## Qué se ejecutó realmente
### 1. Selección del conjunto de eventos
No se corrió sobre "todo `data/losc`" sin filtrar.
Se construyó un lote auditable mediante intersecciones sucesivas:
1. eventos **visibles** en `data/losc`
2. eventos presentes en `gwtc_quality_events.csv`
3. eventos presentes en `events_with_t0.txt`
Artefacto final de lote:
- `runs/prep_batch_final_20260318T110223Z/external_inputs/events_batch_final.txt`
Cardinalidad final:
- **24 eventos**
### 2. Ejecución batch
Para cada evento del lote se ejecutó:
```bash
python -m mvp.pipeline multimode \
  --event-id <EVENT_ID> \
  --atlas-default \
  --run-id <RUN_ID> \
  --s3b-method spectral_two_pass
python -m mvp.experiment_qnm_221_literature_check --run-id <RUN_ID>
```
Y se agregaron resúmenes en:
- `runs/prep_batch_final_20260318T110223Z/outputs/qnm_221_batch_summary.csv`
### 3. Run centinela analizado en detalle
Se inspeccionó de forma manual y repetida el caso:
- `mvp_GW190521_221check_20260318T080000Z`
- posteriormente `mvp_GW190521_221rerun_20260318T113343Z`
En ese caso se observó:
- `s8a_family_gr_kerr` → `assessment = INCONCLUSIVE`
- `s4c_kerr_consistency` → `status = SKIPPED_MULTIMODE_GATE`
- `multimode_viability.class = SINGLEMODE_ONLY`
- razón explícita:
  - `mode_221_ok=false: overtone posterior not usable for multimode inference`
Eso ya indicaba que el 221 **no era usable para inferencia multimodo** en ese caso, incluso antes de cerrar Gate A.
---
## Qué se encontró
### Hallazgo 1 — El experimento 221 funciona como artefacto post-hoc
El experimento:
- corre,
- respeta IO determinista,
- aborta correctamente con `RUN_VALID != PASS`,
- escribe artefactos auditables,
- y produce veredictos consistentes con los datos que ve.
Es decir:
- **el problema no está en que el experimento no funcione**.
### Hallazgo 2 — El batch inicial devolvió 24/24 `INSUFFICIENT_DATA`
En el agregado batch se obtuvo:
- `N = 24`
- `verdict_counts = {'INSUFFICIENT_DATA': 24}`
- `reason_counts = {'gate_a_not_available:missing_canonical_mf_af': 24}`
Eso mostraba que **Gate A estaba sistemáticamente bloqueado**.
### Hallazgo 3 — Sí existe fuente externa real de remanente
Se auditó el repositorio y sí aparecieron fuentes reales, no circulares, del remanente:
- HDF5 externos bajo `external/.../*.h5`
- JSON canónicos ya materializados en otros runs, por ejemplo:
  - `runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/GW190521.json`
  - `runs/host_phase3_physkey_20260316T001000Z/external_inputs/gwtc_posteriors/GW190521.json`
Además, se verificó que hay utilidades en el repo para trabajar con ese material:
- `mvp/experiment_gwtc_posteriors_fetch.py`
- `mvp/experiment_phase4b_convert_imr_posteriors.py`
- `mvp/experiment_phase4b_prepare_gwtc_posteriors_input.py`
### Hallazgo 4 — El experimento 221 fue ampliado para intentar descubrir el remanente
Se añadió lógica para:
- buscar `external_inputs/gwtc_posteriors/raw/*.h5`
- leer `posterior_samples.final_mass`
- leer `posterior_samples.final_spin`
- usar `p50` como estimador robusto de `Mf` y `af`
Esto quedó respaldado por tests y por un caso real aislado donde se reportó un resultado del tipo:
- `mf_source = external_inputs/gwtc_posteriors/raw/...h5`
- `f221_kerr != null`
- `tau221_kerr != null`
- `verdict = REJECTED`
Es decir:
- **la lógica de extracción existe y es viable en principio**.
### Hallazgo 5 — El flujo real `multimode` no materializa `external_inputs/...` para los runs nuevos
Aquí apareció el bloqueo decisivo.
En runs nuevos rehechos desde cero, por ejemplo:
- `mvp_GW190521_221rerun_20260318T113343Z`
se comprobó que:
- `runs/<run_id>/external_inputs` **no existe**
- por tanto no existe:
  - `external_inputs/gwtc_posteriors/`
  - `external_inputs/gwtc_posteriors/raw/`
  - `external_inputs/gwtc_posteriors/<EVENT>.json`
y el summary seguía dando:
- `mf_source = null`
- `af_source = null`
- `verdict = INSUFFICIENT_DATA`
- `verdict_reason = gate_a_not_available:missing_canonical_mf_af`
### Hallazgo 6 — El fetcher tampoco resuelve el problema por sí solo
Se intentó usar:
```bash
python -m mvp.experiment_gwtc_posteriors_fetch --run-id <RUN_ID>
```
Primero falló por faltar:
- `external_inputs/gwtc_posteriors/required_events.txt`
Después, al materializar ese fichero, falló por faltar:
- `external_inputs/gwtc_posteriors/GW190521.json`
Es decir:
- el fetcher no estaba funcionando aquí como "descargador y convertidor integral" del input canónico,
- sino como validador/organizador de una estructura que **ya debería existir**.
### Hallazgo 7 — El conversor existe, pero introducirlo ya no era un cambio mínimo del flujo de validación 221
`experiment_phase4b_convert_imr_posteriors` sí parece ser la utilidad que escribe el JSON canónico a partir de un HDF5 real, pero requiere:
- `--source-dir`
- `--input-format hdf5`
- mapeo explícito de campos `m1/m2/chi1/chi2`
- `--hdf5-dataset`
- `--write-output`
Eso ya no es una simple "ejecución del experimento 221", sino una **reconstrucción manual del pipeline upstream de posteriors IMR**.
---
## Por qué finalmente no hemos sido capaces de validar los 221
### Razón corta
Porque **faltaba una integración canónica y reproducible del remanente `(Mf, af)` dentro del flujo real de runs `multimode`**, y sin ese oracle externo Gate A no puede cerrarse de forma trazable y no circular.
### Razón técnica detallada
El experimento 221 depende de tres gates:
- **Gate A**: coherencia Kerr frente a `(Mf, af)` del remanente
- **Gate B**: estabilidad en `t0`
- **Gate C**: model selection frente a `220` solo
El bloqueo dominante fue **Gate A**:
- los runs nuevos creados por `pipeline multimode` no materializan `external_inputs/gwtc_posteriors/...`
- sin esa materialización, `Mf` y `af` no entran en el run
- sin `Mf, af`, no hay `f221_kerr`, `tau221_kerr`, `rel_err_f`, `rel_err_tau`
- por tanto el veredicto cae en:
  - `INSUFFICIENT_DATA`
Incluso en el caso centinela, además apareció evidencia interna desfavorable:
- `SINGLEMODE_ONLY`
- `mode_221_ok=false`
- `overtone posterior not usable for multimode inference`
Esto impide ya de por sí vender el 221 como robusto o utilizable.
### Por qué lo dejamos por imposible
No porque sea matemáticamente imposible en abstracto, sino porque **en el estado actual del flujo del repositorio** se necesitan cambios upstream que ya no son un ajuste mínimo del experimento 221.
Para cerrar la validación habría que arreglar, como mínimo, una de estas rutas:
1. **Pipeline / preparación del run**
   - hacer que `multimode` materialice automáticamente:
     - `external_inputs/gwtc_posteriors/required_events.txt`
     - `external_inputs/gwtc_posteriors/<EVENT>.json`
     - o `external_inputs/gwtc_posteriors/raw/*.h5`
2. **Wrapper intermedio reproducible**
   - entre `pipeline multimode` y `experiment_qnm_221_literature_check`, ejecutar formalmente:
     - preparación de posteriors,
     - conversión a JSON canónico,
     - y solo entonces el experimento 221
3. **Revisión contractual del oracle Kerr**
   - fijar oficialmente qué schema es la fuente canónica del remanente
   - documentarlo e integrarlo en el pipeline principal
Como nada de eso estaba ya integrado en el flujo que se estaba usando, continuar habría significado:
- introducir más parches ad hoc,
- reconstruir manualmente inputs externos por run,
- y desdibujar la frontera entre experimento 221 y reparación del pipeline de posteriors.
Por esa razón, **se decidió parar y dejar la validación por imposible en el estado actual**.
---
## Qué sí podemos afirmar con rigor
Sí podemos afirmar todo esto:
1. El experimento post-hoc 221 quedó implementado y probado.
2. La infraestructura de tests del experimento es correcta.
3. El lote batch de 24 eventos fue construido de forma auditable.
4. Los 24 runs evaluados inicialmente quedaron en `INSUFFICIENT_DATA` por falta de `Mf, af` canónicos dentro del run.
5. En `GW190521`, además, la inferencia interna multimodo quedó en:
   - `SINGLEMODE_ONLY`
   - `mode_221_ok=false`
6. Existe fuente canónica real del remanente en otros árboles/experimentos del repo.
7. El flujo `multimode` actual no la integra automáticamente en los runs nuevos que necesitaría el experimento 221.
---
## Qué no debemos afirmar
No debemos afirmar:
- que el 221 haya sido "detectado"
- que los 221 sean coherentes en general
- que el 221 esté universalmente refutado en todos los eventos
- que el fracaso sea por falta de física o por falta de teoría
La formulación correcta es:
> **La validación post-hoc del 221 quedó bloqueada por ausencia de integración canónica del remanente `(Mf, af)` en el flujo real de runs `multimode`, y en el caso centinela analizado la evidencia interna además degradó el caso a `SINGLEMODE_ONLY`. Por ello, el intento se deja por imposible en el estado actual del pipeline.**
---
## Comandos relevantes que sí se ejecutaron
Selección del lote final:
```bash
comm -12 \
  runs/prep_batch_from_quality_csv_20260318T110112Z/external_inputs/events_from_quality_csv_visible.txt \
  runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt \
  > runs/prep_batch_final_20260318T110223Z/external_inputs/events_batch_final.txt
```
Ejecución tipo por evento:
```bash
python -m mvp.pipeline multimode \
  --event-id <EVENT_ID> \
  --atlas-default \
  --run-id <RUN_ID> \
  --s3b-method spectral_two_pass
python -m mvp.experiment_qnm_221_literature_check --run-id <RUN_ID>
```
Resumen batch:
```bash
python - <<'PY'
import csv
from collections import Counter
path = "runs/prep_batch_final_20260318T110223Z/outputs/qnm_221_batch_summary.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("N =", len(rows))
print("verdict_counts =", dict(Counter(r["verdict"] for r in rows)))
print("reason_counts =", dict(Counter(r["verdict_reason"] for r in rows)))
PY
```
Resultado observado en el batch:
```text
N = 24
verdict_counts = {'INSUFFICIENT_DATA': 24}
reason_counts = {'gate_a_not_available:missing_canonical_mf_af': 24}
```
---
## Estado recomendado si se retoma en el futuro
Si en el futuro se quiere reabrir esta línea, el orden correcto sería:
1. Definir oficialmente la **fuente canónica de remanente** (`JSON` o `HDF5 + conversión`) para cada run.
2. Integrar esa materialización en el flujo `multimode` o en un wrapper formal upstream del experimento 221.
3. Repetir primero el centinela `GW190521`.
4. Solo si Gate A deja de caer por `missing_canonical_mf_af`, relanzar el batch.
Hasta que eso no ocurra, repetir el batch solo reproducirá:
- `INSUFFICIENT_DATA`
- o casos internos `SINGLEMODE_ONLY / INCONCLUSIVE`
---
## Conclusión final
**Se implementó, probó y ejecutó un experimento post-hoc serio para validar el modo 221.**
**Sin embargo, la validación no pudo cerrarse porque el flujo real de runs `multimode` no materializa de forma canónica el remanente `(Mf, af)` que Gate A necesita, y además el caso centinela mostró degradación interna a `SINGLEMODE_ONLY`.**
Por tanto:
- **no hemos validado los 221**,
- **no podemos presentarlos como coherentes**,
- y **se deja este intento por imposible en el estado actual del pipeline**.
