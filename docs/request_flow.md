# Flujo de una ejecución ("request") en BASURIN

Este documento resume cómo fluye una ejecución iniciada por CLI (por ejemplo `python mvp/pipeline.py single ...`) a través de los módulos principales del MVP.

## 1) Entrada y orquestación

- `mvp/pipeline.py` parsea el modo (`single`, `multi`, `multimode`) y parámetros de ejecución.
- Para cada run, crea `RUN_VALID/verdict.json` y un `pipeline_timeline.json` inicial.
- Luego ejecuta cada stage como subproceso (`subprocess`), en orden, con semántica **fail-fast** (si una etapa falla, se aborta el pipeline).

## 2) Contratos y gobernanza por etapa

- Cada stage usa `mvp/contracts.py` para:
  - inicialización (`init_stage`),
  - validación de inputs (`check_inputs`),
  - escritura de artefactos de salida (`finalize`),
  - manejo de fallo estandarizado (`abort`).
- `init_stage` valida `run_id`, resuelve `RUNS_ROOT` y exige `RUN_VALID=PASS` para etapas que no son bootstrap.

## 3) Resolución de rutas e IO determinista

- `basurin_io.py` centraliza:
  - resolución de `RUNS_ROOT` (`BASURIN_RUNS_ROOT` o `<cwd>/runs`),
  - escritura JSON atómica,
  - hashing SHA256 de artefactos,
  - helpers de manifest/summary.

## 4) Flujo funcional por etapas (single-event)

1. `s1_fetch_strain.py`
   - Obtiene strain (GWOSC, local HDF5 o sintético).
   - Produce `strain.npz` + `provenance.json`.
2. `s2_ringdown_window.py`
   - Corta la ventana de ringdown usando `t0` y parámetros temporales.
   - Produce `{H1,L1,V1}_rd.npz` + `window_meta.json`.
3. `s3_ringdown_estimates.py`
   - Estima `f`, `tau`, `Q` (Hilbert + bootstrap opcional).
   - Produce `estimates.json`.
4. `s4_geometry_filter.py`
   - Compara observables con atlas teórico usando métrica euclidiana o Mahalanobis.
   - Produce `compatible_set.json`.

## 5) Validaciones clave (qué se valida y dónde)

- Entrada CLI y tipos: en `argparse` de cada script.
- Integridad de run y gating:
  - `validate_run_id` y `require_run_valid` (desde `contracts.init_stage`).
- Existencia de inputs y trazabilidad:
  - `contracts.check_inputs` (además calcula SHA256 para auditoría).
- Validación numérica por etapa:
  - `s2`: `sample_rate_hz > 0`, `duration_s > 0`, ventana dentro de rango, arrays 1-D y finitos.
  - `s3`: tamaño mínimo de muestra, banda válida, ajuste de decaimiento válido, `Q > 0`.
  - `s4`: `f_hz > 0`, `Q > 0`, atlas bien formado, y covarianza invertible para Mahalanobis.
- Validación de métricas:
  - `mvp/distance_metrics.py` impone sigmas positivos/finitos y `|r| < 1` para evitar covarianzas no invertibles.

## 6) Gotchas al modificar

1. **RUNS_ROOT/subruns**: si cambias el cwd o no propagas `BASURIN_RUNS_ROOT`, stages manuales pueden buscar `RUN_VALID` en un árbol incorrecto.
2. **Compatibilidad de incertidumbre**: `s4_geometry_filter` soporta claves legacy y modernas (`sigma_logf` vs `sigma_lnf`, etc.). Cambios de naming pueden romper auditorías/tests de contrato si eliminas alias.


## 7) Experimentos: `phase=run` vs `inventory` vs `finalize`

`mvp/experiment_t0_sweep_full.py` separa ejecución y validación final en tres fases:

- `phase=run`
  - Requiere `--atlas-path` cuando el sweep usa atlas.
  - Ejecuta subruns del barrido y deja estado operativo (`IN_PROGRESS`) mientras exista trabajo pendiente.
- `phase=inventory`
  - No requiere atlas.
  - Solo escanea subruns y escribe `runs/<RUN_ID>/experiment/derived/sweep_inventory.json`.
  - Requiere seeds y grilla explícitas (`--inventory-seeds` + `--t0-grid-ms` o `start/stop/step`).
- `phase=finalize`
  - No requiere atlas.
  - Evalúa decisión PASS/FAIL comparando faltantes con umbrales de aceptación.
  - Escribe la decisión en `sweep_inventory.json` y retorna exit code `2` en FAIL.

Qué valida y dónde:

- `inventory` valida completitud: `expected_pairs` vs `observed_pairs` y reporta `missing_abs`/`missing_frac`.
- `finalize` aplica umbrales: `max_missing_abs` y `max_missing_frac` sobre el inventario previo.

### Ejemplos mínimos por fase

```bash
RUN_ID="mvp_GW150914_20260219T120000Z"
ATLAS_PATH="/ruta/al/atlas"

python mvp/experiment_t0_sweep_full.py \
  --run-id "$RUN_ID" \
  --phase run \
  --atlas-path "$ATLAS_PATH" \
  --t0-grid-ms 0,2,4,6,8 \
  --seed 101
```

```bash
RUN_ID="mvp_GW150914_20260219T120000Z"

python mvp/experiment_t0_sweep_full.py \
  --run-id "$RUN_ID" \
  --phase inventory \
  --inventory-seeds 101,202 \
  --t0-grid-ms 0,2,4,6,8
```

```bash
RUN_ID="mvp_GW150914_20260219T120000Z"

python mvp/experiment_t0_sweep_full.py \
  --run-id "$RUN_ID" \
  --phase finalize \
  --inventory-seeds 101,202 \
  --t0-grid-ms 0,2,4,6,8 \
  --max-missing-abs 0 \
  --max-missing-frac 0.0
```
