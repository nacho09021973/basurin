# BRUNETE

BRUNETE es la fachada operativa nueva y mínima sobre este checkout de BASURIN.
No borra ni renombra código legacy. Expone solo tres entrypoints públicos:

1. `brunete_prepare_events.py`
2. `brunete_run_batch.py`
3. `brunete_classify_geometries.py`

## Contrato Común

Cada stage público escribe solo bajo `runs/<run_id>/<stage>/...` y comparte la
misma estructura base:

- `RUN_VALID/verdict.json`
- `stage_summary.json`
- `manifest.json`
- `outputs/...`
- `external_inputs/...`

El contrato de metadata es homogéneo:

- `RUN_VALID/verdict.json`: `schema_version`, `created`, `stage`, `run_id`, `verdict`, `reason`
- `stage_summary.json`: `schema_version`, `created`, `stage`, `run_id`, `verdict`, `reason`, `parameters`, `results`, `artifacts`, `notes`
- `manifest.json`: `schema_version`, `created`, `stage`, `run_id`, `verdict`, `parameters`, `artifacts`, `hashes`

## Flujo Exacto

### 1. `prepare_events`

Prepara una cohorte local normalizada a partir de:

- `--events-file <ruta>`
- o `--losc-root <ruta>`

Output:

- `runs/<run_id>/prepare_events/external_inputs/events.txt`
- `runs/<run_id>/prepare_events/outputs/events_catalog.json`

### 2. `run_batch` 220

Consume un `prepare_run` válido y ejecuta el batch local para modo `220`.

Output:

- `runs/<run_id>/run_batch/outputs/results.json`
- `runs/<run_id>/run_batch/outputs/results.csv`
- `runs/<run_id>/run_batch/event_runs/<event_run_id>/...`

### 3. `run_batch` 221

Mismo contrato que el batch `220`, pero para modo `221`.

### 4. `classify_geometries`

Cruza dos batch runs válidos, uno `220` y otro `221`, y produce un resumen
geométrico conjunto.

Output:

- `runs/<run_id>/classify_geometries/outputs/geometry_summary.json`
- `runs/<run_id>/classify_geometries/outputs/geometry_summary.csv`

## Ejemplo End-to-End Local

Desde la raíz del repo:

```bash
cd /home/adnac/basurin/work/basurin

./brunete/brunete_prepare_events.py \
  --run-id brunete_prepare_local \
  --losc-root data/losc

./brunete/brunete_run_batch.py \
  --prepare-run brunete_prepare_local \
  --mode 220 \
  --run-id brunete_batch_220_local \
  --losc-root data/losc \
  --atlas-path docs/ringdown/atlas/atlas_berti_v2.json

./brunete/brunete_run_batch.py \
  --prepare-run brunete_prepare_local \
  --mode 221 \
  --run-id brunete_batch_221_local \
  --losc-root data/losc \
  --atlas-path docs/ringdown/atlas/atlas_berti_v2.json

./brunete/brunete_classify_geometries.py \
  --batch-220 brunete_batch_220_local \
  --batch-221 brunete_batch_221_local \
  --run-id brunete_classify_local
```

## Catálogo B5 experimental

Además de los tres entrypoints públicos, BRUNETE expone una fachada experimental
en `brunete/experiment/` para los módulos B5. Esa fachada **no** reemplaza a
`mvp/experiment/`; actúa como puente de entrada desde un run
`classify_geometries` ya materializado.

Contrato operativo del puente B5:

- entrada gobernante: `runs/<classify_run_id>/classify_geometries/outputs/geometry_summary.json`
- resolución de batches vía `batch_220_run_id` y `batch_221_run_id`
- resolución de event runs vía `event_run_id_220` / `event_run_id_221`
- para métricas de soporte conjunto, B5 consume `has_joint_support`, `classification`,
  `support_region_status_221` y `support_region_n_final_221` **tal como los calcula BRUNETE**
- la lógica analítica pesada sigue viviendo en `mvp/experiment/`; el puente B5 solo cambia
  el punto de entrada y la resolución de inputs

Esto permite ejecutar B5 a partir de un `classify_run_id` sin recalcular la semántica
de `classify_geometries` aguas abajo.

## Qué No Depende Ya De BASURIN Legacy

BRUNETE ya no depende, como interfaz pública, de entrypoints legacy rotos o
inestables como:

- `mvp.experiment_losc_quality`
- `mvp.audit_gwosc_losc_quality`
- `mvp.experiment_t0_sweep_full`
- `mvp.experiment_phase3_physkey_common`
- `mvp.experiment_phase4_hawking_area_common_support`

En concreto:

- `prepare_events` no depende de GWOSC online ni de bootstrap `t0`
- `run_batch` no depende de un `t0_catalog` externo ni de `losc_quality`
- `classify_geometries` no depende de los entrypoints públicos de Fase 3/4

Limitación actual:

- `run_batch` sí reutiliza internamente el motor legacy de `mvp.pipeline` para
  ejecutar subruns por evento; lo que se independiza aquí es la interfaz
  pública y el contrato operativo, no toda la implementación física subyacente
