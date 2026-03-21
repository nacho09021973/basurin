# BRUNETE

BRUNETE es la fachada operativa nueva y mínima sobre este checkout de BASURIN.
No borra ni renombra código legacy. Expone cinco entrypoints públicos:

1. `brunete_list_events.py`
2. `brunete_audit_cohort_authority.py`
3. `brunete_prepare_events.py`
4. `brunete_run_batch.py`
5. `brunete_classify_geometries.py`

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

### 1. `list_events`

Materializa una lista canónica y ordenada de los `EVENT_ID` visibles bajo
`data/losc/<EVENT_ID>/` para no tener que reconstruir la cohorte a mano.

Importante: este output es un *snapshot operativo* de `external input` local;
no congela por sí solo una autoridad versionada del repo.

Output:

- `runs/<run_id>/list_events/outputs/visible_events.txt`
- `runs/<run_id>/list_events/outputs/events_catalog.json`

### 2. `audit_cohort_authority`

Emite un veredicto binario auditable sobre si una cohorte tiene o no una fuente
única suficiente para declararla canónica sin reinterpretación humana.

Output:

- `runs/<run_id>/audit_cohort_authority/outputs/authority_report.json`

La autoridad declarativa vive en `brunete/cohorts/authority_registry.json`.
Hoy ese registro fija que `support_multi`, `support_singleton` y
`no_common_region` tienen autoridad versionada en repo, mientras que
`visible_losc_events` devuelve `FAIL` porque la vista local de `data/losc/`
solo puede materializarse como snapshot, no como autoridad única versionada.

### 3. `prepare_events`

Prepara una cohorte local normalizada a partir de:

- `--events-file <ruta>`
- o `--losc-root <ruta>`

Output:

- `runs/<run_id>/prepare_events/external_inputs/events.txt`
- `runs/<run_id>/prepare_events/outputs/events_catalog.json`

### 4. `run_batch` 220

Consume un `prepare_run` válido y ejecuta el batch local para modo `220`.

Output:

- `runs/<run_id>/run_batch/outputs/results.json`
- `runs/<run_id>/run_batch/outputs/results.csv`
- `runs/<run_id>/run_batch/event_runs/<event_run_id>/...`

### 5. `run_batch` 221

Mismo contrato que el batch `220`, pero para modo `221`.

### 6. `classify_geometries`

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

./brunete/brunete_list_events.py \
  --run-id brunete_list_local \
  --losc-root data/losc

./brunete/brunete_audit_cohort_authority.py \
  --run-id brunete_audit_visible_losc \
  --cohort-key visible_losc_events

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

## Qué No Depende Ya De BASURIN Legacy

BRUNETE ya no depende, como interfaz pública, de entrypoints legacy rotos o
inestables como:

- `mvp.experiment_losc_quality`
- `mvp.audit_gwosc_losc_quality`
- `mvp.experiment_t0_sweep_full`
- `mvp.experiment_phase3_physkey_common`
- `mvp.experiment_phase4_hawking_area_common_support`

En concreto:

- `list_events` materializa una lista canónica local de `data/losc` bajo `runs/<run_id>/list_events/`
- `audit_cohort_authority` permite cerrar con PASS/FAIL si una cohorte tiene autoridad única o si esa autoridad no existe todavía
- `prepare_events` no depende de GWOSC online ni de bootstrap `t0`
- `run_batch` no depende de un `t0_catalog` externo ni de `losc_quality`
- `classify_geometries` no depende de los entrypoints públicos de Fase 3/4

Limitación actual:

- `run_batch` sí reutiliza internamente el motor legacy de `mvp.pipeline` para
  ejecutar subruns por evento; lo que se independiza aquí es la interfaz
  pública y el contrato operativo, no toda la implementación física subyacente
