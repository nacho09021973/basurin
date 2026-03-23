# BASURIN — SSOT de rutas e IO

Este documento fija las rutas operativas vigentes y la separación entre inputs externos y outputs internos.
Si una ruta o patrón no aparece aquí, no debe asumirse como contrato estable.

## 1. Principios

- Los inputs externos se leen desde `data/losc/<EVENT_ID>/`.
- Los outputs auditables se escriben solo bajo `runs/<run_id>/...`.
- Los experimentos se escriben solo bajo `runs/<classify_run_id>/experiment/<name>/...`.
- `data/losc/...` es read-only a efectos de pipeline.
- No se deben usar rutas absolutas locales como contrato documental.
- Antes de ejecutar o depurar, verificar la raíz efectiva con:

```bash
git rev-parse --show-toplevel
```

## 2. Inputs externos

La única vista canónica de strain/LOSC consumible por el flujo operativo es:

```text
data/losc/<EVENT_ID>/
```

Cada evento visible debe vivir en su propio directorio bajo `data/losc/`.
Los nombres exactos de los ficheros internos pueden variar, pero la visibilidad operativa del evento se valida con las herramientas del repositorio, no a ojo.

Comandos canónicos:

```bash
python tools/list_losc_events.py --losc-root data/losc
python tools/losc_precheck.py --event-id GW150914 --losc-root data/losc
python tools/fetch_losc_event.py --event-id GW150914 --out-root data/losc
```

Si existe una caché física previa fuera de `data/losc/`, debe exponerse bajo `data/losc/<EVENT_ID>/` mediante symlink o bind mount por evento.
El pipeline no debe redirigirse a otra raíz.

## 3. Outputs internos

Todo stage canónico escribe bajo:

```text
runs/<run_id>/<stage>/
```

Todo experimento escribe bajo:

```text
runs/<classify_run_id>/experiment/<name>/
```

Estructura mínima esperada por stage o experimento:

```text
runs/<run_id>/<stage>/
  manifest.json
  stage_summary.json
  outputs/
```

Gating mínimo:

```text
runs/<run_id>/RUN_VALID/verdict.json
```

## 4. Rutas BRUNETE vigentes

Flujo público BRUNETE:

- `runs/<run_id>/list_events/`
- `runs/<run_id>/audit_cohort_authority/`
- `runs/<run_id>/prepare_events/`
- `runs/<run_id>/run_batch/`
- `runs/<run_id>/classify_geometries/`

Artefactos relevantes:

```text
runs/<run_id>/list_events/outputs/visible_events.txt
runs/<run_id>/audit_cohort_authority/outputs/authority_report.json
runs/<run_id>/prepare_events/outputs/events_catalog.json
runs/<run_id>/run_batch/outputs/results.json
runs/<run_id>/run_batch/event_runs/<event_run_id>/
runs/<run_id>/classify_geometries/outputs/geometry_summary.json
runs/<run_id>/classify_geometries/outputs/geometry_summary.csv
```

## 5. Ejemplos mínimos correctos

Listar eventos visibles:

```bash
python -m brunete.brunete_list_events \
  --run-id brunete_list_local \
  --losc-root data/losc
```

Preparar una cohorte desde `data/losc`:

```bash
python -m brunete.brunete_prepare_events \
  --run-id brunete_prepare_local \
  --losc-root data/losc
```

Ejecutar un batch:

```bash
python -m brunete.brunete_run_batch \
  --prepare-run brunete_prepare_local \
  --mode 220 \
  --run-id brunete_batch_220_local \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

Materializar outputs experimentales sobre un classify run:

```text
runs/<classify_run_id>/experiment/b5a_multi_event_aggregation/
runs/<classify_run_id>/experiment/b5f_verdict_aggregation/
runs/<classify_run_id>/experiment/b5z_gpr_emulator/
```

## 6. Rutas que no deben usarse como contrato operativo

Estas rutas o patrones no deben presentarse como ubicación operativa vigente:

- `gw_events/...` como raíz consumida directamente por el pipeline
- rutas absolutas locales del tipo `/home/.../basurin/...`
- `experiment/...` en raíz del repo como namespace operativo
- cualquier árbol de outputs fuera de `runs/<run_id>/...`

Si una caché previa vive en `gw_events/strain/<EVENT_ID>/`, la solución correcta es exponerla bajo `data/losc/<EVENT_ID>/`, no reconfigurar BRUNETE para consumir `gw_events/strain`.

## 7. Qué documento usar para cada cosa

- Flujo BRUNETE actual: [docs/operativo/brunete_flujo_actual.md](operativo/brunete_flujo_actual.md)
- Índice operativo: [docs/operativo/README.md](operativo/README.md)
- Manual técnico BRUNETE: [brunete/MANUAL_OPERATIVO.md](../brunete/MANUAL_OPERATIVO.md)

