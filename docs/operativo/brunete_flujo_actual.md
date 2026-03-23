# BRUNETE — flujo actual

Este documento describe el flujo ejecutable vigente de BRUNETE.
Su alcance llega hasta la clasificación geométrica conjunta de los batches `220` y `221`.

## 1. Alcance

El flujo público actual de BRUNETE es:

1. listar eventos visibles
2. auditar autoridad de cohorte
3. preparar cohorte
4. ejecutar batch `220`
5. ejecutar batch `221`
6. clasificar geometrías

La interfaz pública son los entrypoints bajo `brunete/`.
Los módulos `mvp/*` pueden intervenir como backend, pero no son la CLI principal de este flujo.

## 2. Prerrequisitos

- raíz del repo válida:

```bash
git rev-parse --show-toplevel
```

- caché externa visible bajo:

```text
data/losc/<EVENT_ID>/
```

- atlas local disponible:

```text
docs/ringdown/atlas/atlas_berti_v2.json
```

- entorno Python operativo con los wrappers BRUNETE importables o ejecutables.

Comprobación mínima:

```bash
python -m brunete.brunete_list_events --help
python -m brunete.brunete_audit_cohort_authority --help
python -m brunete.brunete_prepare_events --help
python -m brunete.brunete_run_batch --help
python -m brunete.brunete_classify_geometries --help
```

Si falla la visibilidad de `data/losc`, validar primero con utilidades auxiliares del repositorio. Estas utilidades no forman parte del flujo principal BRUNETE, pero sí del precheck operativo:

```bash
python tools/list_losc_events.py --losc-root data/losc
python tools/losc_precheck.py --event-id GW150914 --losc-root data/losc
```

## 3. Cohortes

BRUNETE mantiene cohortes versionadas en:

```text
brunete/cohorts/
```

Referencias actualmente visibles en el repo:

- `brunete/cohorts/events_support_multi.txt`
- `brunete/cohorts/events_support_singleton.txt`
- `brunete/cohorts/events_no_common_region.txt`
- `brunete/cohorts/authority_registry.json`

`authority_registry.json` es la referencia declarativa para auditoría de autoridad de cohorte.

## 4. Secuencia operativa

### 4.1 Listar eventos visibles

```bash
python -m brunete.brunete_list_events \
  --run-id brunete_list_local \
  --losc-root data/losc
```

Output principal:

```text
runs/brunete_list_local/list_events/outputs/visible_events.txt
```

### 4.2 Auditar autoridad de cohorte

Ejemplo contra una cohorte declarada:

```bash
python -m brunete.brunete_audit_cohort_authority \
  --run-id brunete_audit_support_multi \
  --cohort-key support_multi
```

Output principal:

```text
runs/brunete_audit_support_multi/audit_cohort_authority/outputs/authority_report.json
```

### 4.3 Preparar cohorte

Desde `data/losc`:

```bash
python -m brunete.brunete_prepare_events \
  --run-id brunete_prepare_local \
  --losc-root data/losc
```

Desde una cohorte versionada:

```bash
python -m brunete.brunete_prepare_events \
  --run-id brunete_prepare_support_multi \
  --events-file brunete/cohorts/events_support_multi.txt
```

Output principal:

```text
runs/<prepare_run_id>/prepare_events/outputs/events_catalog.json
```

### 4.4 Ejecutar batch 220

```bash
python -m brunete.brunete_run_batch \
  --prepare-run brunete_prepare_local \
  --mode 220 \
  --run-id brunete_batch_220_local \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

Outputs principales:

```text
runs/brunete_batch_220_local/run_batch/outputs/results.json
runs/brunete_batch_220_local/run_batch/event_runs/
```

### 4.5 Ejecutar batch 221

```bash
python -m brunete.brunete_run_batch \
  --prepare-run brunete_prepare_local \
  --mode 221 \
  --run-id brunete_batch_221_local \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

Outputs principales:

```text
runs/brunete_batch_221_local/run_batch/outputs/results.json
runs/brunete_batch_221_local/run_batch/event_runs/
```

### 4.6 Clasificar geometrías

```bash
python -m brunete.brunete_classify_geometries \
  --batch-220 brunete_batch_220_local \
  --batch-221 brunete_batch_221_local \
  --run-id brunete_classify_local
```

Outputs principales:

```text
runs/brunete_classify_local/classify_geometries/outputs/geometry_summary.json
runs/brunete_classify_local/classify_geometries/outputs/geometry_summary.csv
```

## 5. Contrato de escritura

Resumen mínimo:

- stages BRUNETE: `runs/<run_id>/<stage>/`
- gating: `runs/<run_id>/RUN_VALID/verdict.json`
- experimentos downstream sobre classify: `runs/<classify_run_id>/experiment/<name>/`

El contrato detallado de rutas e IO está en [docs/readme_rutas.md](../readme_rutas.md).

## 6. Validación mínima

Comprobación de CLI:

```bash
python -m brunete.brunete_list_events --help
python -m brunete.brunete_audit_cohort_authority --help
python -m brunete.brunete_prepare_events --help
python -m brunete.brunete_run_batch --help
python -m brunete.brunete_classify_geometries --help
```

Comprobación de tests directamente relacionados:

```bash
pytest tests/brunete/test_prepare_events.py -q
pytest tests/brunete/test_run_batch.py -q
pytest tests/brunete/test_classify_geometries.py -q
```

## 7. Evidencia de estado actual

La superficie descrita aquí está respaldada por:

- `tests/brunete/test_prepare_events.py`
- `tests/brunete/test_run_batch.py`
- `tests/brunete/test_classify_geometries.py`
- `brunete/cohorts/authority_registry.json`
- `docs/ringdown/atlas/atlas_berti_v2.json`

## 8. Qué no cubre este documento

- No documenta Fase 4 ni Fase 5 en detalle.
- No documenta `mvp/*` como interfaz pública para usuarios BRUNETE.
- No documenta rutas históricas o árboles legacy fuera de `data/losc/...` y `runs/...`.
