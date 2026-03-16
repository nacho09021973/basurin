# Fase 1 — Preparación, auditoría LOSC/t0 y entrada de eventos

## Objetivo

Materializar los **inputs externos auditables** y dejar lista la lista de eventos que pueden entrar en ejecución offline-first sin resolución ambigua de `t0`.

Esta fase existe para responder tres preguntas antes de lanzar runs pesados:

1. ¿Qué eventos están disponibles y auditados?
2. ¿Qué `t0` es canónico para cada evento?
3. ¿Qué lista exacta de eventos se autoriza para batch?

---

## Entradas canónicas

- `data/losc/<EVENT_ID>/...` — HDF5 externos en modo read-only.
- GWOSC/LOSC como fuente externa de metadatos para auditoría.

---

## Salidas canónicas

### Auditoría LOSC/t0

Bajo:

- `runs/<audit_run_id>/experiment/losc_quality/`

Artefactos principales:

- `losc_event_quality.csv`
- `approved_events.txt`
- `gwosc_ready_events.txt`
- `t0_catalog_gwosc_v2.json`

### Preparación de lista final de eventos

Bajo:

- `runs/<prep_run_id>/external_inputs/`

Artefacto principal:

- `events_with_t0.txt`

---

## Gate de salida

La fase 1 se considera utilizable cuando existen y son coherentes:

- `runs/<audit_run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/<prep_run_id>/external_inputs/events_with_t0.txt`

Y cuando la lista batch se construye a partir de eventos realmente auditados.

---

## Comandos canónicos

### 1) Auditoría LOSC/t0

```bash
AUDIT_RUN="audit_gwosc_t0_$(date -u +%Y%m%dT%H%M%SZ)"

python -m mvp.experiment_losc_quality \
  --run "$AUDIT_RUN" \
  --gwosc-api-version v2 \
  --batch-gwosc \
  --write-t0-catalog
```

### 2) Construcción de lista de eventos con `t0`

> Si existe un paso separado de preparación, su output debe vivir bajo `runs/<prep_run_id>/external_inputs/events_with_t0.txt`.

---

## Artefactos que hay que inspeccionar siempre

- `runs/<audit_run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/<audit_run_id>/experiment/losc_quality/approved_events.txt`
- `runs/<audit_run_id>/experiment/losc_quality/gwosc_ready_events.txt`
- `runs/<prep_run_id>/external_inputs/events_with_t0.txt`

---

## Fallos típicos

- evento presente en `data/losc/` pero ausente en `t0_catalog_gwosc_v2.json`
- evento aprobado pero no listo para GWOSC
- lista `events_with_t0.txt` construida fuera de `runs/<run_id>/...`
- mezcla de listas de eventos no auditadas con listas auditadas

---

## Estado operativo que debe dejar documentado el equipo

Cada vez que se cierre esta fase, registrar explícitamente:

- `audit_run_id`
- `prep_run_id`
- ruta exacta de `t0_catalog_gwosc_v2.json`
- ruta exacta de `events_with_t0.txt`
- número de eventos autorizados para batch

---

## Dependencia hacia la fase siguiente

La fase 2 **no debe arrancar** sin:

- `t0_catalog_gwosc_v2.json`
- `events_with_t0.txt`
- trazabilidad completa bajo `runs/<run_id>/...`
