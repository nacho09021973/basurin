# Manual Operativo de BRUNETE

Este manual describe la ejecución operativa vigente de BRUNETE sobre una cohorte local visible en `data/losc/<EVENT_ID>/`.
La interfaz pública recomendada son los wrappers bajo `brunete/`.

## 1. Alcance

El flujo cubierto por este manual es:

1. listar eventos visibles
2. auditar autoridad de cohorte
3. preparar cohorte
4. ejecutar batch `220`
5. ejecutar batch `221`
6. clasificar geometrías

El manual no documenta Fase 4 ni Fase 5 en detalle.
Tampoco documenta `mvp/*` como interfaz pública, aunque algunos stages internos se usen como backend de implementación.

## 2. Requisitos de entorno

Requisitos mínimos:

- repositorio accesible desde la raíz efectiva devuelta por `git rev-parse --show-toplevel`
- Python operativo con los módulos `brunete.*` ejecutables
- atlas local disponible en `docs/ringdown/atlas/atlas_berti_v2.json`
- caché LOSC visible bajo `data/losc/<EVENT_ID>/`
- árbol de escritura auditable disponible bajo `runs/`

Comprobación mínima:

```bash
git rev-parse --show-toplevel
test -f docs/ringdown/atlas/atlas_berti_v2.json
test -d data/losc
python -m brunete.brunete_list_events --help
python -m brunete.brunete_audit_cohort_authority --help
python -m brunete.brunete_prepare_events --help
python -m brunete.brunete_run_batch --help
python -m brunete.brunete_classify_geometries --help
```

Si alguna de esas comprobaciones falla, no continúes con el batch.

## 3. Preparación de `data/losc`

La única vista canónica de inputs externos para BRUNETE es:

```text
data/losc/<EVENT_ID>/
```

Antes de preparar una cohorte, valida la visibilidad real de los eventos con utilidades auxiliares del repositorio:

```bash
python tools/list_losc_events.py --losc-root data/losc
python tools/losc_precheck.py --event-id GW150914 --losc-root data/losc
```

Si falta un evento concreto en la caché canónica:

```bash
python tools/fetch_losc_event.py --event-id GW150914 --out-root data/losc
```

Estas utilidades son prechecks y bootstrap de caché.
No sustituyen el flujo principal BRUNETE.

## 4. Cohortes soportadas

Cohortes versionadas actualmente visibles:

- `brunete/cohorts/events_support_multi.txt`
- `brunete/cohorts/events_support_singleton.txt`
- `brunete/cohorts/events_no_common_region.txt`
- `brunete/cohorts/authority_registry.json`

Uso recomendado:

- `events_support_multi.txt`: cohorte robusta principal
- `events_support_singleton.txt`: cohorte frontera
- `events_no_common_region.txt`: cohorte de investigación o exclusión explícita

La autoridad declarativa de cohortes vive en `brunete/cohorts/authority_registry.json`.

## 5. Secuencia exacta de comandos

Ejemplo base:

```bash
RUN_TS=20260323T120000Z
LIST_RUN=brunete_list_${RUN_TS}
AUDIT_RUN=brunete_audit_support_multi_${RUN_TS}
PREP_RUN=brunete_prepare_support_multi_${RUN_TS}
B220_RUN=brunete_batch_220_${RUN_TS}
B221_RUN=brunete_batch_221_${RUN_TS}
CLASSIFY_RUN=brunete_classify_${RUN_TS}
```

### 5.1 Listar eventos visibles

```bash
python -m brunete.brunete_list_events \
  --run-id "$LIST_RUN" \
  --losc-root data/losc
```

### 5.2 Auditar autoridad de cohorte

```bash
python -m brunete.brunete_audit_cohort_authority \
  --run-id "$AUDIT_RUN" \
  --cohort-key support_multi
```

### 5.3 Preparar cohorte

Desde una cohorte versionada:

```bash
python -m brunete.brunete_prepare_events \
  --run-id "$PREP_RUN" \
  --events-file brunete/cohorts/events_support_multi.txt
```

Desde `data/losc`:

```bash
python -m brunete.brunete_prepare_events \
  --run-id "$PREP_RUN" \
  --losc-root data/losc
```

### 5.4 Ejecutar batch 220

```bash
python -m brunete.brunete_run_batch \
  --prepare-run "$PREP_RUN" \
  --mode 220 \
  --run-id "$B220_RUN" \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

### 5.5 Ejecutar batch 221

```bash
python -m brunete.brunete_run_batch \
  --prepare-run "$PREP_RUN" \
  --mode 221 \
  --run-id "$B221_RUN" \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

### 5.6 Clasificar geometrías

```bash
python -m brunete.brunete_classify_geometries \
  --batch-220 "$B220_RUN" \
  --batch-221 "$B221_RUN" \
  --run-id "$CLASSIFY_RUN"
```

## 6. Validación de outputs

Rutas principales por stage:

```text
runs/<run_id>/list_events/
runs/<run_id>/audit_cohort_authority/
runs/<run_id>/prepare_events/
runs/<run_id>/run_batch/
runs/<run_id>/classify_geometries/
```

Artefactos mínimos a revisar:

- `runs/<run_id>/RUN_VALID/verdict.json`
- `runs/<run_id>/<stage>/manifest.json`
- `runs/<run_id>/<stage>/stage_summary.json`
- `runs/<run_id>/list_events/outputs/visible_events.txt`
- `runs/<run_id>/audit_cohort_authority/outputs/authority_report.json`
- `runs/<run_id>/prepare_events/outputs/events_catalog.json`
- `runs/<run_id>/run_batch/outputs/results.json`
- `runs/<run_id>/run_batch/outputs/results.csv`
- `runs/<run_id>/classify_geometries/outputs/geometry_summary.json`
- `runs/<run_id>/classify_geometries/outputs/geometry_summary.csv`

Comprobación mínima:

```bash
python -m json.tool "runs/$CLASSIFY_RUN/RUN_VALID/verdict.json"
python -m json.tool "runs/$CLASSIFY_RUN/classify_geometries/stage_summary.json"
python -m json.tool "runs/$CLASSIFY_RUN/classify_geometries/outputs/geometry_summary.json"
```

Lectura rápida del CSV final:

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

p = Path("runs") / "brunete_classify_20260323T120000Z" / "classify_geometries" / "outputs" / "geometry_summary.csv"
with p.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
print("n_rows =", len(rows))
print("classification_counts =", dict(Counter(r["classification"] for r in rows)))
PY
```

Sustituye el `run_id` embebido por tu classify run real.

## 7. Fallos habituales y diagnóstico

### 7.1 `data/losc` no visible o incoherente

Síntoma:

- `prepare_events` no encuentra eventos
- `list_events` produce un snapshot vacío

Diagnóstico:

```bash
python tools/list_losc_events.py --losc-root data/losc
python tools/losc_precheck.py --event-id GW150914 --losc-root data/losc
```

### 7.2 Cohorte sin autoridad única

Síntoma:

- `audit_cohort_authority` devuelve FAIL

Diagnóstico:

```bash
python -m brunete.brunete_audit_cohort_authority \
  --run-id brunete_audit_debug \
  --cohort-key support_multi
```

Revisa:

- `runs/brunete_audit_debug/audit_cohort_authority/outputs/authority_report.json`
- `brunete/cohorts/authority_registry.json`

### 7.3 `prepare_events` falla o prepara una cohorte vacía

Síntoma:

- `RUN_VALID=FAIL` en `prepare_events`

Diagnóstico:

```bash
python -m json.tool "runs/$PREP_RUN/prepare_events/stage_summary.json"
python -m json.tool "runs/$PREP_RUN/RUN_VALID/verdict.json"
```

### 7.4 `run_batch` falla en eventos concretos

Síntoma:

- `results.csv` contiene eventos con `status != PASS`

Diagnóstico:

```bash
python - <<'PY'
import csv
from pathlib import Path

p = Path("runs") / "brunete_batch_220_20260323T120000Z" / "run_batch" / "outputs" / "results.csv"
with p.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["status"] != "PASS":
            print(row)
PY
```

Después inspecciona el subrun afectado bajo:

```text
runs/<batch_run_id>/run_batch/event_runs/<event_run_id>/
```

### 7.5 `classify_geometries` falla por batches no válidos

Síntoma:

- `classify_geometries` no materializa `geometry_summary.json`

Diagnóstico:

```bash
python -m json.tool "runs/$B220_RUN/RUN_VALID/verdict.json"
python -m json.tool "runs/$B221_RUN/RUN_VALID/verdict.json"
```

No ejecutes classify si cualquiera de los dos batch previos no tiene `RUN_VALID=PASS`.

## 8. Referencias operativas

- Flujo resumido: [docs/operativo/brunete_flujo_actual.md](../docs/operativo/brunete_flujo_actual.md)
- SSOT de rutas e IO: [docs/readme_rutas.md](../docs/readme_rutas.md)

