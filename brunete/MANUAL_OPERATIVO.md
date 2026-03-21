# Manual Operativo de BRUNETE

## 1. Qué es BRUNETE

BRUNETE es el carril operativo público para ejecutar análisis locales de ringdown sobre una cohorte de eventos ya disponible en `data/losc/<EVENT_ID>/`.
Su flujo mínimo tiene tres fases: preparar la cohorte, correr el batch en modo `220` y `221`, y clasificar el soporte geométrico conjunto.
La interfaz pública son los scripts de `brunete/`.
`mvp.pipeline` sigue existiendo, pero aquí se usa como backend de ejecución por evento y como herramienta de depuración puntual, no como interfaz principal.
Este manual asume Ubuntu, `bash`, raíz del repo en `/home/adnac/basurin/work/basurin` y outputs por defecto en `runs/`.

## 2. Requisitos previos

- Repo clonado en `/home/adnac/basurin/work/basurin`.
- Entorno virtual ya creado en `./.venv/`.
- Atlas local disponible en `docs/ringdown/atlas/atlas_berti_v2.json`.
- Caché LOSC local disponible en `data/losc/<EVENT_ID>/`.
- Rama recomendada para reproducir este estado: `codex/normalize-losc-catalog`.
- Este manual no cubre bootstrap de dependencias si `.venv` no existe.

Comprobación rápida:

```bash
cd /home/adnac/basurin/work/basurin
git rev-parse --show-toplevel
git branch --show-current
test -x ./.venv/bin/python
test -f docs/ringdown/atlas/atlas_berti_v2.json
test -d data/losc
find data/losc -maxdepth 2 -type f \( -name '*.h5' -o -name '*.hdf5' \) | head
./.venv/bin/python brunete/brunete_prepare_events.py --help >/dev/null
./.venv/bin/python brunete/brunete_run_batch.py --help >/dev/null
./.venv/bin/python brunete/brunete_classify_geometries.py --help >/dev/null
```

Si alguna de esas comprobaciones falla, no sigas con el batch.

## 3. Estructura pública de BRUNETE

- `brunete/brunete_list_events.py`
  Materializa bajo `runs/<run_id>/list_events/` la lista canónica de eventos visibles en `data/losc/`.

- `brunete/brunete_prepare_events.py`
  Normaliza una cohorte bajo `runs/<run_id>/prepare_events/` a partir de `--losc-root` o `--events-file`.
  Si usas `--losc-root`, descubre solo directorios con al menos un `.h5` o `.hdf5`.

- `brunete/brunete_run_batch.py`
  Ejecuta la cohorte preparada en modo `220` o `221`.
  `220` llama al backend single-event.
  `221` llama al backend multimode.
  Publica un contrato batch estable en `runs/<run_id>/run_batch/` y deja los subruns por evento en `runs/<run_id>/run_batch/event_runs/`.

- `brunete/brunete_classify_geometries.py`
  Cruza dos batch válidos, uno `220` y otro `221`.
  Publica `geometry_summary.json` y `geometry_summary.csv` en `runs/<run_id>/classify_geometries/`.
  La clasificación actual usa `status`, `n_compatible`, `support_region_status_221` y `support_region_n_final_221`.

- `brunete/cohorts/events_support_multi.txt`
  Cohorte robusta principal versionada en repo.
  Hoy contiene 40 eventos.

- `brunete/cohorts/events_support_singleton.txt`
  Cohorte frontera.
  Hoy contiene 8 eventos.

- `brunete/cohorts/events_no_common_region.txt`
  Cohorte a investigar o excluir según el objetivo.
  Hoy contiene 10 eventos.

## Mapa de archivos operativo

- Carpetas para operar
  `brunete/`: entrypoints públicos.
  `brunete/cohorts/`: cohortes reutilizables y estables.
  `runs/<run_id>/prepare_events/`: cohorte preparada.
  `runs/<run_id>/run_batch/`: contrato batch y resultados por evento.
  `runs/<run_id>/classify_geometries/`: clasificación final y resumen geométrico.

- Carpetas y archivos para depurar
  `runs/<batch_run>/run_batch/event_runs/<event_run_id>/`: subrun real del evento.
  `manifest.json`: qué publicó el stage.
  `stage_summary.json`: parámetros, resumen y error.
  `RUN_VALID/verdict.json`: PASS o FAIL efectivo.

- Backend crítico que sí conviene conocer
  `mvp/s1_fetch_strain.py`
  `mvp/s2_ringdown_window.py`
  `mvp/pipeline.py`

- Qué ignorar casi siempre
  Docs legacy no operativos.
  Módulos experimentales viejos.
  Wrappers históricos fuera de BRUNETE.
  Outputs antiguos en `runs/` que no sean del run actual.

- Corrección operativa importante
  `220` se reintenta con `python -m mvp.pipeline single ...`.
  `221` se reintenta con `python -m mvp.pipeline multimode ...`.

## 4. Flujo mínimo end-to-end

Ejemplo validado con nombres coherentes con los runs ya presentes en `runs/`.
Solo cambia `RUN_TS` para no pisar ejecuciones anteriores.

```bash
cd /home/adnac/basurin/work/basurin

RUN_TS=20260321T001500Z
PREP_RUN=brunete_prepare_e2e_${RUN_TS}
B220_RUN=brunete_batch_220_e2e_${RUN_TS}
B221_RUN=brunete_batch_221_e2e_${RUN_TS}
GEOM_RUN=brunete_geom_e2e_${RUN_TS}
```

Preparar eventos desde `data/losc`:

```bash
./.venv/bin/python brunete/brunete_list_events.py \
  --run-id brunete_list_local \
  --losc-root data/losc

cat runs/brunete_list_local/list_events/outputs/visible_events.txt

./.venv/bin/python brunete/brunete_prepare_events.py \
  --run-id "$PREP_RUN" \
  --losc-root data/losc
```

Correr batch `220`:

```bash
./.venv/bin/python brunete/brunete_run_batch.py \
  --prepare-run "$PREP_RUN" \
  --mode 220 \
  --run-id "$B220_RUN" \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

Correr batch `221`:

```bash
./.venv/bin/python brunete/brunete_run_batch.py \
  --prepare-run "$PREP_RUN" \
  --mode 221 \
  --run-id "$B221_RUN" \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

Clasificar geometrías:

```bash
./.venv/bin/python brunete/brunete_classify_geometries.py \
  --batch-220 "$B220_RUN" \
  --batch-221 "$B221_RUN" \
  --run-id "$GEOM_RUN"
```

Leer `geometry_summary.csv`:

```bash
./.venv/bin/python - <<PY
import csv, collections, pathlib
p = pathlib.Path("runs") / "${GEOM_RUN}" / "classify_geometries" / "outputs" / "geometry_summary.csv"
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("n_rows =", len(rows))
print("classification_counts =", dict(collections.Counter(r["classification"] for r in rows)))
print("joint_support_counts =", dict(collections.Counter(r["has_joint_support"] for r in rows)))
for row in rows[:5]:
    print(row)
PY
```

Usar la cohorte robusta `support_multi`:

```bash
SUPPORT_PREP=brunete_prepare_support_multi_${RUN_TS}
SUPPORT_B220=brunete_batch_220_support_multi_${RUN_TS}
SUPPORT_B221=brunete_batch_221_support_multi_${RUN_TS}

./.venv/bin/python brunete/brunete_prepare_events.py \
  --run-id "$SUPPORT_PREP" \
  --events-file brunete/cohorts/events_support_multi.txt

./.venv/bin/python brunete/brunete_run_batch.py \
  --prepare-run "$SUPPORT_PREP" \
  --mode 220 \
  --run-id "$SUPPORT_B220" \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral

./.venv/bin/python brunete/brunete_run_batch.py \
  --prepare-run "$SUPPORT_PREP" \
  --mode 221 \
  --run-id "$SUPPORT_B221" \
  --losc-root data/losc \
  --epsilon 2500 \
  --estimator spectral
```

## 5. Salidas esperadas

`list_events` escribe en `runs/<run_id>/list_events/`:

- `RUN_VALID/verdict.json`
- `manifest.json`
- `stage_summary.json`
- `outputs/visible_events.txt`
- `outputs/events_catalog.json`

`prepare_events` escribe en `runs/<run_id>/prepare_events/`:

- `RUN_VALID/verdict.json`
- `manifest.json`
- `stage_summary.json`
- `external_inputs/events.txt`
- `outputs/events_catalog.json`

`run_batch` escribe en `runs/<run_id>/run_batch/`:

- `RUN_VALID/verdict.json`
- `manifest.json`
- `stage_summary.json`
- `external_inputs/events.txt`
- `outputs/results.json`
- `outputs/results.csv`
- `event_runs/brunete_<EVENT_ID>_m220/...` o `event_runs/brunete_<EVENT_ID>_m221/...`

`classify_geometries` escribe en `runs/<run_id>/classify_geometries/`:

- `RUN_VALID/verdict.json`
- `manifest.json`
- `stage_summary.json`
- `external_inputs/batch_220_results.json`
- `external_inputs/batch_221_results.json`
- `outputs/geometry_summary.json`
- `outputs/geometry_summary.csv`

Archivos que hay que mirar primero:

- `manifest.json`: contrato del stage y artefactos publicados.
- `stage_summary.json`: parámetros, `verdict`, resumen y error si falla.
- `outputs/results.csv`: estado por evento del batch.
- `outputs/geometry_summary.csv`: clasificación conjunta por evento.

## 6. Cohortes derivadas

- `brunete/cohorts/events_support_multi.txt`
  Corresponde a `common_nonempty_both_221_support_multi`.
  Es la cohorte robusta principal.

- `brunete/cohorts/events_support_singleton.txt`
  Corresponde a `common_nonempty_both_221_support_singleton`.
  Es la cohorte frontera.

- `brunete/cohorts/events_no_common_region.txt`
  Corresponde a `common_nonempty_both_221_no_common_region`.
  Es la cohorte a investigar o excluir según el objetivo.

Usa `support_multi` como entrada por defecto para trabajo reutilizable.
Usa `singleton` para casos límite.
Usa `no_common_region` para depuración, revisión o exclusión explícita.

## 7. Qué hacer cuando falla

1. Para en el primer error real. No sigas downstream si el stage bloqueante no tiene `RUN_VALID=PASS`.
2. Mira primero el resumen del stage:

```bash
./.venv/bin/python -m json.tool runs/<run_id>/<stage>/stage_summary.json | sed -n '1,160p'
./.venv/bin/python -m json.tool runs/<run_id>/<stage>/RUN_VALID/verdict.json
```

3. Si falla un batch, localiza el evento bloqueante en `outputs/results.csv`:

```bash
./.venv/bin/python - <<'PY'
import csv
p = "runs/<batch_run_id>/run_batch/outputs/results.csv"
with open(p, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["status"] != "PASS":
            print(row)
PY
```

4. Inspecciona el subrun bloqueante:

```bash
find "runs/<batch_run_id>/run_batch/event_runs/<event_run_id>" -name stage_summary.json | sort
./.venv/bin/python -m json.tool "runs/<batch_run_id>/run_batch/event_runs/<event_run_id>/RUN_VALID/verdict.json"
```

5. Reintenta solo el evento problemático cuando aplique.

Para depurar un evento dentro del contexto real del batch, reutiliza el árbol
`runs/<batch_run_id>/run_batch/event_runs`.

Para `220`:

```bash
BASURIN_RUNS_ROOT="$(pwd)/runs/<batch_run_id>/run_batch/event_runs" \
./.venv/bin/python -m mvp.pipeline single \
  --event-id <EVENT_ID> \
  --run-id brunete_<EVENT_ID>_m220 \
  --atlas-path docs/ringdown/atlas/atlas_berti_v2.json \
  --offline \
  --epsilon 2500 \
  --estimator spectral
```

Para `221`:

```bash
BASURIN_RUNS_ROOT="$(pwd)/runs/<batch_run_id>/run_batch/event_runs" \
./.venv/bin/python -m mvp.pipeline multimode \
  --event-id <EVENT_ID> \
  --run-id brunete_<EVENT_ID>_m221 \
  --atlas-path docs/ringdown/atlas/atlas_berti_v2.json \
  --offline \
  --epsilon 2500 \
  --estimator spectral
```

No reabras arqueología del repo si el fallo ya es real, local y reproducible.

## 8. Errores reales ya resueltos

- `s1_fetch_strain`: fallback local de `GPS/t0` para no depender de bootstrap remoto.
- Soporte de `event_metadata` y `window_catalog_v1.json` en el carril operativo actual.
- Descarte de detectores enteramente no finitos si otro detector del evento es válido.
- `classify_geometries`: refinamiento de la clasificación usando `support_region_status_221` y `support_region_n_final_221` para no colapsar toda la cohorte en una sola clase.

Si reaparece uno de esos síntomas, compáralo primero con estos precedentes antes de tocar el carril público.

## 9. Uso desde cualquier terminal

- No dependas del `run_id` de la sesión anterior.
- Define cohortes reutilizables desde ficheros versionados en `brunete/cohorts/`, no desde artefactos efímeros dentro de `runs/`.
- En una terminal nueva basta con volver a `cd` al repo, usar `./.venv/bin/python` y lanzar nuevos `run_id`.
- Si quieres repetir un análisis estable, usa `brunete/cohorts/events_support_multi.txt` como entrada y genera runs nuevos.
- `runs/` sirve para trazabilidad y diagnóstico; las cohortes reutilizables deben vivir en `brunete/cohorts/`.

## 10. Checklist de arranque rápido

```bash
cd /home/adnac/basurin/work/basurin
source .venv/bin/activate
git rev-parse --show-toplevel
git branch --show-current
test -f docs/ringdown/atlas/atlas_berti_v2.json && test -d data/losc
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ); PREP_RUN=brunete_prepare_e2e_${RUN_TS}; B220_RUN=brunete_batch_220_e2e_${RUN_TS}; B221_RUN=brunete_batch_221_e2e_${RUN_TS}; GEOM_RUN=brunete_geom_e2e_${RUN_TS}; export RUN_TS PREP_RUN B220_RUN B221_RUN GEOM_RUN
python brunete/brunete_prepare_events.py --run-id "$PREP_RUN" --losc-root data/losc
python brunete/brunete_run_batch.py --prepare-run "$PREP_RUN" --mode 220 --run-id "$B220_RUN" --losc-root data/losc --epsilon 2500 --estimator spectral
python brunete/brunete_run_batch.py --prepare-run "$PREP_RUN" --mode 221 --run-id "$B221_RUN" --losc-root data/losc --epsilon 2500 --estimator spectral
python brunete/brunete_classify_geometries.py --batch-220 "$B220_RUN" --batch-221 "$B221_RUN" --run-id "$GEOM_RUN"
python - <<'PY'
import csv, collections, os
p = f"runs/{os.environ['GEOM_RUN']}/classify_geometries/outputs/geometry_summary.csv"
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(len(rows), dict(collections.Counter(r["classification"] for r in rows)))
PY
```

## 11. Checklist de interpretación rápida

- `¿batch 220 pasó?`
  Mira `runs/<B220_RUN>/run_batch/RUN_VALID/verdict.json`.

- `¿batch 221 pasó?`
  Mira `runs/<B221_RUN>/run_batch/RUN_VALID/verdict.json`.

- `¿cuántos eventos comunes hay?`
  Mira `runs/<GEOM_RUN>/classify_geometries/outputs/geometry_summary.json`, campo `summary.n_events_both`.

- `¿cuántos support_multi?`
  Ejecuta `wc -l brunete/cohorts/events_support_multi.txt`.

- `¿dónde están los 10 no_common_region?`
  Están versionados en `brunete/cohorts/events_no_common_region.txt`.

- `¿qué archivo usar para lanzar el multi?`
  Usa `brunete/cohorts/events_support_multi.txt`.
