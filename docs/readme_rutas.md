# BASURIN — mapa de rutas (versión anti-pérdida de tiempo para IA)

Objetivo: que una IA (o humano) no pierda 5–6 horas diarias por confundir `RUN_ID`, `SUBRUN_ID`, `RUNS_ROOT` y árboles de experimentos.

## Índice: qué busco → ruta exacta (30 segundos)

- **Gating canónico (run válido para downstream):**
  - `runs/<RUN_ID>/RUN_VALID/verdict.json`
- **H5 efectivamente usados por s1 + trazabilidad:**
  - `runs/<RUN_ID>/s1_fetch_strain/inputs/H1.h5`
  - `runs/<RUN_ID>/s1_fetch_strain/inputs/L1.h5`
  - `runs/<RUN_ID>/s1_fetch_strain/outputs/provenance.json`
- **Ventana ringdown (s2):**
  - `runs/<RUN_ID>/s2_ringdown_window/outputs/`
  - `runs/<RUN_ID>/s2_ringdown_window/outputs/window_meta.json` (si existe)
- **Estimaciones clave (s3):**
  - `runs/<RUN_ID>/s3_ringdown_estimates/outputs/estimates.json`
- **Multimode (s3b):**
  - `runs/<RUN_ID>/s3b_multimode_estimates/outputs/`
- **Filtro geométrico (s4):**
  - `runs/<RUN_ID>/s4_geometry_filter/outputs/`
- **Curvatura/diagnóstico (s6/s6b):**
  - `runs/<RUN_ID>/s6*/outputs/curvature*.json`
  - `runs/<RUN_ID>/s6*/outputs/metric_diagnostics*.json`
- **Rutas canónicas (external inputs vs outputs de stages):**
  - `data/losc/<EVENT_ID>/`
  - `runs/<run_id>/external_inputs/...`
  - `runs/<run_id>/<stage>/outputs/`
  - `runs/<run_id>/experiment/<name>/`
- **Ejemplo NetCDF contra release externo:**
  - `runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/<file>.nc`

---

# Atlas (geometrías) — localización inmediata (<60s)

## Rutas canónicas dentro del repo (no inventar “atlas.json”)
Estos ficheros viven versionados en el repo (preferidos por gobernanza). Lista explícita:

- `docs/ringdown/atlas/atlas_real_v2_s4.json`  *(recomendado para `s4_geometry_filter`)*
- `docs/ringdown/atlas/atlas_berti_v2_s4.json`
- `docs/ringdown/atlas/atlas_real_v1_s4.json`
- `docs/ringdown/atlas/atlas_berti_v2.json`
- `docs/ringdown/atlas/atlas_real_v1.json`
- `mvp/test_atlas_fixture.json` *(solo tests; no usar en runs reales)*

**Regla:** no existe (ni debe sugerirse) `atlas.json` en raíz. Si ves docs/ejemplos con `--atlas-path atlas.json`, trátalo como anti-ejemplo heredado.

## Qué atlas usar dónde (regla operativa)
- `s4_geometry_filter`: preferir `docs/ringdown/atlas/*_s4.json`
- tests: `mvp/test_atlas_fixture.json`
- batch/offline (`experiment_offline_batch`): documentar el atlas efectivo usado por esa CLI:
  - hoy está documentado como `docs/ringdown/atlas/atlas_berti_v2.json` en `docs/readme_experiment_4.md`
  - migración a `*_s4.json` solo cuando se haga explícita (no asumir)

## Descubrimiento (copy/paste)
Si dudas de rutas o estás en un checkout distinto:

```bash
find . -maxdepth 6 -type f \( -name "atlas*.json" -o -name "*atlas*.json" \) | sort
```

Comprobación rápida (estructura JSON):

```bash
python -c 'import json; p="docs/ringdown/atlas/atlas_real_v2_s4.json"; print("OK", p, "top_keys=", list(json.load(open(p)).keys())[:10])'
```

## Ejemplo de uso en pipeline (single-event)

```bash
python mvp/pipeline.py single \
  --event-id GW191113_071753 \
  --atlas-path docs/ringdown/atlas/atlas_real_v2_s4.json \
  --run-id <RUN_ID>
```

Si vas con defaults:

- usa `--atlas-default` (si está soportado por tu CLI) en lugar de inventar rutas.

### Comando universal para encontrar outputs sin pensar

```bash
RUN_ID="mvp_GW150914_..."
find "runs/$RUN_ID" -type f \
  \( -name 'verdict.json' \
  -o -name 'H1.h5' -o -name 'L1.h5' \
  -o -name 'provenance.json' \
  -o -name 'window_meta.json' \
  -o -name 'estimates.json' \
  -o -name 'curvature*.json' \
  -o -name 'metric_diagnostics*.json' \)
```

---

# HDF5 (LOSC/GWOSC) en 10 segundos (para que s1 no aborte)

Ruta canónica (input externo *read-only*):

`data/losc/<EVENT_ID>/`

Precheck canónico:

```bash
python tools/losc_precheck.py --event-id "$EVENT_ID" --losc-root data/losc
```

Decisión rápida A/B/C:

- **Caso A (mount/symlink roto o mal apuntado)**: `data/losc` no apunta a la caché real.
  - Reapunta `data/losc` con la estrategia estándar del equipo (symlink o bind mount).
- **Caso B (naming)**: hay `.h5/.hdf5`, pero no casan con H1/L1.
  - Crea symlinks casables `H1.h5` y `L1.h5` dentro del evento, sin renombrar originales:
- **Caso C (carpeta inexistente o vacía)**: `data/losc/<EVENT_ID>/` no existe o no tiene HDF5 válidos.
  - Pobla primero `data/losc/<EVENT_ID>/` con H1/L1.
  - Repite `tools/losc_precheck.py`.
  - Solo después corre `s1_fetch_strain`.

```bash
ln -sf "<archivo_real_H1>.h5" "data/losc/$EVENT_ID/H1.h5"
ln -sf "<archivo_real_L1>.h5" "data/losc/$EVENT_ID/L1.h5"
```

**Solo después del precheck PASS**, continúa offline con `s1` (ejemplo corto):

```bash
python mvp/s1_fetch_strain.py --run <run_id> --event-id <EVENT_ID> --detectors H1,L1 --hdf5-root data/losc --reuse-if-present
```

Procedimiento completo de bootstrap/descarga/poblado: ver `README.md` en la sección "Descarga manual rápida de strain (GWOSC) para modo offline".

**Nota de gobernanza**: `data/losc/...` es input externo. El árbol auditable del run empieza en `runs/<RUN_ID>/...`.


## 0) Regla de oro (léela primero)

Un stage **siempre resuelve rutas como**:

`<RUNS_ROOT>/<run_id>/...`

Si eso no coincide con el árbol real donde está `RUN_VALID/verdict.json`, el stage falla.

---

## Rutas canónicas

- `data/losc/<EVENT_ID>/`: caché local *read-only* de HDF5 (external input). No es generado por el pipeline.
- `runs/<run_id>/external_inputs/...`: anclaje determinista de releases externos (por ejemplo, `siegel_220_210.tar.gz`) con hash verificable para trazabilidad.
- `runs/<run_id>/<stage>/outputs/`: artefactos producidos por stages. Deben convivir con `manifest.json` y `stage_summary.json`, incluyendo hashes SHA256.
- `runs/<run_id>/experiment/<name>/`: espacio para experimentos; no debe mutar artefactos canónicos de stages ya emitidos.

### Rutas de auditoría LOSC/t0 y batch offline

> En CLI de pipeline/batch, usa `--window-catalog` para s2. Alias soportado: `--t0-catalog`.

- `runs/<audit>/experiment/losc_quality/losc_event_quality.csv`
- `runs/<audit>/experiment/losc_quality/approved_events.txt`
- `runs/<run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json` *(ruta canónica)*
- `runs/<audit>/experiment/losc_quality/gwosc_ready_events.txt`
- `runs/<prep_run_id>/external_inputs/events_with_t0.txt` *(lista derivada para batch offline)*
- `runs/<batch>/experiment/offline_batch/results.csv`

Propósito de `t0_catalog_gwosc_v2.json`: catálogo `event_id -> t0_gps` para experimentos offline/batch (por ejemplo `experiment_offline_batch --t0-catalog ...`), tratado como input externo gobernado dentro de `runs/`.

Gobernanza: tanto el catálogo t0 como artefactos derivados (por ejemplo `events_with_t0.txt`) deben vivir bajo `runs/<run_id>/...`; está prohibido escribir fuera del árbol de runs auditable.

Nota: `RUN_VALID` es un directorio; el veredicto canónico siempre vive en `RUN_VALID/verdict.json`.

Verificación mínima (contract-first, tratable/auditable):

```bash
RUN_ID="<run_id>"
STAGE="<stage>"

test -d "data/losc/<EVENT_ID>"
test -d "runs/$RUN_ID/external_inputs"
ls -l "runs/$RUN_ID/$STAGE/manifest.json" "runs/$RUN_ID/$STAGE/stage_summary.json"
sha256sum runs/$RUN_ID/external_inputs/**/* 2>/dev/null || true
```

---

## 1) Resolución de `RUNS_ROOT` (orden exacto)

1. Si existe `BASURIN_RUNS_ROOT` → `RUNS_ROOT=$BASURIN_RUNS_ROOT`
2. Si no existe → `RUNS_ROOT=<cwd>/runs`

Diagnóstico inmediato:

```bash
python -c "import os; print('BASURIN_RUNS_ROOT=', os.environ.get('BASURIN_RUNS_ROOT'))"
pwd
```

---

## 2) Tipos de run y rutas correctas

### A) Run principal

Ruta base:

```text
runs/<RUN_ID>/
```

Gating obligatorio:

```text
runs/<RUN_ID>/RUN_VALID/verdict.json
```

### B) Subrun de experimento (`t0_sweep_full`)

Ruta base típica:

```text
runs/<RUN_ID>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/
```

Gating del subrun:

```text
runs/<RUN_ID>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/RUN_VALID/verdict.json
```

> Error clásico: ejecutar un stage con `--run-id <SUBRUN_ID>` sin redefinir `RUNS_ROOT`; entonces busca en `<cwd>/runs/<SUBRUN_ID>/...` y falla.

---

## 3) Cómo ejecutar un stage sobre subrun (sin romper nada)

```bash
RUN_ID="mvp_GW150914_real_local_20260217T115536Z"
SUBRUN_ID="${RUN_ID}__t0ms0000"
SUBRUNS_ROOT="runs/$RUN_ID/experiment/t0_sweep_full/runs"

# 1) validar gating real
ls -l "$SUBRUNS_ROOT/$SUBRUN_ID/RUN_VALID/verdict.json"

# 2) ejecutar stage con RUNS_ROOT correcto
BASURIN_RUNS_ROOT="$SUBRUNS_ROOT" \
python mvp/s3b_multimode_estimates.py \
  --run-id "$SUBRUN_ID" \
  --n-bootstrap 600 \
  --seed 12345
```

---

## 4) Checklist de 30 segundos antes de lanzar cualquier stage

1. ¿`run_id` corresponde a un run principal o a un subrun?
2. ¿`RUN_VALID/verdict.json` existe exactamente en `<RUNS_ROOT>/<run_id>/RUN_VALID/verdict.json`?
3. ¿`RUNS_ROOT` efectivo es el que crees (`BASURIN_RUNS_ROOT` vs `<cwd>/runs`)?
4. Si es experimento, ¿estás en `.../t0_sweep_full/runs/<SUBRUN_ID>` y no en rutas inventadas?

Comando rápido:

```bash
test -f "<RUNS_ROOT>/<run_id>/RUN_VALID/verdict.json" \
  && echo "OK gating" || echo "ERROR gating"
```

---

## 5) Seed sweep: patrón que más confunde

Con separación por seed puede aparecer:

```text
runs/<BASE_RUN>/experiment/t0_sweep_full_seed<seed>/runsroot/<BASE_RUN>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/...
```

Procedimiento correcto:

1. Validar que exista `runsroot`.
2. Tratar `runsroot` como `scan_root` para inventario/agregados.
3. No inferir por prefijos de nombre; recorrer árbol real.

Snippet:

```bash
BASE_RUN="mvp_GW150914_nofetch_realfix_20260218T150000Z"
seed=101
RUNSROOT="runs/$BASE_RUN/experiment/t0_sweep_full_seed${seed}/runsroot"

test -d "$RUNSROOT" && echo "OK RUNSROOT" || echo "ERROR RUNSROOT"
```

---

## 6) Dónde mirar artefactos agregados (fuente de verdad)

Bajo run principal:

```text
runs/<RUN_ID>/experiment/derived/geometry_table.tsv
runs/<RUN_ID>/experiment/derived/sweep_inventory.json
```

No mezclar con outputs internos de cada subrun al evaluar estado global.

---

## 7) HDF5 externos (LOSC/GWOSC): ubicación canónica

Inputs externos de solo lectura:

```text
data/losc/<EVENT_ID>/
```

Patrones de nombre esperados:

- `*H1*.hdf5` o `*H1*.h5`
- `*L1*.hdf5` o `*L1*.h5`

Precheck obligatorio (mismo bloque canónico):

```bash
EVENT_ID=GW150914
echo "data/losc -> $(readlink -f data/losc 2>/dev/null || echo '(no symlink)')"
test -d "data/losc/$EVENT_ID" || { echo "ERROR: falta data/losc/$EVENT_ID (cache no montada/visible)"; exit 2; }
echo "H1/L1 matches:"
ls -1 "data/losc/$EVENT_ID" | egrep -i 'H1.*\.(h5|hdf5)$|L1.*\.(h5|hdf5)$' || echo "ERROR: hay ficheros pero no casan con H1/L1"
echo "total h5/hdf5:"; find "data/losc/$EVENT_ID" -maxdepth 1 -type f \( -iname '*.h5' -o -iname '*.hdf5' \) | wc -l
```

Si falla: resolver primero **Caso A (mount/symlink)** o **Caso B (nombres con symlinks H1.h5/L1.h5)** y repetir el precheck.

Ejemplo mínimo (s1 exige rutas explícitas si no hay fetch/caché):

```bash
python mvp/s1_fetch_strain.py --run <run_id> --event-id GW150914 --detectors H1,L1 \
  --local-hdf5 "H1=data/losc/GW150914/H-H1_...hdf5" \
  --local-hdf5 "L1=data/losc/GW150914/L-L1_...hdf5"
```

---

## 8) Reglas de gobernanza que NO se negocian

- No relajar `RUN_VALID`: si falta `verdict.json`, debe fallar.
- Todo output de stages vive bajo `<RUNS_ROOT>/<run_id>/...`.
- `data/losc/...` es input externo, no árbol de auditoría del run.

---

## 9) Playbook ultra-corto para IA (copiar/pegar mental)

1. Identifica `run_id` real que quieres procesar.
2. Calcula `RUNS_ROOT` real donde vive su `RUN_VALID/verdict.json`.
3. Exporta `BASURIN_RUNS_ROOT` solo si el run no está bajo `<cwd>/runs`.
4. Ejecuta stage.
5. Verifica `manifest.json` y `stage_summary.json` en la ruta efectiva.

Chequeo final:

```bash
BASE="<RUNS_ROOT>/<run_id>/<stage_name>"
ls -l "$BASE/manifest.json" "$BASE/stage_summary.json"
```

---

## Ejemplo completo: comparar contra un data release externo (NetCDF)

Patrón general (sin crear stage nuevo):

1. Guardar/anclar el tarball externo fuera del pipeline o en `runs/<run_id>/external_inputs/...` y registrar hash (`sha256sum`).
2. Extraer un `.nc` concreto bajo `runs/<run_id>/external_inputs/...`.
3. Inspeccionar cabecera con `ncdump -h` para confirmar grupos/variables.
4. Extraer percentiles con `python3-netcdf4` leyendo `group='posterior'`, variables `M` y `chi`, aplanando `chain×draw`, y calculando `p10/p50/p90`.

Ruta real usada como ejemplo:

`runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/<file>.nc`

Comandos:

```bash
RUN_ID="ext_220_210_20260227T090000Z"
BASE="runs/$RUN_ID/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210"
NC_FILE="$BASE/<file>.nc"

# 1) Integridad del release anclado
sha256sum "runs/$RUN_ID/external_inputs/siegel_220_210"/*.tar.gz

# 2) Confirmar archivo y cabecera NetCDF
test -f "$NC_FILE"
ncdump -h "$NC_FILE" | sed -n '1,120p'

# 3) Percentiles p10/p50/p90 para posterior.M y posterior.chi
python3 - <<'PY'
import numpy as np
from netCDF4 import Dataset

nc_path = "runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/<file>.nc"
with Dataset(nc_path, "r") as ds:
    g = ds.groups["posterior"]
    for name in ("M", "chi"):
        arr = np.array(g.variables[name][:]).reshape(-1)
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        print(f"{name}: p10={p10:.6g}, p50={p50:.6g}, p90={p90:.6g}")
PY
```

Checklist corto de verificación:

- `test -f "$NC_FILE"` debe pasar.
- `ncdump -h` debe mostrar `group: posterior` con variables `M` y `chi`.
- Registrar `sha256sum` del tarball y, si aplica, del `.nc` extraído para mantener artefactos auditables.

Si este archivo se sigue, se eliminan casi todos los errores de “ruta equivocada”.


## 10) Oráculo t0 v1.2: rutas de outputs

Una vez ejecutado:

```bash
python mvp/experiment_oracle_t0_ringdown.py --run-id <RUN_ID>
```

los outputs quedan en el run base:

```text
runs/<RUN_ID>/experiment/oracle_t0_ringdown/outputs/oracle_report.json
runs/<RUN_ID>/experiment/oracle_t0_ringdown/stage_summary.json
runs/<RUN_ID>/experiment/oracle_t0_ringdown/manifest.json
```

Input requerido por el oráculo (debe existir antes):

```text
runs/<RUN_ID>/experiment/t0_sweep_full_seed<seed>/outputs/t0_sweep_full_results.json
```

Si falta ese directorio/JSON, el oráculo imprime la ruta esperada exacta y el comando para regenerar el sweep (`phase=run`)

