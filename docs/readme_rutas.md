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

**Regla**: los HDF5 externos viven como input *solo lectura* en:

`data/losc/<EVENT_ID>/`

**1) Lista los H5 disponibles (H1/L1)**:

```bash
EVENT_ID=GW150914
find "data/losc/$EVENT_ID" -type f \( -iname '*.hdf5' -o -iname '*.h5' \)
```

**2) Escoge uno de H1 y uno de L1** (típicamente contienen `H-H1_...` y `L-L1_...` en el nombre).

**3) Ejecuta s1 con rutas explícitas (copy/paste)**:

```bash
RUN_ID="mvp_${EVENT_ID}_real_local_$(date -u +%Y%m%dT%H%M%SZ)"
H1="/ruta/a/data/losc/$EVENT_ID/H-H1_...hdf5"
L1="/ruta/a/data/losc/$EVENT_ID/L-L1_...hdf5"

python mvp/s1_fetch_strain.py \
  --run "$RUN_ID" \
  --event-id "$EVENT_ID" \
  --detectors H1,L1 \
  --duration-s 32.0 \
  --local-hdf5 "H1=$H1" \
  --local-hdf5 "L1=$L1"
```

**Nota de gobernanza**: `data/losc/...` es input externo. El árbol auditable del run empieza en `runs/<RUN_ID>/...`.

---

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

Verificación:

```bash
find data/losc/GW150914 \( -iname '*.hdf5' -o -iname '*.h5' \) -type f
```

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

Si falta ese directorio/JSON, el oráculo imprime la ruta esperada exacta y el comando para regenerar el sweep (`phase=run`).

=======
## Codex
