# BASURIN — mapa de rutas (versión anti-pérdida de tiempo para IA)

Objetivo: que una IA (o humano) no pierda 5–6 horas diarias por confundir `RUN_ID`, `SUBRUN_ID`, `RUNS_ROOT` y árboles de experimentos.

---

## 0) Regla de oro (léela primero)

Un stage **siempre resuelve rutas como**:

`<RUNS_ROOT>/<run_id>/...`

Si eso no coincide con el árbol real donde está `RUN_VALID/verdict.json`, el stage falla.

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

