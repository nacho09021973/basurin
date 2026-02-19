# BASURIN — Mapa de rutas (runs, experimentos y subruns)

Este documento describe **dónde vive cada artefacto** en BASURIN cuando se ejecutan *runs* “normales” y cuando un **experimento** (p.ej. `experiment/t0_sweep_full`) crea **subruns** anidados.

> Principio rector: **un stage interpreta siempre `--run-id <RID>` como `<RUNS_ROOT>/<RID>/...`**.  
> El `RUNS_ROOT` se resuelve por `basurin_io.resolve_out_root("runs")` y puede variar con `BASURIN_RUNS_ROOT`.

---

## 1) Resolución del `RUNS_ROOT` (crítico)

Un stage resuelve el root de runs así:

1. Si existe `BASURIN_RUNS_ROOT`, entonces:
   - `RUNS_ROOT = $BASURIN_RUNS_ROOT`
2. Si no existe, entonces:
   - `RUNS_ROOT = <cwd>/runs`

Consecuencia práctica:

- Sin `BASURIN_RUNS_ROOT`, un stage buscará `RUN_VALID` en:
  - `<cwd>/runs/<run_id>/RUN_VALID/verdict.json`

- Con `BASURIN_RUNS_ROOT=/path/to/subruns`, buscará `RUN_VALID` en:
  - `/path/to/subruns/<run_id>/RUN_VALID/verdict.json`

---

## 2) Run principal (ejecución “normal”)

Estructura típica:

```
runs/<RUN_ID>/
  RUN_VALID/
    verdict.json
  s1_fetch_strain/
    manifest.json
    stage_summary.json
    inputs/                <-- si usas --local-hdf5, copia auditable de los .h5/.hdf5
    outputs/...
  s2_ringdown_window/
    manifest.json
    stage_summary.json
    outputs/
      H1_rd.npz
      L1_rd.npz
      window_meta.json
  s3_ringdown_estimates/
    manifest.json
    stage_summary.json
    outputs/...
  ...
```

---

## 3) Experimentos y subruns (caso `experiment/t0_sweep_full`)

El experimento vive dentro del run principal:

```
runs/<RUN_ID>/experiment/t0_sweep_full/
  manifest.json
  stage_summary.json
  outputs/
    t0_sweep_full_results.json
  runs/                         <-- SUBRUNS_ROOT (raíz de subruns)
    <SUBRUN_ID>/
      RUN_VALID/verdict.json
      s2_ringdown_window/...
      s3_ringdown_estimates/...
      s3b_multimode_estimates/...
      s4c_kerr_consistency/...
```

### 3.1) Vocabulario

- `RUN_ID` (run principal): p.ej. `mvp_GW150914_real_local_20260217T115536Z`
- `SUBRUNS_ROOT` (raíz donde viven los subruns del experimento):
  - `runs/<RUN_ID>/experiment/t0_sweep_full/runs`
- `SUBRUN_ID` (run_id del subrun):
  - p.ej. `<RUN_ID>__t0ms0000`, `<RUN_ID>__t0ms0005`, …

### 3.2) Caso `seed sweep` con `runsroot`

Cuando se usa separación por semilla, el layout observado puede verse así:

```
runs/<RUN_ID>/experiment/t0_sweep_full_seed<seed>/runsroot/<RUN_ID>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/...
```

Este patrón aparece al materializar subruns por seed dentro de un `runsroot` dedicado.
Para inventarios/agregados, conviene tratar `runsroot` como `scan_root` y no inferir subruns por prefijos.

### 3.3) Dónde viven los agregados deterministas

Bajo el run principal:

```
runs/<RUN_ID>/experiment/derived/
  geometry_table.tsv
  sweep_inventory.json
```

- `geometry_table.tsv`: agregado de geometría (`s6_geometry_table.py`) con escaneo determinista.
- `sweep_inventory.json`: inventario/aceptación/decisión del experimento (`phase=inventory/finalize`).

---

## 4) El error típico (y por qué pasa)

### Síntoma

Ejecutar un stage manualmente con:

```bash
python mvp/s3b_multimode_estimates.py --run-id "<SUBRUN_ID>" ...
```

falla con:

```
RUN_VALID verdict not found: <cwd>/runs/<SUBRUN_ID>/RUN_VALID/verdict.json
```

### Causa

El subrun **no vive** en `<cwd>/runs/<SUBRUN_ID>/...` sino en:

```
runs/<RUN_ID>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/...
```

y, sin `BASURIN_RUNS_ROOT`, el stage **siempre** mira en `<cwd>/runs`.

---

## 5) Cómo ejecutar manualmente stages dentro del árbol de subruns (forma correcta)

### Opción A: exportar `BASURIN_RUNS_ROOT`

```bash
RUN_ID="mvp_GW150914_real_local_20260217T115536Z"
SUBRUN_ID="${RUN_ID}__t0ms0000"
SUBRUNS_ROOT="runs/$RUN_ID/experiment/t0_sweep_full/runs"

# Verifica gating en el árbol correcto
ls -l "$SUBRUNS_ROOT/$SUBRUN_ID/RUN_VALID/verdict.json"

# Ejecuta el stage contra el subruns_root
BASURIN_RUNS_ROOT="$SUBRUNS_ROOT"   python mvp/s3b_multimode_estimates.py     --run-id "$SUBRUN_ID"     --n-bootstrap 600     --seed 12345
```

### Opción B: (si existe en el futuro) flag `--runs-root`

Si se añade un flag `--runs-root`, el equivalente sería:

```bash
python mvp/s3b_multimode_estimates.py   --runs-root "$SUBRUNS_ROOT"   --run-id "$SUBRUN_ID"   --n-bootstrap 600   --seed 12345
```

---

## 6) Verificación de auditoría (artefactos esperados)

Si el stage ejecuta correctamente dentro del subrun, se esperan:

```
<SHOW_ROOT>/<SUBRUN_ID>/s3b_multimode_estimates/
  manifest.json
  stage_summary.json
  outputs/...
```

Con variables:

- `SHOW_ROOT = $BASURIN_RUNS_ROOT` si está definido, si no `<cwd>/runs`.

Comandos:

```bash
BASE="$SUBRUNS_ROOT/$SUBRUN_ID/s3b_multimode_estimates"
ls -l "$BASE/manifest.json" "$BASE/stage_summary.json"
ls -l "$BASE/outputs"
```

---

## 7) Diagnóstico rápido de “dónde estoy ejecutando”

Para comprobar el root efectivo:

```bash
python -c "import os; print('BASURIN_RUNS_ROOT=', os.environ.get('BASURIN_RUNS_ROOT'))"
pwd
```

Y para localizar un subrun por ID:

```bash
RUN_ID="mvp_GW150914_real_local_20260217T115536Z"
SUBRUN_ID="${RUN_ID}__t0ms0000"

find "runs/$RUN_ID/experiment/t0_sweep_full/runs" -maxdepth 2 -type f -path "*/$SUBRUN_ID/RUN_VALID/verdict.json" -print
```

### 7.1) Caso `seed sweep`: valida primero que el `RUNSROOT` exista

Cuando el experimento se ejecuta en árboles separados por semilla (ejemplo: `t0_sweep_full_seed101`),
es fácil construir una ruta sintácticamente válida pero inexistente.

Antes de leer `stage_summary.json` de un subrun, valida en este orden:

```bash
BASE_RUN="mvp_GW150914_nofetch_realfix_20260218T150000Z"
seed=101
RUNSROOT="runs/$BASE_RUN/experiment/t0_sweep_full_seed${seed}/runsroot"
sub="$RUNSROOT/$BASE_RUN/experiment/t0_sweep_full/runs/${BASE_RUN}__t0ms0008"

echo "RUNSROOT=$RUNSROOT"
test -d "$RUNSROOT" && echo "OK: RUNSROOT existe" || echo "ERROR: RUNSROOT no existe"

echo "SUB=$sub"
test -f "$sub/s3b_multimode_estimates/stage_summary.json" \
  && jq -r '.created, (.parameters | {seed, n_bootstrap})' "$sub/s3b_multimode_estimates/stage_summary.json" \
  || echo "ERROR: stage_summary.json no existe en ese subrun"
```

Si `RUNSROOT` o el `stage_summary.json` no existen, no compares seeds todavía: primero hay que
corregir la ruta efectiva (árbol equivocado, `runs_root` no materializado o `subrun_id` distinto).

### 7.2) Recordatorio para `experiment_t0_sweep_full`

- Los stages siguen interpretando `--run-id` como `<RUNS_ROOT>/<run_id>/...`.
- Para `phase=inventory` y `phase=finalize`, el experimento usa `--runs-root/--scan-root` explícitos y no depende de `BASURIN_RUNS_ROOT`.
- En `inventory/finalize`, pasa siempre seeds + grilla explícitas (`--inventory-seeds`, `--t0-grid-ms` o equivalente start/stop/step).

### 7.3) Warning reforzado sobre symlinks

No dependas de symlinks en `runsroot` para agregación/inventario: al agregar, se excluyen ancestros symlink para evitar conteo duplicado o rutas alias.

---

## 8) Nota de gobernanza (recordatorio)

- **No relajar** `RUN_VALID`: si no existe `RUN_VALID/verdict.json` en el root efectivo, el stage debe fallar.
- **IO determinista**: todo lo que el stage escriba debe quedar bajo `<RUNS_ROOT>/<run_id>/...`.

---

## 9) Recordatorio anti-descarga repetida (GW150914/GW150904)

Si ya tienes strain en local para un evento, evita volver a bajar GWOSC en cada experimento:

- ejecuta `s1_fetch_strain.py` con `--local-hdf5 DET=PATH` por detector,
- y activa `--reuse-if-present` para reutilizar `strain.npz` cuando los parámetros/hash coinciden.

Rutas clave para auditoría cuando usas local HDF5:

- copia de entrada: `<RUNS_ROOT>/<run_id>/s1_fetch_strain/inputs/*.h5`
- procedencia: `<RUNS_ROOT>/<run_id>/s1_fetch_strain/outputs/provenance.json`
