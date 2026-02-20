# Manual operativo — `experiment_t0_sweep_full` (BASURIN)

Este manual define el **procedimiento reproducible y auditable** para ejecutar el experimento
`mvp.experiment_t0_sweep_full` sobre datos reales (p.ej. GW150914), siguiendo gobernanza BASURIN:
IO determinista bajo `runs/<run_id>/...`, gating por `RUN_VALID`, y artefactos por stage.

---

## 0) Convenciones y variables

### Variables mínimas
- `RUNS_ROOT`: raíz física donde viven los runs (no escribir fuera).
- `RUN_ID`: identificador del run base.
- `EVENT_ID`: evento LVK (ej. `GW150914`).
- `SEED`: semilla del experimento (ej. `123`).
- `T0_GRID_MS`: grid de offsets en ms (ej. `0,5,10`).

### Nota crítica sobre interfaces
- `experiment_t0_sweep_full` **acepta** `--runs-root`.
- `s1_fetch_strain` (y posiblemente otros stages) **NO acepta** `--runs-root` y toma el root de:
  - `BASURIN_RUNS_ROOT` (env), o
  - su default interno.
  
**Regla operativa:** antes de ejecutar stages `mvp/sX_*.py`, exporta:
```bash
export BASURIN_RUNS_ROOT="$RUNS_ROOT"
```

---

## 1) External inputs (HDF5 LOSC/GWOSC)

### Ubicación canónica recomendada

Guardar HDF5 externos por evento en:

* `./data/losc/<EVENT_ID>/`

Ejemplo real (GW150914):

* `./data/losc/GW150914/H-H1_LOSC_4_V2-1126259446-32.hdf5`
* `./data/losc/GW150914/L-L1_LOSC_4_V2-1126259446-32.hdf5`

Verificación:

```bash
ls -lah ./data/losc/GW150914
```

---

## 2) Flujo completo del experimento (visión general)

El experimento **no es self-contained**: requiere un **run base** con upstream canónico ya materializado.

### Prerrequisitos del run base (`runs/<RUN_ID>/...`)

1. `s1_fetch_strain` **PASS**

   * produce `s1_fetch_strain/outputs/strain.npz`
2. `s2_ringdown_window` **PASS**

   * produce `s2_ringdown_window/manifest.json`
   * produce `s2_ringdown_window/outputs/window_meta.json`
3. (según configuración) puede requerir también `s3_ringdown_estimates` antes de barrer (depende del contrato del experimento).

Si falta cualquiera, el experimento debe **fallar rápido** con mensaje auditable (ruta exacta).

---

## 3) Procedimiento paso a paso (ejecutable)

### 3.1 Crear runs_root y variables

```bash
RUNS_ROOT="$(mktemp -d)/runs"
RUN_ID="exp_t0_sweep_full_GW150914"
EVENT_ID="GW150914"
SEED="123"
T0_GRID_MS="0,5,10"

export BASURIN_RUNS_ROOT="$RUNS_ROOT"
```

### 3.2 Stage s1 — `s1_fetch_strain` (datos reales desde HDF5 local)

#### (A) Selección explícita (recomendado)

```bash
python mvp/s1_fetch_strain.py \
  --run "$RUN_ID" \
  --event-id "$EVENT_ID" \
  --detectors H1,L1 \
  --local-hdf5 "H1=./data/losc/$EVENT_ID/H-H1_LOSC_4_V2-1126259446-32.hdf5" \
  --local-hdf5 "L1=./data/losc/$EVENT_ID/L-L1_LOSC_4_V2-1126259446-32.hdf5" \
  --reuse-if-present
```

#### Verificación mínima (contract-first)

```bash
ls -lah "$RUNS_ROOT/$RUN_ID/s1_fetch_strain/outputs/strain.npz"
cat "$RUNS_ROOT/$RUN_ID/RUN_VALID/verdict.json" 2>/dev/null || true
```

**Artefactos esperados (mínimo):**

* `runs/<RUN_ID>/s1_fetch_strain/outputs/strain.npz`
* `runs/<RUN_ID>/s1_fetch_strain/manifest.json`
* `runs/<RUN_ID>/s1_fetch_strain/stage_summary.json`
* `runs/<RUN_ID>/RUN_VALID/verdict.json` (si el stage lo gestiona)

---

### 3.3 Stage s2 — `s2_ringdown_window` (ventana/metadata para ringdown)

> **Nota:** los flags exactos dependen de vuestro script. Si dudas, usa:

```bash
python mvp/s2_ringdown_window.py --help
```

Invocación típica (ajustar si s2 lee todo del run):

```bash
python mvp/s2_ringdown_window.py \
  --run "$RUN_ID" \
  --event-id "$EVENT_ID" \
  --detector auto
```

#### Verificación mínima

```bash
ls -lah "$RUNS_ROOT/$RUN_ID/s2_ringdown_window/manifest.json"
ls -lah "$RUNS_ROOT/$RUN_ID/s2_ringdown_window/outputs/window_meta.json"
```

**Artefactos esperados (mínimo):**

* `runs/<RUN_ID>/s2_ringdown_window/manifest.json`
* `runs/<RUN_ID>/s2_ringdown_window/stage_summary.json`
* `runs/<RUN_ID>/s2_ringdown_window/outputs/window_meta.json`

---

## 4) Ejecutar `experiment_t0_sweep_full`

### 4.1 Phase `run` (subruns seed-scoped)

Este comando **sí** acepta `--runs-root`.

```bash
python -m mvp.experiment_t0_sweep_full \
  --runs-root "$RUNS_ROOT" \
  --run-id "$RUN_ID" \
  --phase run \
  --seed "$SEED" \
  --t0-grid-ms "$T0_GRID_MS"
```

#### Estructura esperada en disco

* Seed dir:

  * `runs/<RUN_ID>/experiment/t0_sweep_full_seed<SEED>/`
* Subruns dentro del seed dir:

  * `.../experiment/t0_sweep_full_seed<SEED>/runs/<SUBRUN_ID>/...`
* Ejemplo de artefacto crítico por subrun:

  * `.../runs/<SUBRUN_ID>/s2_ringdown_window/outputs/window_meta.json`

#### Verificación rápida

```bash
find "$RUNS_ROOT/$RUN_ID/experiment" -maxdepth 6 -name window_meta.json -print | head
```

---

## 5) Fases adicionales (según uso)

El experimento soporta fases:

* `inventory`: computa inventario (pares seed/t0 etc.).
* `finalize`: consolida resultados.
* `diagnose`: diagnóstico/explicación de bloqueos.
* `backfill_window_meta`: backfill de metadata si faltó.

Listado:

```bash
python -m mvp.experiment_t0_sweep_full --help
```

### Nota sobre atlas

`--atlas-path` **solo** debe exigirse si el plan incluye un stage atlas-dependiente (p.ej. `s4c_kerr_consistency`).
Si no ejecutas ese stage en la fase actual, `--atlas-path` no debe bloquear.

---

## 6) Errores comunes y resolución (auditables)

### 6.1 `RUN_VALID verdict not found: .../RUN_VALID/verdict.json`

**Causa:** fase intenta gatear un run no inicializado.
**Solución:** el experimento debe inicializar RUN_VALID en `phase=run` antes del require. (Si ya está corregido, no debería ocurrir).

### 6.2 `preflight failed: missing base s1 strain NPZ path=.../s1_fetch_strain/outputs/strain.npz`

**Causa:** no se ejecutó `s1_fetch_strain` en el run base.
**Solución:** ejecutar `s1_fetch_strain` y verificar `strain.npz` en la ruta indicada.

### 6.3 `Missing s2 manifest required for phase=run: .../s2_ringdown_window/manifest.json`

**Causa:** no se ejecutó `s2_ringdown_window` en el run base.
**Solución:** ejecutar `s2_ringdown_window` y verificar `manifest.json` y `window_meta.json`.

### 6.4 `s1_fetch_strain.py: error: unrecognized arguments: --runs-root ...`

**Causa:** el stage no acepta `--runs-root`.
**Solución:** exportar `BASURIN_RUNS_ROOT="$RUNS_ROOT"` y usar `--run`.

### 6.5 `No HDF5 found for H1 from ...`

**Causa:** ruta HDF5 incorrecta o fichero no existe.
**Solución:** listar HDF5 reales en `./data/losc/<EVENT_ID>/` y pasar rutas reales a `--local-hdf5`.

---

## 7) Checklist final (antes de abrir PR / auditoría)

* [ ] `runs/<RUN_ID>/s1_fetch_strain/outputs/strain.npz` existe
* [ ] `runs/<RUN_ID>/s2_ringdown_window/manifest.json` existe
* [ ] `runs/<RUN_ID>/s2_ringdown_window/outputs/window_meta.json` existe
* [ ] `python -m mvp.experiment_t0_sweep_full --phase run ...` crea seed dir y subruns
* [ ] Cada stage escribe `manifest.json` + `stage_summary.json` + `outputs/*`
* [ ] No hay escrituras fuera de `runs/<RUN_ID>/...` (excepto lectura de `./data/losc/...`)

---

## 8) Comandos “mínimos” de ejemplo (GW150914, H1/L1)

```bash
RUNS_ROOT="$(mktemp -d)/runs"
RUN_ID="exp_t0_sweep_full_GW150914"
EVENT_ID="GW150914"
SEED="123"
T0_GRID_MS="0,5,10"

export BASURIN_RUNS_ROOT="$RUNS_ROOT"

python mvp/s1_fetch_strain.py \
  --run "$RUN_ID" \
  --event-id "$EVENT_ID" \
  --detectors H1,L1 \
  --local-hdf5 "H1=./data/losc/$EVENT_ID/H-H1_LOSC_4_V2-1126259446-32.hdf5" \
  --local-hdf5 "L1=./data/losc/$EVENT_ID/L-L1_LOSC_4_V2-1126259446-32.hdf5" \
  --reuse-if-present

python mvp/s2_ringdown_window.py \
  --run "$RUN_ID" \
  --event-id "$EVENT_ID" \
  --detector auto

python -m mvp.experiment_t0_sweep_full \
  --runs-root "$RUNS_ROOT" \
  --run-id "$RUN_ID" \
  --phase run \
  --seed "$SEED" \
  --t0-grid-ms "$T0_GRID_MS"
```
