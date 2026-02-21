# Manual operativo — `experiment_t0_sweep_full` (BASURIN) — v2 (2026-02-21)

Este manual define el **procedimiento reproducible y auditable** para ejecutar el experimento
`mvp.experiment_t0_sweep_full` sobre **datos reales** (p.ej. `GW150914`), siguiendo gobernanza BASURIN:

- IO determinista bajo `runs/<run_id>/...` (o `BASURIN_RUNS_ROOT` si se define).
- Gating por `RUN_VALID` (si falla, **no se sigue downstream**).
- Artefactos por stage: `manifest.json` + `stage_summary.json` + `outputs/*`.

> Nota crítica (anti-bucle): **NO es necesario “bajarse 500 veces” los HDF5**.  
> Los HDF5 se tratan como **external input** estable bajo `./data/losc/<EVENT_ID>/`.  
> Una vez existen ahí, el pipeline debe operar con `--local-hdf5 ...` y/o reutilizar artefactos en `runs/` sin re-descargas.

---

## 0) Convenciones y variables

### Variables mínimas
- `RUNS_ROOT`: raíz física donde viven los runs (no escribir fuera).
- `RUN_ID`: identificador del run base.
- `EVENT_ID`: evento LVK (ej. `GW150914`).
- `SEED`: semilla del experimento (ej. `123`).
- `T0_GRID_MS`: offsets del inicio de ringdown en ms (ej. `"0,5,10"`).

### Setup recomendado (shell)
```bash
cd ~/basurin/work

export RUNS_ROOT="$PWD/runs"
export BASURIN_RUNS_ROOT="$RUNS_ROOT"

export EVENT_ID="GW150914"
export RUN_ID="exp_t0_sweep_full_${EVENT_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
export SEED="123"
export T0_GRID_MS="0,5,10"

mkdir -p "$RUNS_ROOT"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "RUN_ID=$RUN_ID"
```

---

## 1) External input — HDF5 (LOSC) **una vez** (sin descargas repetidas)

### 1.1 Ruta canónica local
Los HDF5 deben vivir en:
- `./data/losc/<EVENT_ID>/`

Para `GW150914` esperamos (ejemplo estándar LOSC 32s):
- `H-H1_LOSC_4_V2-1126259446-32.hdf5`
- `L-L1_LOSC_4_V2-1126259446-32.hdf5`

### 1.2 Verificación idempotente (NO descarga si ya existen)
```bash
ls -lah "./data/losc/$EVENT_ID" || true
```

Si ves los HDF5, **NO ejecutes curl/wget**. Pasa directamente a `s1_fetch_strain` con `--local-hdf5`.

### 1.3 Descarga única (solo si faltan)
**Solo si `./data/losc/$EVENT_ID/` no contiene los HDF5**, descárgalos y registra hashes:

```bash
mkdir -p "./data/losc/$EVENT_ID"

curl -L --fail --retry 5 --retry-delay 2   -o "./data/losc/$EVENT_ID/H-H1_LOSC_4_V2-1126259446-32.hdf5"   "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW150914/v2/H-H1_LOSC_4_V2-1126259446-32.hdf5"

curl -L --fail --retry 5 --retry-delay 2   -o "./data/losc/$EVENT_ID/L-L1_LOSC_4_V2-1126259446-32.hdf5"   "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW150914/v2/L-L1_LOSC_4_V2-1126259446-32.hdf5"

sha256sum "./data/losc/$EVENT_ID/"*.hdf5 | tee "./data/losc/$EVENT_ID/SHA256SUMS.txt"
```

---

## 2) Stage s1 — `s1_fetch_strain` (real, usando HDF5 local)

### 2.1 Ejecutar (detectors en formato CSV)
> `--detectors` se pasa como `H1,L1` (coma), no separado por espacios.

```bash
python mvp/s1_fetch_strain.py   --run "$RUN_ID"   --event-id "$EVENT_ID"   --detectors H1,L1   --local-hdf5 "H1=./data/losc/$EVENT_ID/H-H1_LOSC_4_V2-1126259446-32.hdf5"   --local-hdf5 "L1=./data/losc/$EVENT_ID/L-L1_LOSC_4_V2-1126259446-32.hdf5"   --reuse-if-present
```

### 2.2 Verificación (artefactos + gate)
```bash
ls -lah "$RUNS_ROOT/$RUN_ID/s1_fetch_strain/outputs/strain.npz"
ls -lah "$RUNS_ROOT/$RUN_ID/s1_fetch_strain/manifest.json" "$RUNS_ROOT/$RUN_ID/s1_fetch_strain/stage_summary.json"
cat "$RUNS_ROOT/$RUN_ID/RUN_VALID/verdict.json" 2>/dev/null || true
```

Si `verdict != PASS`, se aborta aquí.

---

## 3) Stage s2 — `s2_ringdown_window`

> `s2_ringdown_window.py` **NO** acepta `--detector`. Usa `--strain-npz`.

```bash
python mvp/s2_ringdown_window.py   --run "$RUN_ID"   --event-id "$EVENT_ID"   --strain-npz "$RUNS_ROOT/$RUN_ID/s1_fetch_strain/outputs/strain.npz"
```

Verificación:
```bash
ls -lah "$RUNS_ROOT/$RUN_ID/s2_ringdown_window/outputs/window_meta.json"
ls -lah "$RUNS_ROOT/$RUN_ID/s2_ringdown_window/manifest.json" "$RUNS_ROOT/$RUN_ID/s2_ringdown_window/stage_summary.json"
cat "$RUNS_ROOT/$RUN_ID/RUN_VALID/verdict.json" 2>/dev/null || true
```

---

## 4) Experimento — `experiment_t0_sweep_full`

### 4.1 Fase `run` (ejecución del sweep)

El CLI **NO** soporta `--reuse-if-present`.  
Para reanudación/reutilización usa:
- `--resume-missing` (recomendada)
- (opcional) `--max-retries-per-pair`, `--resume-batch-size`

Comando recomendado:
```bash
python -m mvp.experiment_t0_sweep_full   --runs-root "$RUNS_ROOT"   --run-id "$RUN_ID"   --phase run   --seed "$SEED"   --t0-grid-ms "$T0_GRID_MS"   --detector auto   --resume-missing
```

### 4.2 Verificación auditable (seed dir)
```bash
SEED_DIR="$RUNS_ROOT/$RUN_ID/experiment/t0_sweep_full_seed${SEED}"
ls -lah "$SEED_DIR/stage_summary.json" "$SEED_DIR/manifest.json"
ls -lah "$SEED_DIR/outputs/t0_sweep_full_results.json"
cat "$SEED_DIR/stage_summary.json"
```

---

## 5) Lectura de resultados

- Resumen por seed:
  - `$SEED_DIR/stage_summary.json`
- Detalle por punto:
  - `$SEED_DIR/outputs/t0_sweep_full_results.json`
- Subruns por `t0`:
  - `$SEED_DIR/runs/<run_id>__t0msXXXX/...`

Comandos útiles:
```bash
find "$SEED_DIR/runs" -maxdepth 3 -name "stage_summary.json" | sort | head -n 80
grep -R --line-number -i "verdict\|INSUFFICIENT\|FAIL\|quality_flags" "$SEED_DIR" | head -n 120
```

---

## 6) Reintentos sin recomputar ni re-descargar

### 6.1 Reanudar lo faltante (misma seed)
```bash
python -m mvp.experiment_t0_sweep_full   --runs-root "$RUNS_ROOT"   --run-id "$RUN_ID"   --phase run   --seed "$SEED"   --t0-grid-ms "$T0_GRID_MS"   --detector auto   --resume-missing
```

### 6.2 Repetir con seed nueva (auditabilidad limpia)
```bash
NEW_SEED=$((SEED+1))
python -m mvp.experiment_t0_sweep_full   --runs-root "$RUNS_ROOT"   --run-id "$RUN_ID"   --phase run   --seed "$NEW_SEED"   --t0-grid-ms "$T0_GRID_MS"   --detector auto   --resume-missing
```

---

## 7) Troubleshooting mínimo (contract-first)

Si algo falla, captura:
1) stderr del comando que falla
2) `RUN_VALID` (si existe):
```bash
cat "$RUNS_ROOT/$RUN_ID/RUN_VALID/verdict.json" 2>/dev/null || true
```
3) `stage_summary.json` del stage que falló.

**Regla:** no sigas a fases downstream si el gate no está en PASS.

---

## 8) Apéndice: recordatorio anti-descargas

- Los HDF5 son **external input** y deben mantenerse estables en `./data/losc/<EVENT_ID>/`.
- Si ya existen, la política operativa es: **NO volver a descargar**.
- Si cambias de máquina, copia `./data/losc/` o monta ese directorio; no re-bajes.
