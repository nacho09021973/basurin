# Experimento 5 — Sweep de `epsilon` para modo `(2,2,1)` con `experiment_offline_batch`

## Diagnóstico

En el batch offline del modo `(2,2,1)` (`--mode-filter 221`) aparece una tensión de calibración en `s4_geometry_filter`:

- **`epsilon` bajo** ⇒ colapso de compatibilidad (`len_compatible = 0` en muchos eventos).
- **`epsilon` alto** ⇒ saturación (`len_compatible ≈ n_atlas`, en este caso picos en `730`).

Objetivo del experimento: encontrar un **epsilon de compromiso ("codo")** que evite ambos extremos para maximizar señal discriminativa y estabilidad del batch.

## Acción mínima

Ejecutar un barrido offline-first de 20 valores de `epsilon` log-espaciados para `221`, de forma **secuencial y fail-fast**, y resumir métricas por epsilon sin usar pandas.

## Inputs canónicos

- Catálogo T0 (auditoría GWOSC):
  - `runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- Lista de eventos preparada con T0:
  - `runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt`
- Atlas canónico del stage `s4_geometry_filter`:
  - `docs/ringdown/atlas/atlas_berti_v2.json`

> Nota contract-first: el atlas entra explícitamente como input de stage vía `--atlas-path`, y su hash queda reflejado en `manifest.json`/`stage_summary.json` del batch.

## Comandos

### 1) Definir sweep y ejecutar 20 epsilons (secuencial, fail-fast)

```bash
set -euo pipefail

SWEEP_ID="sweep_221_eps_$(date -u +%Y%m%dT%H%M%SZ)"
T0_CATALOG="runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json"
EVENTS_FILE="runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt"
ATLAS_PATH="docs/ringdown/atlas/atlas_berti_v2.json"

mkdir -p "runs/${SWEEP_ID}"

# 20 puntos log-espaciados en [12, 2500]
EPS_LIST="$(python - << 'PY'
import numpy as np
vals = np.geomspace(12.0, 2500.0, 20)
print(" ".join(f"{x:.6g}" for x in vals))
PY
)"

echo "epsilon,batch_run_id,results_csv" > "runs/${SWEEP_ID}/sweep_summary.csv"

for EPS in ${EPS_LIST}; do
  EPS_TAG="$(printf '%g' "${EPS}" | tr '.' 'p')"
  BATCH_RUN_ID="${SWEEP_ID}_eps${EPS_TAG}"

  python -m mvp.experiment_offline_batch \
    --batch-run-id "${BATCH_RUN_ID}" \
    --events-file "${EVENTS_FILE}" \
    --t0-catalog "${T0_CATALOG}" \
    --atlas-path "${ATLAS_PATH}" \
    --mode-filter 221 \
    --epsilon-default "${EPS}" \
    --epsilon-fallback "${EPS}"

  echo "${EPS},${BATCH_RUN_ID},runs/${BATCH_RUN_ID}/experiment/offline_batch/outputs/results.csv" \
    >> "runs/${SWEEP_ID}/sweep_summary.csv"
done
```

### 2) Auditoría rápida sin pandas (`head`, `awk`)

```bash
set -euo pipefail
SWEEP_ID="<REEMPLAZAR_SWEEP_ID>"

# Ver inventario del sweep
head -n 5 "runs/${SWEEP_ID}/sweep_summary.csv"

# Tabla agregada por epsilon:
# - n: número de eventos
# - median_len_compatible
# - count_0 (colapso)
# - count_len_compatible_730 (saturación en 730)
awk -F, '
  FNR==1 {next}
  {
    eps=$1
    file=$3
    cmd = "awk -F, '\''NR>1{v[++n]=$4; if($4==0)c0++; if($4==730)c730++} END{"
    cmd = cmd "asort(v); med=(n%2?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2);"
    cmd = cmd "printf \"%s,%d,%.0f,%d,%d\\n\",\"" eps "\",n,med,c0,c730}'\'' " file
    system(cmd)
  }
' "runs/${SWEEP_ID}/sweep_summary.csv" \
| awk 'BEGIN{print "epsilon,n,median_len_compatible,count_0,count_len_compatible_730"}1'
```

### 3) Criterio explícito para elegir el epsilon “codo”

Seleccionar `epsilon` que cumpla simultáneamente:

1. **Evitar saturación**: `count_len_compatible_730` bajo.
2. **Evitar colapso**: `count_0` bajo.
3. **Mediana intermedia** de `len_compatible` (ni ~0 ni ~730).

Heurística operativa: priorizar mínimo de `count_0 + count_len_compatible_730`; ante empate, elegir la mediana más centrada en el rango `(0,730)`.

## Resultados observados (sweep ya ejecutado)

Resumen consolidado observado:

| epsilon | mediana `len_compatible` | `count_0` | `count_len_compatible_730` |
|---:|---:|---:|---:|
| 12 .. 1076 | 0 | 44 | 0 |
| 1425 | 312 | 1 | 0 |
| 1888 | 600 | 1 | 0 |
| 2500 | 730 | 0 | 42 |

Interpretación:

- `12..1076`: colapso severo.
- `2500`: saturación fuerte.
- `1425` y `1888`: zona de transición útil.
- **Baseline elegido: `epsilon=1888`** (alta compatibilidad sin saturar masivamente).

## Verificación del batch final (baseline)

- `RUN_ID=batch_with_t0_221_eps1888_20260305T100942Z`
- `results.csv`:
  - ruta: `runs/batch_with_t0_221_eps1888_20260305T100942Z/experiment/offline_batch/outputs/results.csv`
  - `sha256=7b74189c232844bcfb4e75f2ce7ca55264fa9188435c2658a99f374d7640609d`
- Métricas reportadas:
  - `n=44`
  - `mean_accept=0.7276`
  - `median_accept=0.7500`
  - supuesto consistente con `n_atlas=800` en `s4_geometry_filter/outputs/compatible_set.json` de cada subrun PASS.

## Artefactos esperados

### A) Raíz del sweep

- `runs/<SWEEP_ID>/sweep_summary.csv`

### B) Por epsilon (por batch)

Para cada `runs/<SWEEP_ID>_eps<EPS>/experiment/offline_batch/`:

- `manifest.json`
- `stage_summary.json`
- `outputs/results.csv`

> Además, el wrapper también escribe `results.csv` en la raíz del stage (`.../offline_batch/results.csv`), pero el contrato mínimo auditable consume `outputs/results.csv`.

## Tests / checks

```bash
set -euo pipefail

SWEEP_ID="<REEMPLAZAR_SWEEP_ID>"
EPS_RUN="${SWEEP_ID}_eps1888"

# 1) Presencia de artefactos raíz del sweep
test -f "runs/${SWEEP_ID}/sweep_summary.csv"

# 2) Presencia de contrato por batch epsilon
test -f "runs/${EPS_RUN}/experiment/offline_batch/manifest.json"
test -f "runs/${EPS_RUN}/experiment/offline_batch/stage_summary.json"
test -f "runs/${EPS_RUN}/experiment/offline_batch/outputs/results.csv"

# 3) Hash reproducible del baseline final (run fijo)
sha256sum "runs/batch_with_t0_221_eps1888_20260305T100942Z/experiment/offline_batch/outputs/results.csv"

# 4) Verificar n_atlas en compatible_set de un subrun PASS (ejemplo)
EVENT_RUN_ID="<REEMPLAZAR_CON_RUN_EVENTO_PASS_DEL_BATCH>"
python - << 'PY'
import json
from pathlib import Path
p = Path('runs') / '<REEMPLAZAR_CON_RUN_EVENTO_PASS_DEL_BATCH>' / 's4_geometry_filter' / 'outputs' / 'compatible_set.json'
x = json.loads(p.read_text())
print('n_atlas=', x['n_atlas'])
PY
```

## Riesgos / supuestos

- El sweep es **costoso** (20 batches × número de eventos); mantener ejecución secuencial evita ruido de concurrencia y facilita auditoría.
- Si falta algún input canónico (`events_file`, `t0_catalog`, atlas), el experimento debe abortar antes de producir conclusiones.
- La elección del codo depende del universo de eventos (`events_with_t0.txt`); si cambia esa lista, repetir sweep completo.
- Asumimos consistencia de `n_atlas` por evento en `(2,2,1)`; validar en `compatible_set.json` cuando se cambie atlas o métrica.
