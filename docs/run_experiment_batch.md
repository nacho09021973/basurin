# Batch local por EVENT_ID (`tools/run_experiment_batch.py`)

Script para ejecutar por evento:
1. `s1_fetch_strain` (siempre offline y con `--local-hdf5`, `--reuse-if-present`)
2. `s2_ringdown_window` (con `--strain-npz` explícito desde s1)
3. `experiment_t0_sweep_full --phase run` (`--seed` configurable, `--resume-missing`)
4. `s5_event_row` (con `--t0-ms`, default `0`)

Toda la salida queda bajo `runs/<batch_run_id>/...` y el inventario de runs en:
`runs/<batch_run_id>/outputs/run_ids.txt`.

## Setup de `data/losc`

No se descarga H5 nunca. Debes montar los HDF5 locales así:

```text
data/losc/<EVENT_ID>/
  ...H1....hdf5
  ...L1....hdf5
```

Debe existir exactamente **1 archivo H5/HDF5 por detector** (`H1`, `L1`) para cada evento.
Si falta o hay ambigüedad, el script aborta con mensaje de setup.

## Uso

```bash
python tools/run_experiment_batch.py \
  --batch-run-id batch_demo_001 \
  --event-id GW150914 \
  --event-id GW151226 \
  --atlas-path docs/ringdown/atlas/atlas_real_v1.json \
  --seed 101 \
  --t0-ms 0 \
  --t0-grid-ms 0,2,4
```

También puedes pasar eventos con archivo:

```bash
python tools/run_experiment_batch.py \
  --events-file /ruta/events.txt \
  --atlas-path docs/ringdown/atlas/atlas_real_v1.json
```

Formato `events.txt`: un `EVENT_ID` por línea (`#` comentarios permitidos).
