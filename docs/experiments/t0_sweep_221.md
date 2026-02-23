# Experimento `t0_sweep_221`

## Propósito
Benchmark reproducible para los eventos **GW250114** y **GW231028_153006**, evaluando en qué región de `t0` aparece/desaparece la extracción del modo **221**.

## Interpretación
Este barrido **no fuerza** el modo 221. Sirve para mapear sensibilidad del resultado frente al desplazamiento de inicio de ventana (`t0`) y auditar censuras (`flags`) y estabilidad (`valid_fraction`).

## Ejecución

```bash
python mvp/experiment_t0_sweep_221.py \
  --event-id GW250114 \
  --base-run-id <RUN_ID_BASE> \
  --exp-run-id <RUN_ID_EXP> \
  --t0-grid "-3,-2,-1,0,1,2,3,5,8,10" \
  --units M
```

Con offsets en milisegundos:

```bash
python mvp/experiment_t0_sweep_221.py \
  --event-id GW231028_153006 \
  --base-run-id <RUN_ID_BASE> \
  --exp-run-id <RUN_ID_EXP> \
  --t0-grid "-6,-4,-2,0,2,4,6" \
  --units ms
```

> Unidades:
> - `M`: unidades geométricas del remanente final.
> - `ms`: milisegundos.
>
> El script convierte internamente a segundos para `s2_ringdown_window` (parámetro `dt_start_s` relativo al default canónico).

## Lectura de outputs
Ubicación: `runs/<RUN_ID_EXP>/experiment/t0_sweep_221/outputs/`

- `t0_sweep_table.json`: tabla completa por punto de grid con `has_221`, `valid_fraction_221`, `flags`, y `reason`.
- `t0_sweep_table.csv`: misma información en CSV.
- `best_t0.json`: mejor `t0` (máxima `valid_fraction_221` con `has_221=true`).
- `diagnostics.json`: conteos agregados por motivo dominante (`reason`).

Además, cada punto guarda su subrun aislado en:
`runs/<RUN_ID_EXP>/experiment/t0_sweep_221/subruns/t0_<VAL>/...`
con outputs canónicos de `s2` y `s3b` para auditoría.
