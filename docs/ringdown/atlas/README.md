# Atlas real QNM (v1)

Archivos:
- atlas_real_v1.json: atlas completo con provenance
- atlas_real_v1_s4.json: formato compatible con s4_geometry_filter
- SHA256SUMS.txt: hashes para auditoría

Generación:
- Script: mvp/tools/generate_real_atlas.py
- Método: qnm (Leaver / continued fraction) + escalado a masa remanente (GW150914)

## Métricas y umbrales (S4 / eps sweep)

- `metric = mahalanobis_log` opera en log-espacio: `ln(f)` y `ln(Q)`.
- Semántica del umbral: `epsilon` se interpreta como umbral en **d²** y se registra como `threshold_d2 = epsilon` en `compatible_set.json`.
- Trazabilidad: `distance` se materializa como `sqrt(d2_min)` (donde `d2_min` es el mínimo d² observado) para que el artefacto sea auditable.



## Atlas real QNM (v2, s4)

- Archivo: `docs/ringdown/atlas/atlas_real_v2_s4.json`
- Motivo: `atlas_real_v1_s4.json` no contiene puntos explícitos GR de desviación (`delta_f_frac=0.0`, `delta_Q_frac=0.0`), por lo que no permitía comparar de forma directa “GR vs deviations” en el mismo espacio parametrizado.
- Cambio: se añaden entradas canónicas `bK_GR_a<spin>_df+0.00_dQ+0.00` para los spins Kerr presentes (incluye al menos `0.80` y `0.95`).
- Recomendación: usar v2 para análisis interpretables de preferencia relativa vs GR.
- SHA256: `f363903113c92bc0c85783cb0e285f6529719fe317374b6c022e9a393e17626c`
