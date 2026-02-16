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

