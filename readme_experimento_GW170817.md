# Experimento GW170817 (220, 221, interseccion fisica y area)

Fecha de consolidacion: 2026-03-09T11:28:04Z (UTC)

## Objetivo

Documentar, con artefactos canonicos ya generados, los resultados para `GW170817`:

1. compatibles en modo 220,
2. compatibles en modo 221,
3. interseccion fisica `220∩221` (por `phys_key`),
4. estado del filtro de area de Hawking para este evento.

## Fuentes canonicas usadas

- `runs/batch_with_t0_220_eps2500_fixlen_20260304T160054Z/experiment/offline_batch/outputs/results.csv`
- `runs/batch_with_t0_221_eps2500_fixlen_20260304T160617Z/experiment/offline_batch/outputs/results.csv`
- `runs/analysis_common_220_221_20260304T161639Z/experiment/common_geometries/outputs/common_summary.json`
- `runs/analysis_area_entropy_20260304T162929Z/experiment/area_entropy/outputs/area_entropy_summary.json`
- `runs/analysis_area_theorem_20260304T173747Z/experiment/area_theorem/outputs/summary.json`

## Resultados GW170817

### 1) Compatibles 220 (conteo bruto por geometry_id)

- `len_compatible = 156`
- Run asociado: `mvp_GW170817_real_offline_20260304T160054Z`

### 2) Compatibles 221 (conteo bruto por geometry_id)

- `len_compatible = 730`
- Run asociado: `mvp_GW170817_real_offline_20260304T160617Z`

### 3) Interseccion fisica 220∩221 (por phys_key)

Definicion usada en ese analisis:  
`phys_key = (family, source, M_solar, chi)`

Resultados para `GW170817` en `common_summary.json`:

- `k220 = 123`
- `k221 = 730`
- `k_inter = 123`

Nota: `k220=123` no contradice `len_compatible(220)=156`; son metrica distintas:

- `156` = compatibles brutos por `geometry_id`,
- `123` = compatibles unicos tras mapear a `phys_key`.

### 4) Area/entropia sobre la interseccion

En `area_entropy_summary.json` para `GW170817`:

- `counts.k_inter = 123`
- `A_geom.inter.p50 = 245943.24352791507`
- `S_geom.inter.p50 = 61485.81088197877` (consistente con `S=A/4`)

## Aclaracion de la duda (importante)

No hay contradiccion entre lo dicho para `GW170817` y el resultado de `507` geometrías:

- Para `GW170817`, en el run de teorema de area (`analysis_area_theorem_20260304T173747Z`), el evento **no aparece** en `summary.json`.  
  Por tanto, ahi no se calculo filtro Hawking per-event para `GW170817`.
- El valor `507` viene de un experimento **distinto** en `GW150914`, con:
  - dataset/evento distinto,
  - umbrales distintos (`threshold_220=50`, `threshold_221=50`),
  - y un `area_obs` construido como proxy estricto.

En resumen:

- `GW170817`: tenemos 220/221/interseccion fisica y analisis de area/entropia.
- `GW170817`: no hay resultado Hawking canonico en ese run de area theorem.
- `507`: corresponde a `GW150914`, no a `GW170817`.

## Si quieres cerrar Hawking para GW170817

Hace falta ejecutar un flujo per-event de area theorem que incluya `GW170817` y sus insumos IMR (posteriores pre-merger) para poder aplicar formalmente `A_final >= A_initial`.
