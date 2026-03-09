# GW150914 external oracle for early BASURIN validation

## 1. Propósito

Este oráculo externo **no prueba directamente** la validez física final de una geometría golden. Su función es verificar, de forma rápida y auditable, que los resultados intermedios de BASURIN para GW150914 no contradicen señales externas ya publicadas en la literatura de ringdown/área.

## 2. Alcance

Este oráculo aplica únicamente a:

- chequeos tempranos de coherencia externa,
- el primer evento de validación (GW150914),
- interpretación operativa de etiquetas PASS/WARN/FAIL.

Este documento **no sustituye** validación física completa, inferencia bayesiana final ni robustez poblacional multi-evento.

## 3. Referencias externas base

- **Isi et al. 2019**: referencia de contexto ringdown con lectura relativamente optimista de compatibilidad modal; se usa como marco para no descartar prematuramente supervivencias tempranas.
- **Carullo et al. 2019**: referencia conservadora para consistencia de modos y prudencia interpretativa en tests de ringdown.
- **Correia et al. 2024**: referencia reciente de contraste conservador en pruebas de consistencia en ringdown para GW150914.
- **Cabero et al. 2018**: referencia clásica para validaciones ligadas a ley de área en eventos GW.
- **Correia & Capano 2024**: referencia de contexto actualizado para interpretaciones cautas de constraints físicos y consistencia.
- **Nelli et al. 2024**: referencia para enmarcar robustez/inestabilidad de conclusiones cuando se varían hipótesis y supuestos.

## 4. Política PASS / WARN / FAIL por artefacto

### A) `s4g_mode220_geometry_filter`

- **PASS**
  - `n_geometries_accepted > 0`
- **WARN**
  - `n_geometries_accepted == 1`
- **FAIL**
  - `n_geometries_accepted == 0`

### B) `s4h_mode221_geometry_filter`

- **PASS**
  - `n_geometries_accepted > 0`
- **WARN**
  - `n_geometries_accepted == 0`
  - `n_geometries_accepted == 1`
- **FAIL**
  - solo si el stage es internamente inconsistente, corrupto o patológico.

### C) `s4i_common_geometry_intersection`

- **PASS**
  - `n_common_geometries > 1`
- **WARN**
  - `n_common_geometries == 1`
  - `n_common_geometries == 0`
- **FAIL**
  - no aplica por cardinalidad sola; solo por inconsistencia contractual.

### D) `s4j_hawking_area_filter`

- **PASS**
  - `n_golden_geometries > 0`
- **WARN**
  - `n_golden_geometries == 0` y `n_common_geometries > 0`
- **FAIL**
  - violación sistemática y robusta de `delta_area < 0`.

### E) `experiment_single_event_golden_robustness`

- **PASS**
  - `robustness_verdict == "ROBUST_UNIQUE"`
- **WARN**
  - `robustness_verdict == "UNSTABLE_UNIQUE"`
  - `robustness_verdict == "NOT_UNIQUE"`
- **FAIL**
  - `robustness_verdict == "NO_DATA"`

## 5. Cómo interpretar correctamente la matriz

- **Un singleton final no implica verdad física automática.**
- **La etiqueta `ROBUST_UNIQUE` sí eleva significativamente la credibilidad interna del hallazgo.**
- **`WARN` no significa invalidez; significa resultado científicamente interpretable pero todavía no fuerte.**

## 6. Rutas exactas de artefactos

Rutas canónicas existentes en el repo para estos chequeos:

- `runs/<run_id>/s4g_mode220_geometry_filter/outputs/geometries_220.json`
- `runs/<run_id>/s4h_mode221_geometry_filter/outputs/mode221_filter.json`
- `runs/<run_id>/s4i_common_geometry_intersection/outputs/common_intersection.json`
- `runs/<run_id>/s4j_hawking_area_filter/outputs/hawking_area_filter.json`
- `runs/<run_id>/experiment/golden_robustness_<timestamp>/robustness_summary.json`

## 7. Uso recomendado

Checklist operativa mínima:

1. Correr stages del evento GW150914 hasta completar s4g/s4h/s4i/s4j y experimento de robustez.
2. Abrir artefactos JSON de salida en rutas canónicas.
3. Contrastar los resultados con `config/external_oracle_gw150914.yaml`.
4. Registrar PASS/WARN/FAIL en notebook técnico o en un `stage_summary` complementario de auditoría.

## 8. Límites epistemológicos

Este oráculo:

- no sustituye comparación con posterior sampling,
- no sustituye estudios de cobertura,
- no demuestra por sí solo “la geometría”,
- sirve como guardarraíl de coherencia externa temprana.
