# Experimento GW170817 (220/221, interseccion fisica, area de Hawking)

Fecha: 2026-03-09 (UTC)

## Objetivo

Documentar de forma reproducible lo hecho para `GW170817`:

1. Corregir la interseccion `220 ∩ 221` para que sea fisicamente consistente.
2. Construir un atlas mejor (multi-modo por geometria canonica).
3. Re-ejecutar el pipeline golden y verificar si aparece geometria comun.
4. Buscar en bibliografia si existe un resultado externo equivalente a `Kerr_M90_a0.8631`.

---

## Contexto del problema inicial

Con el flujo original:

- `s4g_mode220_geometry_filter` devolvia compatibles en 220.
- `s4h_mode221_geometry_filter` devolvia compatibles en 221.
- `s4i_common_geometry_intersection` devolvia `n_common = 0`.

La causa no era fisica, era de identificadores: en el atlas v2 las geometrías estaban separadas por modo, por ejemplo:

- `Kerr_M62_a0.6600_l2m2n0`
- `Kerr_M62_a0.6600_l2m2n1`

`s4i` intersectaba por string literal y por eso no emparejaba 220 con 221 del mismo `(M,a)`.

---

## Paso 1 — Parche en s4i (interseccion canonica)

Archivo modificado:

- `mvp/s4i_common_geometry_intersection.py`

Cambios:

- Se añade `canonical_geometry_id()` que elimina sufijo de modo `"_lXmYnZ"`.
- `compute_intersection()` pasa a intersectar por ID canonico.
- Se mantiene alias legacy `S4G_OUTPUT_REL` para compatibilidad de imports antiguos.

Tests ajustados:

- `tests/test_golden_geometry_pipeline_integration.py`
  - actualizado contrato de filename (`mode220_filter.json` primario + legacy),
  - nuevo test de interseccion mode-agnostic por sufijo.

Validacion:

- `pytest tests/test_golden_geometry_pipeline_integration.py -q`
- Resultado: `9 passed`.

---

## Paso 2 — Atlas mejorado (v3 multimodo)

Archivo nuevo:

- `mvp/tools/generate_multimode_atlas_v3.py`

Idea:

- tomar atlas v2 (mode-split),
- canonizar `geometry_id` (sin sufijo de modo),
- agrupar entradas de la misma geometria fisica,
- emitir una fila por geometria con `mode_220` y/o `mode_221`.

Comando usado:

```bash
.venv/bin/python -m mvp.tools.generate_multimode_atlas_v3 \
  --in-atlas docs/ringdown/atlas/atlas_berti_v2.json \
  --out docs/ringdown/atlas/atlas_berti_v3_multimode.json
```

Artefactos generados:

- `docs/ringdown/atlas/atlas_berti_v3_multimode.json`
- `docs/ringdown/atlas/atlas_berti_v3_multimode_s4.json`

Resumen de generacion:

- `N_SOURCE = 2637`
- `N_TOTAL = 1037`
- `N_MODE220 = 1037`
- `N_MODE221 = 800`
- `N_BOTH = 800`

---

## Paso 3 — Rerun de GW170817 con atlas v3

Run trabajado:

- `runs/mvp_GW170817_real_offline_20260305T174006Z`

Comandos:

```bash
.venv/bin/python -m mvp.s4g_mode220_geometry_filter \
  --run-id mvp_GW170817_real_offline_20260305T174006Z \
  --atlas-path docs/ringdown/atlas/atlas_berti_v3_multimode.json

.venv/bin/python -m mvp.s4h_mode221_geometry_filter \
  --run-id mvp_GW170817_real_offline_20260305T174006Z \
  --atlas-path docs/ringdown/atlas/atlas_berti_v3_multimode.json

.venv/bin/python -m mvp.s4i_common_geometry_intersection \
  --run-id mvp_GW170817_real_offline_20260305T174006Z

.venv/bin/python -m mvp.s4j_hawking_area_filter \
  --run-id mvp_GW170817_real_offline_20260305T174006Z
```

Resultado (parametros por defecto):

- `s4g`: `n_geometries_accepted = 100`
- `s4h`: `n_passed = 69`
- `s4i`: `n_common = 50` (antes era 0)
- `s4j`: `n_golden = 50`, `verdict = PASS`

Artefactos clave:

- `runs/mvp_GW170817_real_offline_20260305T174006Z/s4i_common_geometry_intersection/outputs/common_intersection.json`
- `runs/mvp_GW170817_real_offline_20260305T174006Z/s4j_hawking_area_filter/outputs/hawking_area_filter.json`

---

## Busqueda de singleton (geometria unica)

Se hizo un barrido amplio en memoria sobre:

- `threshold_220`: `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.605, 6.0, 9.21, 12.0, 20.0]`
- `threshold_221`: `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.605, 6.0, 9.21, 12.0, 20.0]`
- `sigma_scale`: `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]`

Resumen:

- escenarios totales: `1296`
- con golden no vacio: `1031`
- singleton: `5`
- geometria singleton unica: `Kerr_M90_a0.8631` (aparece en los 5 singleton)

Snapshot del barrido:

- `runs/mvp_GW170817_real_offline_20260305T174006Z/experiment/manual_sweep_gw170817_v3.json`

Escenario puntual confirmado (1 escenario):

```bash
.venv/bin/python -m mvp.experiment_single_event_golden_robustness \
  --run-id mvp_GW170817_real_offline_20260305T174006Z \
  --atlas-path docs/ringdown/atlas/atlas_berti_v3_multimode.json \
  --threshold-220-values 2.0 \
  --threshold-221-values 3.0 \
  --sigma-scale-values 0.5 \
  --area-tolerance-values 0.0
```

Resultado:

- `verdict = ROBUST_UNIQUE`
- `robust_unique_geometry_id = Kerr_M90_a0.8631`

Salida:

- `runs/mvp_GW170817_real_offline_20260305T174006Z/experiment/golden_robustness_20260309T121055Z/robustness_summary.json`

---

## Bibliografia: ¿alguien reporta el mismo resultado `Kerr_M90_a0.8631` para GW170817?

### Conclusión corta

No se encontró bibliografia primaria que reporte para `GW170817` una solucion tipo:

- masa remanente `~90 M_sun`
- spin `a ~ 0.863`

Ese punto (`Kerr_M90_a0.8631`) aparece como geometria del atlas/grid interno del experimento, no como estimacion astrofisica publicada para GW170817.

### Evidencia bibliografica consultada

1. **GW170817 observacion original (LIGO/Virgo)**  
   Total del sistema en inspiral: `2.74^{+0.04}_{-0.01} M_sun`, consistente con BNS.  
   Fuente: `arXiv:1710.05832` (LIGO-P170817), PRL 119, 161101 (2017).  
   URL: https://arxiv.org/abs/1710.05832

2. **Propiedades de la fuente y post-merger (LIGO/Virgo)**  
   No deteccion concluyente de señal post-merger; se reportan limites superiores.  
   Fuente: `LIGO-P1800061` / `GW170817_SourceProperties.pdf` (2018).  
   URL: https://dcc.ligo.org/LIGO-P1800061/public

3. **Busqueda especifica de ondas post-merger del remanente (LIGO/Virgo)**  
   No deteccion; limites de strain por encima de la mayoria de modelos.  
   Fuente: `arXiv:1710.09320`.  
   URL: https://arxiv.org/abs/1710.09320

4. **Trabajo no canónico sobre “echoes”**  
   Reporta remanente `2.6–2.7 M_sun` con spin `0.84–0.87` (tentativo).  
   Fuente: `arXiv:1803.10454`.  
   URL: https://arxiv.org/abs/1803.10454  
   Incluso en este caso, el orden de masa es `~3 M_sun`, no `90 M_sun`.

5. **Interpretacion multi-mensajero alternativa**  
   Evidencia calorimetrica para BH rotante de `~3 M_sun`.  
   Fuente: `arXiv:1910.12730` (ApJ Lett. 876 L2).  
   URL: https://arxiv.org/abs/1910.12730

---

## Lectura fisica del resultado interno

`Kerr_M90_a0.8631` debe leerse como:

- un punto del atlas que mejor satisface simultaneamente filtros 220/221 en cierto subespacio de hiperparametros,
- no como “medida directa” publicada del remanente real de GW170817.

Para afirmacion fisica fuerte haria falta:

1. atlas calibrado especificamente a BNS/GW170817 (escala de masas realista del remanente),
2. modelado de remanente y post-merger consistente con EOS y colapso,
3. comparacion con inferencia bayesiana y limites instrumentales oficiales.
