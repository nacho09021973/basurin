# PROMPT CONSOLIDADO S4 — Informatividad + delta_lnL + Three-state gate

```
Actúa como ingeniero contract-first de BASURIN. Vas a hacer 3 cambios
coordinados en mvp/s4_geometry_filter.py que resuelven la saturación del modo 221.

Los 3 cambios son aditivos y no rompen comportamiento existente.

═══════════════════════════════════════════════════════════════
ANTES DE ESCRIBIR CÓDIGO, lee estos ficheros COMPLETOS:
═══════════════════════════════════════════════════════════════

- mvp/s4_geometry_filter.py
- mvp/contracts.py (buscar s4_geometry_filter)
- mvp/distance_metrics.py (si existe)
- tests/unit/ que contengan "s4" o "geometry_filter"
- Un compatible_set.json real si existe en runs/

═══════════════════════════════════════════════════════════════
CONTEXTO CIENTÍFICO (no inventar, es evidencia de runs reales):
═══════════════════════════════════════════════════════════════

- Con metric=mahalanobis_log y epsilon=2500, el modo 221 produce
  n_compatible/n_atlas ≈ 0.91 (saturación) para ~42/44 eventos.
- d² en BASURIN NO es χ²(2). Es un score geométrico. d²_min típico: 1000–5000.
- Necesitamos:
  (A) Diagnosticar saturación (acceptance_fraction + informative flag)
  (B) Ofrecer un criterio alternativo relativo (delta_lnL)
  (C) Clasificar el resultado en 3 estados (SATURATED/EMPTY/OK)
- Los 3 cambios son COMPATIBLES con RUN_VALID. No bloquean nada.

═══════════════════════════════════════════════════════════════
CAMBIO A: acceptance_fraction + informative flag
═══════════════════════════════════════════════════════════════

1. Nuevo argumento CLI:
   --informative-threshold FLOAT (default 0.80)

2. Tras calcular n_compatible y n_atlas:
   acceptance_fraction = n_compatible / n_atlas  # float 0.0–1.0
   informative = (acceptance_fraction <= informative_threshold)

3. Registrar en stage_summary.json:
   "acceptance_fraction": <float>,
   "informative": <bool>,
   "informative_threshold": <float>

═══════════════════════════════════════════════════════════════
CAMBIO B: modo threshold delta_lnL
═══════════════════════════════════════════════════════════════

1. Nuevos argumentos CLI:
   --threshold-mode {d2, delta_lnL}  (default: "d2")
   --delta-lnL FLOAT                 (default: 3.0)

2. Lógica:
   Si threshold_mode == "d2":
     EXACTAMENTE el comportamiento actual. Cero cambios.
   
   Si threshold_mode == "delta_lnL":
     a) Calcular d² para TODAS las geometrías del atlas.
     b) d²_min = min(todos los d²)
     c) Para cada geometría i:
        delta_lnL_i = -0.5 * (d²_i - d²_min)
     d) Compatible si delta_lnL_i >= -delta_threshold
     e) La geometría con d²_min SIEMPRE es compatible (delta_lnL=0).

3. En stage_summary.json:
   "threshold_mode": "d2" | "delta_lnL",
   "delta_lnL_threshold": <float> | null,
   "d2_min": <float>

4. En compatible_set.json, para CADA geometría compatible:
   "delta_lnL": <float>
   (Calcular siempre, incluso en modo d2. Es auditoría gratuita.)

5. NO añadir numpy. Solo stdlib math.

═══════════════════════════════════════════════════════════════
CAMBIO C: three-state gate
═══════════════════════════════════════════════════════════════

1. Tras calcular acceptance_fraction:
   if n_compatible == 0:
       filter_status = "EMPTY"
   elif acceptance_fraction > informative_threshold:
       filter_status = "SATURATED"
   else:
       filter_status = "OK"

2. Registrar en stage_summary.json:
   "filter_status": "OK" | "SATURATED" | "EMPTY"

3. NO cambiar RUN_VALID.
4. NO abortar si SATURATED o EMPTY.

═══════════════════════════════════════════════════════════════
TESTS (en tests/unit/test_s4_desaturation.py)
═══════════════════════════════════════════════════════════════

--- Bloque A: acceptance_fraction ---

Test 1: test_acceptance_fraction_computation
  n_compatible=730, n_atlas=800 → abs(af - 0.9125) < 1e-4

Test 2: test_informative_true_below_threshold
  n_compatible=200, n_atlas=800 → informative == True

Test 3: test_informative_false_when_saturated
  n_compatible=730, n_atlas=800, threshold=0.80 → informative == False

Test 4: test_acceptance_fraction_in_stage_summary
  Verificar 3 campos: acceptance_fraction (float), informative (bool),
  informative_threshold (float)

--- Bloque B: delta_lnL ---

Test 5: test_delta_lnL_desaturates
  Atlas 10 geometrías: d²=[100,105,110,120,150,200,500,800,1000,2000]
  delta_threshold=3.0, d²_min=100
  Compatible si d² <= 106 → n_compatible=2

Test 6: test_d2_mode_unchanged
  Misma fixture, threshold_mode="d2", epsilon=2500 → n_compatible=10

Test 7: test_delta_lnL_always_includes_best
  Cualquier atlas, cualquier delta_threshold >= 0 → d²_min siempre compatible

Test 8: test_delta_lnL_in_stage_summary
  threshold_mode="delta_lnL" → campos threshold_mode, delta_lnL_threshold, d2_min

Test 9: test_delta_lnL_field_in_compatible_set_d2_mode
  threshold_mode="d2" → cada geometría tiene campo delta_lnL

--- Bloque C: three-state ---

Test 10: test_filter_status_ok
  n_compatible=200, n_atlas=800 → "OK"

Test 11: test_filter_status_saturated
  n_compatible=730, n_atlas=800 → "SATURATED"

Test 12: test_filter_status_empty
  n_compatible=0 → "EMPTY"

Test 13: test_filter_status_edge_exactly_at_threshold
  af=0.80, threshold=0.80 → "OK" (not strictly greater)

--- Bloque D: backward compat ---

Test 14: test_default_d2_mode_identical_to_baseline
  Sin pasar --threshold-mode → default "d2"
  Output compatible_set IDÉNTICO al actual MÁS campos nuevos (delta_lnL por geom)

═══════════════════════════════════════════════════════════════
VALIDACIÓN POST-IMPLEMENTACIÓN
═══════════════════════════════════════════════════════════════

# 1. Tests nuevos
pytest -q -o "addopts=" tests/unit/test_s4_desaturation.py -v

# 2. Tests existentes de s4 (regresión)
pytest -q -o "addopts=" tests/unit/ -k "s4" -v

# 3. Smoke test CLI
python mvp/s4_geometry_filter.py --help | grep -E "threshold-mode|delta-lnL|informative"

# 4. Si hay run existente, smoke real:
# python mvp/s4_geometry_filter.py --run-id <run_id> --threshold-mode delta_lnL --delta-lnL 3.0

═══════════════════════════════════════════════════════════════
ARTEFACTOS ESPERADOS
═══════════════════════════════════════════════════════════════

- mvp/s4_geometry_filter.py (modificado: 3 bloques de cambios)
- tests/unit/test_s4_desaturation.py (nuevo: 14 tests)

═══════════════════════════════════════════════════════════════
RIESGOS Y SUPUESTOS
═══════════════════════════════════════════════════════════════

- Riesgo cero en modo d2: comportamiento idéntico al actual + campos extra.
- Si delta_threshold demasiado bajo → n_compatible=1. No es bug, es informativo.
- Si el código actual de s4 no tiene una función separada para "construir compatible_set"
  y la lógica está inline en main(), el refactor mínimo es extraer esa lógica a una
  función con signature clara. Pero solo si es necesario para inyectar threshold_mode.
- Supuesto: d² ya se calcula para todas las geometrías. Si se calcula solo para
  las que pasan un pre-filtro, documentar.
```
