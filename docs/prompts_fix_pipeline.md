# BASURIN — Batería de Prompts de Corrección

**Fecha**: 2026-03-20
**Objetivo**: Dejar el pipeline en estado publicable
**Secuencia**: Los prompts están ordenados por dependencia. Ejecutar en orden.
**Convención**: Cada prompt se ejecuta en Claude Code contra el checkout actual del repo.

---

## MAPA DE DEPENDENCIAS

```
PROMPT 0 (auditoría) ──→ validar hallazgos
       │
PROMPT 1 (gate informatividad + acceptance_fraction en s4)
       │
PROMPT 2 (threshold_mode delta_lnL en s4)
       │
PROMPT 3 (three-state gate: SATURATED/EMPTY/OK)
       │
PROMPT 4 (paridad CLI stage ↔ host pipeline.py)
       │
PROMPT 5 (externalizar hardcodes peligrosos)
       │
PROMPT 6 (s4e kerr_ratio_filter — si no está implementado)
       │
PROMPT 7 (validación end-to-end: GW150914 single + multi)
```

---

## PROMPT 0 — AUDITORÍA FACTUAL

> **Este es el prompt que ya tienes redactado** (el archivo que subiste).
> Ejecútalo PRIMERO. Su salida alimenta las decisiones de los prompts 1–5.
> Si la auditoría revela hallazgos que yo no conozco, tráemelos antes de seguir.

---

## PROMPT 1 — Gate de informatividad y `acceptance_fraction` en s4

```
Actúa como ingeniero contract-first de BASURIN. Vas a modificar mvp/s4_geometry_filter.py
para añadir un diagnóstico de informatividad que NO cambia ningún resultado existente.

CONTEXTO OBLIGATORIO
- El pipeline produce compatible_set.json con n_compatible geometrías de un atlas de n_atlas.
- Con epsilon=2500 y el modo 221, n_compatible/n_atlas ≈ 0.91 (saturación).
- Esto no es un bug: es un resultado no informativo que debe documentarse, no ocultarse.
- El campo acceptance_fraction NO existe todavía en el output.
- No tocar RUN_VALID. No tocar n_compatible. No cambiar la semántica de compatible_set.json.

ANTES DE ESCRIBIR CÓDIGO, lee estos ficheros completos:
- mvp/s4_geometry_filter.py
- mvp/contracts.py (buscar s4_geometry_filter)
- tests/unit/ que contengan "s4" o "geometry_filter"

CAMBIOS EXACTOS (acción mínima):

1. En mvp/s4_geometry_filter.py, tras calcular n_compatible y n_atlas:
   - Calcular: acceptance_fraction = n_compatible / n_atlas (float, 0.0–1.0)
   - Nuevo argumento CLI: --informative-threshold FLOAT (default 0.80)
   - Calcular: informative = (acceptance_fraction <= informative_threshold)
   - Registrar AMBOS campos en stage_summary.json:
     "acceptance_fraction": <float>,
     "informative": <bool>,
     "informative_threshold": <float>

2. NO cambiar compatible_set.json ni su schema.
3. NO añadir nuevos imports. Usar solo stdlib.
4. NO cambiar el comportamiento cuando informative=false (solo registrar).

TESTS (en tests/unit/test_s4_informative_gate.py):

Test 1: test_acceptance_fraction_computation
  - n_compatible=730, n_atlas=800 → acceptance_fraction ≈ 0.9125
  - Assert: abs(result - 0.9125) < 1e-4

Test 2: test_informative_true_when_below_threshold
  - n_compatible=200, n_atlas=800 → acceptance_fraction=0.25
  - informative_threshold=0.80
  - Assert: informative == True

Test 3: test_informative_false_when_saturated
  - n_compatible=730, n_atlas=800 → acceptance_fraction=0.9125
  - informative_threshold=0.80
  - Assert: informative == False

Test 4: test_acceptance_fraction_in_stage_summary
  - Verificar que stage_summary.json contiene los 3 campos nuevos
  - Verificar tipos: float, bool, float

Test 5: test_backward_compatible_without_flag
  - Si no se pasa --informative-threshold, default=0.80
  - Salida idéntica a la actual MÁS los 3 campos nuevos

VALIDACIÓN POST-IMPLEMENTACIÓN:
  pytest -q -o "addopts=" tests/unit/test_s4_informative_gate.py -v
  # Verificar que tests existentes de s4 siguen pasando:
  pytest -q -o "addopts=" tests/unit/ -k "s4" -v

ARTEFACTOS ESPERADOS:
  - mvp/s4_geometry_filter.py (modificado)
  - tests/unit/test_s4_informative_gate.py (nuevo)

RIESGOS: Ninguno. Es aditivo y no cambia outputs existentes.
```

---

## PROMPT 2 — Modo `delta_lnL` en s4_geometry_filter

```
Actúa como ingeniero contract-first de BASURIN. Vas a añadir un segundo modo
de threshold a mvp/s4_geometry_filter.py: delta_lnL (relativo al mejor del atlas).

CONTEXTO OBLIGATORIO
- Actualmente el criterio de compatibilidad es: d² < epsilon (modo "d2").
- d² en BASURIN NO es χ²(2) calibrado. Es un score geométrico sobre atlas discreto.
  Los valores típicos de d²_min están en el rango 1000–5000.
- Para desaturar el modo 221, necesitamos un criterio RELATIVO:
  compatible si delta_lnL = -0.5 * (d² - d²_min) >= -delta_threshold.
  Esto selecciona geometrías cuyo likelihood está dentro de exp(delta_threshold) del mejor.
- El campo delta_lnL (o delta_log_likelihood) YA puede existir en compatible_geometries.
  Lee el código actual para verificar.

ANTES DE ESCRIBIR CÓDIGO, lee estos ficheros completos:
- mvp/s4_geometry_filter.py
- mvp/distance_metrics.py (si existe)
- El output actual compatible_set.json de algún run existente en runs/ (si hay)

CAMBIOS EXACTOS:

1. Nuevos argumentos CLI en s4_geometry_filter.py:
   --threshold-mode {d2, delta_lnL}  (default: d2, backward compatible)
   --delta-lnL FLOAT                 (default: 3.0, solo aplica si threshold-mode=delta_lnL)

2. En la función que construye compatible_set:
   Si threshold_mode == "d2":
     - Comportamiento IDÉNTICO al actual. Sin cambios.
   Si threshold_mode == "delta_lnL":
     - Calcular d² para TODAS las geometrías del atlas.
     - Encontrar d²_min (mejor match).
     - Para cada geometría: delta_lnL_i = -0.5 * (d²_i - d²_min)
     - Compatible si delta_lnL_i >= -delta_threshold
     - La geometría con d²_min SIEMPRE es compatible (delta_lnL=0 >= -delta_threshold).
   
3. En stage_summary.json, registrar:
   "threshold_mode": "d2" | "delta_lnL",
   "delta_lnL_threshold": <float> | null,
   "d2_min": <float>,
   "d2_at_best": <float>,  (es d2_min, redundante para auditoría)
   Mantener TODOS los campos existentes.

4. En compatible_set.json, para cada geometría compatible, añadir:
   "delta_lnL": <float>  (siempre calcular, incluso en modo d2)

5. NO cambiar distance_metrics.py.
6. NO añadir numpy. Solo stdlib math.
7. NO romper el modo d2 existente.

TESTS (en tests/unit/test_s4_delta_lnl.py):

Test 1: test_delta_lnL_desaturates
  - Atlas de 10 geometrías con d²=[100, 105, 110, 120, 150, 200, 500, 800, 1000, 2000]
  - delta_threshold=3.0
  - d²_min=100, delta_lnL = -0.5*(d²-100)
  - Compatible si -0.5*(d²-100) >= -3.0, es decir d² <= 106
  - Expect: n_compatible=2 (d²=100 y d²=105)

Test 2: test_d2_mode_unchanged
  - Misma fixture, threshold_mode="d2", epsilon=2500
  - Expect: n_compatible=10 (todos pasan)
  - Verificar que output es idéntico al comportamiento actual

Test 3: test_delta_lnL_always_includes_best
  - Cualquier atlas, cualquier delta_threshold >= 0
  - La geometría con d²_min SIEMPRE está en compatible_set

Test 4: test_delta_lnL_registered_in_stage_summary
  - threshold_mode="delta_lnL", delta_threshold=5.0
  - Verificar campos en stage_summary.json

Test 5: test_delta_lnL_field_present_even_in_d2_mode
  - threshold_mode="d2"
  - Verificar que cada geometría en compatible_set tiene campo delta_lnL

VALIDACIÓN POST-IMPLEMENTACIÓN:
  pytest -q -o "addopts=" tests/unit/test_s4_delta_lnl.py -v
  pytest -q -o "addopts=" tests/unit/ -k "s4" -v
  # Smoke test manual con GW150914 si hay run disponible:
  python mvp/s4_geometry_filter.py --run-id <run_id> --threshold-mode delta_lnL --delta-lnL 3.0

ARTEFACTOS ESPERADOS:
  - mvp/s4_geometry_filter.py (modificado)
  - tests/unit/test_s4_delta_lnl.py (nuevo)

RIESGOS:
  - Si delta_threshold es demasiado bajo, n_compatible→1. Pero eso es informativo, no un bug.
  - Default d2 sigue funcionando → regresión cero.
```

---

## PROMPT 3 — Three-state gate (SATURATED / EMPTY / OK)

```
Actúa como ingeniero contract-first de BASURIN. Vas a añadir una clasificación
ternaria del resultado de s4_geometry_filter para diagnóstico automático.

PREREQUISITO: Prompts 1 y 2 deben estar implementados (acceptance_fraction + delta_lnL).

ANTES DE ESCRIBIR CÓDIGO, lee:
- mvp/s4_geometry_filter.py (versión actual, con los cambios de Prompts 1 y 2)
- mvp/contracts.py

CONTEXTO:
- acceptance_fraction ya existe en stage_summary.json (Prompt 1).
- Necesitamos una clasificación formal de 3 estados:
  "SATURATED": acceptance_fraction > informative_threshold (demasiado permisivo)
  "EMPTY": n_compatible == 0 (demasiado restrictivo)
  "OK": ni saturado ni vacío (régimen informativo)
- Esta clasificación es DIAGNÓSTICA (no bloquea RUN_VALID).
- Downstream (s5, ex4, ex5f) puede usar este campo para decidir censura.

CAMBIOS EXACTOS:

1. En mvp/s4_geometry_filter.py, tras calcular acceptance_fraction e informative:
   - Calcular:
     if n_compatible == 0:
         filter_status = "EMPTY"
     elif acceptance_fraction > informative_threshold:
         filter_status = "SATURATED"
     else:
         filter_status = "OK"
   - Registrar en stage_summary.json:
     "filter_status": "OK" | "SATURATED" | "EMPTY"
   - Mantener "informative": bool para backward compat (redundante pero no rompe nada).

2. NO cambiar RUN_VALID.
3. NO abortar si SATURATED o EMPTY.
4. NO modificar compatible_set.json.

TESTS (añadir a tests/unit/test_s4_informative_gate.py):

Test 6: test_filter_status_ok
  - n_compatible=200, n_atlas=800 → "OK"

Test 7: test_filter_status_saturated
  - n_compatible=730, n_atlas=800 → "SATURATED"

Test 8: test_filter_status_empty
  - n_compatible=0 → "EMPTY"

Test 9: test_filter_status_edge_at_threshold
  - n_compatible=640, n_atlas=800 → acceptance_fraction=0.80 exacto
  - Con threshold=0.80: acceptance_fraction > 0.80 es false → "OK"
  - Verificar: 0.80 <= 0.80 → informative=True, filter_status="OK"

VALIDACIÓN:
  pytest -q -o "addopts=" tests/unit/test_s4_informative_gate.py -v

ARTEFACTOS ESPERADOS:
  - mvp/s4_geometry_filter.py (modificado)
  - tests/unit/test_s4_informative_gate.py (ampliado)
```

---

## PROMPT 4 — Paridad CLI: s3b ↔ pipeline.py multimode

```
Actúa como auditor de paridad CLI de BASURIN. Vas a verificar y corregir
que TODOS los knobs de mvp/s3b_multimode_estimates.py estén expuestos
en mvp/pipeline.py cuando se ejecuta en modo multimode.

ANTES DE ESCRIBIR CÓDIGO, lee COMPLETOS:
- mvp/s3b_multimode_estimates.py (buscar add_argument y variables con default)
- mvp/pipeline.py (buscar "s3b", "multimode", "221")
- mvp/tools/*.py que mencionen "multimode" o "s3b" (si existen)

MÉTODO:

1. INVENTARIO s3b: Lista TODOS los argumentos de argparse en s3b.
   Para cada uno: nombre, tipo, default, si afecta resultado científico.

2. INVENTARIO pipeline.py multimode: Lista TODOS los parámetros que pipeline.py
   pasa a s3b cuando ejecuta en modo multimode/batch.

3. TABLA DE PARIDAD:
   | Argumento s3b | Default s3b | ¿Expuesto en pipeline.py? | Nombre en pipeline.py | ¿Mismo default? | ¿Afecta veredicto? |
   
4. Para cada fila con "No expuesto" Y "Sí afecta veredicto":
   - Añadir el argumento a pipeline.py con el MISMO default que s3b.
   - Nombrar igual o con prefijo s3b_ si hay colisión.
   - Pasar al subprocess/función que invoca s3b.

5. NO cambiar defaults de s3b.
6. NO renombrar argumentos de s3b.
7. Si pipeline.py invoca s3b como subprocess: verificar que el comando construido
   incluye todos los nuevos flags.
8. Si pipeline.py invoca s3b como función: verificar que los kwargs se pasan.

HACER LO MISMO para s4_geometry_filter.py:
- Verificar que --threshold-mode y --delta-lnL (del Prompt 2) se exponen en pipeline.py.
- Verificar que --informative-threshold (del Prompt 1) se expone en pipeline.py.
- Si no están: añadirlos.

TESTS (en tests/unit/test_cli_parity.py):

Test 1: test_s3b_args_subset_of_pipeline_multimode
  - Parsear argparse de s3b y de pipeline.py
  - Verificar que todo argumento científico de s3b tiene correspondencia en pipeline.py

Test 2: test_s4_new_args_in_pipeline
  - Verificar que --threshold-mode, --delta-lnL, --informative-threshold
    están en pipeline.py multimode

Test 3: test_pipeline_help_includes_all_scientific_knobs
  - python mvp/pipeline.py multi --help | grep "threshold-mode"
  - python mvp/pipeline.py multi --help | grep "delta-lnL"
  - python mvp/pipeline.py multi --help | grep "informative-threshold"

VALIDACIÓN:
  pytest -q -o "addopts=" tests/unit/test_cli_parity.py -v
  python mvp/pipeline.py multi --help  # verificar visualmente

ARTEFACTOS ESPERADOS:
  - mvp/pipeline.py (modificado)
  - tests/unit/test_cli_parity.py (nuevo)

RIESGOS:
  - Si pipeline.py invoca s3b como subprocess con un comando hardcodeado,
    hay riesgo de romper la invocación. Lee el mecanismo actual ANTES de tocar.
```

---

## PROMPT 5 — Externalizar hardcodes peligrosos

```
Actúa como ingeniero contract-first de BASURIN. Este prompt se ejecuta DESPUÉS
de la auditoría (Prompt 0). Vas a externalizar los hardcodes que la auditoría
marque como severidad ALTA.

PREREQUISITO: Ejecutar primero el Prompt 0 de auditoría y tener sus resultados.

ANTES DE ESCRIBIR CÓDIGO, lee:
- La salida de la auditoría (hallazgos con severidad ALTA)
- Los ficheros afectados

REGLAS PARA EXTERNALIZAR:

1. Para cada hardcode ALTA, el cambio mínimo es:
   a) Convertirlo en argumento CLI con add_argument y default = valor actual.
   b) Registrar el valor usado en stage_summary.json bajo un campo nombrado.
   c) Añadir un test que verifique que el default produce output idéntico al actual.

2. NO cambiar el valor default. Solo exponerlo.
3. NO refactorizar lógica adyacente.
4. NO crear ficheros de configuración nuevos (YAML, TOML, etc.). Solo CLI args.
5. Documentar cada nuevo argumento con help= explicativo.

PATRÓN DE CADA HALLAZGO:

Para hallazgo Hxx (severidad ALTA, clase X):
  Fichero: mvp/<nombre>.py
  Variable: <nombre_variable>
  Valor actual: <valor>
  
  Cambio:
    parser.add_argument("--<nombre-cli>", type=<tipo>, default=<valor>,
                        help="<descripción>. Default: <valor>")
  
  En stage_summary:
    "<nombre_campo>": args.<nombre_cli>
  
  Test:
    test_default_<nombre>_produces_identical_output
    - Sin pasar el flag → resultado idéntico al baseline
    - Pasar un valor distinto → resultado diferente (sanity)

VALIDACIÓN:
  pytest -q -o "addopts=" tests/unit/ -k "s4 or s3 or s2 or s1" -v
  # Todos los tests existentes deben seguir pasando (defaults no cambian)

NOTA: Este prompt es genérico. Adáptalo con los hallazgos concretos de la auditoría.
Si la auditoría produce <3 hallazgos ALTA, este prompt puede no ser necesario.
```

---

## PROMPT 6 — s4e Kerr ratio filter (condicional)

```
Actúa como ingeniero contract-first de BASURIN. Vas a implementar s4e_kerr_ratio_filter.

PREREQUISITO: Verificar si s4e ya existe en el checkout.
  ls mvp/s4e*
  grep -r "s4e" mvp/contracts.py
Si ya existe y tiene tests pasando, SALTA este prompt y reporta estado.

ANTES DE ESCRIBIR CÓDIGO, lee:
- mvp/kerr_qnm_fits.py (funciones existentes: qnm_frequency, qnm_quality_factor)
- mvp/s3b_multimode_estimates.py (output: estimates.json, campos exactos)
- mvp/s4_geometry_filter.py (output: compatible_set.json, schema exacto)
- mvp/contracts.py (patrón StageContract)

RESUMEN DE s4e:
s4e consume el compatible_set de s4 (modo 221) + estimates de s3b (f_220, f_221).
Calcula el ratio R_f = f_221/f_220 observado y lo compara con la predicción Kerr
para el spin de cada geometría del atlas. Excluye geometrías cuyo R_f(χ) predicho
cae fuera del intervalo observado.

IMPLEMENTACIÓN:

Parte A: Añadir kerr_ratio_curve() a mvp/kerr_qnm_fits.py
  - Función pura, sin numpy, determinista.
  - Input: chi_grid (list of float) o n_points (int, default 50)
  - Output: dict con chi_grid, Rf_grid, RQ_grid, Rf_range, RQ_range
  - Rf(χ) = F_221(χ) / F_220(χ) usando los fits de Berti que ya existen.
  - Verificar Schwarzschild: Rf(0) ≈ 0.927, RQ(0) ≈ 0.304

Parte B: mvp/s4e_kerr_ratio_filter.py (stage nuevo)
  - Contract: inputs = [s3b estimates.json, s4 compatible_set.json]
  - Output: ratio_filter_result.json con:
    - n_input, n_surviving, n_excluded
    - Rf_observed (float), sigma_Rf (float)
    - Rf_kerr_range (dict min/max)
    - per-geometry: compatible_by_ratio (bool), tension_sigma (float)
    - diagnostics: reduction_fraction, informativity_class
  - Argumento: --sigma-tolerance FLOAT (default 2.0, cuántos sigmas de margen)
  - Si no hay campo spin en atlas: warning, no abort.
  
Parte C: Registrar en contracts.py

Parte D: Tests (12 tests según el spec que ya se discutió)
  - test_kerr_ratio_curve_basic
  - test_kerr_ratio_curve_schwarzschild (Rf≈0.927, RQ≈0.304)
  - test_kerr_ratio_curve_near_extremal (Rf>0.95)
  - test_kerr_ratio_curve_monotonicity
  - test_filter_excludes_incompatible
  - test_filter_keeps_compatible
  - test_filter_no_spin_warning
  - test_filter_abort_no_221
  - test_reduction_fraction
  - test_tension_sigma_calculation
  - test_informativity_class
  - test_backward_compat_contract

VALIDACIÓN:
  python -c "from mvp.kerr_qnm_fits import kerr_ratio_curve; r = kerr_ratio_curve(n_points=5); print(r['Rf_range'])"
  python -c "from mvp.contracts import CONTRACTS; print(CONTRACTS['s4e_kerr_ratio_filter'].name)"
  pytest -q -o "addopts=" tests/unit/test_s4e_kerr_ratio_filter.py -v

ARTEFACTOS ESPERADOS:
  - mvp/kerr_qnm_fits.py (modificado: +kerr_ratio_curve)
  - mvp/s4e_kerr_ratio_filter.py (nuevo)
  - mvp/contracts.py (modificado: +s4e)
  - tests/unit/test_s4e_kerr_ratio_filter.py (nuevo)
```

---

## PROMPT 7 — Validación end-to-end

```
Actúa como validador de BASURIN. Vas a ejecutar el pipeline completo
para verificar que todos los cambios de los Prompts 1–6 son coherentes.

PREREQUISITO: Prompts 1–4 implementados. Prompt 5 y 6 opcionales.

NO IMPLEMENTAR NADA NUEVO. Solo ejecutar y verificar.

PASO 1: Tests unitarios completos
  pytest -q -o "addopts=" tests/unit/ -v 2>&1 | tee /tmp/test_results.txt
  # Reportar: total, passed, failed, errors
  # Si hay failures: PARAR y reportar cuáles son.

PASO 2: Single-event 220 only (baseline)
  python mvp/pipeline.py single \
    --event GW150914 \
    --run-id validation_220_$(date +%Y%m%dT%H%M%SZ) \
    --mode-label 220 \
    2>&1 | tee /tmp/run_220.log
  # Verificar: RUN_VALID=PASS
  # Verificar: stage_summary.json de s4 contiene acceptance_fraction, informative, filter_status

PASO 3: Single-event multimode (220+221, modo d2 con epsilon=2500)
  python mvp/pipeline.py single \
    --event GW150914 \
    --run-id validation_multi_d2_$(date +%Y%m%dT%H%M%SZ) \
    --mode-label multimode \
    --threshold-mode d2 \
    --epsilon 2500 \
    2>&1 | tee /tmp/run_multi_d2.log
  # Verificar: filter_status para 221 es "SATURATED" (esperado)
  # Verificar: acceptance_fraction_221 > 0.80

PASO 4: Single-event multimode (220+221, modo delta_lnL)
  python mvp/pipeline.py single \
    --event GW150914 \
    --run-id validation_multi_dlnl_$(date +%Y%m%dT%H%M%SZ) \
    --mode-label multimode \
    --threshold-mode delta_lnL \
    --delta-lnL 3.0 \
    2>&1 | tee /tmp/run_multi_dlnl.log
  # Verificar: filter_status para 221 es "OK" (desaturado)
  # Verificar: n_compatible_221 << 730 (objetivo: 50–300)
  # Verificar: la geometría con d²_min está en compatible_set

PASO 5: Comparar outputs
  # Verificar que modo d2 produce EXACTAMENTE el mismo compatible_set_220
  # que el baseline del Paso 2.
  # Verificar que modo delta_lnL produce el MISMO compatible_set_220
  # (delta_lnL solo afecta a 221 si epsilon es el mismo para 220).

REPORTE FINAL:
Producir un resumen con:
  - Total tests unitarios: X passed / Y total
  - Run 220: RUN_VALID, n_compatible, filter_status
  - Run multi d2: RUN_VALID, n_compatible_220, n_compatible_221, filter_status_221
  - Run multi delta_lnL: RUN_VALID, n_compatible_220, n_compatible_221, filter_status_221
  - ¿Desaturación lograda? n_compatible_221(delta_lnL) < n_compatible_221(d2)
  - Regresiones detectadas: ninguna / lista

NOTA: Los comandos CLI exactos pueden diferir del patrón arriba si pipeline.py
usa flags distintos. Lee "python mvp/pipeline.py single --help" ANTES de ejecutar.
Adapta los flags al --help real, no a lo que dice este prompt.
```

---

## PROMPT OPCIONAL A — Experiment ex5f (Agregación de Veredictos)

```
Actúa como ingeniero contract-first de BASURIN. Vas a implementar el experimento
ex5f: agregación read-only sobre verdict.json de múltiples runs.

PREREQUISITO: Al menos 3 runs con RUN_VALID=PASS en runs/.

ANTES DE ESCRIBIR CÓDIGO, lee:
- Un verdict.json real de cualquier run existente
- mvp/contracts.py (patrón ExperimentContract si existe, o StageContract)
- experiment/ (directorio, si existe — ver estructura)

DISEÑO:
- Fichero: experiment/ex5f_verdict_aggregation.py
- Input: lista de run_ids (CLI) o --parent-run-id con aggregate.json
- Output bajo: runs/<run_id>/experiment/ex5f_verdict_agg/
  - outputs/verdict_population.json
  - stage_summary.json
  - manifest.json (SHA256 de inputs y outputs)

SCHEMA DE verdict_population.json:
{
  "n_events": int,
  "n_valid_runs": int,
  "family_support": {
    "<family>": {
      "n_events_compatible": int,
      "fraction": float,
      "events": [str]
    }
  },
  "verdict_distribution": {
    "<verdict_code>": int
  },
  "kerr_survival_rate": float | null,
  "metadata": {
    "run_ids_used": [str],
    "timestamp": str,
    "pipeline_version": str
  }
}

GOBERNANZA:
- Read-only sobre artefactos canónicos.
- Verificar RUN_VALID=PASS de cada run antes de incluirlo.
- Si un run no tiene verdict.json: skip con warning, no abort.
- SHA256 de cada verdict.json consumido en manifest.json.

TESTS (4 tests):
1. test_ex5f_produces_valid_schema
2. test_ex5f_skips_invalid_runs
3. test_ex5f_sha256_in_manifest
4. test_ex5f_no_mutation_of_source (verificar que runs originales no se tocan)

VALIDACIÓN:
  pytest -q -o "addopts=" tests/unit/test_ex5f_verdict_aggregation.py -v
```

---

## PROMPT OPCIONAL B — Experiment ex8 (Area Consistency)

```
Este prompt ya fue generado en una sesión anterior.
Buscar en el historial: "prompt_s4e_kerr_ratio_filter.md" o "experimento 8".
Si no lo encuentras, dime y lo regenero con el contexto actualizado.
```

---

## NOTAS OPERATIVAS

1. **Orden de ejecución**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7.
   Los prompts 1, 2, 3 son incrementales sobre el mismo fichero (s4_geometry_filter.py).
   Si Claude Code los ejecuta en secuencia sin conflictos, perfecto.
   Si hay conflictos, ejecutar 1+2+3 en un solo prompt combinado.

2. **Fallback de prompt único**: Si prefieres un solo prompt masivo en vez de 7,
   combina los Prompts 1+2+3 en uno solo (todos tocan s4_geometry_filter.py)
   y los Prompts 4+5 en otro (tocan pipeline.py y hardcodes).

3. **Adaptabilidad**: Los Prompts 5 y 6 dependen de la auditoría (Prompt 0).
   Si la auditoría no produce hallazgos ALTA, el Prompt 5 se omite.
   Si s4e ya existe, el Prompt 6 se omite.

4. **Test runner**: Todos los prompts asumen `pytest -q -o "addopts="` para
   evitar que addopts globales con --cov rompan la ejecución.

5. **Cada prompt instruye a Claude Code a LEER ANTES DE ESCRIBIR**.
   Esto es crítico para evitar inventar schemas o campos que no existen.
