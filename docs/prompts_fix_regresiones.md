# BASURIN — Prompts de Corrección Post-Validación E2E

**Fecha**: 2026-03-20
**Contexto**: El Prompt 7 (validación E2E) detectó 2 regresiones. Estos prompts las corrigen.
**Ejecutar en orden**: FIX-A primero, FIX-B segundo, REVALIDACIÓN tercero.

---

## FIX-A — CLI regression: `pipeline.py single` crashea por atributos faltantes

### Diagnóstico

`pipeline.py single` crashea con:
```
AttributeError: 'Namespace' object has no attribute 'threshold_mode'
```

**Causa raíz**: Los Prompts 1–3 añadieron `--threshold-mode`, `--delta-lnL`, e
`--informative-threshold` al subparser `multimode` pero NO al subparser `single`.
Sin embargo, `main()` pasa `args.threshold_mode`, `args.delta_lnL`, y
`args.informative_threshold` a `run_single_event()` incondicionalmente,
independientemente de si el subparser es `single` o `multimode`.

### Prompt para Claude Code

```
Actúa como ingeniero contract-first de BASURIN. Hay un crash en pipeline.py single.

DIAGNÓSTICO CONFIRMADO:
- "python mvp/pipeline.py single --event-id GW150914 --run-id test --atlas-default"
  crashea con: AttributeError: 'Namespace' object has no attribute 'threshold_mode'
- Los flags --threshold-mode, --delta-lnL, --informative-threshold están definidos
  SOLO en el subparser multimode, pero main() los referencia incondicionalmente.

ANTES DE ESCRIBIR CÓDIGO, lee:
- mvp/pipeline.py completo
- Identificar EXACTAMENTE:
  1. Dónde se define sp_single (el subparser de "single")
  2. Dónde se define sp_multimode (el subparser de "multimode")
  3. Dónde main() referencia args.threshold_mode, args.delta_lnL, args.informative_threshold
  4. Cómo run_single_event() recibe estos parámetros (kwargs, args directo, dict)

HAY DOS OPCIONES DE FIX. Elegir la correcta según lo que veas en el código:

OPCIÓN 1 (preferida si single-event TAMBIÉN usa s4_geometry_filter):
  Añadir los 3 argumentos al subparser sp_single con los MISMOS defaults que multimode:
    sp_single.add_argument("--threshold-mode", choices=["d2", "delta_lnL"], default="d2")
    sp_single.add_argument("--delta-lnL", type=float, default=3.0)
    sp_single.add_argument("--informative-threshold", type=float, default=0.80)

OPCIÓN 2 (si single-event NO pasa por s4 y por tanto no necesita estos flags):
  En main(), antes de pasar estos args a run_single_event(), usar getattr con defaults:
    threshold_mode = getattr(args, "threshold_mode", "d2")
    delta_lnL = getattr(args, "delta_lnL", 3.0)
    informative_threshold = getattr(args, "informative_threshold", 0.80)

PARA DECIDIR: Lee run_single_event() y verifica si invoca s4_geometry_filter.
  - Si sí → Opción 1 (el usuario necesita poder configurar estos flags en single).
  - Si no → Opción 2 (pero esto sería raro; single debería pasar por s4).

REGLA: NO cambiar defaults. NO cambiar run_single_event(). Solo corregir
la interfaz CLI para que los atributos existan cuando main() los referencia.

TEST DE REGRESIÓN (añadir a tests/unit/test_cli_parity.py o crear nuevo):

Test: test_single_mode_has_threshold_args
  import subprocess
  result = subprocess.run(
      ["python", "mvp/pipeline.py", "single", "--help"],
      capture_output=True, text=True
  )
  assert "threshold-mode" in result.stdout
  assert "delta-lnL" in result.stdout
  assert "informative-threshold" in result.stdout

Test: test_single_mode_namespace_has_all_attrs
  # Parsear args de single sin ejecutar:
  # python mvp/pipeline.py single --event-id X --run-id Y --atlas-default --help
  # Verificar que el Namespace tiene threshold_mode, delta_lnL, informative_threshold

VALIDACIÓN:
  python mvp/pipeline.py single --help | grep -E "threshold-mode|delta-lnL|informative"
  # Debe encontrar las 3 líneas.

  python mvp/pipeline.py single --event-id GW150914 --run-id smoke_test --atlas-default 2>&1 | head -20
  # Ya NO debe crashear con AttributeError.
  # Puede fallar en s1_fetch_strain por datos ausentes, pero eso es esperado.

ARTEFACTOS:
  - mvp/pipeline.py (modificado: subparser single ampliado)
  - tests/unit/test_cli_single_regression.py (nuevo, o ampliado en test_cli_parity.py)
```

---

## FIX-B — Abort semantics: RUN_VALID=PASS pese a abort en s1

### Diagnóstico

Cuando `pipeline.py multimode` ejecuta y `s1_fetch_strain` aborta por datos
LOSC ausentes, el pipeline deja en disco:
- `RUN_VALID` con contenido `PASS`
- `verdict.json` con veredicto `PASS`

Esto viola la semántica documentada en el README: si un stage aborta,
el run "no existe" downstream y NO debe tener `RUN_VALID=PASS`.

**Hipótesis de causa raíz** (verificar en código):
- `RUN_VALID` se crea al inicio del run (optimista) y no se revierte al abortar.
- O bien `s0_oracle_mvp` ya escribe `RUN_VALID=PASS` antes de que s1 ejecute.
- O bien el abort de s1 no propaga correctamente al mecanismo de gobernanza.

### Prompt para Claude Code

```
Actúa como ingeniero contract-first de BASURIN. Hay una violación de abort semantics.

DIAGNÓSTICO CONFIRMADO:
- "python mvp/pipeline.py multimode --event-id GW150914 --run-id X --atlas-default
   --threshold-mode d2 --epsilon 2500"
- s0_oracle_mvp pasa. s1_fetch_strain aborta (datos LOSC ausentes).
- PERO el directorio del run queda con RUN_VALID=PASS y verdict.json=PASS.
- Esto es INCORRECTO: un run donde un stage aborta NO debe tener RUN_VALID=PASS.

ANTES DE ESCRIBIR CÓDIGO, investiga el mecanismo completo:

1. Lee mvp/pipeline.py:
   - ¿Dónde se crea/escribe el fichero RUN_VALID?
   - ¿Se escribe al inicio (optimista) o al final (pesimista)?
   - ¿Qué pasa cuando run_single_event() o un stage lanza excepción / retorna abort?

2. Lee mvp/contracts.py:
   - ¿Cómo gestiona StageContract el abort?
   - ¿Hay una función finalize() que escriba RUN_VALID?
   - ¿abort() borra o sobreescribe RUN_VALID?

3. Lee el stage s1_fetch_strain.py:
   - ¿Cómo aborta? ¿Lanza excepción? ¿Retorna código? ¿Llama a contract.abort()?

4. Lee el fichero RUN_VALID real que dejó el run fallido:
   cat runs/validation_multi_d2_20260320T091245Z/RUN_VALID
   cat runs/validation_multi_d2_20260320T091245Z/verdict.json

5. Reportar:
   - ¿Quién escribe RUN_VALID? ¿En qué momento?
   - ¿Quién escribe verdict.json? ¿En qué momento?
   - ¿El abort de s1 propaga correctamente?

CAMBIO MÍNIMO (elegir según hallazgo):

CASO A: RUN_VALID se escribe al inicio (pattern optimista)
  Fix: Mover la escritura de RUN_VALID al FINAL del pipeline,
  SOLO si todos los stages completaron con status != ABORT.
  O bien: escribir RUN_VALID=FAIL explícitamente si cualquier stage aborta.

CASO B: RUN_VALID se escribe al final pero el abort de s1 no se propaga
  Fix: Asegurar que cuando s1 (o cualquier stage) aborta:
    a) La excepción/retorno se propaga hasta el caller en pipeline.py.
    b) pipeline.py atrapa el abort y escribe RUN_VALID=FAIL (no PASS).
    c) verdict.json NO se escribe, O se escribe con verdict_code != "PASS".

CASO C: El abort semántico de s1 no es un abort real (silently continues)
  Fix: Asegurar que s1_fetch_strain use contract.abort() o lance una excepción
  que pipeline.py reconozca como fatal para el run.

REGLA INVIOLABLE:
  Si CUALQUIER stage aborta → RUN_VALID != PASS.
  Si RUN_VALID no se puede garantizar → NO escribirlo (ausencia = no válido).

NO hacer:
  - No borrar el directorio del run (los artefactos parciales son auditoría útil).
  - No cambiar la lógica de abort de s1.
  - No cambiar StageContract sin necesidad.

TESTS:

Test 1: test_aborted_run_has_no_pass_verdict
  - Simular un pipeline donde s1 aborta (mock que lanza excepción o retorna ABORT).
  - Verificar que:
    a) RUN_VALID no existe O contiene "FAIL"
    b) verdict.json no existe O tiene verdict_code != "PASS"

Test 2: test_successful_run_has_pass_verdict
  - Simular un pipeline donde todos los stages completan.
  - Verificar que RUN_VALID contiene "PASS".

Test 3: test_abort_in_middle_stage_propagates
  - Simular pipeline donde s0 pasa pero s2 aborta.
  - Verificar misma semántica: RUN_VALID != PASS.

VALIDACIÓN:
  pytest -q -o "addopts=" tests/unit/ -k "abort" -v
  pytest -q -o "addopts=" tests/unit/ -k "verdict" -v

  # Si hay datos LOSC disponibles, re-run real:
  python mvp/pipeline.py multimode --event-id GW150914 --run-id revalidation_$(date +%Y%m%dT%H%M%SZ) --atlas-default --threshold-mode d2 --epsilon 2500
  cat runs/revalidation_*/RUN_VALID  # debe decir FAIL si s1 aborta

ARTEFACTOS:
  - mvp/pipeline.py (modificado: lógica de escritura de RUN_VALID)
  - Posiblemente mvp/contracts.py (si el mecanismo de propagación está ahí)
  - tests/unit/test_abort_semantics.py (nuevo)

RIESGOS:
  - Si RUN_VALID se usa como lock file por procesos concurrentes, cambiarlo puede
    tener side effects. Verificar si hay lectores de RUN_VALID fuera de pipeline.py.
  - Si verdict.json se escribe en un stage específico (s6, s6b), el fix puede
    ser no tocar verdict.json sino solo RUN_VALID. Verificar quién escribe cada uno.
```

---

## REVALIDACIÓN — Ejecutar tras FIX-A y FIX-B

```
Actúa como validador de BASURIN. Verifica que los fixes FIX-A y FIX-B son correctos.

NO IMPLEMENTAR NADA. Solo ejecutar y reportar.

PASO 1: Unit tests completos
  pytest -q -o "addopts=" tests/unit/ -v 2>&1 | tee /tmp/revalidation_tests.txt
  Reportar: total / passed / failed

PASO 2: Single-event 220 (verificar que FIX-A resuelve el crash)
  python mvp/pipeline.py single --event-id GW150914 --run-id reval_single_$(date +%Y%m%dT%H%M%SZ) --atlas-default 2>&1 | head -30
  
  CRITERIO DE ÉXITO:
  - NO crashea con AttributeError
  - Si falla en s1 por datos ausentes: OK (es el entorno, no el código)
  - Si falla en s1: verificar que RUN_VALID != PASS (FIX-B)

PASO 3: Multimode con datos ausentes (verificar FIX-B)
  python mvp/pipeline.py multimode --event-id GW150914 --run-id reval_multi_$(date +%Y%m%dT%H%M%SZ) --atlas-default --threshold-mode d2 --epsilon 2500 2>&1 | head -30
  
  CRITERIO DE ÉXITO:
  - s0 pasa, s1 aborta (esperado sin datos)
  - RUN_VALID NO contiene "PASS"
  - Si verdict.json existe, verdict_code != "PASS"
  
  VERIFICAR:
  cat runs/reval_multi_*/RUN_VALID 2>/dev/null || echo "RUN_VALID no existe (correcto)"
  cat runs/reval_multi_*/verdict.json 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print('verdict:', d.get('verdict_code','N/A'))" 2>/dev/null || echo "verdict.json no existe (correcto)"

PASO 4: --help de ambos subparsers
  python mvp/pipeline.py single --help | grep -c "threshold-mode"    # debe ser >= 1
  python mvp/pipeline.py multimode --help | grep -c "threshold-mode"  # debe ser >= 1

REPORTE:
  - FIX-A resuelto: sí/no (criterio: single no crashea con AttributeError)
  - FIX-B resuelto: sí/no (criterio: run abortado no tiene RUN_VALID=PASS)
  - Regresiones nuevas: ninguna / lista
  - Tests: X passed / Y total
```

---

## RESUMEN DE SEVERIDAD

| Fix | Severidad | Impacto | Esfuerzo estimado |
|-----|-----------|---------|-------------------|
| FIX-A | **ALTA** | Bloquea TODO uso de `pipeline.py single` | ~10 min (3 líneas de add_argument) |
| FIX-B | **CRÍTICA** | Viola semántica de gobernanza: runs inválidos parecen válidos | ~30-60 min (depende de dónde se escribe RUN_VALID) |

FIX-A es trivial. FIX-B requiere que Claude Code investigue el mecanismo real antes de tocar nada.
