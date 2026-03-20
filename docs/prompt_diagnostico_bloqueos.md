# DIAGNÓSTICO DE BLOQUEOS — Pre-revalidación E2E

```
Actúa como ingeniero contract-first de BASURIN. La revalidación E2E está bloqueada
por 2 problemas. Diagnostica ambos SIN cambiar código hasta tener causa raíz confirmada.

═══════════════════════════════════════════════════════════════
BLOQUEO 1: pytest no colecta tests
═══════════════════════════════════════════════════════════════

SÍNTOMA:
  pytest -q -o "addopts=" tests/unit/ -v
  → 0 collected, 1 error (FileNotFoundError en _pytest/capture.py), exit code 1

DIAGNÓSTICO — ejecutar en orden:

# 1a. Ver el error completo (no truncar)
pytest -q -o "addopts=" tests/unit/ -v --tb=long 2>&1 | tail -60

# 1b. Si el error es en capture.py, puede ser un conftest.py roto
#     o un fixture que abre un fichero que no existe.
pytest -q -o "addopts=" tests/unit/ --co 2>&1 | tail -40
# --co = collect-only, no ejecuta nada, solo lista lo que encontraría

# 1c. Probar un solo fichero de test aislado para aislar si es global o local
ls tests/unit/*.py | head -5
pytest -q -o "addopts=" tests/unit/$(ls tests/unit/*.py | head -1 | xargs basename) -v 2>&1

# 1d. Verificar conftest.py
cat tests/conftest.py 2>/dev/null || echo "No hay conftest.py global"
cat tests/unit/conftest.py 2>/dev/null || echo "No hay conftest.py en unit/"

# 1e. Verificar si hay un conftest que referencia una ruta hardcodeada
grep -rn "open\|Path\|os.path" tests/conftest.py tests/unit/conftest.py 2>/dev/null

# 1f. Verificar si la ruta de AGENTS.md está causando el problema
#     (AGENTS.md dice /home/adnac/basurin/work/basurin pero la máquina real
#      tiene /home/ignac/work/basurin)
grep -rn "adnac\|/home/" tests/ mvp/ conftest.py 2>/dev/null | grep -v ".pyc" | grep -v __pycache__

# 1g. Probar los tests de regresión específicos
pytest -q -o "addopts=" tests/test_cli_single_regression.py -v 2>&1
pytest -q -o "addopts=" tests/test_abort_semantics.py -v 2>&1

REPORTAR:
  - Error exacto (traceback completo, últimas 30 líneas)
  - ¿Es un fichero/ruta que no existe?
  - ¿Es un conftest.py con fixture rota?
  - ¿Afecta a todos los tests o solo a un subset?

SI ENCUENTRAS LA CAUSA:
  Proponer el cambio mínimo (1–3 líneas) para resolverlo.
  Si es una ruta hardcodeada: reemplazar por ruta relativa al repo.
  Si es un conftest.py que asume un directorio: hacer el directorio opcional con exist_ok.
  Si es un import fallido: reportar qué módulo falta.

═══════════════════════════════════════════════════════════════
BLOQUEO 2: Discrepancia CLI flags
═══════════════════════════════════════════════════════════════

SÍNTOMA:
  python mvp/pipeline.py single --help | grep -E "threshold-mode|delta-lnL"
  → no encuentra nada

  python mvp/pipeline.py multimode --help | grep -E "threshold-mode|delta-lnL"
  → encuentra --threshold-mode y --delta-lnL (singular), NO --delta-lnL-220 ni --delta-lnL-221

EXPECTATIVA (según el Summary de FIX-A):
  - single debería tener --threshold-mode, --delta-lnL-220, --delta-lnL-221
  - multimode debería tener --delta-lnL-220 y --delta-lnL-221 (por modo)

DIAGNÓSTICO — ejecutar en orden:

# 2a. Ver EXACTAMENTE qué flags tiene cada subparser
python mvp/pipeline.py single --help 2>&1
python mvp/pipeline.py multimode --help 2>&1

# 2b. Buscar en el código fuente dónde se definen los subparsers
grep -n "add_argument.*threshold\|add_argument.*delta.*lnL\|add_argument.*informative" mvp/pipeline.py

# 2c. Buscar en run_single_event la firma actual
grep -n "def run_single_event" mvp/pipeline.py

# 2d. Ver las líneas alrededor de run_single_event para entender qué parámetros recibe
grep -n -A 20 "def run_single_event" mvp/pipeline.py

# 2e. Buscar dónde main() invoca run_single_event
grep -n "run_single_event(" mvp/pipeline.py

# 2f. Verificar si el commit de FIX-A está realmente aplicado
git log --oneline -5
git diff HEAD~1 --stat  # o el commit que corresponda a FIX-A

# 2g. Verificar si hay cambios no committeados
git status
git diff mvp/pipeline.py | head -80

REPORTAR:
  - ¿FIX-A está committeado o quedó sin commit?
  - ¿Los flags se añadieron a sp_single o solo a sp_multimode?
  - ¿El flag es --delta-lnL (singular) o --delta-lnL-220/--delta-lnL-221 (por modo)?
  - ¿run_single_event() referencia args.threshold_mode directamente?

HAY 3 ESCENARIOS POSIBLES:

ESCENARIO A: FIX-A no se committeó / se aplicó en branch equivocado
  → Verificar branch actual: git branch --show-current
  → Verificar si hay stash: git stash list

ESCENARIO B: FIX-A se committeó pero los flags son distintos a lo reportado
  → El Summary decía --delta-lnL-220/221 pero el código tiene --delta-lnL (uno solo)
  → No es necesariamente un bug. Puede que el flag sea global y s4 lo use
     internamente por modo. Leer la lógica para confirmarlo.

ESCENARIO C: FIX-A se aplicó parcialmente (multimode sí, single no)
  → Completar la aplicación en sp_single con los mismos flags que multimode.

SI EL ESCENARIO ES C (lo más probable dado el síntoma):
  Aplicar el fix mínimo: copiar los add_argument de sp_multimode a sp_single.
  Usar los MISMOS nombres y defaults.

═══════════════════════════════════════════════════════════════
ORDEN DE EJECUCIÓN
═══════════════════════════════════════════════════════════════

1. Diagnosticar Bloqueo 1 (pytest). Si es fix trivial (< 3 líneas), aplicar.
2. Diagnosticar Bloqueo 2 (CLI). Si es fix trivial, aplicar.
3. Re-ejecutar baseline:
   pytest -q -o "addopts=" tests/unit/ -v
   python mvp/pipeline.py single --help | grep -E "threshold-mode|delta"
4. Si ambos pasan: reportar "BLOQUEOS RESUELTOS" con detalle de causa y fix.
5. Si alguno persiste: reportar causa raíz sin intentar más fixes.

NO hacer:
  - No refactorizar pipeline.py
  - No cambiar lógica de stages
  - No tocar tests existentes (excepto conftest si es la causa)
  - No inventar flags que no existan
```
