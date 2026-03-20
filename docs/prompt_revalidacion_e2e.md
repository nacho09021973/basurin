# REVALIDACIÓN E2E — Post FIX-A + FIX-B

```
Actúa como validador de BASURIN. Verifica que los fixes FIX-A y FIX-B son correctos
y que la desaturación del 221 funciona end-to-end.

NO IMPLEMENTAR NADA NUEVO. Solo ejecutar, verificar y reportar.

═══════════════════════════════════════════════════════════════
ANTES DE EJECUTAR: verificar entorno
═══════════════════════════════════════════════════════════════

# 1. Verificar datos LOSC disponibles
ls -la data/losc/GW150914/
# Debe tener ficheros HDF5 para H1 y L1.
# Si no: buscar en subdirectorios o verificar nombres exactos.

# 2. Verificar que los fixes están aplicados
python mvp/pipeline.py single --help | grep -E "threshold-mode|delta-lnL-220|delta-lnL-221"
# Debe encontrar los 3 flags.

# 3. Tests unitarios completos (baseline verde)
pytest -q -o "addopts=" tests/unit/ -v 2>&1 | tee /tmp/reval_tests.txt
pytest -q -o "addopts=" tests/ -k "cli_single_regression or abort_semantics" -v
# Reportar: total / passed / failed. Si hay failures: PARAR.

═══════════════════════════════════════════════════════════════
PASO 1: Single-event 220 only (verificar FIX-A)
═══════════════════════════════════════════════════════════════

# Primero lee el --help real para confirmar los flags exactos:
python mvp/pipeline.py single --help

# Ejecutar con los flags reales que veas en --help:
python mvp/pipeline.py single \
  --event-id GW150914 \
  --run-id reval_220_$(date +%Y%m%dT%H%M%SZ) \
  --atlas-default \
  2>&1 | tee /tmp/reval_220.log

# CRITERIOS DE ÉXITO:
# ✅ NO crashea con AttributeError (FIX-A confirmado)
# ✅ RUN_VALID existe
# ✅ Si el run completa: verificar stage_summary.json de s4 contiene:
#    - acceptance_fraction (float)
#    - informative (bool)
#    - filter_status ("OK" | "SATURATED" | "EMPTY")
#    - threshold_mode ("d2")

# Verificar artefactos:
RUN_ID_220=$(ls -td runs/reval_220_* | head -1)
echo "=== RUN_VALID ==="
cat "$RUN_ID_220/RUN_VALID"
echo "=== s4 stage_summary ==="
python -c "
import json, glob
ss = glob.glob('$RUN_ID_220/s4_geometry_filter/stage_summary.json')
if ss:
    d = json.load(open(ss[0]))
    for k in ['acceptance_fraction','informative','filter_status','threshold_mode','n_compatible','n_atlas']:
        print(f'  {k}: {d.get(k, \"MISSING\")}')
else:
    print('  stage_summary.json NOT FOUND')
"

═══════════════════════════════════════════════════════════════
PASO 2: Multimode d2 con epsilon=2500 (confirmar saturación)
═══════════════════════════════════════════════════════════════

# Lee el --help de multimode para confirmar flags exactos:
python mvp/pipeline.py multimode --help

python mvp/pipeline.py multimode \
  --event-id GW150914 \
  --run-id reval_multi_d2_$(date +%Y%m%dT%H%M%SZ) \
  --atlas-default \
  --threshold-mode d2 \
  --epsilon 2500 \
  2>&1 | tee /tmp/reval_multi_d2.log

# CRITERIOS:
# ✅ RUN_VALID = PASS (si completa)
# ✅ filter_status para 220: probablemente "OK"
# ✅ filter_status para 221: esperamos "SATURATED" (acceptance_fraction > 0.80)
# ✅ n_compatible_221 ≈ 700–730 (saturación conocida)

# Verificar:
RUN_ID_D2=$(ls -td runs/reval_multi_d2_* | head -1)
echo "=== RUN_VALID ==="
cat "$RUN_ID_D2/RUN_VALID"

# Buscar stage_summary de s4 para cada modo (220 y 221).
# La estructura puede ser:
#   runs/<id>/s4_geometry_filter/stage_summary.json  (si hay uno por modo)
#   o con subdirectorios por modo
# Lee el código o el output real para encontrar la ruta correcta.
echo "=== Buscando stage_summaries de s4 ==="
find "$RUN_ID_D2" -name "stage_summary.json" -path "*/s4*" | while read f; do
    echo "--- $f ---"
    python -c "
import json
d = json.load(open('$f'))
for k in ['mode_label','acceptance_fraction','informative','filter_status','threshold_mode','n_compatible','n_atlas','d2_min']:
    print(f'  {k}: {d.get(k, \"MISSING\")}')
"
done

═══════════════════════════════════════════════════════════════
PASO 3: Multimode delta_lnL (verificar desaturación)
═══════════════════════════════════════════════════════════════

python mvp/pipeline.py multimode \
  --event-id GW150914 \
  --run-id reval_multi_dlnl_$(date +%Y%m%dT%H%M%SZ) \
  --atlas-default \
  --threshold-mode delta_lnL \
  --delta-lnL-220 5.0 \
  --delta-lnL-221 3.0 \
  2>&1 | tee /tmp/reval_multi_dlnl.log

# CRITERIOS:
# ✅ RUN_VALID = PASS (si completa)
# ✅ filter_status para 221: "OK" (desaturado)
# ✅ n_compatible_221 << 730 (objetivo: 50–300)
# ✅ La geometría con d²_min está en compatible_set (delta_lnL=0 >= -3.0)
# ✅ n_compatible_220 puede también cambiar vs d2 mode (es esperado con delta_lnL)

# Verificar:
RUN_ID_DLNL=$(ls -td runs/reval_multi_dlnl_* | head -1)
echo "=== RUN_VALID ==="
cat "$RUN_ID_DLNL/RUN_VALID"

echo "=== stage_summaries s4 ==="
find "$RUN_ID_DLNL" -name "stage_summary.json" -path "*/s4*" | while read f; do
    echo "--- $f ---"
    python -c "
import json
d = json.load(open('$f'))
for k in ['mode_label','acceptance_fraction','informative','filter_status','threshold_mode','n_compatible','n_atlas','d2_min','delta_lnL_threshold']:
    print(f'  {k}: {d.get(k, \"MISSING\")}')
"
done

═══════════════════════════════════════════════════════════════
PASO 4: Verificar FIX-B (abort semantics)
═══════════════════════════════════════════════════════════════

# Forzar un abort de s1 con un evento que NO tenga datos LOSC descargados.
# Elegir un evento que sepamos que no está en data/losc/:
python mvp/pipeline.py single \
  --event-id GW170817 \
  --run-id reval_abort_$(date +%Y%m%dT%H%M%SZ) \
  --atlas-default \
  2>&1 | tee /tmp/reval_abort.log

# CRITERIO FIX-B:
# ✅ s1 aborta (datos ausentes)
# ✅ RUN_VALID NO existe O contiene "FAIL"
# ✅ verdict.json NO existe O verdict_code != "PASS"

RUN_ID_ABORT=$(ls -td runs/reval_abort_* | head -1)
echo "=== RUN_VALID (debe ser FAIL o no existir) ==="
cat "$RUN_ID_ABORT/RUN_VALID" 2>/dev/null || echo "RUN_VALID no existe (CORRECTO)"
echo "=== verdict.json ==="
python -c "
import json
try:
    d = json.load(open('$RUN_ID_ABORT/verdict.json'))
    vc = d.get('verdict_code', 'N/A')
    print(f'  verdict_code: {vc}')
    if vc == 'PASS':
        print('  ⚠️  FIX-B NO RESUELTO: verdict sigue siendo PASS tras abort')
    else:
        print('  ✅ FIX-B confirmado')
except FileNotFoundError:
    print('  verdict.json no existe (CORRECTO)')
" 2>/dev/null

═══════════════════════════════════════════════════════════════
PASO 5: Comparar outputs 220 entre modos
═══════════════════════════════════════════════════════════════

# Si los 3 runs completaron, comparar n_compatible del modo 220 entre:
# - reval_220 (single, d2 default)
# - reval_multi_d2 (multimode, d2 explícito)
# - reval_multi_dlnl (multimode, delta_lnL)
#
# En modo d2 con epsilon=2500, los dos primeros deben dar n_compatible_220 IDÉNTICO.
# En modo delta_lnL, n_compatible_220 puede diferir (el criterio es distinto).

echo "=== COMPARACIÓN n_compatible_220 ==="
for run_dir in "$RUN_ID_220" "$RUN_ID_D2" "$RUN_ID_DLNL"; do
    echo "--- $(basename $run_dir) ---"
    find "$run_dir" -name "stage_summary.json" -path "*/s4*" | while read f; do
        python -c "
import json
d = json.load(open('$f'))
ml = d.get('mode_label', '?')
nc = d.get('n_compatible', '?')
af = d.get('acceptance_fraction', '?')
fs = d.get('filter_status', '?')
tm = d.get('threshold_mode', '?')
print(f'  mode={ml}  n_compatible={nc}  af={af}  status={fs}  threshold={tm}')
"
    done
done

═══════════════════════════════════════════════════════════════
REPORTE FINAL — Rellenar y reportar
═══════════════════════════════════════════════════════════════

# Producir un resumen estructurado con:

echo "
========================================
REPORTE REVALIDACIÓN E2E — $(date)
========================================

1. TESTS UNITARIOS
   Total: ___  Passed: ___  Failed: ___

2. FIX-A (CLI single)
   ¿single ejecuta sin AttributeError? ___

3. FIX-B (abort semantics)
   ¿Run abortado tiene RUN_VALID=FAIL? ___

4. RUN 220 (single, baseline)
   RUN_VALID: ___
   n_compatible: ___
   filter_status: ___

5. RUN MULTI D2 (multimode, epsilon=2500)
   RUN_VALID: ___
   n_compatible_220: ___  filter_status_220: ___
   n_compatible_221: ___  filter_status_221: ___
   acceptance_fraction_221: ___

6. RUN MULTI DELTA_LNL (multimode, delta_lnL-221=3.0)
   RUN_VALID: ___
   n_compatible_220: ___  filter_status_220: ___
   n_compatible_221: ___  filter_status_221: ___
   acceptance_fraction_221: ___

7. DESATURACIÓN LOGRADA
   n_compatible_221(d2): ___
   n_compatible_221(delta_lnL): ___
   Ratio de reducción: ___

8. REGRESIONES NUEVAS: ___
"
```
