# PLAN_PRE_REAL_V0_v1 — Cronograma de gates (Ringdown → Real v0)

## Propósito
Definir el conjunto mínimo de **experimentos-gate** que deben pasar en sintético/controlado antes de ejecutar “real v0” (GW150914 ringdown), bajo gobernanza BASURIN.

Regla soberana:
- Ningún experimento se ejecuta si `RUN_VALID != PASS`.
- Un FAIL invalida el run para downstream.

## Ops quick commands (para no buscar rutas)
- Comando:
  python tools/basurin_where.py --run "$RUN_ID" --ringdown-min
- Regla: si READY: NO, no ejecutar gates downstream.

## Artefactos canónicos requeridos (upstream)
- `geometry/outputs/*.h5` (p.ej. `ads_puro.h5`)
- `spectrum/outputs/spectrum.h5` (Bloque B)

## Track QNM (validación de condiciones de contorno)
Estos gates se incorporan al pre-real porque validan el paso “closed → open BC” y la consistencia interna de la modelización de absorción.

### EXP_RINGDOWN_QNM_00 — Open BC “horizon absorber” gate
- Objetivo: verificar decaimiento (Im(ω) < 0) y estabilidad cross-grid.
- Contratos:
  - C1: decay (signo + calidad de fit)
  - C2: stability (variación acotada)
- PASS requerido para continuar.

### EXP_RINGDOWN_QNM_01 — Closed↔Open limit validation
- Objetivo: en el límite de absorción → 0 recuperar el espectro cerrado (ω_R² → M²) y monotonicidad del decaimiento con absorción.
- Contratos:
  - C3: closed limit recovery (ω_R² ~ M², ω_I → 0)
  - C4: monotonicity (|ω_I| no decrece con absorción)

### Evidencia de referencia (run validado)
- Run: `runs/2026-02-01__qnm_validation_fix2/`
- Resultado: `EXP_QNM_00=PASS`, `EXP_QNM_01=PASS`.

## Track Ringdown (pre-real v0)
(Completar aquí con los gates del cronograma original: EXP_RINGDOWN_00..08, cuando toque consolidarlo.)

## Criterio “Go/No-Go” a Real v0
Go si:
- Todos los gates del track QNM están en PASS
- Todos los gates del track Ringdown (00..08) están en PASS
- RUN_VALID PASS y artefactos canónicos trazados (path+hash)
