# Fase 2 — Predicción de informatividad y automatización de búsqueda

## Objetivo

Resolver el problema detectado en GW150914: el pipeline ejecutaba configuraciones
(t₀, T) algebraicamente incapaces de producir resultados informativos, desperdiciando
cómputo y generando rechazos del gate multimodo que eran inevitables por diseño.

La Fase 2 introduce un **preflight Fisher** que predice analíticamente, antes de
ejecutar el pipeline, si una configuración dada puede producir un run informativo.

---

## Componentes implementados

### 1. Documento teórico (`docs/metodologia_informatividad_predictiva.md`)

Desarrollo completo del marco de informatividad predictiva:

- **Diagnóstico raíz:** el run de GW150914 falla porque t₀ = 23 ms es 4.2×τ₂₂₀;
  solo queda 0.022% de la SNR² del modo fundamental en la ventana.
- **Conexión Cramér-Rao → gate:** rel_iqr ≈ 0.304 × α / (Q × ρ_eff), por lo que
  la condición de informatividad Q × ρ_eff > 0.61 × α es evaluable sin pipeline.
- **Fórmula cerrada de t₀_max:** t₀_max = (τ/2) × ln(Q²ρ² / umbral²).
- **Mapa de viabilidad (t₀, T):** superficie analítica que delimita la región
  donde la informatividad es posible.
- **Calibración empírica de α_safety:** método para conectar la teoría (CR ideal)
  con la práctica (bootstrap real).

### 2. Módulo de preflight (`mvp/preflight_viability.py`)

Módulo pure-function (sin IO) con:

| Función | Propósito |
|---------|-----------|
| `snr_fraction_eta(t0, T, tau)` | Fracción de SNR² capturada en ventana |
| `rho_effective(t0, T, tau, rho)` | SNR efectiva en ventana |
| `rel_iqr_predicted(Q, rho_eff, alpha)` | Predicción de rel_iqr vía CR |
| `t0_max_informative(tau, Q, rho, alpha)` | t₀ máximo para informatividad |
| `T_min_resolution(tau)` | Duración mínima para resolución espectral |
| `band_min_hz(f, Q)` | Banda mínima para capturar el Lorentziano |
| `assess_mode_viability(...)` | Evaluación completa por modo |
| `preflight_viability(...)` | Evaluación por evento (todos los modos) |
| `catalog_viability_table(events)` | Tabla de viabilidad para N eventos |
| `calibrate_alpha_from_runs(obs)` | Calibración empírica de α_safety |
| `viable_t0_domain(...)` | Dominio viable de t₀ para sweep |

Verdicts: `VIABLE`, `MARGINAL`, `INVIABLE`, `DOMAIN_EMPTY`.

### 3. Integración en pipeline (`mvp/pipeline.py`)

- **`_run_preflight_viability()`**: ejecuta el preflight Fisher como Stage 0.5,
  después de s2 y antes de s3. Emite `preflight_viability.json`.
- **Validación del t₀ sweep**: si el t₀ seleccionado excede t₀_max del preflight,
  emite warning de que el run será no-informativo.
- **Timeline**: el resultado del preflight se registra en el timeline del run.

### 4. Catálogo de eventos GWTC

- **`gwtc_events_t0.json`**: metadatos de 55+ eventos con M_final, χ_final,
  SNR de red, t₀ de referencia.
- **`gwtc_quality_events.csv`**: tabla filtrada de eventos con calidad suficiente.
- **`mvp/gwtc_events.py`**: lookup de parámetros por event_id.

### 5. Herramientas de sincronización GWOSC

- **`tools/gwosc_sync_losc.py`**: descarga strain data de GWOSC/LOSC.
- **`download_gw_events.py`**: script de descarga masiva de eventos.

### 6. Correcciones de gates (`mvp/s3b_multimode_estimates.py`, `mvp/s4c_kerr_consistency.py`)

- Semántica corregida del gate `RINGDOWN_NONINFORMATIVE` en s4c y timeline:
  un evento no-informativo no es un fallo del pipeline sino una censura legítima.
- Propagación consistente del veredicto de viabilidad multimodo a stages
  downstream (s4c, s4d).

### 7. Tests

| Archivo | Tests | Cobertura |
|---------|-------|-----------|
| `tests/test_preflight_viability.py` | 28 | Todas las funciones del preflight, coherencia con gate multimodo, GW150914 como caso de validación |
| `tests/test_multimode_wiring_unittest.py` | 13 | Cableado de viabilidad entre s3b → s4c/s4d |
| `tests/test_s3b_multimode_estimates.py` | +4 | Nuevos tests para campos de viabilidad |
| `tests/test_s4c_kerr_consistency_unit.py` | +6 | Tests de semántica NONINFORMATIVE |
| `tests/test_contracts_*.py` | 5 fixes | Actualización de contratos (count=25, s4 outputs) |

---

## Resultado cuantitativo para GW150914

| Métrica | Config actual (t₀=23ms) | Config óptima (t₀=5ms) |
|---------|------------------------|----------------------|
| η (fracción SNR²) | 0.022% | ~16% |
| ρ_eff | 0.12 | ~3.2 |
| Q × ρ_eff | 0.52 | ~13.8 |
| rel_iqr predicho | ~1.5 | ~0.06 |
| Veredicto preflight | INVIABLE | VIABLE |

El preflight identifica correctamente que t₀ = 23 ms está fuera del dominio
viable (t₀_max ≈ 18 ms para α=2) y recomienda t₀ ∈ [3, 11] ms.

---

## Archivos nuevos/modificados

```
Nuevos:
  docs/metodologia_informatividad_predictiva.md   — Marco teórico completo
  mvp/preflight_viability.py                      — Módulo de predicción Fisher
  tests/test_preflight_viability.py               — 28 tests del preflight
  tests/test_multimode_wiring_unittest.py         — 13 tests de cableado
  gwtc_events_t0.json                             — Catálogo de eventos
  gwtc_quality_events.csv                         — Eventos filtrados
  download_gw_events.py                           — Descarga de eventos
  tools/gwosc_sync_losc.py                        — Sincronización GWOSC

Modificados:
  mvp/pipeline.py                                 — Integración del preflight
  mvp/s3b_multimode_estimates.py                  — Semántica NONINFORMATIVE
  mvp/s4c_kerr_consistency.py                     — Propagación de verdicts
  tests/test_s3b_multimode_estimates.py            — Tests adicionales
  tests/test_s4c_kerr_consistency_unit.py          — Tests adicionales
  tests/test_contracts_completeness.py             — Actualización de contratos
  tests/test_contracts_fase4.py                    — Actualización de contratos
  tests/test_contracts_fase5.py                    — Actualización de contratos
  tests/test_mvp_contracts.py                      — Actualización de contratos
```

---

## Estado de tests

586 passed, 0 failed, 27 skipped (los skips son por `numpy` no instalado en
el entorno de CI ligero; no afectan la funcionalidad del preflight).

---

## Trabajo futuro (no incluido en Fase 2)

1. **Sweep restringido**: integrar `viable_t0_domain()` como restricción dura
   en `experiment_t0_sweep.py` para que solo explore la región viable.
2. **PSD-aware preflight**: usar la PSD medida (post-s1) en lugar de la
   aproximación de PSD constante, incorporando el factor conforme Ω del
   Método Brunete.
3. **Tabla de viabilidad del catálogo**: generar la tabla de viabilidad completa
   como artefacto persistente para planificación de campañas de análisis.
4. **Calibración empírica de α_safety**: ejecutar `calibrate_alpha_from_runs()`
   sobre los runs existentes para fijar α_safety definitivo.
