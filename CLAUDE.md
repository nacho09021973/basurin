# BASURIN/BRUNETE — Instrucciones para agente

## Proyecto

Pipeline contract-first de análisis de ringdown de agujeros negros.
Objetivo: tests de Relatividad General usando quasi-normal modes (QNM)
del modo dominante (2,2,0) sobre cohorte O4/O4b de ondas gravitacionales.

## Estructura

```
~/work/basurin/
├── mvp/                          # Código del pipeline
│   ├── pipeline.py               # Orquestador principal
│   ├── s3b_multimode_estimates.py # Stage clave: extracción QNM
│   ├── kerr_qnm_fits.py         # Referencia Kerr (Berti fits)
│   └── ...
├── runs/                         # Artefactos inmutables por run
│   └── <run_id>/
│       ├── RUN_VALID
│       ├── s2_ringdown_window/
│       ├── s3_ringdown_estimates/
│       ├── s3b_multimode_estimates/
│       │   ├── stage_summary.json
│       │   └── outputs/multimode_estimates.json
│       └── experiment/           # Diagnósticos y A/B tests
├── gwtc_events_t0.json           # Catálogo canónico
└── .venv/                        # Entorno Python
```

## Entorno

```bash
cd ~/work/basurin
source .venv/bin/activate
```

Verificar siempre que el venv está activo antes de ejecutar cualquier cosa.

## Convenciones contract-first (OBLIGATORIAS)

1. Todo stage emite `stage_summary.json` con `status: "PASS"` o `"FAIL"`.
2. Los artefactos bajo `runs/` son **inmutables** una vez materializados.
3. **Nunca** sobrescribir runs originales.
4. Para A/B tests: crear subruns derivados con symlinks a `RUN_VALID`,
   `s2_ringdown_window`, `s3_ringdown_estimates` del run original.
5. Resultados experimentales van a `runs/<run_id>/experiment/`.
6. Todo script nuevo va a la raíz del proyecto o a `experiment/`.
7. Los scripts solo **leen** del pipeline; no modifican código en `mvp/`
   salvo autorización explícita.

## Fuentes de verdad

- **QNM Kerr:** `mvp/kerr_qnm_fits.py`
  - Modo (2,2,0): Q y f dependen de Mf y af
  - Modo (2,2,1): Q_221(af) = 0.1000 + 0.5436 * (1 - af)^(-0.4731)
- **Catálogo:** `gwtc_events_t0.json`
  - Campos por evento: SNR, Mf_source, af, chi_eff, detectors, GPS, etc.
- **Cohorte actual:** 82 eventos O4/O4b
  - Preparada en `runs/brunete_prepare_20260323T1545Z/`
  - Catálogo de cohorte: `prepare_events/outputs/events_catalog.json`

## Interfaz CLI del pipeline

```bash
# Ejecutar s3b para un evento (subrun derivado)
python -m mvp.s3b_multimode_estimates \
    --runs-root ~/work/basurin/runs \
    --run-id <RUN_ID> \
    --n-bootstrap 200 \
    --seed 12345 \
    --method hilbert_peakband \
    --mode-221-topology <rigid_spectral_split|shared_band_early_taper> \
    --bootstrap-221-residual-strategy <refit_220_each_iter|fixed_220_template> \
    --band-strategy kerr_centered_overlap \
    [... más parámetros en s3b_multimode_estimates.py:1580]
```

## Estado actual del proyecto

### Carril 220: OPERATIVO
- 82/82 eventos PASS en batch 220.
- Fisher poblacional pendiente de ejecución.
- Objetivo: σ(δf₂₂₀) < 1%, σ(δτ₂₂₀) < 10% combinados.

### Carril 221: CERRADO COMO NO INFORMATIVO
- 82/82 eventos SINGLEMODE_ONLY. Resultado correcto.
- Causa raíz: Q_221_Kerr ≈ 1 para af ≈ 0.6-0.7 (sub-ciclo).
- Estimador espectral no puede medir Q < 2.
- Diagnóstico completo en `README_diario_221.md`.
- **No tocar sin autorización explícita.**

## Reglas de ejecución

### Antes de empezar cualquier tarea
1. Leer `README_diario_221.md` (diario del proyecto).
2. Identificar la tarea actual en la sección "Pendientes".
3. Verificar que los artefactos de entrada existen antes de ejecutar.

### Durante la ejecución
4. Ejecutar tests/verificaciones antes de declarar éxito.
5. Si algo falla, documentar el fallo exacto antes de reintentar.
6. No reintentar más de 2 veces con la misma estrategia.
7. Si un batch da resultado uniforme inesperado (ej. 100% FAIL),
   **no reintentar con variaciones**. Documentar y esperar instrucciones.

### Después de cada tarea
8. Actualizar `README_diario_221.md` con:
   - Fecha y nombre de la tarea
   - Hechos observados (no inferencias)
   - Artefactos generados (con rutas)
   - Pendientes actualizados
9. Hacer `git add` + `git commit` con mensaje descriptivo.

## Criterios de éxito por tarea

### Auditoría cruzada Q_220 (Prompt 3)
- Input: 3 eventos (GW230708_230935, GW231004_232346, GW230914_111401)
- Calcular Q_220_Kerr y f_220_Kerr usando `kerr_qnm_fits.py`
- Extraer Q_220 y f_220 medidos de `multimode_estimates.json`
- PASS si ratio medido/Kerr para Q_220 está entre 0.3 y 3.0 en 3/3
- Output: `runs/brunete_prepare_20260323T1545Z/experiment/q220_kerr_crosscheck.json`

### Batch poblacional Fisher
- Input: 82 eventos con carril 220 PASS
- Ejecutar: `run_population_fisher.py` (verificar existencia y args)
- PASS si se materializa `fisher_total.json` y `combined_constraints.json`
- Output esperado: σ(δf₂₂₀), σ(δτ₂₂₀)

### Pre-gate Q_221_Kerr
- Input: `af_distribution_and_pregate.json` ya materializado
- Verificación: 0 eventos pasan con umbral Q > 2.0 (resultado conocido)
- Implementar gate en `pipeline.py` o como stage nuevo

## Qué NO hacer (NUNCA)

- No inventar datos ni asumir artefactos que no existan en disco.
- No modificar código en `mvp/` sin autorización.
- No relajar thresholds de gates sin documentar justificación completa.
- No ejecutar pipeline completo (costoso) — ejecutar solo stages necesarios.
- No borrar ni sobrescribir runs existentes.
- No afirmar que algo funciona sin verificar el artefacto de salida.
- No decir "se necesita más información" sin proponer qué inspeccionar.

## Progreso y memoria

El diario del proyecto vive en `README_diario_221.md`.
Cada sesión añade una sección `## YYYY-MM-DD — Descripción`.
Los enfoques fallidos se documentan explícitamente para no repetirlos.

## Contexto científico mínimo

- QNM = Quasi-Normal Mode: oscilación amortiguada del agujero negro remanente.
- Modo (ℓ,m,n): (2,2,0) es el fundamental, (2,2,1) es el primer overtone.
- Q = factor de calidad: ciclos antes de decaer. Q > 2 = medible; Q ≈ 1 = no.
- f = frecuencia del modo. Depende de masa y spin del remanente.
- δf, δτ = desviaciones fraccionarias respecto a predicción Kerr (GR).
- Fisher matrix: aproximación gaussiana a la posterior; combinable entre eventos.
- Pipeline mide (f, Q) por evento con bootstrap, luego combina con Fisher.
