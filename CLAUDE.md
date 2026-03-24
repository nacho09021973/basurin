# BASURIN/BRUNETE — Instrucciones para agente

## Proyecto

Pipeline contract-first de análisis de ringdown de agujeros negros.
Objetivo: tests de Relatividad General usando quasi-normal modes (QNM)
del modo dominante (2,2,0) sobre cohorte O4/O4b de ondas gravitacionales.

## Estructura

```
~/work/basurin/
├── mvp/                          # Código del pipeline (NO MODIFICAR)
│   ├── pipeline.py               # Orquestador principal
│   ├── s3b_multimode_estimates.py # Stage clave: extracción QNM
│   ├── s3_ringdown_estimates.py   # Estimador base (hilbert_peakband)
│   ├── s2_ringdown_window.py      # Ventana de ringdown (dt_start CLI)
│   ├── s5_aggregate.py            # Agregación poblacional (δf, δQ vs Kerr)
│   ├── s7_beyond_kerr_deviation_score.py  # Desviaciones beyond-Kerr
│   ├── kerr_qnm_fits.py          # Referencia Kerr (Berti fits)
│   └── ...
├── runs/                          # Artefactos inmutables por run
│   └── <run_id>/
│       ├── RUN_VALID
│       ├── s2_ringdown_window/
│       ├── s3_ringdown_estimates/
│       ├── s3b_multimode_estimates/
│       │   ├── stage_summary.json
│       │   └── outputs/multimode_estimates.json
│       └── experiment/            # Diagnósticos y A/B tests
├── docs/
│   └── README_diario_nacho_martin.md   # DIARIO PRINCIPAL (leer siempre)
├── gwtc_events_t0.json            # Catálogo canónico
├── run_population_fisher.py       # Fisher forecast v0.2 (analítica)
├── select_sentinel_events.py      # Selector de centinela estratificado
├── CLAUDE.md                      # Este archivo
├── launch_agent.sh                # Launcher tmux
└── .venv/                         # Entorno Python
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

- **QNM Kerr (2,2,0):** `mvp/kerr_qnm_fits.py:27`
  - Q_220(af) = 0.7000 + 1.4187 * (1 - af)^(-0.4990)
  - F_220(af) = 1.5251 - 1.1568 * (1 - af)^(0.1292)
  - f_220 = F_220 / (2π × Mf_source × MSUN_S)
  - Convención: Q = πfτ (igual que el estimador)
- **QNM Kerr (2,2,1):** `mvp/kerr_qnm_fits.py:40`
  - Q_221(af) = 0.1000 + 0.5436 * (1 - af)^(-0.4731)
- **Catálogo:** `gwtc_events_t0.json` (90 eventos totales)
  - Campos por evento: SNR, Mf_source, af, chi_eff, detectors, GPS, etc.
- **Cohorte actual:** 82 eventos O4/O4b
  - Preparada en `runs/brunete_prepare_20260323T1545Z/`
  - Catálogo de cohorte: `prepare_events/outputs/events_catalog.json`
    (formato: dict con clave `event_ids` → lista de strings)

## Interfaz CLI del pipeline

```bash
# s2: ventana de ringdown (dt_start es CLI directo)
python -m mvp.s2_ringdown_window \
    --runs-root ~/work/basurin/runs \
    --run-id <RUN_ID> \
    --dt-start-s 0.003 \
    --t0-shift-ms 0.0

# s3b: extracción QNM multimode
python -m mvp.s3b_multimode_estimates \
    --runs-root ~/work/basurin/runs \
    --run-id <RUN_ID> \
    --n-bootstrap 200 \
    --seed 12345 \
    --method hilbert_peakband \
    --mode-221-topology <rigid_spectral_split|shared_band_early_taper> \
    --bootstrap-221-residual-strategy <refit_220_each_iter|fixed_220_template> \
    --band-strategy kerr_centered_overlap

# Fisher forecast (NO usa s3b, usa Kerr teórico)
python run_population_fisher.py \
    --catalog gwtc_events_t0.json \
    --cohort-file runs/brunete_prepare_20260323T1545Z/prepare_events/outputs/events_catalog.json \
    --output-dir runs/brunete_prepare_20260323T1545Z/experiment/population_fisher_forecast/ \
    --snr-fraction 0.3
```

## Estado actual del proyecto (actualizado 2026-03-23)

### Carril 220: OPERATIVO con sesgo conocido en Q
- 82/82 eventos PASS en batch 220.
- **SESGO DESCUBIERTO:** hilbert_peakband mide Q_220 ≈ 4.5× Q_Kerr.
  - Frecuencia bien medida (ratio f_medido/f_Kerr ≈ 0.95–1.06).
  - Q sistemáticamente sobreestimado.
  - Sesgo NO viene del merger (dt_start scan lo descarta — Q sube al
    retrasar t0, no baja).
  - Causa probable: estimador mide Q efectivo de señal+ruido en banda,
    no del QNM aislado.
  - Evidencia: auditoría cruzada 3 eventos + dt_start scan.
  - Artefactos: `experiment/q220_kerr_crosscheck.json`,
    `experiment/q220_dtstart_scan_GW230708_230935.json`
- **Consecuencia:** δf vs Kerr es interpretable. δτ/δQ vs Kerr NO sin
  calibración (s5_aggregate y s7 definen desviaciones vs Kerr).
- **Fisher forecast COMPLETADO:** σ(δf₂₂₀) = 0.58%, σ(δτ₂₂₀) = 3.88%
  con 81 eventos. Forecast teórico, NO afectado por sesgo de s3b.
  Artefacto: `experiment/population_fisher_forecast/`

### Carril 221: CERRADO COMO NO INFORMATIVO
- 82/82 eventos SINGLEMODE_ONLY. Resultado físicamente correcto.
- Causa raíz: Q_221_Kerr < 1.5 para toda la cohorte (af 0.61–0.84).
- 0/82 eventos pasan pre-gate Q_221_Kerr > 1.5 (ni siquiera > 2.0).
- Diagnóstico completo con 4 A/B tests y distribuciones bootstrap.
- **No tocar sin autorización explícita.**

### Paper: pivote estratégico definido
Framing: "Ringdown-only population analysis of O4 events: methodology,
constraints on δf₂₂₀, and systematic characterization of overtone and
damping measurability limits."
Tres contribuciones: método (Fisher ringdown-only), resultado (σ(δf)=0.58%),
diagnóstico (221 + sesgo Q).

## Tareas pendientes (por prioridad)

### PRIORIDAD ALTA — Paper
1. **Extender auditoría Q_220 a 10 centinela.**
   Verificar que ratio Q_medido/Q_Kerr ≈ 4.5× es estable en función de
   masa, spin y SNR. Usa los 10 centinela de
   `experiment/sentinel_selection.json`.
   Para cada evento: leer Q_220 y f_220 de multimode_estimates.json
   del batch 221 (la ruta real es
   `runs/brunete_batch_221_20260323T1605Z/run_batch/event_runs/brunete_<EVENT>_m221/s3b_multimode_estimates/outputs/multimode_estimates.json`).
   Campo: `modes[220].ln_Q` (= mediana bootstrap lnQ_p50).
   Calcular Q_Kerr con la fórmula de arriba usando Mf_source y af del catálogo.
   Output: `experiment/q220_kerr_crosscheck_10sentinela.json`

2. **Investigar calibración del sesgo Q_220.**
   Si el ratio es estable (~4.5×), evaluar si se puede aplicar factor de
   calibración empírico. Si depende de SNR o Mf, documentar la dependencia.
   Output: `experiment/q220_bias_characterization.json`

3. **Evaluar incorporación de GW250114.**
   GW250114 (O4b, enero 2025, SNR~80, af~0.68) es el evento más ruidoso
   y el único con detección de overtone a 4.1σ. Verificar:
   - ¿Está en el catálogo canónico? Si no, ¿hay datos públicos?
   - ¿Puede procesarse con el pipeline actual?
   Output: nota en el diario.

4. **Redactar outline del paper.**
   Tres secciones principales:
   - Método: pipeline contract-first, Fisher ringdown-only, 82 eventos O4
   - Resultado: σ(δf₂₂₀) = 0.58% (forecast), δf₂₂₀ observado (medición real)
   - Diagnóstico: 221 no medible (Q_Kerr sub-ciclo), Q_220 sesgado
     (hilbert_peakband), dominio de validez del estimador
   Output: `docs/paper_outline.md`

### PRIORIDAD MEDIA — Estimador
5. **Investigar alternativas para medir Q/τ.**
   - Fit paramétrico en dominio temporal (damped sinusoid)
   - Estimación de τ por decay de envolvente con sustracción de suelo de ruido
   - ¿El sesgo Q escala como función de SNR? Si es predecible → calibración.

### PRIORIDAD MEDIA — Infraestructura
6. **Implementar pre-gate Q_221_Kerr en pipeline.py.**
   Evita 82 extracciones inútiles de 221. Datos ya disponibles en
   `experiment/af_distribution_and_pregate.json`.

7. **Especificación contratos carril 221_detection_candidate.**
   3 contratos: detectability_gate, targeted_extractor, detection_verdict.
   Validación requiere inyecciones sintéticas o GW250114.

### PRIORIDAD BAJA — Futuro
8. Estimador dominio temporal para 221.
9. Evaluar modo (3,3,0) como alternativa para espectroscopía.

## Reglas de ejecución

### Antes de empezar cualquier tarea
1. Leer `docs/README_diario_nacho_martin.md` (diario del proyecto).
2. Identificar la tarea actual en "Tareas pendientes" arriba.
3. Verificar que los artefactos de entrada existen antes de ejecutar.

### Durante la ejecución
4. Ejecutar tests/verificaciones antes de declarar éxito.
5. Si algo falla, documentar el fallo exacto antes de reintentar.
6. No reintentar más de 2 veces con la misma estrategia.
7. Si un batch da resultado uniforme inesperado (ej. 100% FAIL),
   **no reintentar con variaciones**. Documentar y esperar instrucciones.

### Después de cada tarea
8. Actualizar `docs/README_diario_nacho_martin.md` con:
   - Fecha y nombre de la tarea
   - Hechos observados (no inferencias)
   - Artefactos generados (con rutas)
   - Pendientes actualizados
9. Hacer `git add` + `git commit` con mensaje descriptivo.

## Qué NO hacer (NUNCA)

- No inventar datos ni asumir artefactos que no existan en disco.
- No modificar código en `mvp/` sin autorización.
- No relajar thresholds de gates sin documentar justificación completa.
- No ejecutar pipeline completo (costoso) — ejecutar solo stages necesarios.
- No borrar ni sobrescribir runs existentes.
- No afirmar que algo funciona sin verificar el artefacto de salida.
- No decir "se necesita más información" sin proponer qué inspeccionar.
- No tocar el carril 221 sin autorización explícita.
- No interpretar δτ/δQ de s5/s7 como test de GR sin calibrar sesgo Q.

## Hechos clave descubiertos (2026-03-23)

Estos son hechos verificados, no hipótesis. No contradecirlos sin evidencia nueva.

1. **Q_221_Kerr < 1.5 para toda la cohorte O4** (af 0.61–0.84).
   Para Q > 2.0 se requiere af > 0.929. Máximo observado: af = 0.843.

2. **hilbert_peakband sobreestima Q_220 por ~4.5×.**
   f correcta (ratio ~1.0), Q sesgado. Sesgo no es de merger (dt_start scan:
   Q sube al retrasar t0). Convención verificada: ambos usan Q = πfτ.

3. **Bootstrap Q_221 centrado en Q ≈ 9–11, no en Q_Kerr ≈ 1.**
   El estimador no "ve" el overtone — ajusta artefacto espectral del residuo.

4. **s5_aggregate y s7 definen δ respecto a Kerr, no al valor medido.**
   Con sesgo Q ~4.5×, δQ_rel ≈ 3.5 (350% de "desviación" aparente).

5. **Fisher analítica: Γ = ρ² × diag(2Q², 1/2).**
   σ(δf/f) = 1/(ρQ√2), σ(δτ/τ) = √2/ρ. Forecast NO usa s3b.

## Contexto científico mínimo

- QNM = Quasi-Normal Mode: oscilación amortiguada del agujero negro remanente.
- Modo (ℓ,m,n): (2,2,0) es el fundamental, (2,2,1) es el primer overtone.
- Q = factor de calidad (Q = πfτ): ciclos antes de decaer. Q > 2 = medible.
- f = frecuencia del modo. Depende de masa y spin del remanente.
- δf, δτ = desviaciones fraccionarias respecto a predicción Kerr (GR).
- Fisher matrix: aproximación gaussiana a la posterior; combinable entre eventos.
- Pipeline mide (f, Q) por evento con bootstrap, luego combina con Fisher.
- LVK GWTC-4.0 TGR (marzo 2026): análisis full-IMR con pSEOBNRv5PHM.
  BRUNETE es complementario: ringdown-only con Fisher explícita.
