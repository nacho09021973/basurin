# EXPERIMENTO_03 — 2026-01-29 — dual_spectrum_min_observable

## Objetivo
Determinar si el observable adicional “dual spectrum” (Dirichlet/Neumann) rompe la degeneración efectiva 1D del inverso basado en espectro.

## Run
- run_id: `2026-01-29__EXPERIMENTO_03__dual_spectrum_min_observable`
- verdict artifact: `runs/2026-01-29__EXPERIMENTO_03__dual_spectrum_min_observable/experiment/EXPERIMENTO_03/outputs/verdict.json`

## Observable
Se computan tres canales espectrales:
- `M2` (base)
- `M2_D` (Dirichlet)
- `M2_N` (Neumann)

El veredicto se evalúa sobre `mean_modes` por punto (media sobre modos).

## Criterio (ejecutivo)
PASS si:
- `effective_rank_mean_modes >= 2` y
- `sigma_ratio_2_1 >= 0.05`

## Resultados
VERDICT: PASS  
effective_rank_mean_modes: 2  
sigma_ratio_2_1: 0.11316729307901596  
sigma3_relative: 6.992570690500708e-17  

param_axes:
- delta_uv: `spectrum.h5/delta_uv`
- m2L2: `spectrum.h5/m2L2`

## Interpretación mínima
El observable dual introduce un segundo eje informativo independiente (subespacio 2D). La tercera dirección es numéricamente nula.

## Limitación
Los parámetros generativos `alpha`/`delta` no están persistidos en este run; el veredicto se formula sobre ejes canónicos presentes en `spectrum.h5` (`delta_uv`, `m2L2`).
