# C3b adaptive threshold

## Objetivo

El contrato C3b compara el ciclo completo **r → Δ̂ → r̂** y exige un umbral que refleje la sensibilidad del mapeo y la incertidumbre del inverso. El umbral fijo único no escala bien cuando la sensibilidad \|dr/dΔ\| cambia o cuando σΔ del inverso aumenta.

## Fórmula del umbral adaptativo

Cuando `--c3-adaptive-threshold` está activo, el umbral usado por C3b se calcula como:

```
noise_floor_r = median_sigma (según --c3-noise-floor-metric)
sigma_delta_used = p90(σΔ) si está disponible; de lo contrario, mean(σΔ)
s_p90 = p90 de ||dr/dΔ||_2 estimado sobre datos

tol_cycle = max(noise_floor_r, s_p90 * sigma_delta_used)
threshold_used = --c3-threshold-factor * tol_cycle
```

Cuando `--c3-adaptive-threshold` está desactivado, C3b usa el umbral fijo `--c3-threshold` (sin modificar C3a).

## Estimación de sensibilidad

La sensibilidad se estima sobre los datos (Δ_i, r_i) usando diferencias finitas robustas:

```
Ordenar por Δ
s_i = ||r_{i+1} - r_{i-1}||_2 / (Δ_{i+1} - Δ_{i-1})
(Bordes: forward/backward)
```

Se reportan cuantiles `s_p50`, `s_p90`, `s_p95` y `s_max` para inspección rápida.

## Persistencia en validation.json

El archivo `runs/<run>/dictionary/validation.json` ahora incluye:

- `bootstrap.sigma_delta_mean`, `bootstrap.sigma_delta_p90`, `bootstrap.sigma_delta_max`.
- `C3_spectral.threshold_used`, `C3_spectral.tol_cycle`.
- `C3_spectral.sensitivity` con `s_p50` y `s_p90`.
- `C3_spectral.sigma_delta.used` indicando el σΔ usado en el cálculo.

Esto permite auditar el umbral aplicado y verificar que el contrato C3b usa un criterio escalado por sensibilidad e incertidumbre.
