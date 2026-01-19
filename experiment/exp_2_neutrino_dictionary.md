# Experimento 2 — Bloque C sobre sandbox “neutrino” (`ir_neutrino`)

## Objetivo

Validar el **Bloque C (diccionario holográfico emergente)** en un caso “sandbox” controlado (neutrino proxy), comprobando:

1. que el ajuste inverso **ratios → Δ** es estable (C2),
2. que la verificación de **compatibilidad espectral/ciclo** (C3) pasa con margen,
3. y que el contrato C1 (Ising 3D) se comporta correctamente como **filtro de dominio** (*OUT_OF_DOMAIN* cuando los targets están fuera del rango de Δ disponible).

Este experimento **no pretende** extraer física real de neutrinos ni inferir Ising 3D: es un **test de infraestructura** para Bloque C en un dataset limpio.

---

## Entradas y configuración

### Dataset de entrada

- Run: `ir_neutrino`
- Archivo: `runs/ir_neutrino/spectrum/outputs/spectrum.h5`
- Rango: `Δ ∈ [1.55, 5.50]`
- Tamaño: `n_delta=80`, `n_modes=5`
- Features usadas: `k_features=3` (ratios `r_1..r_3`)

### Comando reproducible

```bash
python 04_diccionario.py \
  --run ir_neutrino \
  --enable-c3 \
  --k-features 3 \
  --n-bootstrap 200
```

### Parámetros relevantes (según `runs/ir_neutrino/dictionary/stage_summary.json`)

- `models_evaluated = [linear, poly2]`
- `cv_folds=5`
- `random_seed=42`
- C1: `tau_delta=0.02`, `sigma_cap=0.1`
- C3: `metric=rmse`, `weights=none`, `threshold=0.05`
- C3 (modelo directo proxy): polinomial grado 4 (`direct_model=poly`, `direct_degree=4`)

**Nota de honestidad (importante):** el modelo directo Δ→ratios usado en C3 es un **proxy suave local** únicamente para evaluar invertibilidad/ciclo; **no es física**. Fallos en C3a pueden indicar proxy/modelo insuficiente, no necesariamente inconsistencia física.

---

## Resultados del modelo (ratios → Δ)

### Selección por BIC

Se evaluaron dos modelos inversos:

- `linear`: BIC ≈ -1012.45
- `poly2`: BIC ≈ -1357.45

Se selecciona **`poly2`** por BIC (menor es mejor).

### Métricas

- `cv_rmse ≈ 4.15e-05`
- `R² ≈ 0.999999999`
- `n_params = 10`

Interpretación: el mapeo ratios→Δ en este sandbox es casi perfectamente determinista en el rango cubierto.

### Incertidumbre (bootstrap)

Con `n_bootstrap=200`:

- `σ_Δ(mean) ≈ 9.66e-06`
- `σ_Δ(max) ≈ 4.15e-05`

Interpretación: la incertidumbre estimada es extremadamente pequeña, coherente con un dataset sin ruido relativo (`noise_rel=0`) y un mapeo suave.

---

## Contratos

Los contratos se evalúan según `runs/ir_neutrino/dictionary/validation.json` (v1.4.1).

### C1 — Compatibilidad puntual con Ising 3D

- Target σ: `Δσ = 0.518148806`
- Target ε: `Δε = 1.41262528`

Ambos están **fuera** del dominio del diccionario (`Δ_min = 1.55`), por lo que:

- `C1_sigma_status = OUT_OF_DOMAIN`
- `C1_epsilon_status = OUT_OF_DOMAIN`

**Lectura correcta:** C1 aquí actúa como **gate de dominio**. No hay fallo: el diccionario **no extrapola** fuera del soporte. El diseño explícito del contrato indica que C1 solo cuenta como “hard contract” cuando está *IN_DOMAIN*.

---

### C2 — Consistencia interna del diccionario

- `cv_rmse = 4.15e-05`
- Umbral: `threshold_cv_rmse = 0.05`
- Estado: `consistency_ok = true`

**Resultado:** C2 **PASS** con margen enorme.

---

### C3 — Compatibilidad espectral (cycle-consistency)

Configuración:
- Métrica: `rmse`
- Pesos: `none` (uniformes en las 3 ratios usadas)
- Umbral usuario: `0.05`
- Estado global: `PASS`

Componentes:
- **C3a (decoder)**: `rmse ≈ 1.28e-08`
- **C3b (cycle)**: `rmse ≈ 2.32e-06`

Noise floor (estimación robusta, eps=0.001):
- `aggregate_distance ≈ 6.91e-05`

**Lectura cuantitativa:** el error de ciclo C3b (`~2.3e-06`) está ~30× por debajo del *noise floor* (`~6.9e-05`), y muchísimo más bajo que el umbral fijo 0.05. Por tanto, C3 pasa **con margen**.

Sensibilidad (diagnóstico):
- `norm_median ≈ 0.118`
- `norm_max ≈ 0.142`

Interpretación: las ratios responden de manera razonable a perturbaciones pequeñas; no hay señales de degeneración severa en el rango cubierto.

---

## Estado global y criterio de verificación

Resumen global (`overall`):

- C1 σ/ε: `OUT_OF_DOMAIN`
- C2: `true`
- C3: `PASS`
- Definición de hard contracts: “C2 siempre; C3 si activo; C1 solo si IN_DOMAIN”
- `all_hard_contracts_pass = true`

**Conclusión:** el diccionario se considera **consistente y verificado** bajo la definición contractual vigente.

---

## Artefactos producidos (rutas y hashes)

Outputs (según `stage_summary.json`):
- `runs/ir_neutrino/dictionary/dictionary.h5`
- `runs/ir_neutrino/dictionary/atlas.json`
- `runs/ir_neutrino/dictionary/ising_comparison.json`
- `runs/ir_neutrino/dictionary/validation.json`
- `runs/ir_neutrino/dictionary/stage_summary.json`
- `runs/ir_neutrino/dictionary/manifest.json`

Hashes registrados (para auditoría):
- `dictionary.h5`: `8f4caaf88de5ccb42d2116d208c3ee6ad6e4b745ae5292cadf7524955f595dd1`
- `atlas.json`: `773309c60feafa55841ab6f8a6b80223cb2a5c53f8ff8df2a7699bd354504d45`
- `ising_comparison.json`: `43e92fe8f54456e51eb6a8eef0e8210f5ab9e0521739b449c3e1275c15465c2a`
- `validation.json`: `d81070b346757fc61a87054de84ebdcb09fd6d0ed0520d0a7e3125e016c99408`

---

## Riesgos, límites y falsabilidad

1. **No extrapolación (C1 OUT_OF_DOMAIN):** este run no puede informar sobre Ising 3D porque los targets están fuera del rango Δ del sandbox. Eso es una limitación **de soporte**, no del algoritmo.
2. **Proxy en C3:** C3 usa un modelo directo Δ→ratios como aproximación local; un fallo en C3a podría deberse a proxy/modelo insuficiente. Esto está explicitado en las notas del stage.
3. **Caso “limpio”:** el sandbox presenta mapeos casi deterministas; este experimento valida infraestructura, no robustez frente a ruido o no-inyectividad.

**Falsabilidad operacional:** este experimento sirve como *test de regresión*. Si futuras versiones rompen C2 o C3 aquí, la hipótesis “Bloque C funciona en un caso controlado” queda refutada de manera inmediata y diagnosticable.

---

## Checklist de reproducción

1. Confirmar que existe: `runs/ir_neutrino/spectrum/outputs/spectrum.h5`
2. Ejecutar el comando de Bloque C (arriba)
3. Verificar en `runs/ir_neutrino/dictionary/validation.json`:
   - `C2_consistency.consistency_ok == true`
   - `C3_spectral.status == "PASS"`
   - `overall.all_hard_contracts_pass == true`
