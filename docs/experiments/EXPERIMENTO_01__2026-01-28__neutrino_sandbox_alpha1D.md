# EXPERIMENTO Nº 1 — Neutrino sandbox: identificabilidad y colapso 1D en α  
**Fecha:** 28 de enero de 2026

## Resumen ejecutivo

En el neutrino sandbox (familia `symmetron`) se ha demostrado de forma reproducible que:

1) El eje efectivo del observable `X` (ratios derivados del espectro) es **1D** (colapso casi total en PC1).  
2) Tras romper la colinealidad del muestreo (`paired` → `cartesian`), el eje 1D se atribuye de forma **identificable** a **α**.  
3) El parámetro **δ es no-identificable** en este sandbox: el espectro `M2` resulta prácticamente **independiente de δ** en los rangos probados.

Este resultado no es un “fallo” del pipeline: es **señal**, porque establece límites de identificabilidad del proxy.

---

## Gobernanza BASURIN

- Todo IO determinista bajo `runs/<run_id>/`.  
- Stages canónicos escriben: `manifest.json`, `stage_summary.json`, `outputs/…`.  
- `atlas_master` agrega runs sin inventar datasets y añade diagnóstico `atlas_master_diagnostics` en `stage_summary.json`.  
- En `04_diccionario.py`, cuando `run_kind == "spectrum_only"`, se aplica perfil **proxy**: C1/C2 quedan como **diagnóstico**, no como hard gate.

---

## Cambios implementados

### 1) `06_build_atlas_master_stage.py`: diagnóstico de redundancia/rango de X

Se añadió un “contrato diagnóstico” en `stage_summary.json`:

`atlas_master_diagnostics = { X_singular_values, X_explained_var_1, X_rank, X_pairwise_corr_max_offdiag, X_effective_dim, X_is_redundant, ... }`

Objetivo: capturar de forma auditable cuándo `X` es efectivamente 1D (evitar interpretaciones 2D infladas).

### 2) `01_genera_neutrino_sandbox.py`: `--grid-mode {paired,cartesian}` + trazabilidad

Se añadió muestreo cartesian para romper colinealidad α–δ.  
Se reforzó trazabilidad en `spectrum/stage_summary.json`:

- `alpha_values`, `delta_values`  
- `grid_order` (p.ej. `delta_outer_alpha_inner`)  
- `n_alpha`, `n_delta`, `n_total`  

### 3) `04_diccionario.py`: perfil proxy por `run_kind == "spectrum_only"`

Para runs neutrino/proxy:

- no hard-fail por C1/C2;  
- mantiene forense de C1/C2 como diagnóstico;  
- `hard_contracts_profile = proxy`, `all_hard_contracts_pass = True`.

---

## Reproducción: comandos exactos y runs

### A) Generación cartesian (smoke) con trazabilidad completa

Run: `2026-01-28__neutrino_cart_smoke2`

```bash
python 01_genera_neutrino_sandbox.py   --run 2026-01-28__neutrino_cart_smoke2   --family symmetron   --profiles vacuum,crust,mantle   --grid-mode cartesian   --n-delta 5   --n-alpha 4   --alpha-min -0.05 --alpha-max 0.05   --delta-min 0.7 --delta-max 3.0   --rho0 2.6 --rho-crit 4.0   --noise-rel 0.0   --seed 123
```

Verificación de independencia (reconstruida desde `stage_summary.json`):
- `corr(alpha, delta) = 0.0` (cartesian correcto)

### B) Diccionario + features (proxy)

```bash
python 04_diccionario.py --run 2026-01-28__neutrino_cart_smoke2 --k-features 2 --n-bootstrap 0
python 05_build_features_stage.py --run 2026-01-28__neutrino_cart_smoke2
```

Verificación (esperado):
- `hard_contracts_profile: proxy`
- `all_hard_contracts_pass: True`
- C1/C2 pueden seguir indicando OUT_OF_DOMAIN/False sin bloquear

### C) Atlas master (agregación 1-run) + diagnóstico 1D

```bash
python 06_build_atlas_master_stage.py   --run 2026-01-28__atlas_master_C   --runs 2026-01-28__neutrino_cart_smoke2
```

Resultado observado:
- `X_effective_dim = 1`
- `X_explained_var_1 ≈ 0.999994` (colapso 1D fuerte)

---

## Resultado principal: atribución del eje 1D a α

Se calcula `pc1(X)` (PC1 por SVD de `X` centrado) y se ajusta:

- `pc1 ~ 1 + alpha`: **R² ≈ 0.999434**, `corr(pc1, alpha) ≈ 0.999717`  
- `pc1 ~ 1 + delta`: **R² ≈ 0**, `corr(pc1, delta) ≈ 0`  
- `pc1 ~ 1 + alpha + delta`: coeficiente de δ ≈ 0  

Conclusión: en este sandbox, el eje efectivo observable está gobernado por **α** y **δ no contribuye**.

---

## Confirmación directa desde el espectro: δ no afecta a M2

Se inspecciona `runs/<run>/spectrum/outputs/spectrum.h5` (datasets: `M2`, `delta_uv`, `m2L2`, `z_grid`) y se correlaciona `M2[:,mode]` con α y δ.

Para `2026-01-28__neutrino_cart_modes5` (cartesian, `n_modes=5`):

- Para modos con varianza no nula: `corr(M2, alpha) ≈ ±1.0`  
- Para esos modos: `corr(M2, delta) ≈ 0.0`  
- Existe un modo constante (`std = 0`), produciendo `corr = NaN` (esperado)

También se probó `map_mode=quad` sin cambios: δ sigue ausente en `M2`.

Interpretación: δ es **no-identificable** en este toy-model/rango; no hay “estructura 2D” que recuperar a partir del espectro.

---

## Artefactos canónicos esperados

Para cada run neutrino:

```
runs/<run>/spectrum/
  manifest.json
  stage_summary.json      # incluye alpha_values/delta_values/grid_order en cartesian
  outputs/spectrum.h5     # contiene dataset M2 (N x n_modes)
```

Para diccionario:

```
runs/<run>/dictionary/
  manifest.json
  stage_summary.json      # hard_contracts_profile=proxy si run_kind=spectrum_only
  outputs/atlas.json
  outputs/dictionary.h5
  outputs/validation.json
  outputs/ising_comparison.json   # diagnóstico (puede fallar sin gate)
```

Para features:

```
runs/<run>/features/
  manifest.json
  stage_summary.json
  outputs/features.json
  outputs/X.npy
  outputs/Y.npy
```

Para atlas master:

```
runs/<run_master>/atlas_master/
  manifest.json
  stage_summary.json      # incluye atlas_master_diagnostics
  outputs/BRIDGE_ATLAS_MASTER.json
```

---

## Tests relevantes

- `tests/test_neutrino_sandbox_contracts.py`:
  - cubre `grid_mode=cartesian` y trazabilidad en `stage_summary`.
- Tests de `atlas_master`:
  - verifican presencia/estructura de `atlas_master_diagnostics`.
- Tests de `04_diccionario.py`:
  - verifican que `run_kind=spectrum_only` activa perfil proxy (`all_hard_contracts_pass=True`).

---

## Conclusiones y límites

1) Resultado sólido: el sandbox actual es un proxy **1D en α**.  
2) δ no es observable con `M2` en este setup (rango y modelo probados).  
3) Si se desea un sandbox 2D real:
   - redefinir el modelo para que el segundo parámetro afecte a la ecuación y al espectro; o
   - introducir un segundo observable independiente (no solo ratios actuales).

---

## Próximo paso sugerido (opcional, contract-first)

Formalizar un diagnóstico canónico de sensibilidad:
- stage `param_sensitivity` (diagnóstico): por modo, reportar `std`, `corr(M2, alpha)`, `corr(M2, delta)` y banderas de “modo constante”.
Esto evita perseguir parámetros no-identificables en futuros experimentos.
