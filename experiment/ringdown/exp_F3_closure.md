# Contrato C6 — F3-closure (cierre de Fase 3)

## Propósito

Este contrato define un experimento mínimo, reproducible y falsable para cerrar la **Fase 3 (contrastación externa)** en BASURIN sin introducir un mapa físico no justificado entre espacios de features incompatibles.

El cierre se establece mediante un único veredicto binario: **PASS/FAIL**, acompañado de diagnósticos informativos.

Artefacto árbitro: `runs/<run_id>/f3_closure/outputs/decision.json` (ver ejemplo de referencia en `runs/2026-01-20__F3_closure/...`).

## Principio de honestidad (no inyección)

- **No se asume** ningún feature-map físico **QNM ↔ ratios**.
- Si `feature_key` o `dim` no coinciden, el pipeline **debe parar** con `INCOMPATIBLE_FEATURE_SPACE`.
- La compatibilidad en un espacio común solo está permitida como **control sintético** explícito (`feature_key="common_latent"`).

## Definición del experimento

El experimento ejecuta tres casos y exige que todos cumplan su expectativa:

### Caso A (REAL) — Control negativo estructural

**Input:** `atlas_points` (ratios) vs `ringdown_features` (qnm)

**Expectativa:** `C6a FAIL` con `failure_mode=INCOMPATIBLE_FEATURE_SPACE`; `C6b SKIP`.

Ejemplo verificado:
- `atlas_feature_key="ratios"`, `atlas_dim=3`
- `ext_feature_key="qnm"`, `ext_dim=4`
- Resultado: `C6a FAIL / INCOMPATIBLE_FEATURE_SPACE`, `C6b SKIP`.

**Nota sobre `atlas_dim`:**
- En este cierre (run de referencia) se espera explícitamente **`atlas_dim = 3`**, consistente con un atlas ratios construido con **k=3** (ratios).
- Si se cambia el generador de atlas (p. ej. k=9), este contrato puede parametrizarse; el criterio sigue siendo: **mismatch ⇒ INCOMPATIBLE_FEATURE_SPACE**.

### Caso B (POS) — Control positivo compatible (sintético, espacio común)

**Input:** `atlas_synth` vs `external_synth_inlier` en `feature_key="common_latent"`, `dim=4`

**Expectativa:** `C6a PASS` y `C6b PASS`.

`C6b` usa calibración interna:
- `tau = p_tau(D_self)` donde `D_self` son distancias leave-one-out dentro del atlas.
- `PASS` si `frac_inlier >= frac_inlier_pass` **y** `d_med <= tau`.

### Caso C (NEG-OOD) — Control negativo OOD (sintético, espacio común)

**Input:** `atlas_synth` vs `external_synth_ood` en `feature_key="common_latent"`, `dim=4`

**Expectativa:** `C6a PASS` y `C6b FAIL_OOD`.

## Criterio formal de cierre (PASS/FAIL)

Sea `decision.json` el árbitro.

**Fase 3 cerrada ⇔** se cumple simultáneamente:

- `overall_status == "PASS"`
- `checks.ok_A == true`
- `checks.ok_B == true`
- `checks.ok_C == true`

Cualquier desviación implica **Fase 3 NO cerrada** y requiere diagnóstico según el caso que falle:

- Si A no da `INCOMPATIBLE_FEATURE_SPACE`: el gating estructural es incorrecto (riesgo de falsa compatibilidad).
- Si B falla: el contraste en espacio común no funciona (falso negativo).
- Si C pasa: el contraste OOD es laxo o defectuoso (falso positivo).

## Ejecución (run plan)

Ejemplo (ajusta rutas según tus runs):

```bash
python experiment/ringdown/stage_F3_closure.py \
  --run 2026-01-20__F3_closure \
  --atlas-points runs/<latest_dictionary_run>/dictionary/outputs/atlas_points.json \
  --ringdown-features runs/<ringdown_run>/ringdown_features/outputs/features.json \
  --seed 42 \
  --p-tau 95 \
  --frac-inlier-pass 0.90
```

Outputs esperados:

```
runs/<run_id>/f3_closure/
  manifest.json
  stage_summary.json
  outputs/
    decision.json
    calibration.json
    inputs_hashes.json
    cases/
      A_real.json
      B_positive.json
      C_negative_ood.json
```

## Referencia verificada (evidencia)

Run de referencia:
- `runs/2026-01-20__F3_closure/f3_closure/outputs/decision.json` con:
  - `overall_status = PASS`
  - `checks.ok_A = true`, `checks.ok_B = true`, `checks.ok_C = true`
  - Diagnóstico A_real: `INCOMPATIBLE_FEATURE_SPACE` (ratios dim=3 vs qnm dim=4)
  - B_positive: PASS (inliers)
  - C_negative_ood: FAIL_OOD (OOD contundente)

Este run constituye evidencia suficiente para declarar cerrado el criterio operacional de Fase 3 bajo C6.
