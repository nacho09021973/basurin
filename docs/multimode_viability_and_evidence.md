# BASURIN: semántica de `multimode_viability`, `systematics_gate` y `science_evidence`

## Objetivo de este documento
Este documento define, en términos de contrato auditable, la separación de responsabilidades entre tres bloques de `stage_summary.json` en el flujo multimodo de BASURIN. Está basado en el informe teórico v3 y en la implementación actual del repositorio.

**Principio rector:** el pipeline separa explícitamente:
1. **informatividad de datos** (`multimode_viability`),
2. **control de sistemáticas** (`systematics_gate`),
3. **cálculo de evidencia formal** (`science_evidence`).

No mezclar estos planos evita conclusiones espurias y facilita auditoría reproducible.

---

## 1) Separación explícita de responsabilidades

## 1.1 `multimode_viability` (¿hay información multimodo utilizable?)
Bloque de clasificación determinista sobre calidad/estabilidad de la extracción 220+221.

- Produce una clase: `MULTIMODE_OK`, `SINGLEMODE_ONLY` o `RINGDOWN_NONINFORMATIVE`.
- Usa métricas y umbrales operativos (fracciones válidas, estabilidad, consistencia geométrica, flags severos).
- **No** emite veredicto sobre nueva física.

Interpretación operacional:
- `MULTIMODE_OK`: se puede continuar a chequeo de sistemáticas.
- `SINGLEMODE_ONLY`: continuar sólo como resultado monomodo.
- `RINGDOWN_NONINFORMATIVE`: no hay base para análisis multimodo downstream.

## 1.2 `systematics_gate` (¿el contexto instrumental/estimador permite evaluar?)
Bloque determinista de control de riesgos conocidos.

- Evalúa checks como `t0_plateau`, `psd_sanity`, `estimator_resolution`.
- Produce `verdict_auto`: `PASS`, `FAIL` o `NOT_AVAILABLE`.
- **No** calcula evidencia científica: sólo habilita o bloquea evaluación.

## 1.3 `science_evidence` (¿se evalúa o no se evalúa evidencia formal?)
Bloque de salida formal de evidencia, condicionado por los dos gates anteriores.

- Siempre existe en `stage_summary.json`.
- Si los gates no pasan: `status = NOT_EVALUATED` con causas explícitas.
- Si los gates pasan: `status = EVALUATED` y se calcula el objeto determinista mínimo (`H1_min.delta_Rf`) junto con `future_slots` reservados.

Regla de composición (resumen):
- Si `multimode_viability.class != MULTIMODE_OK` → `NOT_EVALUATED`.
- Si `systematics_gate.verdict_final != PASS` → `NOT_EVALUATED`.
- Sólo con ambos en verde → `EVALUATED`.

---

## 2) Definición de estados `EVALUATED` / `NOT_EVALUATED`

## 2.1 `EVALUATED`
`science_evidence.status = EVALUATED` significa únicamente:

- se cumplieron precondiciones de informatividad multimodo y sistemáticas,
- se computó el objeto de evidencia definido por contrato,
- el artefacto es auditable y reproducible.

**No significa** validación científica de una hipótesis física. Significa "evaluación habilitada y ejecutada".

## 2.2 `NOT_EVALUATED`
`science_evidence.status = NOT_EVALUATED` significa que la evaluación formal **no** se ejecutó por contrato.

Causas típicas (`reason_if_skipped`):
- `MULTIMODE_GATE`
- `SYSTEMATICS_FAIL`
- `SYSTEMATICS_NOT_AVAILABLE`
- `RF_NOT_COMPUTABLE`
- `RF_KERR_GRID_INVALID`

Interpretación de auditoría: ausencia de evaluación es una salida válida y explícita; no es un error silencioso.

---

## 3) Override humano: sólo degradante (nunca promocionante)

## 3.1 Regla
El override humano puede **vetar** evaluación (`PASS` automático → `FAIL` final), pero no puede promover:

- `FAIL` automático **no** puede convertirse a `PASS`.
- `NOT_AVAILABLE` automático **no** puede convertirse a `PASS`.

Esto mantiene determinismo operacional y evita bypass manual de controles automáticos.

## 3.2 Ejemplo de payload con hash
Ruta recomendada del archivo: `runs/<run_id>/annotations/systematics_override.json`.

```json
{
  "schema_version": "systematics_override_v1",
  "verdict_human": "FAIL",
  "reason": "non_stationary_segment_detected_in_manual_review",
  "analyst_id": "ops-01",
  "timestamp_utc": "2026-03-03T12:34:56Z",
  "sha256_self": "8f45e4a6f4ad1b2141f3a47a2f36c77bb0f31abcead3e31f36e4d0d1f55a46c0"
}
```

Uso de auditoría:
- registrar el hash del documento de override,
- enlazar hash y razón en el informe de revisión,
- verificar que el efecto fue degradar a `NOT_EVALUATED` y no promover.

---

## 4) Checklist de auditoría (10 ítems)

1. **Ubicación canónica:** confirmar `stage_summary.json` bajo `runs/<run_id>/<stage>/stage_summary.json`.
2. **Bloques presentes:** existen `multimode_viability`, `systematics_gate`, `science_evidence`.
3. **Schema versions:** cada bloque reporta `schema_version` esperado.
4. **Separación semántica:** `multimode_viability` no contiene claims de evidencia.
5. **Gate de sistemáticas:** `systematics_gate.verdict_auto` consistente con checks y thresholds.
6. **Estado explícito:** `science_evidence.status` es `EVALUATED` o `NOT_EVALUATED` (sin estados implícitos).
7. **Razones de no evaluación:** si `NOT_EVALUATED`, `reason_if_skipped` no está vacío y es coherente.
8. **Override degradante:** si hay override humano, validar que sólo degrada y está hasheado.
9. **Trazabilidad de parámetros:** `thresholds_used` y métricas usadas están serializadas.
10. **Consistencia de artefactos:** `manifest.json` y `stage_summary.json` referencian outputs con hashes SHA256.

---

## 5) Ejemplos de `stage_summary.json`

## 5.1 Ejemplo mínimo (válido, no evaluado)

```json
{
  "stage": "s3b_multimode_estimates",
  "status": "PASS",
  "multimode_viability": {
    "schema_version": "multimode_viability_v1",
    "class": "SINGLEMODE_ONLY",
    "reasons": ["valid_fraction_221 below threshold"],
    "metrics": {},
    "thresholds_used": {}
  },
  "systematics_gate": {
    "schema_version": "systematics_gate_v1",
    "verdict_auto": "NOT_AVAILABLE",
    "checks": {}
  },
  "science_evidence": {
    "schema_version": "science_evidence_v1",
    "status": "NOT_EVALUATED",
    "reason_if_skipped": ["MULTIMODE_GATE", "SYSTEMATICS_NOT_AVAILABLE"],
    "H1_min": {"delta_Rf": null},
    "future_slots": {
      "delta_f_221": null,
      "delta_tau_221": null,
      "log_bayes_factor": null
    }
  }
}
```

## 5.2 Ejemplo completo (evaluado)

```json
{
  "stage": "s3b_multimode_estimates",
  "status": "PASS",
  "multimode_viability": {
    "schema_version": "multimode_viability_v1",
    "class": "MULTIMODE_OK",
    "reasons": ["overtone passes minimum viability checks"],
    "metrics": {
      "valid_fraction_220": 0.86,
      "valid_fraction_221": 0.61,
      "kerr_consistent": true,
      "delta_bic": 3.7
    },
    "thresholds_used": {
      "MIN_VALID_FRAC_220": 0.5,
      "MIN_VALID_FRAC_221": 0.3
    }
  },
  "systematics_gate": {
    "schema_version": "systematics_gate_v1",
    "verdict_auto": "PASS",
    "checks": {
      "t0_plateau": {"verdict": "PASS", "metric": 4.1, "threshold": 8.0},
      "psd_sanity": {"verdict": "PASS", "metric": 0.19, "threshold": 0.25},
      "estimator_resolution": {"verdict": "PASS", "metric": 2.4, "threshold": 1.2}
    },
    "thresholds_used": {
      "T_T0_STD_MAX": 8.0,
      "T_CHI_PSD_MAX": 0.25,
      "T_Q221_MIN": 1.2
    }
  },
  "science_evidence": {
    "schema_version": "science_evidence_v1",
    "status": "EVALUATED",
    "reason_if_skipped": [],
    "H1_min": {
      "delta_Rf": {
        "value": 0.006,
        "interval": [-0.011, 0.024],
        "quantiles": [0.05, 0.95],
        "chi_star": 0.71,
        "Rf_kerr_at_chi_star": 0.941,
        "contains_zero": true,
        "definition": "Rf_obs_median - Rf_Kerr(chi_star)"
      }
    },
    "future_slots": {
      "delta_f_221": null,
      "delta_tau_221": null,
      "log_bayes_factor": null
    }
  },
  "annotations": {
    "systematics_override_ref": null,
    "systematics_override_sha256": null,
    "kerr_inconsistency_is_not_fail": true
  }
}
```

---

## 6) Reglas de interpretación para informes

- Reportar primero el estado contractual (`EVALUATED`/`NOT_EVALUATED`).
- Diferenciar siempre "falta de evaluación" de "resultado evaluado".
- Tratar `delta_Rf` como métrica formal de desvío dentro del contrato, no como claim físico concluyente.
- Cualquier lectura científica final debe hacerse fuera de este gate técnico, con revisión humana y metodología adicional.
