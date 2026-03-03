# BASURIN — Schemas JSON v1 canónicos (multimode/systematics/science/kerr-ratios)

**Estado:** propuesta contractual v1 (SSOT de estructura).  
**Fuente base:** `docs/informe_multimode_viability_v3.md`.  
**Regla transversal:** todos los numéricos en ejemplos son *placeholders explícitos* (no datos GW reales).

---

## Convenciones generales

- Tipado estilo pseudo-JSON Schema:
  - `string`, `number`, `integer`, `boolean`, `object`, `array<T>`, `null`.
  - `enum[...]` para catálogos cerrados.
- `schema_version` es obligatorio y fija el contrato.
- `additionalProperties: false` recomendado para v1 en objetos canónicos.
- `sha256` en minúsculas hex (`^[a-f0-9]{64}$`).
- `path_ref` es ruta relativa al root del repo/run (no absoluta).

---

## A) `stage_summary.json` bloques canónicos

## A1) `multimode_viability` (`multimode_viability_v1`)

### Typed spec

```text
multimode_viability_v1 := object {
  schema_version: string = "multimode_viability_v1";                    # required
  class: enum["MULTIMODE_OK", "SINGLEMODE_ONLY", "RINGDOWN_NONINFORMATIVE"]; # required
  reasons: array<string>;                                                # required, minItems=0
  metrics: object<string, number|boolean|null>;                          # required
  thresholds_used: object<string, number|boolean>;                       # required
  Rf_obs_interval: [number, number] | null;                              # required
  Rf_kerr_band: [number, number];                                        # required
}
```

### Invariantes

1. `Rf_kerr_band[0] <= Rf_kerr_band[1]`.
2. Si `Rf_obs_interval != null`, entonces `Rf_obs_interval[0] <= Rf_obs_interval[1]`.
3. Si `class == "RINGDOWN_NONINFORMATIVE"`, se permite `Rf_obs_interval = null`.
4. `reasons` debe contener al menos 1 motivo cuando `class != "MULTIMODE_OK"`.

### Mínimo

```json
{
  "schema_version": "multimode_viability_v1",
  "class": "MULTIMODE_OK",
  "reasons": [],
  "metrics": {},
  "thresholds_used": {},
  "Rf_obs_interval": [0.91, 0.95],
  "Rf_kerr_band": [0.88, 1.00]
}
```

### Completo

```json
{
  "schema_version": "multimode_viability_v1",
  "class": "SINGLEMODE_ONLY",
  "reasons": [
    "placeholder: valid_fraction_221 below threshold",
    "placeholder: n_severe_flags reached limit"
  ],
  "metrics": {
    "valid_fraction_220": 0.78,
    "valid_fraction_221": 0.21,
    "rel_iqr_f220": 0.15,
    "rel_iqr_f221": 0.71,
    "kerr_consistent": true,
    "delta_bic": 0.9,
    "informative_score": 0.11,
    "spin_at_floor_frac_221": 0.34,
    "n_severe_flags": 2
  },
  "thresholds_used": {
    "MIN_VALID_FRAC_220": 0.50,
    "MIN_VALID_FRAC_221": 0.30,
    "MAX_REL_IQR_F220": 0.50,
    "MAX_REL_IQR_F221": 0.60,
    "MAX_SPIN_FLOOR_FRAC": 0.30,
    "INFORMATIVE_THRESHOLD": 0.30,
    "DELTA_BIC_SUPPORTIVE": 2.0,
    "SEVERE_COUNT_LIMIT": 2
  },
  "Rf_obs_interval": [0.90, 0.98],
  "Rf_kerr_band": [0.88, 1.00]
}
```

### Compatibility notes

- v1 permite `metrics` extensible (claves nuevas) sin romper consumidores tolerantes.
- En v2 podría formalizarse `metrics` con sub-objetos tipados (`quality`, `kerr_test`, `severity_flags`).
- Mantener `class` estable como enum cerrado evita ambigüedad operativa en gating downstream.

---

## A2) `systematics_gate` (`systematics_gate_v1`)

### Typed spec

```text
systematics_gate_v1 := object {
  schema_version: string = "systematics_gate_v1";                        # required
  verdict_auto: enum["PASS", "FAIL", "NOT_APPLICABLE"];               # required
  verdict_final: enum["PASS", "FAIL", "NOT_EVALUATED"];               # required
  checks: object {                                                        # required
    t0_plateau?: check_v1;
    psd_sanity?: check_v1;
    estimator_resolution?: check_v1;
    [future_check: string]: check_v1;
  };
  thresholds_used: object<string, number|boolean>;                        # required
  inputs: object<string, string|null>;                                    # required
  reasons: array<string>;                                                 # optional
}

check_v1 := object {
  verdict: enum["PASS", "FAIL", "NOT_EVALUATED"];                     # required
  metric: number | null;                                                  # required
  threshold: number | null;                                               # required
  note?: string;                                                          # optional
}
```

### Invariantes

1. Si `verdict_auto == "PASS"`, entonces ningún `check.verdict` puede ser `FAIL`.
2. `verdict_final` puede diferir de `verdict_auto` solo por override degradante (a `FAIL` o `NOT_EVALUATED`).
3. Si `multimode_viability.class != MULTIMODE_OK`, se recomienda `verdict_auto = NOT_APPLICABLE` y `verdict_final = NOT_EVALUATED`.

### Mínimo

```json
{
  "schema_version": "systematics_gate_v1",
  "verdict_auto": "PASS",
  "verdict_final": "PASS",
  "checks": {},
  "thresholds_used": {},
  "inputs": {}
}
```

### Completo

```json
{
  "schema_version": "systematics_gate_v1",
  "verdict_auto": "FAIL",
  "verdict_final": "FAIL",
  "checks": {
    "t0_plateau": {
      "verdict": "PASS",
      "metric": 2.2,
      "threshold": 5.0,
      "note": "placeholder"
    },
    "psd_sanity": {
      "verdict": "FAIL",
      "metric": 0.25,
      "threshold": 0.10,
      "note": "placeholder: chi_psd above max"
    },
    "estimator_resolution": {
      "verdict": "PASS",
      "metric": 1.8,
      "threshold": 1.5
    }
  },
  "thresholds_used": {
    "T_T0_STD_MAX": 5.0,
    "T_CHI_PSD_MAX": 0.10,
    "T_Q221_MIN": 1.5
  },
  "inputs": {
    "t0_sweep_ref": "runs/<run_id>/experiment/oracle_t0_ringdown/outputs/oracle_report.json",
    "psd_ref": "runs/<run_id>/s6c_brunete_psd_curvature/outputs/curvature_diagnostics.json",
    "multimode_ref": "runs/<run_id>/s3b_multimode_estimates/outputs/multimode_summary.json"
  },
  "reasons": [
    "placeholder: psd_sanity check failed"
  ]
}
```

### Compatibility notes

- `checks` debe aceptar extensiones de nombres (ej. `stationarity_sanity`) sin romper v1.
- En v2 podría añadirse `confidence` por check para separar hard-fail de warning.

---

## A3) `science_evidence` (`science_evidence_v1`)

### Typed spec

```text
science_evidence_v1 := object {
  schema_version: string = "science_evidence_v1";                        # required
  status: enum["EVALUATED", "NOT_EVALUATED"];                           # required
  reason_if_skipped: array<string>;                                       # required, minItems=0
  H1_min: object {                                                        # required
    delta_Rf: object {
      value: number | null;                                               # required
      interval: [number, number] | null;                                  # required
      quantiles: [number, number] | null;                                 # required
      chi_star: number | null;                                             # required
      Rf_kerr_at_chi_star: number | null;                                  # required
      contains_zero: boolean | null;                                       # required
      definition: string;                                                  # required
    }
  };
  future_slots: object {                                                  # required
    delta_f_221: number | null;
    delta_tau_221: number | null;
    log_bayes_factor: number | null;
  };
}
```

### Invariantes

1. `status == "EVALUATED"` exige `H1_min.delta_Rf.value != null` e `interval != null`.
2. `status == "NOT_EVALUATED"` permite campos numéricos en `null`.
3. Si `interval != null`, entonces `interval[0] <= interval[1]`.
4. Si `contains_zero == true`, entonces `interval` debe incluir 0.
5. Gating semántico: solo evaluar si `multimode_viability.class == MULTIMODE_OK` **y** `systematics_gate.verdict_final == PASS`.

### Mínimo

```json
{
  "schema_version": "science_evidence_v1",
  "status": "NOT_EVALUATED",
  "reason_if_skipped": ["placeholder: systematics gate not pass"],
  "H1_min": {
    "delta_Rf": {
      "value": null,
      "interval": null,
      "quantiles": null,
      "chi_star": null,
      "Rf_kerr_at_chi_star": null,
      "contains_zero": null,
      "definition": "Rf_obs_median - Rf_Kerr(chi_star)"
    }
  },
  "future_slots": {
    "delta_f_221": null,
    "delta_tau_221": null,
    "log_bayes_factor": null
  }
}
```

### Completo

```json
{
  "schema_version": "science_evidence_v1",
  "status": "EVALUATED",
  "reason_if_skipped": [],
  "H1_min": {
    "delta_Rf": {
      "value": 0.004,
      "interval": [-0.010, 0.019],
      "quantiles": [0.05, 0.95],
      "chi_star": 0.66,
      "Rf_kerr_at_chi_star": 0.936,
      "contains_zero": true,
      "definition": "Rf_obs_median - Rf_Kerr(chi_star)"
    }
  },
  "future_slots": {
    "delta_f_221": null,
    "delta_tau_221": null,
    "log_bayes_factor": null
  }
}
```

### Compatibility notes

- `future_slots` reservado en v1 evita breaking changes al activar sampler más adelante.
- En v2 puede añadirse `method`/`estimator_version` para reproducibilidad de inferencia.

---

## A4) `annotations` (refs + sha256)

### Typed spec

```text
annotations_v1 := object {
  systematics_override_ref: string | null;                                # required
  systematics_override_sha256: string | null;                             # required
  refs?: array<object {path_ref: string; sha256: string}>;                # optional
}
```

### Invariantes

1. Si `systematics_override_ref != null`, entonces `systematics_override_sha256` es obligatorio no-null y válido.
2. Si `systematics_override_ref == null`, entonces `systematics_override_sha256 == null`.

### Mínimo

```json
{
  "systematics_override_ref": null,
  "systematics_override_sha256": null
}
```

### Completo

```json
{
  "systematics_override_ref": "runs/<run_id>/annotations/systematics_override.json",
  "systematics_override_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "refs": [
    {
      "path_ref": "runs/<run_id>/s3b_multimode_estimates/outputs/model_comparison.json",
      "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    {
      "path_ref": "runs/<run_id>/outputs/kerr_ratio_reference.json",
      "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    }
  ]
}
```

### Compatibility notes

- `refs` opcional en v1: consumidores legacy pueden ignorarlo.
- v2 podría unificarlo con un bloque general `provenance_refs` compartido entre stages.

---

## B) `outputs/kerr_ratio_reference.json` (canónico)

### Typed spec

```text
kerr_ratio_reference_v1 := object {
  schema_version: string = "kerr_ratio_reference_v1";                    # required
  generator: object {                                                     # required
    tool: string;                                                         # required
    tool_version: string;                                                 # required
    chi_grid_spec: string;                                                # required
  };
  chi_grid: array<number>;                                                # required, minItems=2
  Rf: array<number>;                                                      # required, same length as chi_grid
  RQ: array<number>;                                                      # required, same length as chi_grid
  Q220: array<number>;                                                    # required, same length as chi_grid
  Q221: array<number>;                                                    # required, same length as chi_grid
  kerr_band: object {                                                     # required
    Rf_min: number;
    Rf_max: number;
    RQ_min: number;
    RQ_max: number;
  };
  sha256_self?: string;                                                   # optional (hash del propio archivo)
}
```

### Invariantes

1. Todas las series (`Rf`, `RQ`, `Q220`, `Q221`) tienen longitud exacta de `chi_grid`.
2. `kerr_band.Rf_min = min(Rf)`, `Rf_max = max(Rf)` (análogo para `RQ`).
3. `chi_grid` monotónica creciente y dentro de `[0.0, 0.998]` para v1.

### Mínimo

```json
{
  "schema_version": "kerr_ratio_reference_v1",
  "generator": {
    "tool": "placeholder_qnm_tool",
    "tool_version": "v0-placeholder",
    "chi_grid_spec": "placeholder grid with 2 points"
  },
  "chi_grid": [0.0, 0.998],
  "Rf": [0.88, 1.00],
  "RQ": [0.25, 0.85],
  "Q220": [2.0, 12.0],
  "Q221": [0.8, 8.0],
  "kerr_band": {
    "Rf_min": 0.88,
    "Rf_max": 1.00,
    "RQ_min": 0.25,
    "RQ_max": 0.85
  }
}
```

### Completo

```json
{
  "schema_version": "kerr_ratio_reference_v1",
  "generator": {
    "tool": "placeholder_qnm_tool",
    "tool_version": "v0-placeholder",
    "chi_grid_spec": "linspace(0.0, 0.998, 5) placeholder"
  },
  "chi_grid": [0.0, 0.2495, 0.499, 0.7485, 0.998],
  "Rf": [0.88, 0.91, 0.94, 0.97, 1.00],
  "RQ": [0.25, 0.38, 0.52, 0.69, 0.85],
  "Q220": [2.0, 3.5, 5.5, 8.0, 12.0],
  "Q221": [0.8, 1.4, 2.5, 4.8, 8.0],
  "kerr_band": {
    "Rf_min": 0.88,
    "Rf_max": 1.00,
    "RQ_min": 0.25,
    "RQ_max": 0.85
  },
  "sha256_self": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}
```

### Compatibility notes

- En v2 podría migrar a representación por filas (`samples:[{chi,Rf,RQ,...}]`) para interoperabilidad tabular.
- Mantener `kerr_band` precalculado evita recomputación en stages consumidores.

---

## C) `outputs/kerr_ratio_observed.json` (canónico)

### Typed spec

```text
kerr_ratio_observed_v1 := object {
  schema_version: string = "kerr_ratio_observed_v1";                     # required
  source_ref: string;                                                     # required (ej. s3b outputs)
  source_sha256: string;                                                  # required
  ratio_name: string = "Rf";                                             # required v1
  quantiles: object {                                                     # required
    q05: number;
    q50: number;
    q95: number;
  };
  interval: [number, number];                                             # required
  bootstrap_samples_count: integer;                                       # required, >=1
  notes?: array<string>;                                                  # optional
}
```

### Invariantes

1. `interval == [quantiles.q05, quantiles.q95]`.
2. Orden de cuantiles: `q05 <= q50 <= q95`.
3. `ratio_name` fijo en v1 a `Rf` para evitar ambigüedad.

### Mínimo

```json
{
  "schema_version": "kerr_ratio_observed_v1",
  "source_ref": "runs/<run_id>/s3b_multimode_estimates/outputs/estimates.json",
  "source_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "ratio_name": "Rf",
  "quantiles": {
    "q05": 0.90,
    "q50": 0.93,
    "q95": 0.96
  },
  "interval": [0.90, 0.96],
  "bootstrap_samples_count": 100
}
```

### Completo

```json
{
  "schema_version": "kerr_ratio_observed_v1",
  "source_ref": "runs/<run_id>/s3b_multimode_estimates/outputs/estimates.json",
  "source_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "ratio_name": "Rf",
  "quantiles": {
    "q05": 0.905,
    "q50": 0.936,
    "q95": 0.952
  },
  "interval": [0.905, 0.952],
  "bootstrap_samples_count": 400,
  "notes": [
    "placeholder: synthetic values for contract example",
    "placeholder: no real GW data"
  ]
}
```

### Compatibility notes

- v2 podría admitir `ratio_name in ["Rf","RQ"]` con bloque por ratio.
- Si se añade distribución completa (no solo cuantiles), mantener cuantiles por backward compatibility.

---

## D) `annotations/systematics_override.json` (`systematics_override_v1`)

> Override **solo degradante**: nunca puede promover evaluación/canal científico.

### Typed spec

```text
systematics_override_v1 := object {
  schema_version: string = "systematics_override_v1";                    # required
  verdict_human: enum["FORCE_FAIL", "FORCE_NOT_EVALUATED"];             # required
  reason: string;                                                         # required
  analyst_id: string;                                                     # required
  timestamp_utc: string;                                                  # required, ISO-8601 UTC
  affects: object {                                                       # required
    systematics_gate_verdict_final: enum["FAIL", "NOT_EVALUATED"];      # required
    science_evidence_status: enum["NOT_EVALUATED"];                      # required
  };
  sha256_self?: string;                                                   # optional
}
```

### Invariantes

1. **No-promoción**: nunca permite `systematics_gate_verdict_final = PASS`.
2. **No-promoción**: nunca permite `science_evidence_status = EVALUATED`.
3. Debe existir trazabilidad humana (`analyst_id`, `timestamp_utc`, `reason`).

### Mínimo

```json
{
  "schema_version": "systematics_override_v1",
  "verdict_human": "FORCE_NOT_EVALUATED",
  "reason": "placeholder: manual veto due to non-stationarity review",
  "analyst_id": "placeholder_analyst",
  "timestamp_utc": "2026-01-01T00:00:00Z",
  "affects": {
    "systematics_gate_verdict_final": "NOT_EVALUATED",
    "science_evidence_status": "NOT_EVALUATED"
  }
}
```

### Completo

```json
{
  "schema_version": "systematics_override_v1",
  "verdict_human": "FORCE_FAIL",
  "reason": "placeholder: identified instrumental artifact in review checklist",
  "analyst_id": "placeholder_analyst_team_A",
  "timestamp_utc": "2026-01-01T00:00:00Z",
  "affects": {
    "systematics_gate_verdict_final": "FAIL",
    "science_evidence_status": "NOT_EVALUATED"
  },
  "sha256_self": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
}
```

### Compatibility notes

- v1 concentra override en efectos de alto nivel (`systematics_gate`, `science_evidence`).
- v2 podría incluir `scope` por check (`psd_sanity`, `t0_plateau`) sin permitir promoción.

---

## Reglas de coherencia cruzada (entre archivos)

1. `stage_summary.annotations.systematics_override_ref` y `..._sha256` deben corresponder exactamente al archivo de D) si existe.
2. `multimode_viability.Rf_kerr_band` debe ser consistente con B) (`kerr_band.Rf_min/max`).
3. `science_evidence.H1_min.delta_Rf` debe usar C) + B) (ratio observado vs referencia Kerr).
4. Si hay override D), el resultado efectivo nunca puede terminar en `science_evidence.status = EVALUATED`.

