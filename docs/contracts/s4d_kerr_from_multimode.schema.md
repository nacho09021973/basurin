# s4d_kerr_from_multimode — Output Schemas (Draft)

## outputs/kerr_from_multimode.json
- schema_name: kerr_from_multimode
- schema_version: 1
- json_strict: true (no NaN/Infinity; null permitido)
- purpose: infer (M_f, a_f) from {(f220,tau220),(f221,tau221)} with uncertainties and multimode consistency.

### Required top-level fields
- schema_name (str)
- schema_version (int)
- created_utc (str, ISO-8601, Z)
- run_id (str)
- stage (str="s4d_kerr_from_multimode")
- source:
  - multimode_estimates:
    - relpath (str) = "s3b_multimode_estimates/outputs/multimode_estimates.json"
    - sha256 (str)
  - model_comparison (optional):
    - relpath (str) = "s3b_multimode_estimates/outputs/model_comparison.json"
    - sha256 (str)
- conventions:
  - units: {f_hz, tau_s, mass_solar, spin_dimensionless}
  - mode_labels: ["220","221"]
  - mapping: description of QNM inversion (root-find / interpolation) and parameterization (M_f, a_f)
- estimates:
  - per_mode:
    - "220": {f_hz: {p10,p50,p90}, tau_s: {p10,p50,p90}}
    - "221": {f_hz: {p10,p50,p90}, tau_s: {p10,p50,p90}}
  - kerr:
    - M_f_solar: {p10,p50,p90}
    - a_f: {p10,p50,p90}
    - covariance (optional): 2x2 packed or null
- consistency:
  - metric_name (str) e.g. "delta_kerr"
  - value (float)
  - threshold (float)
  - pass (bool)
- trace:
  - inversion:
    - method (str)
    - grid_or_solver (str)
    - seed (int)
    - tie_break (str)

## outputs/kerr_from_multimode_diagnostics.json
- schema_name: kerr_from_multimode_diagnostics
- schema_version: 1
- json_strict: true
- purpose: audit trail for inversion diagnostics (solver status, rejected samples, numerical conditioning).

### Required top-level fields
- schema_name (str)
- schema_version (int)
- created_utc (str, ISO-8601, Z)
- run_id (str)
- stage (str)
- diagnostics:
  - solver_status (dict)
  - conditioning (dict)
  - rejected_fraction (float)
  - notes (list[str])
