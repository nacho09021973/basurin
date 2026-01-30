# EXP_RINGDOWN_00 Schema — Stability Sweep

**Status:** CLOSED (contract-first)
**Date:** 2026-01-30
**Version:** 1.0.0

---

## Purpose

EXP_RINGDOWN_00 is a **stability gate** that validates ringdown parameter extraction
is robust against variations in preprocessing (window, bandpass, whitening).

---

## A) Dependencies (Inputs)

### A1: Event Source — Canonical Upstream

```
runs/<run_id>/ringdown_synth/outputs/synthetic_event.json
```

**Decision:** A1.a — EXP_00 consumes a canonical synthetic event from upstream.
It does NOT generate events internally.

**Rationale:** Separation of concerns. The generator (`ringdown_synth_stage.py`)
is a distinct stage with its own manifest/summary.

### A2: Required Upstream Artifacts

| Artifact | Path | Required |
|----------|------|----------|
| RUN_VALID | `runs/<run_id>/RUN_VALID/outputs/run_valid.json` | YES |
| Synthetic Event | `runs/<run_id>/ringdown_synth/outputs/synthetic_event.json` | YES |

If either is missing or RUN_VALID != PASS, EXP_00 aborts with exit code 2.

---

## B) Metrics — Contractual Set M

### B1: Primary Metrics (stability gate)

| Metric | Symbol | Unit | Statistic |
|--------|--------|------|-----------|
| Frequency | `f_220` | Hz | median |
| Decay time | `tau_220` | ms | median |

### B2: Secondary Metric (consistency check)

| Metric | Symbol | Derivation | Purpose |
|--------|--------|------------|---------|
| Quality factor | `Q_220` | Q = π·f·τ | Internal consistency |

**Decision:** Q_220 is a **secondary consistency check**, not a primary gate metric.
If `|Q_computed - Q_derived| / Q_derived > 0.02`, emit WARN but do NOT fail gate.

**Rationale:** Avoids double-counting violations (f and tau already capture the physics).

---

## C) PASS/FAIL Criteria

### C1: Baseline Reference

- **Baseline case:** `case_000` (canonical preprocessing parameters)
- All deviations measured relative to baseline

### C2: Tolerances

| Metric | `rel_tol` | Interpretation |
|--------|-----------|----------------|
| `f_220` | 0.02 (2%) | `|f - f_baseline| / f_baseline < 0.02` |
| `tau_220` | 0.05 (5%) | `|τ - τ_baseline| / τ_baseline < 0.05` |

### C3: SNR Policy

```python
snr_min = 8.0
policy = "SKIP_LOW_SNR"
```

- Cases with `snr_effective < snr_min` are **skipped** (not counted as violations)
- Skipped cases appear in `diagnostics.json` with `status: "SKIP_LOW_SNR"`
- Skipped cases do NOT affect PASS/FAIL verdict

### C4: Aggregate Rule

```python
pass_rule = "ALL"  # 100% of valid (non-skipped) cases must pass
```

- **PASS:** All valid cases satisfy tolerances
- **FAIL:** Any valid case violates any tolerance

---

## D) Sweep Grid (OFAT)

### D1: Dimensions

| Dimension | Parameter | Baseline | Variations |
|-----------|-----------|----------|------------|
| Window start | `t_ref_shift` | 0.0 s | -0.05, +0.05 s |
| Window duration | `duration` | 0.5 s | 0.3, 0.8 s |
| Bandpass low | `f_low` | 20 Hz | 30 Hz |
| Bandpass high | `f_high` | 500 Hz | 400 Hz |
| Whitening | `whitening_method` | `median_psd` | `welch_psd` |

**Note on t_ref_shift:** Interpreted as offset relative to internal reference marker.
The extraction window is `[t_ref + t_ref_shift, t_ref + t_ref_shift + duration]`.
Negative shifts are valid as long as data exists. If padding would be required,
the case emits `status: "SKIP_NO_DATA"` instead of silently padding.

### D2: Case Grid (8 cases, OFAT)

| Case ID | t_ref_shift | duration | f_low | f_high | whitening |
|---------|-------------|----------|-------|--------|-----------|
| `case_000` | 0.0 | 0.5 | 20 | 500 | median_psd |
| `case_001` | -0.05 | 0.5 | 20 | 500 | median_psd |
| `case_002` | +0.05 | 0.5 | 20 | 500 | median_psd |
| `case_003` | 0.0 | 0.3 | 20 | 500 | median_psd |
| `case_004` | 0.0 | 0.8 | 20 | 500 | median_psd |
| `case_005` | 0.0 | 0.5 | 30 | 500 | median_psd |
| `case_006` | 0.0 | 0.5 | 20 | 400 | median_psd |
| `case_007` | 0.0 | 0.5 | 20 | 500 | welch_psd |

### D3: Canonical Order

Cases are always processed in lexicographic order by `case_id`.
This ensures deterministic comparison across runs.

---

## E) Experiment Numbering

### E1: Clean Renumbering (00-07)

| ID | Name | Purpose |
|----|------|---------|
| `EXP_RINGDOWN_00` | stability_sweep | Preprocessing robustness gate |
| `EXP_RINGDOWN_01` | injection_recovery | Synthetic → estimated validation |
| `EXP_RINGDOWN_02` | model_selection | Overtone selection (logZ) |
| `EXP_RINGDOWN_03` | multidetector_weighting | L1/H1 weighting strategies |
| `EXP_RINGDOWN_04` | priors_sensitivity | Flat vs informed priors |
| `EXP_RINGDOWN_05` | glitches_nonstationary | Non-Gaussian noise robustness |
| `EXP_RINGDOWN_06` | cross_psd | PSD estimation sensitivity |
| `EXP_RINGDOWN_07` | scaling_benchmark | Performance vs n_samples |

---

## F) Determinism

### F1: Seed Strategy

```python
SEED_GLOBAL = 42
seed_case = SEED_GLOBAL + case_idx
```

### F2: Manifest Hashing

The following are included in manifest hash computation:

- `sweep_plan.json` (full plan)
- `diagnostics.json` (aggregate results)
- `contract_verdict.json` (final verdict)
- `per_case/*.json` (each case individually)

---

## G) Output Artifacts

### G1: Directory Structure

```
runs/<run_id>/experiment/exp_ringdown_00_stability/
├── manifest.json
├── stage_summary.json
└── outputs/
    ├── sweep_plan.json
    ├── diagnostics.json
    ├── contract_verdict.json
    └── per_case/
        ├── case_000.json
        ├── case_001.json
        └── ...
```

### G2: sweep_plan.json Schema

```json
{
  "schema_version": "1.0.0",
  "experiment": "exp_ringdown_00_stability",
  "n_cases": 8,
  "baseline_case": "case_000",
  "seed_global": 42,
  "cases": [
    {
      "case_id": "case_000",
      "seed": 42,
      "params": {
        "t_ref_shift": 0.0,
        "duration": 0.5,
        "f_low": 20,
        "f_high": 500,
        "whitening_method": "median_psd"
      }
    }
  ]
}
```

### G3: per_case/case_XXX.json Schema

```json
{
  "case_id": "case_000",
  "status": "OK",
  "snr_effective": 25.3,
  "metrics": {
    "f_220": {"median": 251.2, "ci68": [248.1, 254.3]},
    "tau_220": {"median": 4.12, "ci68": [3.98, 4.26]},
    "Q_220": {"computed": 3.25, "derived": 3.24, "consistency": "OK"}
  },
  "deviations_from_baseline": {
    "f_220_rel": 0.008,
    "tau_220_rel": 0.023
  },
  "violations": []
}
```

### G4: contract_verdict.json Schema

```json
{
  "schema_version": "1.0.0",
  "experiment": "exp_ringdown_00_stability",
  "verdict": "PASS",
  "summary": {
    "total_cases": 8,
    "valid_cases": 8,
    "skipped_cases": 0,
    "violations": 0
  },
  "assumptions": [
    "snr_min=8.0 with SKIP_LOW_SNR policy",
    "t_ref_shift interpreted as relative offset (no silent padding)"
  ],
  "violations_detail": []
}
```

---

## H) Tests (Permanent Gate)

| Test | Description |
|------|-------------|
| `test_abort_if_run_invalid` | RUN_VALID != PASS → exit 2, no outputs |
| `test_abort_if_synth_missing` | Missing synthetic_event.json → exit 2 |
| `test_sweep_plan_deterministic` | Same seed → identical sweep_plan.json hash |
| `test_verdict_always_present` | contract_verdict.json always written |
| `test_no_write_outside_run` | All writes under runs/<run_id>/ |
| `test_skip_low_snr` | snr < 8 → SKIP_LOW_SNR, not FAIL |
| `test_tolerances_enforced` | rel_tol violations → FAIL verdict |

---

## I) Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-30 | Initial schema closure |
