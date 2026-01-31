# EXP_RINGDOWN_QNM_01 — Closed ↔ Open Limit Schema

**Version:** v1.0
**Stage:** `experiment/ringdown/EXP_RINGDOWN_QNM_01_closed_open_limit`
**Type:** Internal consistency validation

## Motivation

Before connecting the QNM pipeline to real gravitational wave data, we need
to verify internal consistency: the open-boundary model should recover the
closed-boundary spectrum in the appropriate limit.

| Regime | Boundary | Operator | Spectrum |
|--------|----------|----------|----------|
| Closed (Bloque B) | "Nail" (Dirichlet/Neumann) | Hermitian | Real M² |
| Open (QNM) | "Horizon" (absorbing) | Non-Hermitian | Complex ω |
| **Limit** | absorption → 0 | Hermitian recovered | ω_R² → M², ω_I → 0 |

This experiment validates the transition between regimes.

---

## Inputs

### Required

| Input | Path | Description |
|-------|------|-------------|
| spectrum.h5 | `runs/<run>/spectrum/outputs/spectrum.h5` | Bloque B eigenvalues M² |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gamma-sweep` | `0.0,0.01,0.1,1.0,10.0,100.0,250.0` | Absorption values |
| `--mode-indices` | `0,1,2` | Which M² modes to test |
| `--delta-index` | `0` | Which Δ slice from spectrum |

---

## Outputs

### Directory Structure

```
runs/<run_id>/experiment/ringdown/EXP_RINGDOWN_QNM_01_closed_open_limit/
├── manifest.json
├── stage_summary.json
└── outputs/
    ├── comparison.json
    ├── per_mode_results.json
    └── contract_verdict.json
```

### comparison.json Schema

```json
{
  "schema_version": "closed_open_comparison_v1",
  "created": "<ISO8601>",
  "spectrum_source": "<path>",
  "spectrum_sha256": "<hash>",
  "delta": <float>,
  "delta_index": <int>,
  "mode_indices": [0, 1, 2],
  "gamma_values": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 250.0],
  "n_results": <int>,
  "summary": {
    "closed_limit_valid": true,
    "monotonicity_valid": true
  },
  "bloque_b": {
    "d": 3,
    "L": 1.0,
    "bc_uv": "dirichlet",
    "bc_ir": "dirichlet",
    "n_modes": 10
  }
}
```

### per_mode_results.json Schema

```json
{
  "mode_0": [
    {
      "gamma": 0.0,
      "omega_I_true": 0.0,
      "omega_R_true": 12.566,
      "M2_target": 157.91,
      "omega_R_fit": 12.565,
      "omega_I_fit": -0.0001,
      "omega_R_sq_fit": 157.88,
      "omega_R_sq_error": 0.0002,
      "omega_I_error": 0.0001,
      "mode_index": 0,
      "delta": 1.55
    },
    ...
  ],
  "mode_1": [...],
  "mode_2": [...]
}
```

---

## Contracts

### Contract C3 — Closed Limit Recovery

**Purpose:** Verify that when absorption → 0, the open-BC model recovers
the closed-BC (Hermitian) spectrum.

**Condition:**
```
PASS if and only if, for all cases with gamma < 1.0:
  - |ω_R² - M²| / M² < omega_R_sq_rel_tol  (default: 5%)
  - For gamma < 0.1: |ω_I| < omega_I_zero_tol + gamma  (default: 1e-6)
```

**Physical interpretation:**
- Closing the horizon (reducing absorption) should recover real eigenvalues
- ω_R² → M² means the oscillation frequency matches the Bloque B mass
- ω_I → 0 means no decay (energy is conserved, not absorbed)

### Contract C4 — Monotonicity

**Purpose:** Verify that decay rate increases with absorption.

**Condition:**
```
PASS if and only if:
  |ω_I(gamma_i)| ≤ |ω_I(gamma_{i+1})| + tol

  for all consecutive gamma values (sorted ascending)
```

**Physical interpretation:**
- More absorption → more decay (larger |ω_I|)
- This is the expected physical behavior: a "leakier" boundary loses energy faster
- Violations would indicate numerical artifacts or model inconsistency

---

## CLI Usage

```bash
# Basic: use spectrum from same run
python experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py \
  --run "2026-01-30__closed_open_test"

# With upstream spectrum
python experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py \
  --run "2026-01-30__closed_open_test" \
  --upstream-run "2026-01-30__bloque_b_run"

# Explicit spectrum path
python experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py \
  --run "2026-01-30__closed_open_test" \
  --in-spectrum "runs/my_run/spectrum/outputs/spectrum.h5"

# Custom sweep
python experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py \
  --run "2026-01-30__fine_sweep" \
  --gamma-sweep "0.0,0.001,0.01,0.1,0.5,1.0,5.0,10.0" \
  --mode-indices "0,1,2,3,4"
```

---

## Physical Picture

```
                    Closed (Bloque B)              Open (QNM)
                    ================              ==========
Boundary:           "Nail" (reflects)      →      "Horizon" (absorbs)
Operator:           Hermitian              →      Non-Hermitian
Eigenvalues:        Real M²                →      Complex ω = ω_R + iω_I
Energy:             Conserved              →      Decays

                         ↑
                         |
                    LIMIT: γ → 0
                         |
                         ↓

                    ω_R² → M²
                    ω_I → 0
```

This experiment verifies the arrow can be reversed: starting from QNM,
we can recover the Hermitian limit.

---

## BASURIN Compliance

### What this experiment validates:

1. **Pipeline consistency** — Bloque B and QNM experiments agree in overlap
2. **Physical correctness** — Limit behavior matches theory
3. **Numerical stability** — Smooth transition between regimes

### Dependencies:

- Requires `spectrum.h5` from Bloque B (03_sturm_liouville.py)
- Uses same surrogate signal model as EXP_RINGDOWN_QNM_00

### Future extensions:

If C3 and C4 pass:
- We can trust the QNM model for real GW data
- The "horizon as absorber" interpretation is validated internally
- Ready to proceed to GW150914 ringdown analysis

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-01-30 | Initial schema |
