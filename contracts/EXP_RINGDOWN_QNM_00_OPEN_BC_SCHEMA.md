# EXP_RINGDOWN_QNM_00 — Open Boundary Condition Schema

**Version:** v1.0
**Stage:** `experiment/ringdown/EXP_RINGDOWN_QNM_00_open_bc`
**Type:** Experimental gate (does not contaminate canonical Bloque B)

## Motivation

This experiment introduces "horizon-like" absorbing boundary conditions to produce
complex eigenfrequencies (QNMs: Quasi-Normal Modes). The analogy:

| Physical System | Boundary Condition | Operator Type | Spectrum |
|-----------------|-------------------|---------------|----------|
| String with nail | Dirichlet/Neumann | Hermitian | Real eigenvalues |
| Horizon (absorber) | Outgoing/Absorbing | Non-Hermitian | Complex eigenvalues |

The key insight: changing from closed (nail) to open (horizon) boundary conditions
produces exponentially decaying modes with `omega_I < 0`.

---

## Inputs

### Required

| Input | Path | Description |
|-------|------|-------------|
| RUN_VALID | `runs/<run_id>/RUN_VALID/verdict.json` | Sovereign gate (optional for experiments) |

### Optional

| Input | Path | Description |
|-------|------|-------------|
| spectrum.h5 | `runs/<upstream>/spectrum/outputs/spectrum.h5` | Upstream Bloque B spectrum (for traceability) |
| config.json | User-provided | Open BC parameters (grid, window, tolerances) |
| strain/PSD | `runs/<run_id>/ringdown_synth/outputs/...` | Alternative: synthetic ringdown data |

---

## Outputs

### Directory Structure

```
runs/<run_id>/experiment/ringdown/EXP_RINGDOWN_QNM_00_open_bc/
├── manifest.json
├── stage_summary.json
└── outputs/
    ├── qnm_fit.json
    ├── contract_verdict.json
    └── per_case/
        ├── case_000.json
        ├── case_001.json
        └── ...
```

### qnm_fit.json Schema

```json
{
  "schema_version": "qnm_fit_v1",
  "created": "<ISO8601>",
  "model": "open_bc_resonance",
  "omega_complex": [omega_R, omega_I],
  "omega_R": <float>,
  "omega_I": <float>,
  "omega_R_std": <float>,
  "omega_I_std": <float>,
  "f_hz": <float>,
  "tau_s": <float | null>,
  "n_cases": <int>,
  "grid_sizes": [1024, 2048, 4096],
  "window_ids": ["w1", "w2", "w3"],
  "truth": {
    "f_hz": <float>,
    "tau_s": <float>,
    "omega_R_true": <float>,
    "omega_I_true": <float>
  },
  "stability_metrics": {
    "omega_R": {"mean": <float>, "std": <float>, "rel_var": <float>, "p50": <float>, "p90_deviation": <float>},
    "omega_I": {"mean": <float>, "std": <float>, "rel_var": <float>, "p50": <float>, "p90_deviation": <float>}
  }
}
```

### contract_verdict.json Schema

```json
{
  "schema_version": "contract_verdict_v1",
  "created": "<ISO8601>",
  "verdict": "PASS" | "FAIL",
  "contracts": {
    "C1_horizon_decay": {
      "contract": "C1_horizon_decay",
      "verdict": "PASS" | "FAIL",
      "thresholds": {
        "decay_eps": 1e-8,
        "fit_r2_min": 0.90
      },
      "violations": [...]
    },
    "C2_stability": {
      "contract": "C2_stability",
      "verdict": "PASS" | "FAIL",
      "thresholds": {
        "omega_R_rel_tol": 0.05,
        "omega_I_rel_tol": 0.10
      },
      "statistics": {...},
      "violations": [...]
    }
  },
  "assumptions": [...],
  "inputs": {
    "spectrum": {"path": "...", "sha256": "..."},
    "config": {"path": "...", "sha256": "..."},
    "RUN_VALID": {"path": "...", "sha256": "..."}
  }
}
```

---

## Contracts

### Contract C1 — Horizon-like Decay

**Purpose:** Verify that the system produces decaying modes (absorption at horizon).

**Condition:**
```
PASS if and only if:
  - omega_I < -eps  (eps default: 1e-8)
  - fit_r2 >= fit_r2_min  (default: 0.90)

for ALL cases in the grid/window sweep.
```

**Physical interpretation:**
- `omega_I < 0` means exponential decay `exp(omega_I * t)`
- The "horizon" absorbs energy, unlike a "nail" (closed boundary) which reflects it
- Complex eigenvalues arise from non-Hermitian operators (absorbing BC)

### Contract C2 — Resonance Stability

**Purpose:** Verify that QNM frequencies are stable under numerical perturbations.

**Condition:**
```
PASS if and only if:
  - rel_var(omega_R) < omega_R_rel_tol  (default: 5%)
  - rel_var(omega_I) < omega_I_rel_tol  (default: 10%)

where rel_var = std / |mean|
```

**Sweep parameters:**
- Grid sizes: `N = 1024, 2048, 4096` (discretization convergence)
- Window functions: `w1` (none), `w2` (Hann), `w3` (Tukey 0.5)

**Physical interpretation:**
- Stable resonances should not depend strongly on numerical details
- Large variations suggest numerical artifacts, not physical QNMs

---

## CLI Usage

```bash
# Basic execution
python experiment/ringdown/exp_ringdown_qnm_00_open_bc.py \
  --run "2026-01-30__EXP_QNM_00__test" \
  --f-hz 250.0 \
  --tau-s 0.004

# With upstream spectrum (for traceability)
python experiment/ringdown/exp_ringdown_qnm_00_open_bc.py \
  --run "2026-01-30__EXP_QNM_00__full" \
  --in-spectrum "runs/upstream_run/spectrum/outputs/spectrum.h5" \
  --grid-sweep "N=512,1024,2048,4096" \
  --window-sweep "w1,w2,w3"

# Custom tolerances
python experiment/ringdown/exp_ringdown_qnm_00_open_bc.py \
  --run "2026-01-30__EXP_QNM_00__strict" \
  --decay-eps 1e-6 \
  --omega-r-rel-tol 0.02 \
  --omega-i-rel-tol 0.05 \
  --fit-r2-min 0.95
```

---

## BASURIN Compliance

### What this experiment does correctly:

1. **Experimental gate** — Does not modify canonical Bloque B spectrum solver
2. **Contract-first** — Defines pass/fail criteria before implementation
3. **Deterministic** — Same inputs produce same outputs (seeded RNG)
4. **Traceable** — Manifest with SHA256 hashes for all artifacts
5. **Sandboxed** — All outputs under `runs/<run_id>/experiment/...`

### Risks and mitigations:

| Risk | Mitigation |
|------|------------|
| Mixing Hermitian/non-Hermitian regimes | Keep experiment isolated from Bloque B |
| Numerical instability | Grid/window sweep validates convergence |
| Surrogate vs real QNMs | Explicit assumption in contract_verdict |

---

## Future Evolution

If this experiment succeeds:

1. Define canonical artifact `resonance_spectrum.h5` (complex eigenvalues)
2. Promote to `stages/stage_qnm_spectrum.py`
3. Add to main pipeline with proper gate chaining

Until then, this remains experimental and does not block downstream stages.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-01-30 | Initial experimental schema |
