# BASURIN — Executive, Contract-Governed Research Pipeline

**Audience:** automated agents (IA), reproducible runners, and auditors.  
**Goal:** enable experiment design and execution *without interactive clarification*.

This document is **authoritative**. Any execution or experiment that contradicts it is **invalid by definition**.

---

## 1. Core Principle (Non-Negotiable)

> **If an executive contract fails, the run does not exist.**

There is no notion of “partial validity” or “interesting results” from an invalid run.

All artifacts, stages, and experiments are governed by this rule.

---

## 2. Deterministic IO Protocol

All executions live under a single root:

```
runs/<run_id>/
```

No stage, experiment, or helper is allowed to write outside this directory.

### 2.1 Mandatory Structure per Stage

Every stage **must** produce:

```
runs/<run_id>/<stage_name>/
├── manifest.json
├── stage_summary.json
└── outputs/
    └── <artifacts>
```

- `manifest.json`  
  Explicit index of produced artifacts (relative paths, hashes).
- `stage_summary.json`  
  Parameters, versions, inputs (path + hash), and verdict.
- `outputs/`  
  Binary or structured artifacts (HDF5, JSON, etc.).

If any of these are missing → **RUN_INVALID**.

---

## 3. Canonical Stages (v1)

### 3.1 Geometry Stage

```
runs/<run_id>/geometry/
├── manifest.json
├── stage_summary.json
└── outputs/
    └── geometry.h5
```

Produces the **only allowed synthetic data source** in the pipeline.

### 3.2 Spectrum Stage

```
runs/<run_id>/spectrum/
├── manifest.json
├── stage_summary.json
└── outputs/
    └── spectrum.h5
```

Consumes geometry, produces spectral data (no geometry mutation).

### 3.3 Dictionary Stage (Optional / Downstream)

Consumes spectrum; never governs run existence.

### 3.4 Derived Canonical Stage: RUN_INDEX (Run Inventory / Audit Index)

Purpose: deterministic inventory of stages and experiments for a run, gated by `RUN_VALID == PASS`.

IO layout:

```
runs/<run_id>/RUN_INDEX/
├── manifest.json
├── stage_summary.json
└── outputs/
    └── index.json
```

Reference artifacts:
- Schema: `schemas/run_index.schema.json`
- Builder: `stages/stage_run_index.py`
- Tests: `tests/test_run_index.py`

---

## 4. Canonical Artifact Semantic Contracts

### 4.1 Geometry Artifact
`runs/<run_id>/geometry/outputs/geometry.h5`

#### Guaranteed Capabilities

- Existence of a **monotonic radial coordinate** (UV → IR).
- Sufficient information to reconstruct a **domain-wall–equivalent metric** in \( d+1 \) dimensions.
- Deterministic content.

#### Not Guaranteed

- Dataset names.
- Explicit warp factor storage.
- Explicit \( l_{AdS} \) or \( G_N \).

Experiments may require **reconstructibility**, never a specific key.

---

### 4.2 Spectrum Artifact
`runs/<run_id>/spectrum/outputs/spectrum.h5`

#### Guaranteed Capabilities

- Finite discrete spectrum associated with upstream geometry.
- Ability to reconstruct UV conformal dimensions \( \Delta \).
- Full traceability (path + hash).

#### Mandatory Minimal Content

- `delta_uv` **or**
- autovalues (`masses`, `M2`, etc.) + solver parameters.

No guarantees on ordering or cardinality.

---

### 4.3 Dimension \( d \)

- Input parameter of geometry.
- Must appear in `geometry/stage_summary.json` **or** be deterministically inferable.
- If unavailable → experiment aborts.

---

## 5. Derived Physical Invariants (Non-Canonical)

Derived invariants are **legal outputs of experiments**, but:

- Are not canonical artifacts.
- Do not propagate downstream unless formalized as a new stage.

Examples:
- Effective \( l_{AdS} \)
- Holographic \( f \)-function
- Integrated density of states
- Cardy-like exponents

All must be fully documented and traceable.

---

## 6. Experiments — Canonical Folder Format

All experiments live under:

```
runs/<run_id>/experiment/<experiment_name>/
```

### 6.1 Mandatory Structure

```
experiment/<experiment_name>/
├── manifest.json
├── stage_summary.json
└── outputs/
    ├── <derived_results>.json
    └── <diagnostics>/
```

Experiments:
- **must not** generate new synthetic data,
- **must not** mutate canonical artifacts,
- **must abort** if `RUN_VALID != PASS`.

---

## 7. Executive Pre-Checks (Required Order)

Every experiment must implement, in this order:

1. Verify `RUN_VALID == PASS`.
2. Verify existence of required canonical artifacts.
3. Verify declared semantic capabilities.
4. Abort on any failure.
5. Only then perform numerical analysis.

Skipping a check invalidates the experiment.

---

## Uso del atlas/diccionario (v1): selección manual dim4/dim6

El índice soberano de atlas master es:

- `runs/2026-01-29__ATLAS_MASTER__INDEX__v1/atlas_index/outputs/ATLAS_INDEX.json`

Este índice **no asigna runs → dim**; solo cataloga los masters disponibles (`dim4`, `dim6`) con `path`+`sha256`.
La selección de dimensión es explícita y auditada vía `atlas_select`.

### Flujo canónico (sin automatismo)

1) **Gate soberano**: el run consumidor debe tener `RUN_VALID=PASS`:

- `runs/<run_id>/RUN_VALID/outputs/run_valid.json` (legacy) o `runs/<run_id>/RUN_VALID/verdict.json` (preferred)

2) **Seleccionar atlas master** (manual):

```bash
python 07_atlas_select_stage.py \
  --run <run_id> \
  --atlas-index runs/2026-01-29__ATLAS_MASTER__INDEX__v1/atlas_index/outputs/ATLAS_INDEX.json \
  --force-dim 4



## Appendix A — Canonical Example 1  
### Derived Holographic f-Consistency (B4 Pattern)

**Purpose:**  
Test consistency between geometry-derived and spectrum-derived \( f \).

**Declared Capabilities:**

- Geometry: warp reconstructible, UV region detectable.
- Spectrum: \( \Delta \) available or reconstructible.

**Outputs:**

```
outputs/
├── f_geometry.json
├── f_spectrum.json
└── comparison.json
```

Includes uncertainties, methods, tolerances, verdict.

---

## Appendix B — Canonical Example 2  
### Capability Audit / Schema Probe (Safety Gate)

**Purpose:**  
Decide *if* an experiment may run, without doing science.

**Outputs:**

```
outputs/
└── capability_audit.json
```

Lists each capability with PASS / FAIL / UNKNOWN and evidence.

This stage is authoritative and non-scientific.

---

## Appendix C — Failure Semantics

- Any FAIL in executive checks → abort.
- No downstream stages may run.
- Partial outputs are ignored.
- Logs exist only for forensic purposes.

---

## Final Note (IA-Oriented)

This document is designed so that an automated agent can:

- design an experiment,
- verify legality,
- execute deterministically,
- and fail correctly,

**without asking a human a single question**.
