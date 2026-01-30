# BASURIN — Executive, Contract-Governed Research Pipeline

**Audience:** automated agents (IA), reproducible runners, and auditors.  
**Goal:** enable experiment design and execution *without interactive clarification*.

This document is **authoritative**. Any execution or experiment that contradicts it is **invalid by definition**.

Subsystem IO specifications (e.g. ringdown) may be defined in **versioned documents under `docs/`**.
When referenced by a canonical stage or contract, those specs are **normative** for producers/consumers,
and must not contradict this master document.

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

Produces the **only allowed synthetic data source** for the geometry→spectrum→dictionary pipeline.

### 3.1b Ringdown Synth Stage (Canonical Synthetic Event Index)

```
runs/<run_id>/ringdown_synth/
├── manifest.json
├── stage_summary.json
└── outputs/
    ├── synthetic_events.json
    └── cases/<case_id>/
        ├── strain_<DET>.npz
        ├── psd_<DET>.npz          # optional (only if the run uses PSD as an input)
        └── meta.json              # optional (non-numeric metadata)
```

`ringdown_synth` is the **only authorized generator** of ringdown synthetic datasets.

It must publish a canonical case index:

- `runs/<run_id>/ringdown_synth/outputs/synthetic_events.json`

**Executive rules:**

- Ringdown experiments under `runs/<run_id>/experiment/ringdown/*` **must not** reconstruct paths; they **must** consume paths from `synthetic_events.json`.
- Any new synthetic condition (multi-detector, PSD families, glitches, labels, etc.) must be formalized as:
  - an extension of `ringdown_synth`, or
  - a new canonical stage (with its own deterministic IO and contracts).
- `synthetic_events.json` + referenced NPZ files are subject to contract validation; if the ringdown data contract fails, downstream ringdown experiments are **invalid**.

Reference spec (normative for IO details):
- `docs/ringdown/IO_RINGDOWN_SYNTH_v1.md`

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

**Ringdown-specific executive constraints:**

- Experiments under `runs/<run_id>/experiment/ringdown/*` consume exclusively canonical artifacts produced by `ringdown_synth`.
- They **must abort** (invalid) if `ringdown_synth/outputs/synthetic_events.json` is missing or fails its data contract.
- They **must not** infer/estimate missing canonical inputs ad-hoc (e.g. PSD) unless an explicit canonical artifact exists.

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

## 8. Atlas / Dictionary Usage (v1) — Manual dim4/dim6 selection

This section documents the *operational* usage of the atlas/dictionary inventory for downstream consumers.
It does **not** introduce automation: the user explicitly selects the dimension (dim4 vs dim6).

### 8.1 Sovereign atlas index (catalog of masters)

The sovereign inventory of atlas masters is:

- `runs/2026-01-29__ATLAS_MASTER__INDEX__v1/atlas_index/outputs/ATLAS_INDEX.json`

This index **does not assign runs → dim**. It only catalogs available masters (`dim4`, `dim6`) with `path` + `sha256`.

### 8.2 Canonical flow (no automation)

#### 1) Sovereign gate: run must exist (`RUN_VALID == PASS`)

A consuming run must have:

- `runs/<run_id>/RUN_VALID/outputs/run_valid.json` (legacy), or
- `runs/<run_id>/RUN_VALID/verdict.json` (preferred)

If missing or not PASS → abort downstream stages.

#### 2) Select atlas master (manual)

From repo root:

```bash
python 07_atlas_select_stage.py   --run <run_id>   --atlas-index runs/2026-01-29__ATLAS_MASTER__INDEX__v1/atlas_index/outputs/ATLAS_INDEX.json   --force-dim 4
```

For dim6:

```bash
python 07_atlas_select_stage.py   --run <run_id>   --atlas-index runs/2026-01-29__ATLAS_MASTER__INDEX__v1/atlas_index/outputs/ATLAS_INDEX.json   --force-dim 6
```

#### 3) Canonical selection artifact

`atlas_select` produces:

- `runs/<run_id>/atlas_select/outputs/ATLAS_SELECTION.json`

It must contain at minimum:

- `index_source.path` + `index_source.sha256`
- `inputs.RUN_VALID.path` + `inputs.RUN_VALID.sha256` + `inputs.RUN_VALID.verdict`
- `selected.dim`
- `selected.atlas_master_path`
- `selected.atlas_master_sha256` (from index)
- `selected.atlas_master_sha256_computed` (computed from file; must match)

#### 4) Resolve the selected atlas master (“dictionary”)

```bash
MASTER=$(jq -r '.selected.atlas_master_path' runs/<run_id>/atlas_select/outputs/ATLAS_SELECTION.json)
jq -r '.schema_version, .run_id, .stage' "$MASTER"
```

The selected `BRIDGE_ATLAS_MASTER.json` is the consumable atlas/dictionary for downstream stages/experiments.

### 8.3 Governance invariants (non-negotiable)

- If `RUN_VALID` is missing or not PASS → `atlas_select` must abort (exit=2) and produce no outputs.
- If the computed `sha256` of the selected master does not match the index → `atlas_select` must abort (exit=2).
- `atlas_select` must write only under `runs/<run_id>/atlas_select/`.

---

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
