# IO_RINGDOWN_SYNTH — Specification v1

**Status:** NORMATIVE
**Date:** 2026-01-30
**Governance:** Referenced by `BASURIN_README_SUPER.md` section 3.1b

---

## 1. Scope

This document specifies the **IO contract** for the `ringdown_synth` canonical stage.

It is normative for:
- Producers (`ringdown_synth` stage implementations)
- Consumers (experiments under `runs/<run_id>/experiment/ringdown/*`)

This document **must not contradict** the master README (`BASURIN_README_SUPER.md`).

---

## 2. Stage Output Layout

```
runs/<run_id>/ringdown_synth/
├── manifest.json               # required (BASURIN standard)
├── stage_summary.json          # required (BASURIN standard)
└── outputs/
    ├── synthetic_events.json   # required (canonical case index)
    └── cases/
        └── <case_id>/
            ├── strain_<DET>.npz    # required per detector
            ├── psd_<DET>.npz       # optional
            └── meta.json           # optional
```

---

## 3. Canonical Case Index (`synthetic_events.json`)

### 3.1 Schema

```json
{
  "schema_version": "ringdown_synth_index_v1",
  "created": "<ISO8601 timestamp>",
  "run_id": "<run_id>",
  "stage": "ringdown_synth",
  "cases": [
    {
      "case_id": "<case_id>",
      "detectors": ["H1", "L1"],
      "paths": {
        "strain_H1": "cases/<case_id>/strain_H1.npz",
        "strain_L1": "cases/<case_id>/strain_L1.npz",
        "psd_H1": "cases/<case_id>/psd_H1.npz",
        "psd_L1": "cases/<case_id>/psd_L1.npz",
        "meta": "cases/<case_id>/meta.json"
      },
      "truth": {
        "f_220": 250.0,
        "tau_220": 0.004,
        "Q_220": 3.1415926
      },
      "injection_params": {
        "snr_target": 12.0,
        "seed": 42
      }
    }
  ],
  "notes": {}
}
```

### 3.2 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Must be `ringdown_synth_index_v1` |
| `created` | string | ISO8601 timestamp |
| `run_id` | string | Parent run identifier |
| `stage` | string | Must be `ringdown_synth` |
| `cases` | array | List of synthetic event cases |

### 3.3 Case Object Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_id` | string | yes | Unique case identifier (e.g., `case_000`) |
| `detectors` | array | yes | List of detector codes (e.g., `["H1", "L1"]`) |
| `paths` | object | yes | Relative paths to data files |
| `truth` | object | yes | Ground truth ringdown parameters |
| `injection_params` | object | no | Injection configuration |

### 3.4 Path Resolution

All paths in `synthetic_events.json` are **relative** to:
```
runs/<run_id>/ringdown_synth/outputs/
```

Consumers **must not**:
- Reconstruct paths independently
- Assume path patterns
- Navigate outside the declared paths

---

## 4. Data File Formats

### 4.1 Strain Files (`strain_<DET>.npz`)

NPZ archive containing:

| Key | Type | Description |
|-----|------|-------------|
| `strain` | float64 array | Time-domain strain data |
| `sample_rate` | float64 | Sampling rate in Hz |
| `t0` | float64 | Reference time (GPS or arbitrary) |

### 4.2 PSD Files (`psd_<DET>.npz`) — Optional

NPZ archive containing:

| Key | Type | Description |
|-----|------|-------------|
| `freqs` | float64 array | Frequency bins in Hz |
| `psd` | float64 array | Power spectral density values |
| `method` | string | PSD estimation method (e.g., `welch`, `median`) |

### 4.3 Meta Files (`meta.json`) — Optional

Free-form JSON for non-numeric metadata.

---

## 5. Truth Object Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `f_220` | float | Hz | Fundamental (2,2,0) mode frequency |
| `tau_220` | float | s | Damping time |
| `Q_220` | float | — | Quality factor (derived: π × f × τ) |

Additional mode parameters (higher overtones, different (l,m)) may be included with corresponding naming convention: `f_lmn`, `tau_lmn`, `Q_lmn`.

---

## 6. Contract Validation

### 6.1 Pre-conditions (for consumers)

Before consuming `ringdown_synth` outputs, experiments **must** verify:

1. `RUN_VALID == PASS` (standard BASURIN gate)
2. `synthetic_events.json` exists
3. `synthetic_events.json` parses as valid JSON
4. `schema_version == "ringdown_synth_index_v1"`
5. All declared paths in `paths` exist
6. At least one case exists in `cases`

### 6.2 Failure semantics

If any pre-condition fails:
- Consumer **must abort** with exit code `2`
- Consumer **must not** produce outputs
- Consumer **must not** attempt to infer/estimate missing data

---

## 7. Extension Protocol

To add new synthetic conditions (multi-detector, PSD families, glitches, labels):

1. Propose extension to this spec as `IO_RINGDOWN_SYNTH_v2.md`
2. Ensure backward compatibility or explicit version gate
3. Update `schema_version` in producers and consumers
4. Reference new spec version in master README

---

## 8. Changelog

| Version | Date | Description |
|---------|------|-------------|
| v1 | 2026-01-30 | Initial specification |
