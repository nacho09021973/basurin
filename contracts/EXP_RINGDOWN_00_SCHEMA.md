# EXP_RINGDOWN_00 Schema — Stability Sweep (Preprocessing Robustness)

**Status:** CLOSED (contract-first)  
**Date:** 2026-01-30  
**Version:** 1.0.0  

---

## Purpose

EXP_RINGDOWN_00 es un **gate de estabilidad** que valida que la extracción de parámetros ringdown es robusta
ante variaciones de preprocesado: ventana temporal, duración, bandpass y whitening/PSD.

Este stage **no genera** datos sintéticos: consume un evento canónico upstream.

---

## A) Dependencies (Hard inputs)

### A1) Run existence gate

Requiere:

- `runs/<run_id>/RUN_VALID/verdict.json` con `verdict == "PASS"`

Si `RUN_VALID != PASS`, el stage **aborta** con exit code `2` y **no produce artefactos**.

### A2) Canonical synthetic event

Requiere:

- `runs/<run_id>/ringdown_synth/outputs/synthetic_event.json`

Si falta, el stage **aborta** con exit code `2`.

---

## B) Contractual metrics (M)

- `f_220` (Hz): estadístico `median` (CI68 opcional)
- `tau_220` (s): estadístico `median` (CI68 opcional)

`Q_220 = pi * f_220 * tau_220` es check secundario (WARN), no gobierna PASS/FAIL.

---

## C) Tolerances

Comparación contra baseline `case_000`:

- `f_220_rel_tol = 0.02`
- `tau_220_rel_tol = 0.05`

---

## D) SNR policy

- `snr_min = 8.0`
- Si `snr_effective < snr_min` ⇒ `SKIP_LOW_SNR` (no cuenta como violación, pero se reporta)

---

## E) No silent padding

Si la ventana cae fuera del soporte ⇒ `SKIP_NO_DATA`. No padding silencioso.

---

## F) Pass rule

PASS si y solo si **100% de los casos válidos** (`status="OK"`) cumplen tolerancias para `f_220` y `tau_220`.
Casos `SKIP_*` no cuentan. Baseline `case_000` debe ser OK.

---

## G) Sweep grid (OFAT; 8 cases)

| case_id  | t0 (s) | duration (s) | f_low | f_high | whitening   |
|----------|--------|--------------|-------|--------|-------------|
| case_000 | 0.0    | 0.5          | 20    | 500    | median_psd  |
| case_001 | -0.1   | 0.5          | 20    | 500    | median_psd  |
| case_002 | +0.1   | 0.5          | 20    | 500    | median_psd  |
| case_003 | 0.0    | 0.3          | 20    | 500    | median_psd  |
| case_004 | 0.0    | 0.8          | 20    | 500    | median_psd  |
| case_005 | 0.0    | 0.5          | 30    | 500    | median_psd  |
| case_006 | 0.0    | 0.5          | 20    | 400    | median_psd  |
| case_007 | 0.0    | 0.5          | 20    | 500    | welch_psd   |

---

## H) Output layout

`runs/<run_id>/experiment/ringdown/EXP_RINGDOWN_00__stability_sweep/`

- `manifest.json`
- `stage_summary.json`
- `outputs/sweep_plan.json`
- `outputs/per_case/case_XXX.json`
- `outputs/diagnostics.json`
- `outputs/contract_verdict.json`

---

## I) Minimal JSON example (synthetic_event.json)

```json
{
  "schema_version": "ringdown_synth_event_v1",
  "seed": 42,
  "snr_target": 12.0,
  "truth": {"f_220": 250.0, "tau_220": 0.004, "Q_220": 3.1415926},
  "notes": {"source": "stages/ringdown_synth_stage.py"}
}

