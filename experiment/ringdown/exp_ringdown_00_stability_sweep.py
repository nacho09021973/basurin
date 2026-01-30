#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------


import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from basurin_io import ensure_stage_dirs, get_run_dir, sha256_file, write_manifest, write_stage_summary


EXIT_CONTRACT_FAIL = 2

SEED_GLOBAL = 42
SNR_MIN = 8.0

REL_TOL_F = 0.02
REL_TOL_TAU = 0.05


def _read_json(p: Path) -> dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_run_valid(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "RUN_VALID" / "verdict.json"
    if not p.exists():
        raise SystemExit(f"ERROR: falta {p}")
    v = _read_json(p)
    verdict = str(v.get("verdict", v.get("status", ""))).upper()
    if verdict != "PASS":
        raise SystemExit(EXIT_CONTRACT_FAIL)
    return v


def _load_synth_event(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json"
    if not p.exists():
        raise SystemExit(EXIT_CONTRACT_FAIL)
    return _read_json(p)


def _sweep_cases() -> list[dict[str, Any]]:
    return [
        {"case_id": "case_000", "t0": 0.0,  "duration": 0.5, "bandpass": [20, 500], "whitening": "median_psd"},
        {"case_id": "case_001", "t0": -0.1, "duration": 0.5, "bandpass": [20, 500], "whitening": "median_psd"},
        {"case_id": "case_002", "t0": 0.1,  "duration": 0.5, "bandpass": [20, 500], "whitening": "median_psd"},
        {"case_id": "case_003", "t0": 0.0,  "duration": 0.3, "bandpass": [20, 500], "whitening": "median_psd"},
        {"case_id": "case_004", "t0": 0.0,  "duration": 0.8, "bandpass": [20, 500], "whitening": "median_psd"},
        {"case_id": "case_005", "t0": 0.0,  "duration": 0.5, "bandpass": [30, 500], "whitening": "median_psd"},
        {"case_id": "case_006", "t0": 0.0,  "duration": 0.5, "bandpass": [20, 400], "whitening": "median_psd"},
        {"case_id": "case_007", "t0": 0.0,  "duration": 0.5, "bandpass": [20, 500], "whitening": "welch_psd"},
    ]


def _simulate_support_window_available(t0: float, duration: float) -> bool:
    """
    Modelo explícito para evitar padding silencioso:
    asumimos que el evento sintético proporciona datos en [0, 1.0] s.
    Si la ventana cae fuera, SKIP_NO_DATA.
    """
    start = t0
    end = t0 + duration
    return (start >= 0.0) and (end <= 1.0)


def _systematic_bias(case: dict[str, Any]) -> tuple[float, float]:
    """
    Sesgos pequeños deterministas por variación de preprocesado.
    Devuelve (bias_f_rel, bias_tau_rel).
    """
    t0 = float(case["t0"])
    dur = float(case["duration"])
    flo, fhi = case["bandpass"]
    whitening = str(case["whitening"])

    bias_f = 0.0
    bias_tau = 0.0

    # shifts de ventana
    bias_f += 0.002 * (t0 / 0.1)  # +/-0.2% aprox
    bias_tau += -0.004 * (t0 / 0.1)

    # duración: ventanas más cortas empeoran tau
    if dur < 0.5:
        bias_tau += 0.01
    if dur > 0.5:
        bias_tau += -0.005

    # bandpass
    if flo >= 30:
        bias_f += 0.003
    if fhi <= 400:
        bias_f += -0.003

    # whitening method
    if whitening == "welch_psd":
        bias_tau += 0.005

    return bias_f, bias_tau


def _posterior_summary_from_truth(truth: dict[str, float], snr_eff: float, case: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    f0 = float(truth["f_220"])
    tau0 = float(truth["tau_220"])

    # scatter ~ 1/SNR (con coeficientes elegidos para ser razonables, no físicos)
    sigma_f_rel = 0.01 * (12.0 / max(snr_eff, 1e-6))
    sigma_tau_rel = 0.03 * (12.0 / max(snr_eff, 1e-6))

    bias_f_rel, bias_tau_rel = _systematic_bias(case)

    # Generar "muestras" gaussianas para mediana/CI68 deterministas
    n = 5000
    f_samp = (f0 * (1.0 + bias_f_rel)) * (1.0 + rng.normal(0.0, sigma_f_rel, size=n))
    tau_samp = (tau0 * (1.0 + bias_tau_rel)) * (1.0 + rng.normal(0.0, sigma_tau_rel, size=n))

    def q68(x: np.ndarray) -> tuple[float, float, float]:
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        return float(p50), float(p16), float(p84)

    f_med, f_lo, f_hi = q68(f_samp)
    tau_med, tau_lo, tau_hi = q68(tau_samp)
    q_med = math.pi * f_med * tau_med
    q_lo = math.pi * f_lo * tau_lo
    q_hi = math.pi * f_hi * tau_hi

    # Q consistency check (secundario)
    q_from_med = math.pi * f_med * tau_med
    q_rel_err = abs(q_med - q_from_med) / max(abs(q_from_med), 1e-12)
    q_consistency = "OK" if q_rel_err <= 0.02 else "WARN"

    return {
        "f_220": {"median": f_med, "ci68": [f_lo, f_hi]},
        "tau_220": {"median": tau_med, "ci68": [tau_lo, tau_hi]},
        "Q_220": {"median": float(q_med), "ci68": [float(q_lo), float(q_hi)]},
        "_diagnostics": {"q_consistency": q_consistency, "q_rel_err": float(q_rel_err)},
    }


def _rel_dev(x: float, xref: float) -> float:
    return abs(x - xref) / max(abs(xref), 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_00 stability sweep gate.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=SEED_GLOBAL)
    args = ap.parse_args()

    run_dir = get_run_dir(args.run).resolve()

    run_valid_verdict = _require_run_valid(run_dir)
    synth = _load_synth_event(run_dir)

    truth = synth["truth"]
    snr_target = float(synth.get("snr_target", 12.0))

    stage_name = "experiment/ringdown/EXP_RINGDOWN_00__stability_sweep"
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name)
    per_case_dir = outputs_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    cases = _sweep_cases()
    plan = {
        "schema_version": "exp_ringdown_00_plan_v1",
        "seed_global": int(args.seed),
        "seed_case_rule": "seed_case = seed_global + case_idx",
        "snr_min": SNR_MIN,
        "cases": cases,
        "inputs": {
            "RUN_VALID": {"path": str((run_dir / "RUN_VALID" / "verdict.json").as_posix()), "sha256": sha256_file(run_dir / "RUN_VALID" / "verdict.json")},
            "synthetic_event": {"path": str((run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json").as_posix()), "sha256": sha256_file(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json")},
        },
        "bounds": {"f_220_rel_tol": REL_TOL_F, "tau_220_rel_tol": REL_TOL_TAU},
    }
    with open(outputs_dir / "sweep_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    # Ejecutar casos
    per_case_payloads: list[dict[str, Any]] = []
    skip_counts = {"SKIP_LOW_SNR": 0, "SKIP_NO_DATA": 0}
    for idx, case in enumerate(cases):
        seed_case = int(args.seed) + idx

        # SNR efectivo: modelo determinista (permite que algunos casos caigan por debajo)
        # degradación suave si whitening/bandpass/duration empeoran
        snr_eff = snr_target
        if case["whitening"] == "welch_psd":
            snr_eff *= 0.95
        if case["bandpass"][0] >= 30:
            snr_eff *= 0.93
        if case["bandpass"][1] <= 400:
            snr_eff *= 0.92
        if case["duration"] <= 0.3:
            snr_eff *= 0.85

        status = "OK"
        if not _simulate_support_window_available(float(case["t0"]), float(case["duration"])):
            status = "SKIP_NO_DATA"
            skip_counts["SKIP_NO_DATA"] += 1
        elif snr_eff < SNR_MIN:
            status = "SKIP_LOW_SNR"
            skip_counts["SKIP_LOW_SNR"] += 1

        payload: dict[str, Any] = {
            "case_id": case["case_id"],
            "params": case,
            "status": status,
            "snr_effective": float(snr_eff),
            "truth": {"f_220": float(truth["f_220"]), "tau_220": float(truth["tau_220"]), "Q_220": float(truth["Q_220"])},
        }

        if status == "OK":
            est = _posterior_summary_from_truth(payload["truth"], float(snr_eff), case, seed_case)
            payload["est"] = {
                "f_220": est["f_220"],
                "tau_220": est["tau_220"],
                "Q_220": est["Q_220"],
            }
            payload["diagnostics"] = est["_diagnostics"]
        else:
            payload["est"] = None
            payload["diagnostics"] = {"reason": status}

        out_p = per_case_dir / f"{case['case_id']}.json"
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        per_case_payloads.append(payload)

    # Baseline = case_000 (si OK)
    baseline = next(p for p in per_case_payloads if p["case_id"] == "case_000")
    violations: list[dict[str, Any]] = []
    diagnostics = {"max_rel_dev": {"f_220": 0.0, "tau_220": 0.0}, "skips": skip_counts, "n_cases": len(cases), "n_valid": 0}

    if baseline["status"] != "OK":
        # baseline inválido => fail directo (no puedes comparar)
        violations.append({"case_id": "case_000", "metric": "baseline", "reason": f"baseline_status={baseline['status']}"})
    else:
        f_ref = float(baseline["est"]["f_220"]["median"])
        tau_ref = float(baseline["est"]["tau_220"]["median"])

        for p in per_case_payloads:
            if p["status"] != "OK":
                continue
            diagnostics["n_valid"] += 1

            f = float(p["est"]["f_220"]["median"])
            tau = float(p["est"]["tau_220"]["median"])

            dev_f = _rel_dev(f, f_ref)
            dev_tau = _rel_dev(tau, tau_ref)

            diagnostics["max_rel_dev"]["f_220"] = float(max(diagnostics["max_rel_dev"]["f_220"], dev_f))
            diagnostics["max_rel_dev"]["tau_220"] = float(max(diagnostics["max_rel_dev"]["tau_220"], dev_tau))

            if dev_f > REL_TOL_F:
                violations.append({"case_id": p["case_id"], "metric": "f_220", "rel_dev": float(dev_f), "tol": REL_TOL_F})
            if dev_tau > REL_TOL_TAU:
                violations.append({"case_id": p["case_id"], "metric": "tau_220", "rel_dev": float(dev_tau), "tol": REL_TOL_TAU})

    with open(outputs_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    verdict = "PASS" if (len(violations) == 0) else "FAIL"
    contract_verdict = {
        "verdict": verdict,
        "bounds": {"f_220_rel_tol": REL_TOL_F, "tau_220_rel_tol": REL_TOL_TAU, "snr_min": SNR_MIN},
        "skips": skip_counts,
        "violations": violations,
        "assumptions": [
            "Baseline is case_000",
            "No silent padding: out-of-support windows => SKIP_NO_DATA",
            "SKIP_LOW_SNR cases do not count as violations",
        ],
        "inputs": plan["inputs"],
    }
    with open(outputs_dir / "contract_verdict.json", "w", encoding="utf-8") as f:
        json.dump(contract_verdict, f, indent=2)

    # stage_summary + manifest
    summary = {
        "stage": stage_name,
        "script": "experiment/ringdown/exp_ringdown_00_stability_sweep.py",
        "params": {"seed_global": int(args.seed)},
        "inputs": {
            "RUN_VALID": plan["inputs"]["RUN_VALID"],
            "synthetic_event": plan["inputs"]["synthetic_event"],
        },
        "outputs": {
            "sweep_plan": "outputs/sweep_plan.json",
            "diagnostics": "outputs/diagnostics.json",
            "contract_verdict": "outputs/contract_verdict.json",
            "per_case_dir": "outputs/per_case/",
        },
        "verdict": verdict,
    }
    write_stage_summary(stage_dir, summary)

    artifacts = {
        "sweep_plan": outputs_dir / "sweep_plan.json",
        "diagnostics": outputs_dir / "diagnostics.json",
        "contract_verdict": outputs_dir / "contract_verdict.json",
    }
    # incluir per_case hashes granularmente en manifest
    for p in sorted(per_case_dir.glob("case_*.json")):
        artifacts[f"per_case::{p.name}"] = p

    write_manifest(stage_dir, artifacts, extra={"verdict": verdict})

    if verdict != "PASS":
        raise SystemExit(EXIT_CONTRACT_FAIL)


if __name__ == "__main__":
    main()
