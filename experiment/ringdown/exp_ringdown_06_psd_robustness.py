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
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

from experiment.ringdown.exp_ringdown_02_recovery_robustness import (
    recover_ringdown_from_series,
)


EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_06__psd_robustness"
DEFAULT_MAX_REL_RMSE = 0.25
DEFAULT_PSD_SWEEP = "v1"
DEFAULT_N_MAX_CASES = 24
PSD_NPERSEG = 1024
PSD_OVERLAP = 0.5
PSD_FLOOR = 1.0e-12


def abort_contract(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                abort_contract(f"invalid JSON on line {idx} of {path}: {exc}")
            if not isinstance(row, dict):
                abort_contract(f"line {idx} of {path} is not an object")
            yield row


def _snr_key(x: float) -> str:
    return f"{float(x):.3f}"


def _load_seed_snr_to_strain(run_dir: Path) -> Tuple[Dict[Tuple[int, str], str], Path]:
    idx = run_dir / "ringdown_synth" / "outputs" / "synthetic_events_list.json"
    if not idx.exists():
        abort_contract(f"missing {idx}")

    with open(idx, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        abort_contract("synthetic_events_list.json invalid/empty")

    out: Dict[Tuple[int, str], str] = {}
    collisions: List[Dict[str, object]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        seed = it.get("seed")
        snr = it.get("snr_target", it.get("snr"))
        rel = it.get("strain_npz")
        if seed is None or snr is None or not isinstance(rel, str) or not rel:
            continue
        k = (int(seed), _snr_key(float(snr)))
        if k in out and out[k] != rel:
            collisions.append(
                {
                    "key": {"seed": int(seed), "snr": _snr_key(float(snr))},
                    "a": out[k],
                    "b": rel,
                }
            )
        out[k] = rel

    if collisions:
        abort_contract(f"synthetic_events_list has collisions for (seed,snr): {collisions[:3]}")

    return out, idx


def _resolve_recovery_cases(run_dir: Path) -> Path:
    preferred = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    legacy = (
        run_dir
        / "experiment"
        / "ringdown_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    abort_contract(f"recovery_cases.jsonl missing at {preferred} (legacy {legacy})")


def _resolve_run_valid_path(run_dir: Path) -> Path:
    preferred = run_dir / "RUN_VALID" / "verdict.json"
    legacy = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    return preferred if preferred.exists() else legacy


def _load_strain(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    data = np.load(path)
    keys = set(data.files)

    if "strain" in keys:
        x = np.asarray(data["strain"], dtype=float)
    elif "h" in keys:
        x = np.asarray(data["h"], dtype=float)
    else:
        x = None
        for k in data.files:
            arr = np.asarray(data[k])
            if arr.ndim == 1 and arr.size > 32:
                x = np.asarray(arr, dtype=float)
                break
        if x is None:
            abort_contract(f"strain_npz sin serie 1D utilizable: {path}")

    if "t" in keys:
        t = np.asarray(data["t"], dtype=float)
        if t.shape != x.shape:
            abort_contract(f"t shape != strain shape en {path}")
        dt = float(np.median(np.diff(t)))
    elif "dt" in keys:
        dt = float(np.asarray(data["dt"]).reshape(-1)[0])
        t = dt * np.arange(x.size, dtype=float)
    else:
        dt = 1.0 / 4096.0
        t = dt * np.arange(x.size, dtype=float)

    if not np.isfinite(dt) or dt <= 0:
        abort_contract(f"dt inválido en {path}")

    return x.astype(float), t.astype(float), float(dt)


def _welch_psd(strain: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(strain)
    nperseg = min(int(PSD_NPERSEG), n)
    if nperseg < 8:
        raise ValueError("nperseg too small for PSD")
    step = max(1, int(nperseg * (1.0 - PSD_OVERLAP)))
    window = np.hanning(nperseg)
    window_norm = float(np.sum(window**2))
    if window_norm <= 0:
        raise ValueError("window normalization invalid")
    segments = []
    for start in range(0, n - nperseg + 1, step):
        segment = strain[start : start + nperseg]
        if segment.shape[0] != nperseg:
            continue
        seg = segment * window
        fft = np.fft.rfft(seg)
        power = (np.abs(fft) ** 2) * (dt / window_norm)
        if power.size > 2:
            power[1:-1] *= 2.0
        segments.append(power)
    if not segments:
        raise ValueError("no PSD segments")
    psd = np.mean(np.vstack(segments), axis=0)
    freqs = np.fft.rfftfreq(nperseg, dt)
    return psd.astype(float), freqs.astype(float)


def _whiten_series(strain: np.ndarray, dt: float, psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    n = len(strain)
    fft_full = np.fft.rfft(strain)
    freqs_full = np.fft.rfftfreq(n, dt)
    psd_interp = np.interp(freqs_full, freqs, psd, left=psd[0], right=psd[-1])
    psd_interp = np.maximum(psd_interp, PSD_FLOOR)
    whitened = np.fft.irfft(fft_full / np.sqrt(psd_interp), n=n)
    return whitened.astype(float)


def _psd_variants_v1() -> List[Dict[str, Any]]:
    return [
        {"id": "scale_up_10", "type": "scale", "params": {"a": 0.10}},
        {"id": "scale_down_10", "type": "scale", "params": {"a": -0.10}},
        {"id": "tilt_up_015", "type": "tilt", "params": {"b": 0.15, "f0": 100.0}},
        {"id": "tilt_down_015", "type": "tilt", "params": {"b": -0.15, "f0": 100.0}},
        {
            "id": "bump_fc150",
            "type": "bump",
            "params": {"c": 0.20, "fc": 150.0, "sigma": 0.30},
        },
    ]


def _apply_psd_variant(psd: np.ndarray, freqs: np.ndarray, variant: Dict[str, Any]) -> np.ndarray:
    vtype = variant.get("type")
    params = variant.get("params") or {}
    if vtype == "scale":
        a = float(params.get("a", 0.0))
        out = psd * (1.0 + a)
    elif vtype == "tilt":
        b = float(params.get("b", 0.0))
        f0 = float(params.get("f0", 100.0))
        safe_freqs = np.where(freqs > 0, freqs, f0)
        out = psd * (safe_freqs / f0) ** b
    elif vtype == "bump":
        c = float(params.get("c", 0.0))
        fc = float(params.get("fc", 150.0))
        sigma = float(params.get("sigma", 0.30))
        safe_freqs = np.where(freqs > 0, freqs, fc)
        log_term = np.log(safe_freqs / fc)
        bump = c * np.exp(-(log_term**2) / (2.0 * sigma**2))
        out = psd * (1.0 + bump)
    else:
        raise ValueError(f"unknown PSD variant type: {vtype}")
    return np.maximum(out, PSD_FLOOR)


def _compute_rmse(truth: Dict[str, float], estimate: Dict[str, Any]) -> float:
    f0 = float(truth["f_220"])
    tau0 = float(truth["tau_220"])
    f_hat = float(estimate["f_220_hat"])
    tau_hat = float(estimate["tau_220_hat"])
    err_f = (f_hat - f0) / f0
    err_tau = (tau_hat - tau0) / tau0
    return float(math.sqrt((err_f**2 + err_tau**2) / 2.0))


def _maybe_git_sha() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if res.returncode != 0:
        return None
    value = res.stdout.strip()
    return value or None


def _stage_created_timestamp(stage_dir: Path) -> Optional[str]:
    summary_path = stage_dir / "stage_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = _read_json(summary_path)
    except Exception:
        return None
    created = payload.get("created")
    return str(created) if created else None


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_06 PSD robustness (contract-first)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-rel-rmse", type=float, default=DEFAULT_MAX_REL_RMSE)
    ap.add_argument("--psd-sweep", default=DEFAULT_PSD_SWEEP)
    ap.add_argument("--n-max-cases", type=int, default=DEFAULT_N_MAX_CASES)
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    seed_snr2strain, synth_idx_path = _load_seed_snr_to_strain(run_dir)
    recovery_cases_path = _resolve_recovery_cases(run_dir)

    cases: List[Dict[str, Any]] = []
    n_unmapped = 0
    for idx, row in enumerate(_read_jsonl(recovery_cases_path)):
        if len(cases) >= int(args.n_max_cases):
            break
        case_id = str(row.get("case_id") or row.get("id") or f"case_{idx:04d}")
        truth = row.get("truth") or {}
        f0 = truth.get("f_220")
        tau0 = truth.get("tau_220")
        seed = row.get("seed")
        snr = row.get("snr_target", row.get("snr"))
        if f0 is None or tau0 is None or seed is None or snr is None:
            continue
        k = (int(seed), _snr_key(float(snr)))
        rel_strain = seed_snr2strain.get(k)
        if not rel_strain:
            n_unmapped += 1
            continue
        strain_path = (run_dir / "ringdown_synth" / "outputs" / rel_strain).resolve()
        expected_prefix = (run_dir / "ringdown_synth" / "outputs").resolve()
        try:
            strain_path.relative_to(expected_prefix)
        except ValueError:
            n_unmapped += 1
            continue
        if not strain_path.exists():
            n_unmapped += 1
            continue
        cases.append(
            {
                "case_id": case_id,
                "seed": int(seed),
                "snr": float(snr),
                "truth": {"f_220": float(f0), "tau_220": float(tau0)},
                "strain_path": strain_path,
            }
        )

    if not cases:
        abort_contract("recovery_cases.jsonl has zero usable entries")

    if args.psd_sweep != "v1":
        abort_contract(f"unsupported psd_sweep={args.psd_sweep}")
    variants = _psd_variants_v1()

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    per_case_path = outputs_dir / "psd_cases.jsonl"
    sweep_path = outputs_dir / "psd_sweep_metrics.json"
    contract_path = outputs_dir / "contract_verdict.json"

    case_baselines: List[float] = []
    case_worsts: List[float] = []
    n_total_rows = 0
    n_ok_rows = 0

    with open(per_case_path, "w", encoding="utf-8") as f:
        for case in cases:
            strain_path = case["strain_path"]
            try:
                strain, t, dt = _load_strain(strain_path)
                psd, freqs = _welch_psd(strain, dt)
                whitened = _whiten_series(strain, dt, psd, freqs)
                baseline_est = recover_ringdown_from_series(whitened, t, dt)
                baseline_metric = _compute_rmse(case["truth"], baseline_est)
            except SystemExit as exc:
                baseline_metric = None
                baseline_error = str(exc)
            except Exception as exc:
                baseline_metric = None
                baseline_error = str(exc)
            else:
                baseline_error = ""

            per_case_variant_metrics: List[float] = []

            for variant in variants:
                n_total_rows += 1
                row: Dict[str, Any] = {
                    "case": {
                        "case_id": case["case_id"],
                        "seed": case["seed"],
                        "snr": case["snr"],
                    },
                    "psd_variant": {
                        "id": variant["id"],
                        "params": variant.get("params", {}),
                    },
                    "metric_baseline": None,
                    "metric_variant": None,
                    "rel_degradation": None,
                    "status": "SKIP",
                }

                if baseline_metric is None or not np.isfinite(baseline_metric) or baseline_metric <= 0:
                    row["status"] = "SKIP"
                    row["notes"] = f"baseline_invalid:{baseline_error}".strip(":")
                    f.write(json.dumps(row) + "\n")
                    continue

                row["metric_baseline"] = float(baseline_metric)

                try:
                    psd_variant = _apply_psd_variant(psd, freqs, variant)
                    whitened_variant = _whiten_series(strain, dt, psd_variant, freqs)
                    est_variant = recover_ringdown_from_series(whitened_variant, t, dt)
                    metric_variant = _compute_rmse(case["truth"], est_variant)
                except SystemExit as exc:
                    row["status"] = "ERROR"
                    row["notes"] = str(exc)
                    f.write(json.dumps(row) + "\n")
                    continue
                except Exception as exc:
                    row["status"] = "ERROR"
                    row["notes"] = str(exc)
                    f.write(json.dumps(row) + "\n")
                    continue

                if not np.isfinite(metric_variant):
                    row["status"] = "ERROR"
                    row["notes"] = "metric_variant non-finite"
                    f.write(json.dumps(row) + "\n")
                    continue

                row["metric_variant"] = float(metric_variant)
                rel = (metric_variant - baseline_metric) / baseline_metric
                row["rel_degradation"] = float(rel)
                row["status"] = "OK"
                n_ok_rows += 1
                per_case_variant_metrics.append(float(metric_variant))
                f.write(json.dumps(row) + "\n")

            if baseline_metric is None or not np.isfinite(baseline_metric) or baseline_metric <= 0:
                continue
            if per_case_variant_metrics:
                case_baselines.append(float(baseline_metric))
                case_worsts.append(max(per_case_variant_metrics))

    baseline_metric_mean = float(np.mean(case_baselines)) if case_baselines else float("nan")
    worst_metric_mean = float(np.mean(case_worsts)) if case_worsts else float("nan")
    if case_baselines and baseline_metric_mean > 0 and np.isfinite(worst_metric_mean):
        worst_rel_degradation = float((worst_metric_mean - baseline_metric_mean) / baseline_metric_mean)
    else:
        worst_rel_degradation = float("inf")

    sweep_payload = {
        "run_id": args.run,
        "n_total": int(n_total_rows),
        "n_usable": int(n_ok_rows),
        "psd_sweep_id": args.psd_sweep,
        "variants": variants,
        "aggregate": {
            "baseline_metric_mean": baseline_metric_mean,
            "worst_metric_mean": worst_metric_mean,
            "worst_rel_degradation": worst_rel_degradation,
        },
    }

    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    robustness_pass = (
        case_baselines
        and math.isfinite(worst_rel_degradation)
        and worst_rel_degradation <= float(args.max_rel_rmse)
    )

    inputs = {
        "run_valid": {
            "path": str(_resolve_run_valid_path(run_dir).relative_to(run_dir)),
            "sha256": sha256_file(_resolve_run_valid_path(run_dir)),
        },
        "synthetic_events_list": {
            "path": str(synth_idx_path.relative_to(run_dir)),
            "sha256": sha256_file(synth_idx_path),
        },
        "recovery_cases": {
            "path": str(recovery_cases_path.relative_to(run_dir)),
            "sha256": sha256_file(recovery_cases_path),
        },
    }

    diagnostics_pass = bool(
        variants
        and n_total_rows > 0
        and "run_valid" in inputs
        and "synthetic_events_list" in inputs
        and "recovery_cases" in inputs
    )

    contracts = [
        {
            "id": "R06_PSD_ROBUSTNESS",
            "verdict": "PASS" if robustness_pass else "FAIL",
            "violations": [] if robustness_pass else ["max_rel_rmse_exceeded"],
            "metrics": {
                "baseline_metric": baseline_metric_mean,
                "worst_metric": worst_metric_mean,
                "rel_degradation": worst_rel_degradation,
                "threshold": float(args.max_rel_rmse),
            },
        },
        {
            "id": "R06_DIAGNOSTICS_COMPLETE",
            "verdict": "PASS" if diagnostics_pass else "FAIL",
            "violations": [] if diagnostics_pass else ["diagnostics_incomplete"],
            "metrics": {
                "variants": variants,
                "n_total": int(n_total_rows),
                "n_usable": int(n_ok_rows),
                "n_cases": int(len(cases)),
                "coverage": {
                    "usable": int(n_ok_rows),
                    "total": int(n_total_rows),
                },
            },
        },
    ]

    overall_verdict = "PASS" if all(c["verdict"] == "PASS" for c in contracts) else "FAIL"

    contract_payload = {
        "overall_verdict": overall_verdict,
        "contracts": contracts,
    }

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "inputs": inputs,
        "parameters": {
            "max_rel_rmse": float(args.max_rel_rmse),
            "psd_sweep": args.psd_sweep,
            "n_max_cases": int(args.n_max_cases),
            "psd_floor": float(PSD_FLOOR),
            "psd_nperseg": int(PSD_NPERSEG),
            "psd_overlap": float(PSD_OVERLAP),
        },
        "outputs": {
            "psd_sweep_metrics": "outputs/psd_sweep_metrics.json",
            "psd_cases": "outputs/psd_cases.jsonl",
            "contract_verdict": "outputs/contract_verdict.json",
        },
        "results": {
            "overall_verdict": overall_verdict,
            "contracts": contracts,
        },
        "version": {
            "git_sha": _maybe_git_sha(),
        },
    }

    existing_created = _stage_created_timestamp(stage_dir)
    if existing_created is not None:
        summary["created"] = existing_created

    summary_written = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "psd_sweep_metrics": sweep_path,
            "psd_cases": per_case_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    if overall_verdict != "PASS":
        abort_contract("EXP_RINGDOWN_06 FAIL: PSD robustness contract not satisfied")

    print("OK: EXP_RINGDOWN_06 PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
