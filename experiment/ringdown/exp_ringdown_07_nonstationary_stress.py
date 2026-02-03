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
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

from experiment.ringdown.exp_ringdown_02_recovery_robustness import (
    recover_ringdown_from_series,
)

EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_07__nonstationary_stress"
DEFAULT_SWEEP = "v1"
DEFAULT_MAX_FAIL_RATE = 0.10
DEFAULT_MAX_BIAS_REL = 0.20
DEFAULT_N_MAX_CASES = 24

ALLOWED_FAIL_CODES = {
    "nonconvergence",
    "nan_inference",
    "numerical_instability",
    "invalid_input",
    "timeout",
    "exception",
}


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


def _resolve_nonstationary_variants(run_dir: Path) -> Optional[Path]:
    path = run_dir / "nonstationary_noise" / "outputs" / "nonstationary_variants.json"
    return path if path.exists() else None


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


def _compute_rmse(truth: Dict[str, float], estimate: Dict[str, Any]) -> float:
    f0 = float(truth["f_220"])
    tau0 = float(truth["tau_220"])
    f_hat = float(estimate["f_220_hat"])
    tau_hat = float(estimate["tau_220_hat"])
    err_f = (f_hat - f0) / f0
    err_tau = (tau_hat - tau0) / tau0
    return float(math.sqrt((err_f**2 + err_tau**2) / 2.0))


def _variants_v1(line_f0: float) -> List[Dict[str, Any]]:
    line_f0 = float(line_f0)
    return [
        {
            "id": "drift_lowfreq",
            "type": "drift_lowfreq",
            "params": {"amplitude_scale": 0.20, "f_hz": 0.5},
        },
        {
            "id": "line_f0",
            "type": "line",
            "params": {"f_hz": line_f0, "amplitude_scale": 0.10},
        },
        {
            "id": "line_2f0",
            "type": "line",
            "params": {"f_hz": 2.0 * line_f0, "amplitude_scale": 0.08},
        },
        {
            "id": "glitch_gaussian_pulse",
            "type": "glitch_gaussian_pulse",
            "params": {"amplitude_scale": 0.50, "t0_rel": 0.35, "sigma_s": 0.008},
        },
        {
            "id": "glitch_chirp_short",
            "type": "glitch_chirp_short",
            "params": {
                "amplitude_scale": 0.35,
                "f0_hz": 30.0,
                "f1_hz": 180.0,
                "t0_rel": 0.15,
                "duration_s": 0.05,
            },
        },
    ]


def _apply_variant(
    strain: np.ndarray,
    t: np.ndarray,
    variant: Dict[str, Any],
    line_f0: float,
) -> np.ndarray:
    vtype = variant.get("type")
    params = variant.get("params") or {}
    out = np.asarray(strain, dtype=float).copy()
    amp_scale = float(params.get("amplitude_scale", 0.1))
    amp_base = float(np.std(out)) if np.std(out) > 0 else 1.0
    amp = amp_scale * amp_base
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0

    if vtype == "drift_lowfreq":
        f_hz = float(params.get("f_hz", 0.5))
        out = out + amp * np.sin(2.0 * math.pi * f_hz * t)
    elif vtype == "line":
        f_hz = float(params.get("f_hz", line_f0))
        out = out + amp * np.sin(2.0 * math.pi * f_hz * t)
    elif vtype == "glitch_gaussian_pulse":
        t0_rel = float(params.get("t0_rel", 0.35))
        sigma = float(params.get("sigma_s", 0.008))
        t0 = float(t[0] + t0_rel * duration)
        pulse = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        out = out + amp * pulse
    elif vtype == "glitch_chirp_short":
        t0_rel = float(params.get("t0_rel", 0.15))
        dur = float(params.get("duration_s", 0.05))
        f0 = float(params.get("f0_hz", 30.0))
        f1 = float(params.get("f1_hz", 180.0))
        t0 = float(t[0] + t0_rel * duration)
        t1 = t0 + dur
        mask = (t >= t0) & (t <= t1)
        if np.any(mask):
            tt = t[mask] - t0
            k = (f1 - f0) / max(dur, 1e-9)
            phase = 2.0 * math.pi * (f0 * tt + 0.5 * k * tt**2)
            chirp = np.sin(phase)
            out[mask] = out[mask] + amp * chirp
    else:
        raise ValueError(f"unknown variant type: {vtype}")
    return out


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


def _classify_failure(message: str) -> str:
    msg = message.lower()
    if "nan" in msg:
        return "nan_inference"
    if "dt" in msg or "shape" in msg or "invalid" in msg:
        return "invalid_input"
    if "pocos puntos" in msg or "fit" in msg or "tau" in msg or "banda" in msg:
        return "nonconvergence"
    if "overflow" in msg or "instability" in msg:
        return "numerical_instability"
    return "exception"


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
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_07 nonstationary stress (contract-first)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--max-fail-rate", type=float, default=DEFAULT_MAX_FAIL_RATE)
    ap.add_argument("--max-bias-rel", type=float, default=DEFAULT_MAX_BIAS_REL)
    ap.add_argument("--n-max-cases", type=int, default=DEFAULT_N_MAX_CASES)
    ap.add_argument("--line-f0", type=float, default=50.0)
    ap.add_argument("--dry-run", action="store_true")
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
    nonstationary_variants_path = _resolve_nonstationary_variants(run_dir)

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

    if args.sweep != "v1":
        abort_contract(f"unsupported sweep={args.sweep}")

    if nonstationary_variants_path is not None:
        variants_payload = _read_json(nonstationary_variants_path)
        variants = list(variants_payload.get("variants", []))
        if not variants:
            abort_contract("nonstationary_variants.json is empty")
    else:
        variants = _variants_v1(float(args.line_f0))

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    failure_catalog_path = outputs_dir / "failure_catalog.jsonl"
    report_path = outputs_dir / "nonstationary_report.json"
    contract_path = outputs_dir / "contract_verdict.json"

    case_baselines: List[float] = []
    case_worsts: List[float] = []
    n_total_rows = 0
    n_fail_rows = 0
    bias_values: List[float] = []
    by_variant: Dict[str, Dict[str, Any]] = {}

    with open(failure_catalog_path, "w", encoding="utf-8") as f:
        for case in cases:
            strain_path = case["strain_path"]
            baseline_metric = None
            baseline_error = ""
            try:
                strain, t, dt = _load_strain(strain_path)
                baseline_est = recover_ringdown_from_series(strain, t, dt)
                baseline_metric = _compute_rmse(case["truth"], baseline_est)
            except SystemExit as exc:
                baseline_error = str(exc)
            except Exception as exc:
                baseline_error = str(exc)

            per_case_variant_metrics: List[float] = []

            for variant in variants:
                n_total_rows += 1
                variant_id = str(variant.get("id", "variant"))
                vtype = str(variant.get("type", "unknown"))
                params = variant.get("params", {})
                stats = by_variant.setdefault(
                    variant_id,
                    {
                        "n_total": 0,
                        "n_fail": 0,
                        "worst_rel_degradation": float("-inf"),
                        "max_bias_rel": float("-inf"),
                    },
                )
                stats["n_total"] += 1
                row: Dict[str, Any] = {
                    "case": {
                        "case_id": case["case_id"],
                        "seed": case["seed"],
                        "snr": case["snr"],
                    },
                    "variant": {"id": variant_id, "type": vtype, "params": params},
                    "status": "SKIP",
                    "metric_baseline": None,
                    "metric_variant": None,
                    "rel_degradation": None,
                    "bias_rel": None,
                    "fail_reason_code": "",
                    "notes": "",
                }

                if baseline_metric is None or not np.isfinite(baseline_metric) or baseline_metric <= 0:
                    row["status"] = "SKIP"
                    row["notes"] = f"baseline_invalid:{baseline_error}".strip(":")
                    f.write(json.dumps(row) + "\n")
                    continue

                row["metric_baseline"] = float(baseline_metric)

                try:
                    corrupted = _apply_variant(strain, t, variant, float(args.line_f0))
                    est_variant = recover_ringdown_from_series(corrupted, t, dt)
                    metric_variant = _compute_rmse(case["truth"], est_variant)
                except SystemExit as exc:
                    code = _classify_failure(str(exc))
                    row["status"] = "ERROR"
                    row["fail_reason_code"] = code
                    if code == "exception":
                        row["exc_type"] = "SystemExit"
                    row["notes"] = str(exc)
                    n_fail_rows += 1
                    stats["n_fail"] += 1
                    f.write(json.dumps(row) + "\n")
                    continue
                except Exception as exc:
                    code = _classify_failure(str(exc))
                    row["status"] = "ERROR"
                    row["fail_reason_code"] = code
                    if code == "exception":
                        row["exc_type"] = type(exc).__name__
                    row["notes"] = str(exc)
                    n_fail_rows += 1
                    stats["n_fail"] += 1
                    f.write(json.dumps(row) + "\n")
                    continue

                if not np.isfinite(metric_variant):
                    row["status"] = "FAIL"
                    row["fail_reason_code"] = "nan_inference"
                    row["notes"] = "metric_variant non-finite"
                    n_fail_rows += 1
                    stats["n_fail"] += 1
                    f.write(json.dumps(row) + "\n")
                    continue

                row["metric_variant"] = float(metric_variant)
                rel = (metric_variant - baseline_metric) / baseline_metric
                row["rel_degradation"] = float(rel)
                row["bias_rel"] = float(rel)
                row["status"] = "OK"
                per_case_variant_metrics.append(float(metric_variant))
                bias_values.append(float(rel))
                f.write(json.dumps(row) + "\n")

                stats["worst_rel_degradation"] = max(stats["worst_rel_degradation"], float(rel))
                stats["max_bias_rel"] = max(stats["max_bias_rel"], float(rel))

            if baseline_metric is None or not np.isfinite(baseline_metric) or baseline_metric <= 0:
                continue
            if per_case_variant_metrics:
                case_baselines.append(float(baseline_metric))
                case_worsts.append(max(per_case_variant_metrics))

    if case_baselines:
        baseline_mean = float(np.mean(case_baselines))
        worst_mean = float(np.mean(case_worsts)) if case_worsts else float("nan")
    else:
        baseline_mean = float("nan")
        worst_mean = float("nan")

    max_bias_rel = float(np.max(bias_values)) if bias_values else float("inf")
    fail_rate = float(n_fail_rows / n_total_rows) if n_total_rows else 1.0

    for stats in by_variant.values():
        n_total = int(stats["n_total"])
        n_fail = int(stats["n_fail"])
        stats["fail_rate"] = float(n_fail / n_total) if n_total else 0.0

    report_payload = {
        "run_id": args.run,
        "sweep_id": args.sweep,
        "n_cases": int(len(cases)),
        "n_total": int(n_total_rows),
        "n_fail": int(n_fail_rows),
        "fail_rate": fail_rate,
        "max_fail_rate_threshold": float(args.max_fail_rate),
        "bias_summary": {
            "baseline_mean": baseline_mean,
            "worst_mean": worst_mean,
            "max_bias_rel": max_bias_rel,
        },
        "variants": variants,
        "by_variant": by_variant,
    }

    if not args.dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, sort_keys=True)
            f.write("\n")

    robustness_pass = (
        n_total_rows > 0
        and math.isfinite(fail_rate)
        and fail_rate <= float(args.max_fail_rate)
        and math.isfinite(max_bias_rel)
        and max_bias_rel <= float(args.max_bias_rel)
    )

    categorized_pass = True
    violations: List[str] = []
    for row in _read_jsonl(failure_catalog_path):
        status = str(row.get("status", ""))
        if status not in {"FAIL", "ERROR"}:
            continue
        code = str(row.get("fail_reason_code", ""))
        if code not in ALLOWED_FAIL_CODES:
            categorized_pass = False
            violations.append(f"invalid_code:{code or '<empty>'}")
        if code == "exception" and not row.get("exc_type"):
            categorized_pass = False
            violations.append("missing_exc_type")

    contracts = [
        {
            "id": "R07_ROBUST_UNDER_NONSTATIONARY",
            "verdict": "PASS" if robustness_pass else "FAIL",
            "violations": [] if robustness_pass else ["fail_rate_or_bias_threshold"],
            "metrics": {
                "n_total": int(n_total_rows),
                "n_fail": int(n_fail_rows),
                "fail_rate": fail_rate,
                "max_fail_rate_threshold": float(args.max_fail_rate),
                "max_bias_rel": max_bias_rel,
                "max_bias_rel_threshold": float(args.max_bias_rel),
                "baseline_mean": baseline_mean,
                "worst_mean": worst_mean,
            },
        },
        {
            "id": "R07_FAIL_CATEGORIZED",
            "verdict": "PASS" if categorized_pass else "FAIL",
            "violations": [] if categorized_pass else violations[:5],
            "metrics": {
                "allowed_codes": sorted(ALLOWED_FAIL_CODES),
            },
        },
    ]

    overall_verdict = "PASS" if all(c["verdict"] == "PASS" for c in contracts) else "FAIL"
    contract_payload = {
        "overall_verdict": overall_verdict,
        "contracts": contracts,
        "timestamp": utc_now_iso(),
        "schema_version": "exp_ringdown_07_v1",
    }

    if not args.dry_run:
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract_payload, f, indent=2, sort_keys=True)
            f.write("\n")

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

    if nonstationary_variants_path is not None:
        inputs["nonstationary_variants"] = {
            "path": str(nonstationary_variants_path.relative_to(run_dir)),
            "sha256": sha256_file(nonstationary_variants_path),
        }

    summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "inputs": inputs,
        "parameters": {
            "sweep": args.sweep,
            "max_fail_rate": float(args.max_fail_rate),
            "max_bias_rel": float(args.max_bias_rel),
            "n_max_cases": int(args.n_max_cases),
            "line_f0": float(args.line_f0),
            "dry_run": bool(args.dry_run),
        },
        "outputs": {
            "nonstationary_report": "outputs/nonstationary_report.json",
            "failure_catalog": "outputs/failure_catalog.jsonl",
            "contract_verdict": "outputs/contract_verdict.json",
        },
        "results": {
            "overall_verdict": overall_verdict,
            "contracts": contracts,
            "n_unmapped": int(n_unmapped),
        },
        "version": {"git_sha": _maybe_git_sha()},
    }

    existing_created = _stage_created_timestamp(stage_dir)
    if existing_created is not None:
        summary["created"] = existing_created

    if not args.dry_run:
        summary_written = write_stage_summary(stage_dir, summary)
        write_manifest(
            stage_dir,
            {
                "nonstationary_report": report_path,
                "failure_catalog": failure_catalog_path,
                "contract": contract_path,
                "stage_summary": summary_written,
            },
            extra={"version": "1"},
        )

    if overall_verdict != "PASS":
        abort_contract("EXP_RINGDOWN_07 FAIL: nonstationary stress contract not satisfied")

    print("OK: EXP_RINGDOWN_07 PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
