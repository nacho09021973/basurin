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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from basurin_io import (
    resolve_out_root,
    validate_run_id,
    ensure_stage_dirs,
    write_manifest,
    write_stage_summary,
    utc_now_iso,
)

from experiment.ringdown.exp_ringdown_01_injection_recovery import (
    recover_ringdown as recover_ringdown_npz,
)


EXIT_CONTRACT_FAIL = 2


def abort_contract(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_overall_verdict(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    # canonical stage_summary_v1
    v = (payload.get("results") or {}).get("overall_verdict")
    if v is not None:
        return v
    # compat fallbacks
    v = payload.get("overall_verdict")
    if v is not None:
        return v
    v = (payload.get("results") or {}).get("verdict")
    if v is not None:
        return v
    v = payload.get("verdict")
    if v is not None:
        return v
    return None


def parse_csv_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_strs(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def check_run_valid_stage_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "RUN_VALID" / "stage_summary.json"
    if not path.exists():
        abort_contract(f"RUN_VALID stage_summary.json missing at {path}")
    payload = read_json(path)
    verdict = _extract_overall_verdict(payload)
    if verdict != "PASS":
        abort_contract(f"RUN_VALID overall_verdict={verdict}")
    return payload


def resolve_events_path(run_dir: Path) -> Path:
    path = (run_dir / "ringdown_synth" / "outputs" / "synthetic_events.json").resolve()
    if not path.exists():
        abort_contract(f"synthetic_events.json missing at {path}")
    return path


def resolve_exp01_contract(run_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    preferred = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_01_injection_recovery"
        / "outputs"
        / "contract_verdict.json"
    )
    legacy = (
        run_dir
        / "experiment"
        / "ringdown_01_injection_recovery"
        / "outputs"
        / "contract_verdict.json"
    )
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        abort_contract(f"EXP01 contract_verdict.json missing at {preferred} (legacy {legacy})")
    payload = read_json(path)
    verdict = _extract_overall_verdict(payload)
    if verdict != "PASS":
        abort_contract(f"EXP01 contract_verdict={verdict}")
    return path, payload


def resolve_exp01_cases_path(run_dir: Path) -> Optional[Path]:
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
    return None


def load_strain_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
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


def recover_ringdown_from_series(x: np.ndarray, t: np.ndarray, dt: float) -> Dict[str, Any]:
    """
    Recuperador determinista idéntico a EXP_RINGDOWN_01,
    operando en memoria para evitar I/O.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if x.shape != t.shape:
        abort_contract("t shape != strain shape (in-memory)")

    x = x - float(np.mean(x))

    freqs = np.fft.rfftfreq(x.size, d=dt)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    band = (freqs >= 20.0) & (freqs <= 500.0)
    if not np.any(band):
        abort_contract("banda 20..500 Hz vacía (n demasiado pequeño)")
    f_hat = float(freqs[band][int(np.argmax(mag[band]))])

    i0 = int(np.argmax(np.abs(x)))
    n = x.size
    i1 = min(n, i0 + int(max(128, round(0.12 / dt))))
    xw = x[i0:i1]
    tw = t[i0:i1]
    if xw.size < 128:
        abort_contract("ventana post-t0 demasiado corta para estimar tau")

    nw = xw.size
    Xfw = np.fft.fft(xw)
    hw = np.zeros(nw)
    if nw % 2 == 0:
        hw[0] = 1.0
        hw[nw // 2] = 1.0
        hw[1:nw // 2] = 2.0
    else:
        hw[0] = 1.0
        hw[1:(nw + 1) // 2] = 2.0
    xa = np.fft.ifft(Xfw * hw)

    tt = (tw - float(tw[0]))
    z = xa * np.exp(-1j * 2.0 * np.pi * f_hat * tt)
    env = np.abs(z).astype(float)

    env_max = float(np.max(env))
    if not np.isfinite(env_max) or env_max <= 0:
        abort_contract("envolvente inválida tras demodulación")

    tmax = 0.08
    base_mask = tt <= tmax

    thr = env_max * 0.30
    mask = base_mask & (env >= thr)
    if np.count_nonzero(mask) < 32:
        thr = env_max * 0.15
        mask = base_mask & (env >= thr)
    if np.count_nonzero(mask) < 32:
        thr = env_max * 0.08
        mask = base_mask & (env >= thr)

    if np.count_nonzero(mask) < 32:
        abort_contract("pocos puntos para fit de tau (ni con thr=8%)")

    yy = np.log(env[mask])
    xx = tt[mask]

    b, a = np.polyfit(xx, yy, deg=1)
    if not np.isfinite(b) or b >= 0:
        abort_contract("pendiente no negativa en log(envolvente demodulada); tau no identificable")
    tau_hat = float(-1.0 / b)
    q_hat = float(np.pi * f_hat * tau_hat)

    return {"f_220_hat": f_hat, "tau_220_hat": tau_hat, "Q_220_hat": q_hat}


def shift_series(x: np.ndarray, dt_samples: int) -> np.ndarray:
    if dt_samples == 0:
        return x.copy()
    n = x.size
    if abs(dt_samples) >= n:
        return np.zeros_like(x)
    if dt_samples > 0:
        return np.concatenate([np.zeros(dt_samples, dtype=float), x[:-dt_samples]])
    return np.concatenate([x[-dt_samples:], np.zeros(-dt_samples, dtype=float)])


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def percentile(xs: Iterable[float], q: float) -> float:
    xs2 = sorted(xs)
    if not xs2:
        return float("nan")
    k = (len(xs2) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs2) - 1)
    a = k - lo
    return xs2[lo] * (1 - a) + xs2[hi] * a


def read_exp01_baseline(cases_path: Path, theta_keys: List[str]) -> Dict[str, Dict[str, float]]:
    baseline: Dict[str, Dict[str, float]] = {}
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "OK":
                continue
            case_id = str(row.get("case_id", ""))
            est = row.get("estimate") or {}
            values: Dict[str, float] = {}
            for key in theta_keys:
                if key in est:
                    values[key] = float(est[key])
            if values:
                baseline[case_id] = values
    return baseline


@dataclass(frozen=True)
class Thresholds:
    max_rel_diff_scale: float
    p95_rel_diff_dt: float
    max_rel_diff_dt: float
    consistency_tol: float


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_02 recovery robustness (contract-first)")
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--stage-name",
        default="experiment/ringdown/EXP_RINGDOWN_02_recovery_robustness",
    )
    ap.add_argument("--scales", default="0.5,1.0,2.0")
    ap.add_argument("--dt-samples", default="-2,-1,0,1,2")
    ap.add_argument("--theta-keys", default="f_220_hat,tau_220_hat")
    ap.add_argument("--max-rel-diff-scale", type=float, default=1e-3)
    ap.add_argument("--p95-rel-diff-dt", type=float, default=5e-3)
    ap.add_argument("--max-rel-diff-dt", type=float, default=2e-2)
    ap.add_argument("--consistency-tol", type=float, default=1e-6)
    ap.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    check_run_valid_stage_summary(run_dir)

    events_path = resolve_events_path(run_dir)
    events = read_json(events_path)
    if not isinstance(events, list) or not events:
        abort_contract(f"events-json invalid or empty list: {events_path}")

    exp01_contract_path, _ = resolve_exp01_contract(run_dir)
    exp01_cases_path = resolve_exp01_cases_path(run_dir)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, args.stage_name, base_dir=out_root)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scales = parse_csv_floats(args.scales)
    dt_samples = parse_csv_ints(args.dt_samples)
    theta_keys = parse_csv_strs(args.theta_keys)
    thr = Thresholds(
        max_rel_diff_scale=float(args.max_rel_diff_scale),
        p95_rel_diff_dt=float(args.p95_rel_diff_dt),
        max_rel_diff_dt=float(args.max_rel_diff_dt),
        consistency_tol=float(args.consistency_tol),
    )

    baseline_exp01: Dict[str, Dict[str, float]] = {}
    if exp01_cases_path is not None:
        baseline_exp01 = read_exp01_baseline(exp01_cases_path, theta_keys)

    per_case_path = outputs_dir / "per_case.jsonl"

    scale_rel_diffs: Dict[str, List[float]] = {k: [] for k in theta_keys}
    dt_rel_diffs: Dict[str, List[float]] = {k: [] for k in theta_keys}
    consistency_abs_diffs: Dict[str, List[float]] = {k: [] for k in theta_keys}

    with open(per_case_path, "w", encoding="utf-8") as f:
        for ev in events:
            case_id = str(ev.get("case_id") or "")
            rel_strain = ev.get("strain_npz") or ev.get("strain_path")
            if not rel_strain:
                abort_contract(f"missing strain_npz for case {case_id}")

            expected_prefix = (run_dir / "ringdown_synth" / "outputs").resolve()
            if str(rel_strain).startswith("runs/"):
                strain_path = (Path.cwd() / rel_strain).resolve()
            else:
                strain_path = (expected_prefix / rel_strain).resolve()
            try:
                strain_path.relative_to(run_dir)
            except ValueError:
                abort_contract(f"strain_npz escapes run dir: {strain_path}")
            if not strain_path.exists():
                abort_contract(f"strain_npz not found: {strain_path}")

            x, t, dt = load_strain_npz(strain_path)

            base_est = recover_ringdown_npz(strain_path)

            by_scale: Dict[str, Dict[str, float]] = {}
            for scale in scales:
                scaled = x * float(scale)
                est = recover_ringdown_from_series(scaled, t, dt)
                by_scale[str(scale)] = {k: float(est[k]) for k in theta_keys if k in est}

            by_dt: Dict[str, Dict[str, float]] = {}
            for dts in dt_samples:
                shifted = shift_series(x, int(dts))
                est = recover_ringdown_from_series(shifted, t, dt)
                by_dt[str(dts)] = {k: float(est[k]) for k in theta_keys if k in est}

            rel_diffs_scale: Dict[str, Dict[str, float]] = {k: {} for k in theta_keys}
            rel_diffs_dt: Dict[str, Dict[str, float]] = {k: {} for k in theta_keys}
            for k in theta_keys:
                base_val = float(base_est[k])
                for scale in scales:
                    est_val = by_scale[str(scale)].get(k)
                    if est_val is None:
                        continue
                    diff = rel_diff(float(est_val), base_val)
                    rel_diffs_scale[k][str(scale)] = diff
                    scale_rel_diffs[k].append(diff)
                for dts in dt_samples:
                    est_val = by_dt[str(dts)].get(k)
                    if est_val is None:
                        continue
                    diff = rel_diff(float(est_val), base_val)
                    rel_diffs_dt[k][str(dts)] = diff
                    dt_rel_diffs[k].append(diff)

            baseline_payload = baseline_exp01.get(case_id)
            if baseline_payload:
                for k in theta_keys:
                    if k in baseline_payload:
                        diff = abs(float(base_est[k]) - float(baseline_payload[k]))
                        consistency_abs_diffs[k].append(diff)

            row: Dict[str, Any] = {
                "case_id": case_id,
                "baseline": {k: float(base_est[k]) for k in theta_keys if k in base_est},
                "by_scale": by_scale,
                "by_dt": by_dt,
                "rel_diffs": {
                    "scale": rel_diffs_scale,
                    "dt": rel_diffs_dt,
                },
            }
            if baseline_payload:
                row["baseline_exp01"] = baseline_payload
            f.write(json.dumps(row) + "\n")

    scale_max_by_theta = {
        k: (max(scale_rel_diffs[k]) if scale_rel_diffs[k] else float("nan")) for k in theta_keys
    }
    dt_p95_by_theta = {
        k: percentile(dt_rel_diffs[k], 0.95) if dt_rel_diffs[k] else float("nan")
        for k in theta_keys
    }
    dt_max_by_theta = {
        k: (max(dt_rel_diffs[k]) if dt_rel_diffs[k] else float("nan")) for k in theta_keys
    }
    consistency_max_by_theta = {
        k: (max(consistency_abs_diffs[k]) if consistency_abs_diffs[k] else float("nan"))
        for k in theta_keys
    }

    r02a_pass = all(
        np.isfinite(scale_max_by_theta[k]) and scale_max_by_theta[k] <= thr.max_rel_diff_scale
        for k in theta_keys
    )
    r02b_pass = all(
        np.isfinite(dt_p95_by_theta[k])
        and np.isfinite(dt_max_by_theta[k])
        and dt_p95_by_theta[k] <= thr.p95_rel_diff_dt
        and dt_max_by_theta[k] <= thr.max_rel_diff_dt
        for k in theta_keys
    )

    r02c_applicable = any(consistency_abs_diffs[k] for k in theta_keys)
    r02c_pass = False
    if r02c_applicable:
        r02c_pass = all(
            np.isfinite(consistency_max_by_theta[k])
            and consistency_max_by_theta[k] <= thr.consistency_tol
            for k in theta_keys
        )

    contracts: List[Dict[str, Any]] = [
        {
            "id": "R02A_scale_invariance",
            "verdict": "PASS" if r02a_pass else "FAIL",
            "metrics": {
                "max_rel_diff_by_theta": scale_max_by_theta,
                "threshold": thr.max_rel_diff_scale,
            },
        },
        {
            "id": "R02B_t0_shift_robustness",
            "verdict": "PASS" if r02b_pass else "FAIL",
            "metrics": {
                "p95_abs_rel_diff_by_theta": dt_p95_by_theta,
                "max_abs_rel_diff_by_theta": dt_max_by_theta,
                "thresholds": {"p95": thr.p95_rel_diff_dt, "max": thr.max_rel_diff_dt},
            },
        },
    ]

    if r02c_applicable:
        contracts.append(
            {
                "id": "R02C_consistency_with_EXP01",
                "verdict": "PASS" if r02c_pass else "FAIL",
                "metrics": {
                    "max_abs_diff_by_theta": consistency_max_by_theta,
                    "threshold": thr.consistency_tol,
                },
            }
        )
    else:
        contracts.append(
            {
                "id": "R02C_consistency_with_EXP01",
                "verdict": "SKIP",
                "metrics": {
                    "max_abs_diff_by_theta": consistency_max_by_theta,
                    "threshold": thr.consistency_tol,
                },
                "reason_skip": "baseline per-case missing; using exp02 baseline only",
            }
        )

    if r02c_applicable:
        overall_verdict = "PASS" if (r02a_pass and r02b_pass and r02c_pass) else "FAIL"
    else:
        overall_verdict = "PASS" if (r02a_pass and r02b_pass) else "FAIL"

    robustness_summary = {
        "theta_keys": theta_keys,
        "metrics": {
            "scale": {"max_rel_diff_by_theta": scale_max_by_theta},
            "dt": {
                "p95_abs_rel_diff_by_theta": dt_p95_by_theta,
                "max_abs_rel_diff_by_theta": dt_max_by_theta,
            },
            "consistency": {
                "max_abs_diff_by_theta": consistency_max_by_theta,
                "applicable": r02c_applicable,
            },
        },
        "thresholds": {
            "scale": thr.max_rel_diff_scale,
            "dt": {"p95": thr.p95_rel_diff_dt, "max": thr.max_rel_diff_dt},
            "consistency": thr.consistency_tol,
        },
        "inputs": {
            "synthetic_events": str(events_path),
            "exp01_contract": str(exp01_contract_path),
            "exp01_cases": str(exp01_cases_path) if exp01_cases_path else None,
        },
        "recoverer": "experiment.ringdown.exp_ringdown_01_injection_recovery.recover_ringdown",
        "notes": {
            "consistency_relaxed": False if r02c_applicable else True,
        },
    }

    contract_verdict = {
        "contract_name": "EXP_RINGDOWN_02_recovery_robustness",
        "run": args.run,
        "stage": args.stage_name,
        "created": utc_now_iso(),
        "overall_verdict": overall_verdict,
        "contracts": contracts,
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    contract_path = outputs_dir / "contract_verdict.json"
    robustness_path = outputs_dir / "robustness_summary.json"

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_verdict, f, indent=2)
        f.write("\n")
    with open(robustness_path, "w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)
        f.write("\n")

    stage_summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "inputs": {
            "RUN_VALID": "RUN_VALID/stage_summary.json",
            "synthetic_events": str(events_path),
            "exp01_contract": str(exp01_contract_path),
            "exp01_cases": str(exp01_cases_path) if exp01_cases_path else None,
        },
        "parameters": {
            "scales": scales,
            "dt_samples": dt_samples,
            "theta_keys": theta_keys,
            **thr.__dict__,
        },
        "results": {
            "overall_verdict": overall_verdict,
            "contracts": [{"id": c["id"], "verdict": c["verdict"]} for c in contracts],
        },
    }
    summary_path = write_stage_summary(stage_dir, stage_summary)

    manifest_path = write_manifest(
        stage_dir,
        {
            "contract_verdict": contract_path,
            "robustness_summary": robustness_path,
            "per_case": per_case_path,
            "stage_summary": summary_path,
        },
        extra={"version": "1"},
    )

    if overall_verdict != "PASS" and args.fail_fast:
        abort_contract("EXP_RINGDOWN_02 FAIL")

    print("OK: EXP_RINGDOWN_02 PASS")
    print(f"  outputs: {outputs_dir}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
