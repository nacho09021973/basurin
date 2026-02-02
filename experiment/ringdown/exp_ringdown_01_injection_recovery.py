# experiment/ringdown/exp_ringdown_01_injection_recovery.py
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
from typing import Any, Dict, List, Optional

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

import numpy as np

from basurin_io import (
    resolve_out_root,
    validate_run_id,
    ensure_stage_dirs,
    write_manifest,
    write_stage_summary,
    require_run_valid,
)


def abort_contract(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_run_valid(out_root: Path, run_id: str) -> Dict[str, Any]:
    try:
        return require_run_valid(out_root, run_id)
    except Exception as e:
        abort_contract(str(e))

def percentile(xs: List[float], q: float) -> float:
    xs2 = sorted(xs)
    if not xs2:
        return float("nan")
    k = (len(xs2) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs2) - 1)
    a = k - lo
    return xs2[lo] * (1 - a) + xs2[hi] * a


def in_interval(x: float, lo: float, hi: float) -> bool:
    return (x >= lo) and (x <= hi)


def recover_ringdown(strain_npz_path: Path) -> Dict[str, Any]:
    """
    Recuperador determinista mínimo (contract-first).

    Estimación:
      - f_220: pico espectral rFFT en banda 20..500 Hz.
      - tau_220: señal analítica (Hilbert por FFT) en ventana post-t0,
        demodulación a banda base con f_hat y ajuste lineal temprano de log(envolvente).

    Notas:
      - Sin aleatoriedad.
      - Evita cola/noise mediante umbral de envolvente y ventana temporal.
    """
    data = np.load(strain_npz_path)
    keys = set(data.files)

    # serie
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
            abort_contract(f"strain_npz sin serie 1D utilizable: {strain_npz_path}")

    # tiempo / dt
    if "t" in keys:
        t = np.asarray(data["t"], dtype=float)
        if t.shape != x.shape:
            abort_contract(f"t shape != strain shape en {strain_npz_path}")
        dt = float(np.median(np.diff(t)))
    elif "dt" in keys:
        dt = float(np.asarray(data["dt"]).reshape(-1)[0])
        t = dt * np.arange(x.size, dtype=float)
    else:
        dt = 1.0 / 4096.0
        t = dt * np.arange(x.size, dtype=float)

    if not np.isfinite(dt) or dt <= 0:
        abort_contract(f"dt inválido en {strain_npz_path}")

    x = x - float(np.mean(x))

    # f_hat
    freqs = np.fft.rfftfreq(x.size, d=dt)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    band = (freqs >= 20.0) & (freqs <= 500.0)
    if not np.any(band):
        abort_contract("banda 20..500 Hz vacía (n demasiado pequeño)")
    f_hat = float(freqs[band][int(np.argmax(mag[band]))])

    # --- estimar tau: demodulación a f_hat + fit temprano (determinista) ---
    i0 = int(np.argmax(np.abs(x)))
    n = x.size
    # ventana corta (tau real ~0.02s => con 0.12s hay varias e-folds sin caer a noise)
    i1 = min(n, i0 + int(max(128, round(0.12 / dt))))
    xw = x[i0:i1]
    tw = t[i0:i1]
    if xw.size < 128:
        abort_contract("ventana post-t0 demasiado corta para estimar tau")

    # Hilbert por FFT en ventana
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
    xa = np.fft.ifft(Xfw * hw)  # señal analítica

    # demodular a banda base usando f_hat (ya estimado arriba)
    tt = (tw - float(tw[0]))
    z = xa * np.exp(-1j * 2.0 * np.pi * f_hat * tt)
    env = np.abs(z).astype(float)

    env_max = float(np.max(env))
    if not np.isfinite(env_max) or env_max <= 0:
        abort_contract("envolvente inválida tras demodulación")

    # fit temprano: limitar tiempo y evitar cola
    tmax = 0.08  # s (determinista)
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


@dataclass(frozen=True)
class Thresholds:
    min_cases: int
    bias_p50_max: float
    bias_p90_max: float
    cov68_min: float
    cov68_max: float
    cov95_min: float


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_01 injection–recovery gate (contract-first)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--events-json", required=True, help="Path to ringdown_synth outputs/synthetic_events.json")
    ap.add_argument("--min-cases", type=int, default=24)
    ap.add_argument("--bias-p50-max", type=float, default=0.03)
    ap.add_argument("--bias-p90-max", type=float, default=0.08)
    ap.add_argument("--cov68-min", type=float, default=0.55)
    ap.add_argument("--cov68-max", type=float, default=0.85)
    ap.add_argument("--cov95-min", type=float, default=0.85)
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)

    run_dir = (out_root / args.run).resolve()
    run_valid = check_run_valid(out_root, args.run)

    stage_name = "experiment/ringdown_01_injection_recovery"
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name, base_dir=out_root)

    events_path = Path(args.events_json)
    if not events_path.is_absolute():
        events_path = (Path.cwd() / events_path).resolve()

    # Enforce within runs/<run_id> (read-only input, but still: no escapes)
    expected_prefix = (run_dir / "ringdown_synth" / "outputs").resolve()
    try:
        events_path.relative_to(expected_prefix)
    except ValueError:
        abort_contract(f"events-json must be under {expected_prefix}, got {events_path}")

    events = read_json(events_path)
    if not isinstance(events, list) or not events:
        abort_contract(f"events-json invalid or empty list: {events_path}")

    thr = Thresholds(
        min_cases=int(args.min_cases),
        bias_p50_max=float(args.bias_p50_max),
        bias_p90_max=float(args.bias_p90_max),
        cov68_min=float(args.cov68_min),
        cov68_max=float(args.cov68_max),
        cov95_min=float(args.cov95_min),
    )

    cases_path = outputs_dir / "recovery_cases.jsonl"
    summary_path = outputs_dir / "recovery_summary.json"
    contract_path = outputs_dir / "contract_verdict.json"
    contract_compat_path = outputs_dir / "recovery_contract.json"

    err_f_abs: List[float] = []
    err_tau_abs: List[float] = []
    cov68_f_hits = cov68_tau_hits = 0
    cov95_f_hits = cov95_tau_hits = 0
    cov68_n = cov95_n = 0
    n_effective = 0

    with open(cases_path, "w", encoding="utf-8") as f:
        for ev in events:
            case_id = str(ev.get("case_id") or "")
            snr = ev.get("snr")
            seed = ev.get("seed")
            truth = ev.get("truth") or {}
            f0 = truth.get("f_220")
            tau0 = truth.get("tau_220")

            rel_strain = ev.get("strain_npz") or ev.get("strain_path")
            if not rel_strain:
                row = {"case_id": case_id, "status": "SKIP", "reason": "missing strain_npz", "snr": snr, "seed": seed}
                f.write(json.dumps(row) + "\n")
                continue

            strain_path = (expected_prefix / rel_strain).resolve() if not str(rel_strain).startswith("runs/") else (Path.cwd() / rel_strain).resolve()
            # common layout: expected_prefix/cases/<case_id>/strain.npz
            if not strain_path.exists():
                row = {"case_id": case_id, "status": "SKIP", "reason": "strain_npz not found", "path": str(strain_path)}
                f.write(json.dumps(row) + "\n")
                continue

            if f0 is None or tau0 is None:
                row = {"case_id": case_id, "status": "SKIP", "reason": "missing truth f_220/tau_220"}
                f.write(json.dumps(row) + "\n")
                continue

            est = recover_ringdown(strain_path)
            fhat = float(est["f_220_hat"])
            tauhat = float(est["tau_220_hat"])

            err_f = (fhat - float(f0)) / float(f0)
            err_tau = (tauhat - float(tau0)) / float(tau0)

            row = {
                "case_id": case_id,
                "status": "OK",
                "snr": snr,
                "seed": seed,
                "truth": {"f_220": f0, "tau_220": tau0, "Q_220": truth.get("Q_220")},
                "estimate": {"f_220_hat": fhat, "tau_220_hat": tauhat, "Q_220_hat": est.get("Q_220_hat")},
                "errors": {"err_f_rel": err_f, "err_tau_rel": err_tau},
                "ci68": est.get("ci68"),
                "ci95": est.get("ci95"),
            }

            # coverage (optional)
            ci68 = est.get("ci68") or {}
            ci95 = est.get("ci95") or {}
            if isinstance(ci68, dict) and "f_220" in ci68 and "tau_220" in ci68:
                cov68_n += 1
                cov68_f_hits += 1 if in_interval(float(f0), float(ci68["f_220"][0]), float(ci68["f_220"][1])) else 0
                cov68_tau_hits += 1 if in_interval(float(tau0), float(ci68["tau_220"][0]), float(ci68["tau_220"][1])) else 0
            if isinstance(ci95, dict) and "f_220" in ci95 and "tau_220" in ci95:
                cov95_n += 1
                cov95_f_hits += 1 if in_interval(float(f0), float(ci95["f_220"][0]), float(ci95["f_220"][1])) else 0
                cov95_tau_hits += 1 if in_interval(float(tau0), float(ci95["tau_220"][0]), float(ci95["tau_220"][1])) else 0

            err_f_abs.append(abs(float(err_f)))
            err_tau_abs.append(abs(float(err_tau)))
            n_effective += 1

            f.write(json.dumps(row) + "\n")

    # aggregate
    summary: Dict[str, Any] = {
        "n_total": len(events),
        "n_effective": n_effective,
        "bias_abs": {
            "f_220": {"p50": percentile(err_f_abs, 0.50), "p90": percentile(err_f_abs, 0.90)},
            "tau_220": {"p50": percentile(err_tau_abs, 0.50), "p90": percentile(err_tau_abs, 0.90)},
        },
        "coverage": {
            "has_ci68": cov68_n > 0,
            "has_ci95": cov95_n > 0,
            "ci68": {
                "n": cov68_n,
                "f_220": (cov68_f_hits / cov68_n) if cov68_n else None,
                "tau_220": (cov68_tau_hits / cov68_n) if cov68_n else None,
            },
            "ci95": {
                "n": cov95_n,
                "f_220": (cov95_f_hits / cov95_n) if cov95_n else None,
                "tau_220": (cov95_tau_hits / cov95_n) if cov95_n else None,
            },
        },
    }

    # contract evaluation (coverage enforced only if available)
    fail_reasons: List[str] = []
    if n_effective < thr.min_cases:
        fail_reasons.append(f"n_effective<{thr.min_cases} (got {n_effective})")

    b50f = summary["bias_abs"]["f_220"]["p50"]
    b90f = summary["bias_abs"]["f_220"]["p90"]
    b50t = summary["bias_abs"]["tau_220"]["p50"]
    b90t = summary["bias_abs"]["tau_220"]["p90"]

    if not (b50f <= thr.bias_p50_max and b50t <= thr.bias_p50_max):
        fail_reasons.append("bias_p50_exceeds_threshold")
    if not (b90f <= thr.bias_p90_max and b90t <= thr.bias_p90_max):
        fail_reasons.append("bias_p90_exceeds_threshold")

    cov = summary["coverage"]
    if cov["has_ci68"]:
        c68f = cov["ci68"]["f_220"]
        c68t = cov["ci68"]["tau_220"]
        if not (thr.cov68_min <= c68f <= thr.cov68_max and thr.cov68_min <= c68t <= thr.cov68_max):
            fail_reasons.append("ci68_coverage_out_of_band")
    if cov["has_ci95"]:
        c95f = cov["ci95"]["f_220"]
        c95t = cov["ci95"]["tau_220"]
        if not (c95f >= thr.cov95_min and c95t >= thr.cov95_min):
            fail_reasons.append("ci95_coverage_below_min")

    verdict = "PASS" if not fail_reasons else "FAIL"

    contract = {
        "overall_verdict": verdict,
        "thresholds": thr.__dict__,
        "fail_reasons": fail_reasons,
        "inputs": {
            "events_json": str(events_path),
            "run_valid_summary": "RUN_VALID/stage_summary.json",
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    with open(contract_compat_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    stage_summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "inputs": {
            "RUN_VALID": "RUN_VALID/stage_summary.json",
            "events_json": str(events_path),
        },
        "parameters": {
            **thr.__dict__,
        },
        "results": {
            "overall_verdict": verdict,
        },
    }
    summary_written = write_stage_summary(stage_dir, stage_summary)
    write_manifest(
        stage_dir,
        {
            "cases": cases_path,
            "summary": summary_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    if verdict != "PASS":
        abort_contract(f"EXP_RINGDOWN_01 FAIL: {fail_reasons}")

    print("OK: EXP_RINGDOWN_01 PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
