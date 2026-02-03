#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

import numpy as np

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    get_run_dir,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)


EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_05__prior_hyperparam_sweep"
DEFAULT_TMAX = 0.08
DEFAULT_ENV_THR = 0.30
DEFAULT_BAND = (20.0, 500.0)
DEFAULT_MIN_CASES = 24


@dataclass(frozen=True)
class RecoverConfig:
    config_id: str
    tmax: float
    env_thr: float
    band: Tuple[float, float]


def abort_contract(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def percentile(xs: List[float], q: float) -> float:
    xs2 = sorted(xs)
    if not xs2:
        return float("nan")
    k = (len(xs2) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs2) - 1)
    a = k - lo
    return xs2[lo] * (1 - a) + xs2[hi] * a


def _snr_key(x: float) -> str:
    return f"{float(x):.3f}"


def load_seed_snr_to_strain(run_dir: Path) -> Tuple[Dict[Tuple[int, str], str], Path]:
    idx = run_dir / "ringdown_synth" / "outputs" / "synthetic_events_list.json"
    if not idx.exists():
        raise SystemExit(f"ERROR: missing {idx}")

    with open(idx, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise SystemExit("ERROR: synthetic_events_list.json invalid/empty")

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
            collisions.append({"key": {"seed": int(seed), "snr": _snr_key(float(snr))}, "a": out[k], "b": rel})
        out[k] = rel

    if collisions:
        raise SystemExit(f"ERROR: synthetic_events_list has collisions for (seed,snr): {collisions[:3]}")

    return out, idx


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


def _resolve_strain_path(run_dir: Path, rel: str) -> Optional[Path]:
    candidate = Path(rel)
    if candidate.is_absolute() or str(rel).startswith("runs/"):
        resolved = candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
    else:
        resolved = (run_dir / "ringdown_synth" / "outputs" / candidate).resolve()
    try:
        assert_within_runs(run_dir, resolved)
    except ValueError:
        return None
    return resolved if resolved.exists() else None


def _load_strain(path: Path) -> Tuple[np.ndarray, float, np.ndarray]:
    with np.load(path) as data:
        if "strain" in data:
            strain = np.asarray(data["strain"], dtype=float)
        elif "h" in data:
            strain = np.asarray(data["h"], dtype=float)
        else:
            raise ValueError("strain key missing (expected 'strain' or 'h')")
        if "t" in data:
            t = np.asarray(data["t"], dtype=float)
            if t.ndim != 1 or len(t) < 2:
                raise ValueError("t array invalid")
            dt = float(np.median(np.diff(t)))
        elif "dt" in data:
            dt = float(np.asarray(data["dt"]).reshape(-1)[0])
            t = dt * np.arange(strain.size, dtype=float)
        else:
            dt = 1.0 / 4096.0
            t = dt * np.arange(strain.size, dtype=float)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt invalid")
    strain = np.asarray(strain, dtype=float).reshape(-1)
    if strain.size < 8 or not np.all(np.isfinite(strain)):
        raise ValueError("strain invalid")
    return strain, dt, t


def _recover_ringdown(strain_path: Path, cfg: RecoverConfig) -> Dict[str, float]:
    strain, dt, t = _load_strain(strain_path)
    strain = strain - float(np.mean(strain))

    if cfg.tmax <= 0:
        raise ValueError("tmax must be positive")

    freqs = np.fft.rfftfreq(strain.size, d=dt)
    nyquist = float(freqs[-1]) if freqs.size else 0.0
    band_lo = float(cfg.band[0])
    band_hi = min(float(cfg.band[1]), nyquist)
    if not np.isfinite(band_lo) or not np.isfinite(band_hi) or band_hi <= band_lo:
        raise ValueError("invalid band limits")

    band = (freqs >= band_lo) & (freqs <= band_hi)
    if not np.any(band):
        raise ValueError("band empty for FFT")

    spectrum = np.fft.rfft(strain)
    mag = np.abs(spectrum)
    f_hat = float(freqs[band][int(np.argmax(mag[band]))])

    i0 = int(np.argmax(np.abs(strain)))
    n = strain.size
    i1 = min(n, i0 + int(max(128, round(cfg.tmax / dt))))
    xw = strain[i0:i1]
    tw = t[i0:i1]
    if xw.size < 128:
        raise ValueError("window too short for tau fit")

    nw = xw.size
    Xfw = np.fft.fft(xw)
    hw = np.zeros(nw)
    if nw % 2 == 0:
        hw[0] = 1.0
        hw[nw // 2] = 1.0
        hw[1 : nw // 2] = 2.0
    else:
        hw[0] = 1.0
        hw[1 : (nw + 1) // 2] = 2.0
    xa = np.fft.ifft(Xfw * hw)

    tt = tw - float(tw[0])
    z = xa * np.exp(-1j * 2.0 * np.pi * f_hat * tt)
    env = np.abs(z).astype(float)
    env_max = float(np.max(env))
    if not np.isfinite(env_max) or env_max <= 0:
        raise ValueError("invalid envelope")

    base_mask = tt <= cfg.tmax
    thr = env_max * cfg.env_thr
    mask = base_mask & (env >= thr)
    if np.count_nonzero(mask) < 32:
        raise ValueError("insufficient samples above env_thr")

    yy = np.log(env[mask])
    xx = tt[mask]
    b, _a = np.polyfit(xx, yy, deg=1)
    if not np.isfinite(b) or b >= 0:
        raise ValueError("non-negative slope in log envelope")
    tau_hat = float(-1.0 / b)
    q_hat = float(np.pi * f_hat * tau_hat)

    return {"f_220_hat": f_hat, "tau_220_hat": tau_hat, "Q_220_hat": q_hat}


def _default_configs() -> List[RecoverConfig]:
    return [
        RecoverConfig("cfg_00_baseline", DEFAULT_TMAX, DEFAULT_ENV_THR, DEFAULT_BAND),
        RecoverConfig("cfg_01_env_thr_hi", DEFAULT_TMAX, 0.40, DEFAULT_BAND),
        RecoverConfig("cfg_02_env_thr_lo", DEFAULT_TMAX, 0.15, DEFAULT_BAND),
        RecoverConfig("cfg_03_tmax_short", 0.05, DEFAULT_ENV_THR, DEFAULT_BAND),
        RecoverConfig("cfg_04_tmax_long", 0.12, DEFAULT_ENV_THR, DEFAULT_BAND),
        RecoverConfig("cfg_05_band_tight", DEFAULT_TMAX, DEFAULT_ENV_THR, (30.0, 400.0)),
        RecoverConfig("cfg_06_band_wide", DEFAULT_TMAX, DEFAULT_ENV_THR, (20.0, 800.0)),
    ]


def _metric_delta(val: float, base: float) -> float:
    if not (math.isfinite(val) and math.isfinite(base)):
        return float("nan")
    return float(val - base)


def _score_config(deltas: Dict[str, float]) -> float:
    values = [v for v in deltas.values() if math.isfinite(v)]
    return max(values) if values else float("nan")


def _run_valid_path(run_dir: Path) -> Path:
    preferred = run_dir / "RUN_VALID" / "verdict.json"
    legacy = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    if preferred.exists():
        return preferred
    return legacy


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EXP_RINGDOWN_05 prior/hyperparam sweep (contract-first)",
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES)
    ap.add_argument("--delta-p50-tau-max", type=float, default=0.02)
    ap.add_argument("--delta-p90-tau-max", type=float, default=0.05)
    ap.add_argument("--delta-p50-f-max", type=float, default=0.01)
    ap.add_argument("--delta-p90-f-max", type=float, default=0.02)
    ap.add_argument("--fail-rate-cap-abs", type=float, default=0.10)
    ap.add_argument("--fail-rate-cap-delta", type=float, default=0.05)
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)
    run_dir = get_run_dir(args.run, base_dir=out_root).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    seed_snr2strain, synth_idx_path = load_seed_snr_to_strain(run_dir)

    recovery_cases_path = (
        run_dir
        / "experiment"
        / "ringdown_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    if not recovery_cases_path.exists():
        abort_contract(f"missing recovery_cases.jsonl at {recovery_cases_path}")

    cases: List[Dict[str, Any]] = []
    n_mapped = 0
    n_unmapped = 0
    for idx, row in enumerate(_read_jsonl(recovery_cases_path)):
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
        n_mapped += 1
        cases.append(
            {
                "case_id": case_id,
                "truth": {"f_220": float(f0), "tau_220": float(tau0)},
                "strain_path": strain_path,
            }
        )

    if len(cases) < int(args.min_cases):
        abort_contract(
            "recovery_cases.jsonl has fewer than min_cases usable entries: "
            f"{len(cases)} < {args.min_cases}"
        )

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    per_case_path = outputs_dir / "per_case.jsonl"
    sweep_path = outputs_dir / "prior_sweep.json"
    contract_path = outputs_dir / "contract_verdict.json"

    configs = _default_configs()
    metrics_by_cfg: Dict[str, Dict[str, Any]] = {}

    with open(per_case_path, "w", encoding="utf-8") as f:
        for cfg in configs:
            err_f_abs: List[float] = []
            err_tau_abs: List[float] = []
            n_effective = 0
            for case in cases:
                case_id = case["case_id"]
                truth = case["truth"]
                result: Dict[str, Any] = {
                    "config_id": cfg.config_id,
                    "config": {
                        "tmax": cfg.tmax,
                        "env_thr": cfg.env_thr,
                        "band": list(cfg.band),
                    },
                    "case_id": case_id,
                    "truth": truth,
                }
                try:
                    est = _recover_ringdown(case["strain_path"], cfg)
                except Exception as exc:
                    result.update({"status": "FAIL", "reason": str(exc)})
                else:
                    f_hat = float(est["f_220_hat"])
                    tau_hat = float(est["tau_220_hat"])
                    err_f = (f_hat - truth["f_220"]) / truth["f_220"]
                    err_tau = (tau_hat - truth["tau_220"]) / truth["tau_220"]
                    result.update(
                        {
                            "status": "OK",
                            "estimate": est,
                            "errors": {
                                "err_f_rel": float(err_f),
                                "err_tau_rel": float(err_tau),
                            },
                        }
                    )
                    err_f_abs.append(abs(float(err_f)))
                    err_tau_abs.append(abs(float(err_tau)))
                    n_effective += 1
                f.write(json.dumps(result) + "\n")

            n_total = len(cases)
            fail_rate = (n_total - n_effective) / n_total if n_total else 1.0
            metrics_by_cfg[cfg.config_id] = {
                "config_id": cfg.config_id,
                "config": {
                    "tmax": cfg.tmax,
                    "env_thr": cfg.env_thr,
                    "band": list(cfg.band),
                },
                "n_total": n_total,
                "n_effective": n_effective,
                "fail_rate": float(fail_rate),
                "bias_abs_p50_f": float(percentile(err_f_abs, 0.50)),
                "bias_abs_p90_f": float(percentile(err_f_abs, 0.90)),
                "bias_abs_p50_tau": float(percentile(err_tau_abs, 0.50)),
                "bias_abs_p90_tau": float(percentile(err_tau_abs, 0.90)),
            }

    baseline = metrics_by_cfg["cfg_00_baseline"]
    for cfg in configs:
        m = metrics_by_cfg[cfg.config_id]
        m["delta"] = {
            "p50_f": _metric_delta(m["bias_abs_p50_f"], baseline["bias_abs_p50_f"]),
            "p90_f": _metric_delta(m["bias_abs_p90_f"], baseline["bias_abs_p90_f"]),
            "p50_tau": _metric_delta(m["bias_abs_p50_tau"], baseline["bias_abs_p50_tau"]),
            "p90_tau": _metric_delta(m["bias_abs_p90_tau"], baseline["bias_abs_p90_tau"]),
            "fail_rate": _metric_delta(m["fail_rate"], baseline["fail_rate"]),
        }

    ranking = []
    for cfg in configs:
        deltas = metrics_by_cfg[cfg.config_id]["delta"]
        score = _score_config(
            {
                "p50_f": deltas["p50_f"],
                "p90_f": deltas["p90_f"],
                "p50_tau": deltas["p50_tau"],
                "p90_tau": deltas["p90_tau"],
            }
        )
        ranking.append({"config_id": cfg.config_id, "score": score})
    ranking_sorted = sorted(ranking, key=lambda r: (float("inf") if not math.isfinite(r["score"]) else -r["score"]))
    worst_case = ranking_sorted[0] if ranking_sorted else {}

    sweep_payload = {
        "stage": STAGE_NAME,
        "baseline": "cfg_00_baseline",
        "configs": [metrics_by_cfg[cfg.config_id] for cfg in configs],
        "ranking": ranking_sorted,
        "worst_case": worst_case,
    }

    sweep_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

    violations: List[str] = []
    contracts: Dict[str, Any] = {}

    min_cases_effective = baseline["n_effective"] >= int(args.min_cases)
    if not min_cases_effective:
        violations.append(
            "baseline effective cases below min_cases: "
            f"{baseline['n_effective']} < {args.min_cases}"
        )

    bounds_sensitivity = {
        "delta_p50_tau_max": float(args.delta_p50_tau_max),
        "delta_p90_tau_max": float(args.delta_p90_tau_max),
        "delta_p50_f_max": float(args.delta_p50_f_max),
        "delta_p90_f_max": float(args.delta_p90_f_max),
    }
    sensitivity_failures: List[Dict[str, Any]] = []
    for cfg in configs:
        deltas = metrics_by_cfg[cfg.config_id]["delta"]
        checks = {
            "delta_p50_tau": deltas["p50_tau"],
            "delta_p90_tau": deltas["p90_tau"],
            "delta_p50_f": deltas["p50_f"],
            "delta_p90_f": deltas["p90_f"],
        }
        for key, val in checks.items():
            bound_key = key + "_max"
            bound = bounds_sensitivity[bound_key]
            if not math.isfinite(val) or val > bound:
                sensitivity_failures.append(
                    {
                        "config_id": cfg.config_id,
                        "metric": key,
                        "value": val,
                        "bound": bound,
                    }
                )
                break

    sensitivity_verdict = "PASS" if min_cases_effective and not sensitivity_failures else "FAIL"

    contracts["R05_PRIOR_SENSITIVITY_BOUNDED"] = {
        "verdict": sensitivity_verdict,
        "bounds": bounds_sensitivity,
        "worst_case": worst_case,
        "details": {
            "failures": sensitivity_failures,
        },
    }

    fail_rate_cap = min(
        float(args.fail_rate_cap_abs),
        float(baseline["fail_rate"]) + float(args.fail_rate_cap_delta),
    )
    fail_rate_failures: List[Dict[str, Any]] = []
    for cfg in configs:
        fail_rate = metrics_by_cfg[cfg.config_id]["fail_rate"]
        if not math.isfinite(fail_rate) or fail_rate > fail_rate_cap:
            fail_rate_failures.append(
                {
                    "config_id": cfg.config_id,
                    "fail_rate": fail_rate,
                    "cap": fail_rate_cap,
                }
            )

    contracts["R05_FAILURE_MODE_CAP"] = {
        "verdict": "PASS" if min_cases_effective and not fail_rate_failures else "FAIL",
        "bounds": {
            "fail_rate_cap_abs": float(args.fail_rate_cap_abs),
            "fail_rate_cap_delta": float(args.fail_rate_cap_delta),
        },
        "worst_case": worst_case,
    }

    if sensitivity_failures:
        violations.append("prior sensitivity bounds exceeded")
    if fail_rate_failures:
        violations.append("fail_rate exceeded cap")

    overall_verdict = "PASS" if not violations and all(c["verdict"] == "PASS" for c in contracts.values()) else "FAIL"

    run_valid_path = _run_valid_path(run_dir)
    contract_payload = {
        "verdict": overall_verdict,
        "contracts": contracts,
        "inputs": {
            "RUN_VALID": {
                "path": str(run_valid_path),
                "sha256": sha256_file(run_valid_path),
            },
            "synthetic_events_list": {
                "path": str(synth_idx_path),
                "sha256": sha256_file(synth_idx_path),
            },
            "recovery_cases": {
                "path": str(recovery_cases_path),
                "sha256": sha256_file(recovery_cases_path),
            },
        },
        "outputs": {
            "prior_sweep": str(sweep_path.relative_to(stage_dir)),
            "per_case": str(per_case_path.relative_to(stage_dir)),
            "contract_verdict": str(contract_path.relative_to(stage_dir)),
        },
        "violations": violations,
    }

    contract_path.write_text(json.dumps(contract_payload, indent=2), encoding="utf-8")

    summary = {
        "results": {
            "overall_verdict": overall_verdict,
            "n_cases": len(cases),
            "baseline_effective": baseline["n_effective"],
            "n_mapped": n_mapped,
            "n_unmapped": n_unmapped,
        },
        "artifacts": {
            "prior_sweep": str(sweep_path.relative_to(stage_dir)),
            "per_case": str(per_case_path.relative_to(stage_dir)),
            "contract_verdict": str(contract_path.relative_to(stage_dir)),
        },
    }
    write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "prior_sweep": sweep_path,
            "per_case": per_case_path,
            "contract_verdict": contract_path,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
