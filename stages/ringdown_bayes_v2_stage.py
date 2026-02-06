#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1], _here.parents[2]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (  # noqa: E402
    assert_within_runs,
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2
STAGE_VERSION = "2.0"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _damped_model(t: np.ndarray, p: np.ndarray, t_ref: float, with_slope: bool) -> np.ndarray:
    A, tau, f, phi, c0 = p[:5]
    base = A * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi) + c0
    if with_slope:
        return base + p[5] * (t - t_ref)
    return base


def _residuals(p: np.ndarray, t: np.ndarray, y: np.ndarray, t_ref: float, with_slope: bool) -> np.ndarray:
    return _damped_model(t, p, t_ref, with_slope) - y


def _initial_guess(t: np.ndarray, y: np.ndarray, priors: dict[str, list[float]], with_slope: bool) -> np.ndarray:
    n = int(y.size)
    c0 = float(np.mean(y))
    amp = float(np.max(np.abs(y - c0))) if n > 0 else 0.0
    A0 = min(max(amp, 1e-8), priors["A"][1])
    tau0 = float(min(max(0.02, priors["tau"][0] * 1.1), priors["tau"][1] * 0.9))
    if n > 1 and t[-1] > t[0]:
        dt = float(np.mean(np.diff(t)))
        if dt > 0:
            freqs = np.fft.rfftfreq(n, dt)
            mag = np.abs(np.fft.rfft(y - c0))
            idx = int(np.argmax(mag))
            f0 = float(freqs[idx])
        else:
            f0 = 100.0
    else:
        f0 = 100.0
    f0 = float(min(max(f0, priors["f"][0]), priors["f"][1]))
    phi0 = 0.0
    c00 = float(min(max(c0, priors["C0"][0]), priors["C0"][1]))
    if with_slope:
        return np.array([A0, tau0, f0, phi0, c00, 0.0], dtype=float)
    return np.array([A0, tau0, f0, phi0, c00], dtype=float)


def _fit_and_logz(
    t: np.ndarray,
    y: np.ndarray,
    priors: dict[str, list[float]],
    t_ref: float,
    with_slope: bool,
    sigma_mode: str,
) -> dict[str, Any]:
    names = ["A", "tau", "f", "phi", "C0"] + (["C1"] if with_slope else [])
    lower = np.array([priors[n][0] for n in names], dtype=float)
    upper = np.array([priors[n][1] for n in names], dtype=float)
    p0 = _initial_guess(t, y, priors, with_slope)

    res = least_squares(
        _residuals,
        p0,
        bounds=(lower, upper),
        args=(t, y, t_ref, with_slope),
        method="trf",
        loss="linear",
        jac="2-point",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )

    if not res.success and res.status not in (1, 2, 3, 4):
        raise RuntimeError(f"least_squares failed: {res.message}")

    r = np.asarray(res.fun, dtype=float)
    J = np.asarray(res.jac, dtype=float)
    n = int(y.size)
    k = int(len(names))
    dof = int(n - k)
    ss_res = float(np.sum(r * r))

    if sigma_mode == "fixed_1":
        sigma2 = 1.0
        # For whitened/normalized residuals: p(y|theta) ∝ exp(-0.5 * ss_res)
        logL_hat = float(-0.5 * ss_res)
    else:
        if dof <= 0:
            raise RuntimeError("dof <= 0 for sigma2 estimation")
        sigma2 = float(ss_res / dof)
        if sigma2 <= 0 or not math.isfinite(sigma2):
            raise RuntimeError("invalid sigma2 estimate")
        # Gaussian iid with estimated sigma2.
        logL_hat = float(-0.5 * (ss_res / sigma2 + n * math.log(2.0 * math.pi * sigma2)))

    H = (J.T @ J) / sigma2
    pd_ok = False
    logdet_H = None
    try:
        L = np.linalg.cholesky(H)
        pd_ok = True
        logdet_H = float(2.0 * np.sum(np.log(np.diag(L))))
    except Exception:
        eig = np.linalg.eigvalsh((H + H.T) / 2.0)
        if np.all(eig > 1e-12):
            pd_ok = True
            sign, ld = np.linalg.slogdet(H)
            if sign <= 0:
                raise RuntimeError("Hessian slogdet sign <= 0")
            logdet_H = float(ld)
        else:
            raise RuntimeError("Hessian not PD")

    ranges = [priors[n][1] - priors[n][0] for n in names]
    if not all(rng > 0 for rng in ranges):
        raise RuntimeError("invalid prior ranges")
    log_prior_volume = float(sum(math.log(float(rng)) for rng in ranges))

    logZ = float(logL_hat + (k / 2.0) * math.log(2.0 * math.pi) - 0.5 * logdet_H - log_prior_volume)
    if not (_is_finite(logZ) and _is_finite(logL_hat) and _is_finite(logdet_H)):
        raise RuntimeError("non-finite evidence quantities")

    return {
        "logZ": logZ,
        "logL_hat": float(logL_hat),
        "k": k,
        "log_prior_volume": float(log_prior_volume),
        "logdet_H": float(logdet_H),
        "pd_ok": bool(pd_ok),
        "sigma2": float(sigma2),
        "dof": dof,
        "n": n,
        "ss_res": float(ss_res),
    }


def _load_window_arrays(det: str, det_payload: dict[str, Any], run_dir: Path) -> tuple[np.ndarray, np.ndarray, str | None]:
    window = det_payload.get("window")
    if isinstance(window, dict) and isinstance(window.get("t"), list) and isinstance(window.get("y"), list):
        t = np.asarray(window["t"], dtype=float)
        y = np.asarray(window["y"], dtype=float)
        return t, y, None

    npz_rel = det_payload.get("window_npz_relpath")
    if isinstance(npz_rel, str) and npz_rel:
        npz_path = (run_dir / npz_rel).resolve()
        assert_within_runs(run_dir, npz_path)
        if not npz_path.exists():
            raise RuntimeError(f"missing window npz for detector {det}: {npz_rel}")
        npz = np.load(npz_path)
        if "t" not in npz or "y" not in npz:
            raise RuntimeError(f"window npz missing t/y for detector {det}: {npz_rel}")
        return np.asarray(npz["t"], dtype=float), np.asarray(npz["y"], dtype=float), npz_rel

    raise RuntimeError(
        f"missing window arrays for detector {det}; require canonical window npz path in inference_report"
    )


def _get_created_utc(run_valid: dict[str, Any], report: dict[str, Any]) -> str:
    for payload in (report, run_valid):
        for key in ("created_utc", "created", "timestamp", "timestamp_utc"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return "1970-01-01T00:00:00+00:00"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BASURIN ringdown Bayes v2 (Laplace)")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--inference-report-relpath", default="ringdown_real_inference_v0/outputs/inference_report.json")
    parser.add_argument("--out-stage", default="ringdown_bayes_v2")
    args = parser.parse_args(argv)

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, args.out_stage, base_dir=out_root)
    assert_within_runs(run_dir, stage_dir)
    assert_within_runs(run_dir, outputs_dir)

    reasons: list[str] = []
    verdict = "PASS"
    bayes_payload: dict[str, Any] | None = None
    output_path = outputs_dir / "bayes_v2.json"

    inputs: list[dict[str, str]] = []

    run_valid_rel = Path("RUN_VALID") / "outputs" / "run_valid.json"
    run_valid_path = (run_dir / run_valid_rel).resolve()
    assert_within_runs(run_dir, run_valid_path)
    if run_valid_path.exists():
        inputs.append({"path": str(run_valid_rel), "sha256": sha256_file(run_valid_path)})
    else:
        reasons.append(f"missing required input: {run_valid_rel}")

    inf_rel = Path(args.inference_report_relpath)
    inf_path = (run_dir / inf_rel).resolve()
    assert_within_runs(run_dir, inf_path)
    if inf_path.exists():
        inputs.append({"path": str(inf_rel), "sha256": sha256_file(inf_path)})
    else:
        reasons.append(f"missing required input: {inf_rel}")

    run_valid: dict[str, Any] = {}
    report: dict[str, Any] = {}
    if not reasons:
        run_valid = json.loads(run_valid_path.read_text(encoding="utf-8"))
        rv = run_valid.get("overall_verdict") or run_valid.get("verdict") or run_valid.get("status")
        if rv != "PASS":
            reasons.append(f"RUN_VALID != PASS (got {rv})")

    if not reasons:
        report = json.loads(inf_path.read_text(encoding="utf-8"))
        per_detector = report.get("per_detector")
        if not isinstance(per_detector, dict) or not per_detector:
            reasons.append("inference_report missing per_detector map")

    priors = {
        "A": [0.0, 1.0],
        "tau": [1e-4, 0.2],
        "f": [0.0, 2048.0],
        "phi": [-math.pi, math.pi],
        "C0": [-1.0, 1.0],
        "C1": [-50.0, 50.0],
    }
    t_ref = 0.0

    if not reasons:
        per_detector_res: dict[str, Any] = {}
        ok_detectors: list[str] = []
        detector_bfs: list[float] = []

        assert isinstance(report["per_detector"], dict)
        for det in sorted(report["per_detector"].keys()):
            det_payload = report["per_detector"][det]
            if not isinstance(det_payload, dict):
                reasons.append(f"invalid detector payload: {det}")
                continue
            status = det_payload.get("status", "OK")
            det_out: dict[str, Any] = {
                "status": status,
                "pd_ok": False,
                "M0": None,
                "M1": None,
                "logBF_10": None,
                "diagnostics": {},
            }
            per_detector_res[det] = det_out

            if status != "OK":
                continue

            ok_detectors.append(det)
            try:
                t, y, npz_rel = _load_window_arrays(det, det_payload, run_dir)
                if npz_rel is not None:
                    npz_path = (run_dir / npz_rel).resolve()
                    inputs.append({"path": npz_rel, "sha256": sha256_file(npz_path)})
                if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 8:
                    raise RuntimeError("invalid t/y shapes; require 1D equal-length arrays with n>=8")
                if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
                    raise RuntimeError("non-finite values in t/y arrays")

                whitened = bool(report.get("whitened", False) or report.get("residuals_normalized", False) or det_payload.get("whitened", False) or det_payload.get("residuals_normalized", False))
                sigma_mode = "fixed_1" if whitened else "estimated"

                m0 = _fit_and_logz(t, y, priors, t_ref=t_ref, with_slope=False, sigma_mode=sigma_mode)
                m1 = _fit_and_logz(t, y, priors, t_ref=t_ref, with_slope=True, sigma_mode=sigma_mode)
                logbf = float(m1["logZ"] - m0["logZ"])
                if not _is_finite(logbf):
                    raise RuntimeError("non-finite logBF_10")

                det_out["M0"] = {k: m0[k] for k in ["logZ", "logL_hat", "k", "log_prior_volume", "logdet_H"]}
                det_out["M1"] = {k: m1[k] for k in ["logZ", "logL_hat", "k", "log_prior_volume", "logdet_H"]}
                det_out["logBF_10"] = logbf
                det_out["pd_ok"] = bool(m0["pd_ok"] and m1["pd_ok"])
                det_out["diagnostics"] = {
                    "sigma2": float(m1["sigma2"] if sigma_mode == "fixed_1" else m0["sigma2"]),
                    "dof0": int(m0["dof"]),
                    "dof1": int(m1["dof"]),
                    "hessian_pd": bool(det_out["pd_ok"]),
                    "n": int(m0["n"]),
                    "ss_res0": float(m0["ss_res"]),
                    "ss_res1": float(m1["ss_res"]),
                }
                detector_bfs.append(logbf)
            except Exception as exc:
                reasons.append(f"{det}: {exc}")

        if not ok_detectors:
            reasons.append("no detectors with status OK in inference_report")

        for det in ok_detectors:
            det_res = per_detector_res.get(det, {})
            if not isinstance(det_res, dict):
                reasons.append(f"{det}: missing result block")
                continue
            if det_res.get("M0") is None or det_res.get("M1") is None:
                reasons.append(f"{det}: could not compute logZ for both models")
                continue
            m0z = det_res["M0"]["logZ"]
            m1z = det_res["M1"]["logZ"]
            lbf = det_res["logBF_10"]
            if not (_is_finite(m0z) and _is_finite(m1z) and _is_finite(lbf)):
                reasons.append(f"{det}: non-finite logZ/logBF")

        if reasons:
            verdict = "FAIL"
        else:
            logz0 = float(sum(per_detector_res[d]["M0"]["logZ"] for d in ok_detectors))
            logz1 = float(sum(per_detector_res[d]["M1"]["logZ"] for d in ok_detectors))
            logbf_global = float(logz1 - logz0)
            if not (_is_finite(logz0) and _is_finite(logz1) and _is_finite(logbf_global)):
                verdict = "FAIL"
                reasons.append("global logZ/logBF non-finite")

            if verdict != "FAIL":
                if any(not bool(per_detector_res[d]["pd_ok"]) for d in ok_detectors):
                    verdict = "INSPECT"
                    reasons.append("non-PD Hessian in at least one detector")

                pos = [v for v in detector_bfs if v > 5.0]
                neg = [v for v in detector_bfs if v < -5.0]
                if pos and neg:
                    verdict = "INSPECT"
                    reasons.append("strong detector disagreement: opposite logBF signs with |logBF|>5")

            bayes_payload = {
                "schema_version": STAGE_VERSION,
                "created_utc": _get_created_utc(run_valid, report),
                "stage": "ringdown_bayes_v2",
                "run": args.run,
                "parameters": {
                    "models": {
                        "M0": {"params": ["A", "tau", "f", "phi", "C0"], "formula": "A*exp(-t/tau)*cos(2*pi*f*t+phi)+C0"},
                        "M1": {"params": ["A", "tau", "f", "phi", "C0", "C1"], "formula": "A*exp(-t/tau)*cos(2*pi*f*t+phi)+C0+C1*(t-t_ref)"},
                    },
                    "t_ref": t_ref,
                    "priors": priors,
                    "likelihood": {
                        "family": "gaussian_iid",
                        "sigma_mode_rule": "fixed_1 if whitened/residuals_normalized else estimated ss_res/dof",
                        "logL_formula_sigma1": "-0.5*ss_res",
                        "logL_formula_sigma_est": "-0.5*(ss_res/sigma2 + n*log(2*pi*sigma2))",
                    },
                    "solver": {
                        "method": "trf",
                        "loss": "linear",
                        "jac": "2-point",
                        "ftol": 1e-12,
                        "xtol": 1e-12,
                        "gtol": 1e-12,
                        "max_nfev": 2000,
                    },
                    "evidence": "logZ = logL_hat + (k/2)*log(2*pi) - 0.5*logdet_H - log_prior_volume",
                },
                "inputs": sorted(inputs, key=lambda x: x["path"]),
                "results": {
                    "per_detector": per_detector_res,
                    "global": {"logZ_M0": logz0, "logZ_M1": logz1, "logBF_10": logbf_global},
                },
                "verdict": verdict,
                "reasons": reasons,
            }

    if reasons and bayes_payload is None:
        verdict = "FAIL"

    summary: dict[str, Any] = {
        "stage": args.out_stage,
        "run": args.run,
        "params": {
            "inference_report_relpath": args.inference_report_relpath,
            "out_stage": args.out_stage,
            "out_root": str(args.out_root),
        },
        "inputs": sorted(inputs, key=lambda x: x["path"]),
        "verdict": verdict,
        "reasons": reasons,
        "outputs": {},
        "version": STAGE_VERSION,
    }

    artifacts: dict[str, Path] = {}
    if verdict != "FAIL" and bayes_payload is not None:
        _json_dump(output_path, bayes_payload)
        assert_within_runs(run_dir, output_path)
        summary["outputs"] = {"bayes_v2": "outputs/bayes_v2.json"}
        artifacts["bayes_v2"] = output_path

    summary_path = write_stage_summary(stage_dir, summary)
    artifacts["stage_summary"] = summary_path
    write_manifest(
        stage_dir,
        artifacts,
        extra={
            "version": STAGE_VERSION,
            "params": summary["params"],
            "inputs": summary["inputs"],
            "verdict": verdict,
            "reasons": reasons,
        },
    )

    if verdict == "FAIL":
        return EXIT_CONTRACT_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
