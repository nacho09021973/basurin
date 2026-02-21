#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from mvp.oracles.oracle_v1_plateau import WindowMetrics, run_oracle_v1
from mvp.oracles.t0_input_schema import WindowSummaryV1Error, map_sweep_point_to_window_summary_v1

from basurin_io import require_run_valid, resolve_out_root, sha256_file, validate_run_id, write_json_atomic

STAGE = "experiment/oracle_t0_ringdown"
RESULTS_NAME = "t0_sweep_full_results.json"


def _seed_from_dir_name(seed_dir: Path) -> int:
    name = seed_dir.name
    prefix = "t0_sweep_full_seed"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return 101


def _build_sweep_command(run_id: str, seed: int) -> str:
    return (
        "python mvp/experiment_t0_sweep_full.py "
        f"--run-id {run_id} "
        "--phase run "
        "--atlas-path <ATLAS_PATH> "
        "--t0-grid-ms 0,2,4,6,8 "
        f"--seed {seed}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical t0 oracle from t0_sweep_full seed results")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed-dir", default=None, help="Optional explicit seed dir (.../experiment/t0_sweep_full_seed<seed>)")
    p.add_argument("--runs-root", default=None, help="Optional runs root (else BASURIN_RUNS_ROOT or ./runs)")
    p.add_argument("--sigma-floor-f", type=float, default=1e-6)
    p.add_argument("--sigma-floor-lnq", type=float, default=1e-6)
    p.add_argument("--default-duration-s", type=float, default=1.0)
    p.add_argument("--default-p-white", type=float, default=1.0)
    p.add_argument("--default-n-samples", type=int, default=256)
    p.add_argument("--chi2-coh-max", type=float, default=None)
    return p.parse_args(argv)


def _find_seed_dir(base_run_dir: Path, seed_dir_arg: str | None) -> Path:
    if seed_dir_arg:
        seed_dir = Path(seed_dir_arg).expanduser().resolve()
        if not seed_dir.exists():
            seed = _seed_from_dir_name(seed_dir)
            sweep_cmd = _build_sweep_command(base_run_dir.name, seed)
            raise FileNotFoundError(
                f"missing seed dir: {seed_dir}\n"
                f"expected path: {seed_dir}\n"
                f"generate sweep with: {sweep_cmd}"
            )
    else:
        candidates = sorted((base_run_dir / "experiment").glob("t0_sweep_full_seed*"))
        if len(candidates) != 1:
            expected_seed_dir = (base_run_dir / "experiment" / "t0_sweep_full_seed101").resolve()
            sweep_cmd = _build_sweep_command(base_run_dir.name, 101)
            raise RuntimeError(
                f"expected exactly one seed dir under {base_run_dir / 'experiment'}, got {len(candidates)}; pass --seed-dir\n"
                f"expected path: {expected_seed_dir}\n"
                f"generate sweep with: {sweep_cmd}"
            )
        seed_dir = candidates[0].resolve()

    expected_parent = (base_run_dir / "experiment").resolve()
    if seed_dir.parent != expected_parent:
        raise RuntimeError(f"seed dir must be under {expected_parent}, got {seed_dir}")
    return seed_dir


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _window_metric_from_summary(summary: dict[str, Any], args: argparse.Namespace) -> WindowMetrics:
    theta = summary.get("theta", {})
    sigma = summary.get("sigma_theta", {})

    ln_f = float(theta["ln_f_220"])
    ln_q = float(theta["ln_Q_220"])
    sigma_ln_f = max(float(sigma["sigma_ln_f_220"]), float(args.sigma_floor_f))
    sigma_ln_q = max(float(sigma["sigma_ln_Q_220"]), float(args.sigma_floor_lnq))

    f_median = math.exp(ln_f)
    q_median = math.exp(ln_q)
    tau_median = q_median / (math.pi * f_median)
    tau_sigma = tau_median * math.sqrt((sigma_ln_f * sigma_ln_f) + (sigma_ln_q * sigma_ln_q))

    cond = _safe_float(summary.get("cond"))
    snr = _safe_float(summary.get("snr"))
    t0 = _safe_float(summary.get("t0_ms"))
    t0 = 0.0 if t0 is None else t0
    T = _safe_float(summary.get("T_s"))

    return WindowMetrics(
        t0=t0,
        T=float(args.default_duration_s if T is None else T),
        f_median=f_median,
        f_sigma=max(f_median * sigma_ln_f, float(args.sigma_floor_f)),
        tau_median=tau_median,
        tau_sigma=max(tau_sigma, float(args.sigma_floor_lnq)),
        cond_number=cond,
        delta_bic=20.0,
        p_ljungbox=float(args.default_p_white),
        n_samples=int(args.default_n_samples),
        snr=snr,
    )


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    out_root = resolve_out_root() if args.runs_root is None else Path(args.runs_root).expanduser().resolve()
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    base_run_dir = out_root / args.run_id
    seed_dir = _find_seed_dir(base_run_dir, args.seed_dir)
    results_path = seed_dir / "outputs" / RESULTS_NAME
    if not results_path.exists():
        seed = _seed_from_dir_name(seed_dir)
        sweep_cmd = _build_sweep_command(args.run_id, seed)
        raise FileNotFoundError(
            f"missing seed results json: {results_path}\n"
            f"expected path: {results_path}\n"
            f"generate sweep with: {sweep_cmd}"
        )

    sweep_payload = json.loads(results_path.read_text(encoding="utf-8"))
    points = sweep_payload.get("points", [])

    windows: list[WindowMetrics] = []
    map_errors: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        try:
            row = map_sweep_point_to_window_summary_v1(sweep_payload, point)
            windows.append(_window_metric_from_summary(row, args))
        except WindowSummaryV1Error as exc:
            map_errors.append({"index": idx, "t0_ms": point.get("t0_ms"), "code": "MAP_WINDOW_SUMMARY_V1_ERROR", "detail": str(exc)})

    oracle_report = run_oracle_v1(windows, chi2_coh_max=args.chi2_coh_max)
    oracle_report["input"] = {
        "run_id": args.run_id,
        "seed_dir": str(seed_dir),
        "results_path": str(results_path),
        "n_points": len(points),
        "n_windows_mapped": len(windows),
        "n_mapping_errors": len(map_errors),
    }
    oracle_report["mapping_errors"] = map_errors

    reason_codes: list[str] = []
    if map_errors:
        reason_codes.append("MAP_WINDOW_SUMMARY_V1_ERROR")
    if oracle_report.get("final_verdict") != "PASS":
        reason_codes.append(str(oracle_report.get("fail_global_reason") or "ORACLE_FAIL"))

    stage_verdict = "PASS" if not reason_codes else "FAIL"

    stage_dir = base_run_dir / STAGE
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    oracle_path = outputs_dir / "oracle_report.json"
    stage_summary_path = stage_dir / "stage_summary.json"
    manifest_path = stage_dir / "manifest.json"

    write_json_atomic(oracle_path, oracle_report)

    stage_summary = {
        "stage": STAGE,
        "run": args.run_id,
        "verdict": stage_verdict,
        "reason_codes": reason_codes,
        "results": {
            "n_points": len(points),
            "n_windows_mapped": len(windows),
            "n_mapping_errors": len(map_errors),
            "oracle_final_verdict": oracle_report.get("final_verdict"),
            "oracle_fail_global_reason": oracle_report.get("fail_global_reason"),
        },
    }
    write_json_atomic(stage_summary_path, stage_summary)

    manifest = {
        "schema_version": "mvp_manifest_v1",
        "artifacts": {
            "oracle_report": "outputs/oracle_report.json",
            "stage_summary": "stage_summary.json",
        },
        "hashes": {
            "oracle_report": sha256_file(oracle_path),
            "stage_summary": sha256_file(stage_summary_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return {"oracle_report": oracle_report, "stage_summary": stage_summary, "manifest": manifest}


if __name__ == "__main__":
    run()
