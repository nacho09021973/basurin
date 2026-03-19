#!/usr/bin/env python3
"""Run a minimal, auditable comparison matrix for mode-221 upstream strategies."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[1], _here.parents[2]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import resolve_out_root, sha256_file, validate_run_id, write_json_atomic, write_manifest, write_stage_summary

EXPERIMENT_STAGE = "experiment/compare_221_strategies"
RESULTS_JSON = "compare_221_runs.json"
SUMMARY_CSV = "compare_221_summary.csv"
MATRIX_JSON = "compare_221_matrix.json"
METHODS = ("hilbert_peakband", "spectral_two_pass")
RESIDUAL_STRATEGIES = ("refit_220_each_iter", "fixed_220_template")
BAND_STRATEGIES = ("default_split_60_40", "coherent_harmonic_band")


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _combo_token(method: str, residual_strategy: str, band_strategy: str) -> str:
    return f"m-{_safe_token(method)}__r-{_safe_token(residual_strategy)}__b-{_safe_token(band_strategy)}"


def _build_run_id(base_run_prefix: str, method: str, residual_strategy: str, band_strategy: str) -> str:
    return f"{base_run_prefix}__{_combo_token(method, residual_strategy, band_strategy)}"


def _build_compare_run_id(base_run_prefix: str) -> str:
    return f"{base_run_prefix}__compare221_audit"


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _extract_mode(payload: dict[str, Any], label: str) -> dict[str, Any]:
    modes = payload.get("modes")
    if not isinstance(modes, list):
        return {}
    for item in modes:
        if isinstance(item, dict) and item.get("label") == label:
            return item
    return {}


def _metric_from_mode(mode_payload: dict[str, Any], key: str) -> Any:
    fit = mode_payload.get("fit") if isinstance(mode_payload, dict) else None
    stability = fit.get("stability") if isinstance(fit, dict) else None
    if not isinstance(stability, dict):
        return None
    return stability.get(key)


def _summarize_run(run_id: str, run_dir: Path, method: str, residual_strategy: str, band_strategy: str, returncode: int) -> dict[str, Any]:
    multimode_path = run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    if not multimode_path.exists():
        raise FileNotFoundError(
            "missing expected multimode artifact: "
            f"{multimode_path}\n"
            "Regenerate upstream with:\n"
            f"python -m mvp.pipeline multimode --event-id ... --run-id {run_id} --atlas-default"
        )
    payload = _read_json_dict(multimode_path)
    mode_221 = _extract_mode(payload, "221")
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    residual_meta = source.get("mode_221_residual") if isinstance(source.get("mode_221_residual"), dict) else {}
    band_meta = source.get("band_strategy") if isinstance(source.get("band_strategy"), dict) else {}

    return {
        "run_id": run_id,
        "pipeline_returncode": int(returncode),
        "pipeline_status": "PASS" if int(returncode) == 0 else "FAILED_SUBRUN",
        "method": method,
        "band_strategy": band_strategy,
        "residual_strategy": residual_strategy,
        "mode_221_usable": bool(payload.get("mode_221_usable", False)),
        "mode_221_usable_reason": payload.get("mode_221_usable_reason") or "missing_mode_221_usable_reason",
        "valid_fraction": _metric_from_mode(mode_221, "valid_fraction"),
        "cv_f": _metric_from_mode(mode_221, "cv_f"),
        "cv_Q": _metric_from_mode(mode_221, "cv_Q"),
        "lnf_span": _metric_from_mode(mode_221, "lnf_span"),
        "lnQ_span": _metric_from_mode(mode_221, "lnQ_span"),
        "results_verdict": results.get("verdict"),
        "quality_flags": list(results.get("quality_flags", [])) if isinstance(results.get("quality_flags"), list) else [],
        "mode_221_residual_source": residual_meta,
        "band_strategy_source": band_meta,
        "artifact": str(multimode_path),
        "artifact_sha256": sha256_file(multimode_path),
    }


def _build_cmd(args: argparse.Namespace, *, run_id: str, method: str, residual_strategy: str, band_strategy: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mvp.pipeline",
        "multimode",
        "--event-id",
        args.event_id,
        "--run-id",
        run_id,
        "--duration-s",
        str(float(args.duration_s)),
        "--window-duration-s",
        str(float(args.window_duration_s)),
        "--band-low",
        str(float(args.band_low)),
        "--band-high",
        str(float(args.band_high)),
        "--s3b-n-bootstrap",
        str(int(args.s3b_n_bootstrap)),
        "--s3b-seed",
        str(int(args.s3b_seed)),
        "--s3b-method",
        method,
        "--bootstrap-221-residual-strategy",
        residual_strategy,
        "--band-strategy",
        band_strategy,
    ]
    if args.dt_start_s is not None:
        cmd.extend(["--dt-start-s", str(float(args.dt_start_s))])
    if args.stage_timeout_s is not None:
        cmd.extend(["--stage-timeout-s", str(float(args.stage_timeout_s))])
    if args.atlas_default:
        cmd.append("--atlas-default")
    elif args.atlas_path:
        cmd.extend(["--atlas-path", args.atlas_path])
    if args.offline:
        cmd.append("--offline")
    if args.reuse_strain:
        cmd.append("--reuse-strain")
    if args.with_t0_sweep:
        cmd.append("--with-t0-sweep")
    for mapping in args.local_hdf5:
        cmd.extend(["--local-hdf5", mapping])
    if args.psd_path:
        cmd.extend(["--psd-path", args.psd_path])
    if args.final_mass_msun is not None:
        cmd.extend(["--final-mass-msun", str(float(args.final_mass_msun))])
    if args.redshift is not None:
        cmd.extend(["--redshift", str(float(args.redshift))])
    return cmd


def _iter_matrix() -> list[tuple[str, str, str]]:
    return [
        (method, residual_strategy, band_strategy)
        for method in METHODS
        for residual_strategy in RESIDUAL_STRATEGIES
        for band_strategy in BAND_STRATEGIES
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--base-run-prefix", required=True)
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument("--atlas-default", action="store_true", default=True)
    ap.add_argument("--duration-s", type=float, default=32.0)
    ap.add_argument("--dt-start-s", type=float, default=None)
    ap.add_argument("--window-duration-s", type=float, default=0.06)
    ap.add_argument("--band-low", type=float, default=150.0)
    ap.add_argument("--band-high", type=float, default=400.0)
    ap.add_argument("--s3b-n-bootstrap", type=int, default=200)
    ap.add_argument("--s3b-seed", type=int, default=12345)
    ap.add_argument("--stage-timeout-s", type=float, default=None)
    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--reuse-strain", action="store_true", default=False)
    ap.add_argument("--with-t0-sweep", action="store_true", default=False)
    ap.add_argument("--local-hdf5", action="append", default=[])
    ap.add_argument("--psd-path", default=None)
    ap.add_argument("--final-mass-msun", type=float, default=None)
    ap.add_argument("--redshift", type=float, default=None)
    ap.add_argument("--keep-going", action="store_true", default=False, help="Continue matrix execution after a failed subrun.")
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.atlas_default and not args.atlas_path:
        raise SystemExit("--atlas-path is required when --atlas-default is disabled")

    out_root = resolve_out_root("runs")
    compare_run_id = _build_compare_run_id(args.base_run_prefix)
    validate_run_id(compare_run_id, out_root)

    stage_dir = out_root / compare_run_id / EXPERIMENT_STAGE
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []

    for method, residual_strategy, band_strategy in _iter_matrix():
        run_id = _build_run_id(args.base_run_prefix, method, residual_strategy, band_strategy)
        validate_run_id(run_id, out_root)
        cmd = _build_cmd(args, run_id=run_id, method=method, residual_strategy=residual_strategy, band_strategy=band_strategy)
        record: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "band_strategy": band_strategy,
            "residual_strategy": residual_strategy,
            "cmd": cmd,
        }
        if args.dry_run:
            record["status"] = "DRY_RUN"
            matrix_rows.append(record)
            continue

        proc = subprocess.run(cmd, text=True, capture_output=True)
        record.update(
            {
                "status": "PASS" if proc.returncode == 0 else "FAILED_SUBRUN",
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "started_utc": _utc_now_z(),
            }
        )
        matrix_rows.append(record)

        if proc.returncode != 0 and not args.keep_going:
            results_path = write_json_atomic(outputs_dir / RESULTS_JSON, {"matrix": matrix_rows, "rows": rows})
            manifest = write_manifest(stage_dir, {RESULTS_JSON: results_path}, extra={"experiment_stage": EXPERIMENT_STAGE})
            summary = write_stage_summary(stage_dir, {
                "schema_version": "compare_221_strategies_v1",
                "event_id": args.event_id,
                "base_run_prefix": args.base_run_prefix,
                "compare_run_id": compare_run_id,
                "completed_runs": len(rows),
                "failed_run_id": run_id,
                "status": "FAILED_SUBRUN",
                "updated_utc": _utc_now_z(),
            })
            print(f"subrun failed: run_id={run_id} returncode={proc.returncode}", file=sys.stderr)
            print(f"OUT_ROOT={out_root}")
            print(f"STAGE_DIR={stage_dir}")
            print(f"OUTPUTS_DIR={outputs_dir}")
            print(f"STAGE_SUMMARY={summary}")
            print(f"MANIFEST={manifest}")
            return int(proc.returncode)

        if proc.returncode == 0:
            rows.append(_summarize_run(run_id, out_root / run_id, method, residual_strategy, band_strategy, proc.returncode))
        elif args.keep_going:
            rows.append({
                "run_id": run_id,
                "method": method,
                "band_strategy": band_strategy,
                "residual_strategy": residual_strategy,
                "mode_221_usable": False,
                "mode_221_usable_reason": f"subrun_failed_returncode_{proc.returncode}",
                "valid_fraction": None,
                "cv_f": None,
                "cv_Q": None,
                "lnf_span": None,
                "lnQ_span": None,
                "pipeline_returncode": int(proc.returncode),
                "pipeline_status": "FAILED_SUBRUN",
                "results_verdict": None,
                "quality_flags": [],
            })

    results_path = write_json_atomic(outputs_dir / RESULTS_JSON, {"matrix": matrix_rows, "rows": rows})
    csv_path = _write_csv_atomic(outputs_dir / SUMMARY_CSV, rows, [
        "run_id",
        "method",
        "band_strategy",
        "residual_strategy",
        "mode_221_usable",
        "mode_221_usable_reason",
        "valid_fraction",
        "cv_f",
        "cv_Q",
        "lnf_span",
        "lnQ_span",
    ])
    matrix_path = write_json_atomic(outputs_dir / MATRIX_JSON, {
        "schema_version": "compare_221_strategies_matrix_v1",
        "event_id": args.event_id,
        "base_run_prefix": args.base_run_prefix,
        "compare_run_id": compare_run_id,
        "generated_utc": _utc_now_z(),
        "comparisons": rows,
    })
    summary_path = write_stage_summary(stage_dir, {
        "schema_version": "compare_221_strategies_v1",
        "event_id": args.event_id,
        "base_run_prefix": args.base_run_prefix,
        "compare_run_id": compare_run_id,
        "n_runs": len(rows),
        "all_subruns_passed": all(row.get("pipeline_status") == "PASS" for row in rows),
        "artifacts": {
            "results_json": str(results_path.relative_to(stage_dir)),
            "summary_csv": str(csv_path.relative_to(stage_dir)),
            "matrix_json": str(matrix_path.relative_to(stage_dir)),
        },
        "updated_utc": _utc_now_z(),
    })
    manifest_path = write_manifest(
        stage_dir,
        {
            RESULTS_JSON: results_path,
            SUMMARY_CSV: csv_path,
            MATRIX_JSON: matrix_path,
        },
        extra={
            "experiment_stage": EXPERIMENT_STAGE,
            "event_id": args.event_id,
            "base_run_prefix": args.base_run_prefix,
        },
    )

    print(f"OUT_ROOT={out_root}")
    print(f"STAGE_DIR={stage_dir}")
    print(f"OUTPUTS_DIR={outputs_dir}")
    print(f"STAGE_SUMMARY={summary_path}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
