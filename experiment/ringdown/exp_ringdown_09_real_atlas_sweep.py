#!/usr/bin/env python3
"""
experiment/ringdown/exp_ringdown_09_real_atlas_sweep.py
--------------------------------------------------------
EXP_RINGDOWN_09: Real atlas sweep runner.

Consumes existing runs (runtime user-provided) and sweeps real pipeline
configurations, collecting EXP08 verdicts and optional inference metrics.

Inputs:
  --run <run_id>
  --grid-json <path>  (list of {dt_start_s, duration_s, band_hz})
  --n-max (optional)
  --dry-run

Outputs:
  runs/<run>/experiment/ringdown/EXP_RINGDOWN_09__real_atlas_sweep/
    manifest.json
    stage_summary.json
    outputs/atlas_cases.jsonl
    outputs/atlas_summary.json
    outputs/failure_catalog.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    get_runs_root,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_09__real_atlas_sweep"
EXP08_REPORT_REL = (
    "experiment/ringdown/EXP_RINGDOWN_08__real_v0_smoke/outputs/real_v0_smoke_report.json"
)


def abort_contract(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _format_band_suffix(band_hz: Tuple[float, float]) -> str:
    low = int(round(band_hz[0]))
    high = int(round(band_hz[1]))
    return f"b{low:04d}_{high:04d}"


def _compute_suffix(dt_start_s: float, duration_s: float, band_hz: Tuple[float, float]) -> str:
    dt_ms = int(round(dt_start_s * 1000))
    dur_ms = int(round(duration_s * 1000))
    return f"dt{dt_ms:04d}ms__dur{dur_ms:04d}ms__{_format_band_suffix(band_hz)}"


def _normalize_band(band_hz: Any) -> Tuple[float, float]:
    if isinstance(band_hz, str):
        parts = [p.strip() for p in band_hz.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("band_hz debe tener 2 valores")
        return float(parts[0]), float(parts[1])
    if isinstance(band_hz, (list, tuple)) and len(band_hz) == 2:
        return float(band_hz[0]), float(band_hz[1])
    raise ValueError("band_hz inválido")


def _band_to_cli(band_hz: Tuple[float, float]) -> str:
    return f"{band_hz[0]},{band_hz[1]}"


def _extract_exp08(report_path: Path) -> Tuple[Optional[str], Optional[int]]:
    if not report_path.exists():
        return None, None
    payload = _read_json(report_path)
    verdict = payload.get("overall_verdict")
    n_smoke_ok = None
    smoke = payload.get("smoke_inference")
    if isinstance(smoke, dict) and isinstance(smoke.get("n_smoke_ok"), int):
        n_smoke_ok = int(smoke["n_smoke_ok"])
    return verdict, n_smoke_ok


def _extract_f_peak(report_path: Path) -> Optional[float]:
    if not report_path.exists():
        return None
    payload = _read_json(report_path)
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        return None
    values: List[float] = []
    for det_payload in fit.values():
        if not isinstance(det_payload, dict):
            continue
        f_peak = det_payload.get("f_peak_hz")
        if isinstance(f_peak, (int, float)):
            values.append(float(f_peak))
    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_inference_decision(report_path: Path) -> Tuple[Optional[str], List[str]]:
    if not report_path.exists():
        return None, []
    payload = _read_json(report_path)
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return None, []
    verdict = decision.get("verdict")
    if not isinstance(verdict, str):
        return None, []
    reasons_raw = decision.get("reasons", [])
    if not isinstance(reasons_raw, list):
        reasons_raw = []
    reasons = [str(item) for item in reasons_raw if isinstance(item, (str, int, float))]
    return verdict, reasons


def _band_ratio(band_hz: Tuple[float, float], f_peak_hz: Optional[float]) -> Optional[float]:
    if f_peak_hz is None:
        return None
    low, high = band_hz
    denom = high - low
    if denom == 0:
        return None
    return float((f_peak_hz - low) / denom)


def _relpath(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _load_grid(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError("grid-json debe ser una lista")
    return payload


def _summarize_top(
    cases: Iterable[Dict[str, Any]],
    key: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    filtered = [case for case in cases if isinstance(case.get(key), (int, float))]
    ranked = sorted(filtered, key=lambda c: c[key], reverse=True)
    return [
        {
            "case_id": case.get("case_id"),
            "config": case.get("config"),
            key: case.get(key),
        }
        for case in ranked[:top_k]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_09 real atlas sweep")
    ap.add_argument("--run", required=True)
    ap.add_argument("--grid-json", required=True)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = resolve_out_root("runs", runs_root=get_runs_root())
    validate_run_id(args.run, out_root)

    run_dir = get_run_dir(args.run, base_dir=out_root).resolve()
    try:
        run_valid_payload = require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    grid_path = Path(args.grid_json)
    if not grid_path.is_absolute():
        grid_path = (Path.cwd() / grid_path).resolve()
    try:
        grid = _load_grid(grid_path)
    except Exception as exc:
        abort_contract(f"grid-json inválido: {exc}")

    if args.n_max is not None:
        grid = grid[: max(int(args.n_max), 0)]

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    cases_path = outputs_dir / "atlas_cases.jsonl"
    summary_path = outputs_dir / "atlas_summary.json"
    failure_path = outputs_dir / "failure_catalog.jsonl"

    exp08_report_path = run_dir / EXP08_REPORT_REL

    cases: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    inspect_reasons_count: Dict[str, int] = {}

    for idx, raw_cfg in enumerate(grid, start=1):
        case_id = f"case_{idx:03d}"
        if not isinstance(raw_cfg, dict):
            failures.append({"case_id": case_id, "reason": "invalid_config"})
            continue
        try:
            dt_start_s = float(raw_cfg["dt_start_s"])
            duration_s = float(raw_cfg["duration_s"])
            band_hz = _normalize_band(raw_cfg["band_hz"])
        except Exception:
            failures.append({"case_id": case_id, "reason": "invalid_config"})
            continue

        config = {
            "dt_start_s": dt_start_s,
            "duration_s": duration_s,
            "band_hz": [band_hz[0], band_hz[1]],
        }

        start_t = time.perf_counter()
        runner_status = "SKIP" if args.dry_run else "OK"
        runner_returncode = None
        if not args.dry_run:
            command = [
                "python",
                "tools/basurin_run_real.py",
                "--run",
                args.run,
                "--dt-start-s",
                str(dt_start_s),
                "--duration-s",
                str(duration_s),
                "--band-hz",
                _band_to_cli(band_hz),
            ]
            result = subprocess.run(command, check=False)
            runner_returncode = result.returncode
            if result.returncode != 0:
                runner_status = "FAIL"
                failures.append(
                    {
                        "case_id": case_id,
                        "reason": "runner_failed",
                        "returncode": int(result.returncode),
                    }
                )
            else:
                runner_status = "OK"

        elapsed_s = time.perf_counter() - start_t

        exp08_verdict, n_smoke_ok = _extract_exp08(exp08_report_path)
        if exp08_verdict is None:
            failures.append({"case_id": case_id, "reason": "missing_exp08_report"})

        suffix = _compute_suffix(dt_start_s, duration_s, band_hz)
        inference_stage = f"ringdown_real_inference_v0__{suffix}"
        inference_report_path = (
            run_dir / inference_stage / "outputs" / "inference_report.json"
        )
        f_peak_hz = _extract_f_peak(inference_report_path)
        band_ratio = _band_ratio(band_hz, f_peak_hz)
        inference_verdict, inference_reasons = _extract_inference_decision(
            inference_report_path
        )

        case_verdict = "FAIL"
        if runner_returncode is not None and runner_returncode != 0:
            case_verdict = "FAIL"
        elif exp08_verdict != "PASS":
            case_verdict = "FAIL"
        else:
            if inference_verdict in {"PASS", "INSPECT"}:
                case_verdict = inference_verdict
            else:
                case_verdict = "PASS"

        if case_verdict == "INSPECT":
            for reason in inference_reasons:
                inspect_reasons_count[reason] = inspect_reasons_count.get(reason, 0) + 1

        cases.append(
            {
                "case_id": case_id,
                "config": config,
                "exp08_verdict": exp08_verdict,
                "n_smoke_ok": n_smoke_ok,
                "f_peak_hz": f_peak_hz,
                "band_ratio": band_ratio,
                "elapsed_s": float(elapsed_s),
                "runner_status": runner_status,
                "runner_returncode": runner_returncode,
                "inference_verdict": inference_verdict,
                "inference_reasons": inference_reasons,
                "case_verdict": case_verdict,
                "paths": {
                    "exp08_report": _relpath(run_dir, exp08_report_path),
                    "inference_report": _relpath(run_dir, inference_report_path),
                },
            }
        )

    with open(cases_path, "w", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    with open(failure_path, "w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    n_pass = sum(1 for case in cases if case.get("case_verdict") == "PASS")
    n_inspect = sum(1 for case in cases if case.get("case_verdict") == "INSPECT")
    n_fail = sum(1 for case in cases if case.get("case_verdict") == "FAIL")

    summary = {
        "run_id": args.run,
        "n_cases": len(cases),
        "n_pass": n_pass,
        "n_inspect": n_inspect,
        "n_fail": n_fail,
        "top_by_n_smoke_ok": _summarize_top(cases, "n_smoke_ok"),
        "top_by_band_ratio": _summarize_top(cases, "band_ratio"),
        "inspect_reasons_count": inspect_reasons_count,
        "dry_run": bool(args.dry_run),
        "timestamp": utc_now_iso(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    inputs: Dict[str, Any] = {
        "run_valid": {
            "path": _relpath(run_dir, run_dir / "RUN_VALID" / "outputs" / "run_valid.json"),
            "sha256": sha256_file(run_dir / "RUN_VALID" / "outputs" / "run_valid.json"),
            "verdict": run_valid_payload.get("overall_verdict")
            or run_valid_payload.get("verdict"),
        },
        "grid_json": {
            "path": str(grid_path),
            "sha256": sha256_file(grid_path),
        },
        "exp08_report": {
            "path": _relpath(run_dir, exp08_report_path),
            "sha256": sha256_file(exp08_report_path) if exp08_report_path.exists() else None,
        },
    }

    summary_payload = {
        "stage": stage_dir.name,
        "run": args.run,
        "created": utc_now_iso(),
        "inputs": inputs,
        "parameters": {
            "grid_json": str(grid_path),
            "n_max": args.n_max,
            "dry_run": args.dry_run,
        },
        "outputs": {
            "atlas_cases": "outputs/atlas_cases.jsonl",
            "atlas_summary": "outputs/atlas_summary.json",
            "failure_catalog": "outputs/failure_catalog.jsonl",
        },
        "results": {
            "n_cases": len(cases),
            "n_pass": n_pass,
            "n_inspect": n_inspect,
            "n_fail": n_fail,
        },
        "version": {"git_sha": _maybe_git_sha()},
    }

    summary_written = write_stage_summary(stage_dir, summary_payload)
    write_manifest(
        stage_dir,
        {
            "atlas_cases": cases_path,
            "atlas_summary": summary_path,
            "failure_catalog": failure_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    print(f"OK: EXP_RINGDOWN_09 completed (cases={len(cases)})")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
