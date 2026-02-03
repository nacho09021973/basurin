#!/usr/bin/env python3
"""
experiment/ringdown/exp_ringdown_08_real_v0_smoke.py
----------------------------------------------------
EXP_RINGDOWN_08: Real v0 smoke gate.

This experiment validates that the "real v0" data package is compatible
with the ringdown pipeline. It does NOT attempt scientific accuracy;
it only ensures:

1. The real v0 format/plumbing is compatible with the pipeline (R08_REAL_IO_COMPAT)
2. Minimal diagnostics are present (R08_PIPELINE_SMOKE)
3. All failures are categorized and traceable (R08_FAIL_CATEGORIZED)

Inputs:
  - RUN_VALID/outputs/run_valid.json (must be PASS)
  - ringdown_synth/outputs/synthetic_events_list.json (optional, for control)
  - ringdown_real_v0/outputs/real_v0_events_list.json (or --real-v0-events-json)

Outputs:
  runs/<run_id>/experiment/ringdown/EXP_RINGDOWN_08__real_v0_smoke/
    manifest.json
    stage_summary.json
    outputs/contract_verdict.json
    outputs/real_v0_smoke_report.json
    outputs/failure_catalog.jsonl
"""
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
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_08__real_v0_smoke"
DEFAULT_N_MAX_CASES = 5
DEFAULT_TIMEOUT_S = 30.0

ALLOWED_FAIL_CODES = {
    "exception",
    "invalid_input",
    "nan_inference",
    "nonconvergence",
    "numerical_instability",
    "timeout",
    "missing_real_v0_inputs",
}


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


def _resolve_run_valid_path(run_dir: Path) -> Path:
    return run_dir / "RUN_VALID" / "outputs" / "run_valid.json"


def _resolve_real_v0_events_path(run_dir: Path) -> Optional[Path]:
    """Resolve canonical path for real v0 events list."""
    path = run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json"
    return path if path.exists() else None


def _resolve_synth_events_path(run_dir: Path) -> Optional[Path]:
    """Resolve synthetic events list for control comparison."""
    path = run_dir / "ringdown_synth" / "outputs" / "synthetic_events_list.json"
    return path if path.exists() else None


def _load_run_valid_payload(run_dir: Path) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    run_valid_path = _resolve_run_valid_path(run_dir)
    if not run_valid_path.exists():
        return None, None, f"RUN_VALID missing at {run_valid_path}"
    try:
        payload = json.loads(run_valid_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, None, f"RUN_VALID invalid JSON at {run_valid_path}: {exc}"
    verdict = payload.get("overall_verdict")
    return payload, verdict, None


def _write_run_valid_failure(
    run_id: str,
    out_root: Path,
    run_valid_path: Path,
    reason: str,
    dry_run: bool,
) -> None:
    run_dir = (out_root / run_id).resolve()
    stage_dir, outputs_dir = ensure_stage_dirs(run_id, STAGE_NAME, base_dir=out_root)
    failure_catalog_path = outputs_dir / "failure_catalog.jsonl"
    report_path = outputs_dir / "real_v0_smoke_report.json"
    contract_path = outputs_dir / "contract_verdict.json"

    sentinel_row = {
        "event_id": "<none>",
        "strain_path": "",
        "status": "FAIL",
        "fail_reason_code": "missing_real_v0_inputs",
        "notes": "run_valid_blocked",
        "metrics": {},
    }
    with open(failure_catalog_path, "w", encoding="utf-8") as catalog_f:
        catalog_f.write(json.dumps(sentinel_row, sort_keys=True) + "\n")

    contracts = [
        {
            "id": "R08_RUN_VALID_PASS",
            "verdict": "FAIL",
            "violations": [reason],
            "metrics": {
                "run_valid_path": str(run_valid_path),
            },
        }
    ]

    contract_payload = {
        "overall_verdict": "FAIL",
        "contracts": contracts,
        "timestamp": utc_now_iso(),
        "schema_version": "exp_ringdown_08_v1",
    }
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    report_payload = {
        "run_id": run_id,
        "dry_run": dry_run,
        "io_validation": {
            "n_events_total": 0,
            "n_valid": 0,
            "n_invalid": 0,
            "validation_results": [],
        },
        "smoke_inference": {
            "n_smoke_ok": 0,
            "n_smoke_fail": 0,
            "n_smoke_error": 0,
            "results": [],
        },
        "overall_verdict": "FAIL",
        "run_valid_status": reason,
        "sentinel_failure_emitted": True,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    summary = {
        "stage": stage_dir.name,
        "run": run_id,
        "created": utc_now_iso(),
        "inputs": {
            "run_valid": {"path": str(run_valid_path.relative_to(run_dir))},
        },
        "parameters": {"dry_run": dry_run},
        "outputs": {
            "real_v0_smoke_report": "outputs/real_v0_smoke_report.json",
            "failure_catalog": "outputs/failure_catalog.jsonl",
            "contract_verdict": "outputs/contract_verdict.json",
        },
        "results": {
            "overall_verdict": "FAIL",
            "contracts": contracts,
        },
        "version": {"git_sha": _maybe_git_sha()},
    }

    summary_written = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "real_v0_smoke_report": report_path,
            "failure_catalog": failure_catalog_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )


def _load_strain(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load strain data from .npz file."""
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
            raise ValueError(f"no usable 1D strain array in {path}")

    if "t" in keys:
        t = np.asarray(data["t"], dtype=float)
        if t.shape != x.shape:
            raise ValueError(f"t shape != strain shape in {path}")
        dt = float(np.median(np.diff(t)))
    elif "dt" in keys:
        dt = float(np.asarray(data["dt"]).reshape(-1)[0])
        t = dt * np.arange(x.size, dtype=float)
    elif "fs" in keys:
        fs = float(np.asarray(data["fs"]).reshape(-1)[0])
        dt = 1.0 / fs
        t = dt * np.arange(x.size, dtype=float)
    else:
        dt = 1.0 / 4096.0
        t = dt * np.arange(x.size, dtype=float)

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid dt in {path}")

    return x.astype(float), t.astype(float), float(dt)


def _classify_failure(message: str) -> str:
    """Classify failure into a known category."""
    msg = message.lower()
    if "nan" in msg:
        return "nan_inference"
    if "dt" in msg or "shape" in msg or "invalid" in msg or "no usable" in msg:
        return "invalid_input"
    if "pocos puntos" in msg or "fit" in msg or "tau" in msg or "banda" in msg:
        return "nonconvergence"
    if "overflow" in msg or "instability" in msg:
        return "numerical_instability"
    if "timeout" in msg:
        return "timeout"
    return "exception"


def _smoke_inference(
    strain: np.ndarray,
    t: np.ndarray,
    dt: float,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """
    Perform minimal smoke inference on strain data.

    This does NOT attempt accurate parameter estimation.
    It only validates that:
    - The data can be processed without NaNs
    - FFT and basic operations complete
    - Results are finite
    """
    result: Dict[str, Any] = {
        "status": "OK",
        "fail_reason_code": "",
        "notes": "",
        "metrics": {},
    }

    start_time = time.time()

    try:
        x = np.asarray(strain, dtype=float)
        t = np.asarray(t, dtype=float)

        if x.shape != t.shape:
            result["status"] = "FAIL"
            result["fail_reason_code"] = "invalid_input"
            result["notes"] = "t shape != strain shape"
            return result

        if x.size < 64:
            result["status"] = "FAIL"
            result["fail_reason_code"] = "invalid_input"
            result["notes"] = f"strain too short: {x.size}"
            return result

        if not np.all(np.isfinite(x)):
            result["status"] = "FAIL"
            result["fail_reason_code"] = "nan_inference"
            result["notes"] = "input strain contains NaN/Inf"
            return result

        x = x - float(np.mean(x))

        freqs = np.fft.rfftfreq(x.size, d=dt)
        X = np.fft.rfft(x)
        mag = np.abs(X)

        if not np.all(np.isfinite(mag)):
            result["status"] = "FAIL"
            result["fail_reason_code"] = "nan_inference"
            result["notes"] = "FFT produced NaN/Inf"
            return result

        band = (freqs >= 20.0) & (freqs <= 500.0)
        if not np.any(band):
            result["status"] = "FAIL"
            result["fail_reason_code"] = "invalid_input"
            result["notes"] = "no frequencies in 20-500 Hz band"
            return result

        peak_idx = int(np.argmax(mag[band]))
        f_peak = float(freqs[band][peak_idx])
        peak_power = float(mag[band][peak_idx])

        total_power = float(np.sum(mag**2))
        band_power = float(np.sum(mag[band]**2))

        elapsed = time.time() - start_time
        if elapsed > timeout_s:
            result["status"] = "FAIL"
            result["fail_reason_code"] = "timeout"
            result["notes"] = f"inference exceeded {timeout_s}s"
            return result

        result["metrics"] = {
            "f_peak_hz": f_peak,
            "peak_power": peak_power,
            "total_power": total_power,
            "band_power": band_power,
            "band_ratio": band_power / total_power if total_power > 0 else 0.0,
            "n_samples": int(x.size),
            "dt_s": float(dt),
            "elapsed_s": elapsed,
        }

        if not all(np.isfinite(v) for v in result["metrics"].values()):
            result["status"] = "FAIL"
            result["fail_reason_code"] = "nan_inference"
            result["notes"] = "computed metrics contain NaN/Inf"
            return result

    except Exception as exc:
        result["status"] = "ERROR"
        result["fail_reason_code"] = _classify_failure(str(exc))
        result["notes"] = f"{type(exc).__name__}: {exc}"

    return result


def _validate_real_v0_event(
    event: Dict[str, Any],
    base_dir: Path,
) -> Dict[str, Any]:
    """Validate a single real v0 event entry."""
    result: Dict[str, Any] = {
        "valid": False,
        "error": None,
        "event_id": event.get("event_id", "<unknown>"),
    }

    strain_rel = event.get("strain_npz")
    if not strain_rel:
        result["error"] = "missing_strain_npz_field"
        return result

    strain_path = base_dir / strain_rel
    if not strain_path.exists():
        result["error"] = f"strain_npz_not_found:{strain_path}"
        return result

    try:
        data = np.load(strain_path)
        keys = set(data.files)

        strain = None
        for k in ["strain", "h"]:
            if k in keys:
                strain = np.asarray(data[k], dtype=float)
                break

        if strain is None:
            for k in data.files:
                arr = np.asarray(data[k])
                if arr.ndim == 1 and arr.size > 32:
                    strain = np.asarray(arr, dtype=float)
                    break

        if strain is None:
            result["error"] = "no_strain_array"
            return result

        if strain.ndim != 1:
            result["error"] = "strain_not_1d"
            return result

        if strain.size < 64:
            result["error"] = f"strain_too_short:{strain.size}"
            return result

        if not np.all(np.isfinite(strain)):
            result["error"] = "strain_has_nan_or_inf"
            return result

        result["valid"] = True
        result["shape"] = list(strain.shape)

    except Exception as exc:
        result["error"] = f"exception:{type(exc).__name__}:{exc}"

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EXP_RINGDOWN_08: Real v0 smoke gate (contract-first)"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument(
        "--real-v0-events-json",
        type=str,
        default=None,
        help="Path to real v0 events list JSON (optional; uses canonical path if not provided)",
    )
    ap.add_argument(
        "--n-max-cases",
        type=int,
        default=DEFAULT_N_MAX_CASES,
        help=f"Max cases for smoke test (default: {DEFAULT_N_MAX_CASES})",
    )
    ap.add_argument(
        "--timeout-per-case",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Timeout per case in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate IO only, do not run inference",
    )
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    _, run_valid_verdict, run_valid_error = _load_run_valid_payload(run_dir)
    if run_valid_error is not None or run_valid_verdict != "PASS":
        reason = run_valid_error or f"RUN_VALID overall_verdict={run_valid_verdict}"
        _write_run_valid_failure(
            args.run,
            out_root,
            _resolve_run_valid_path(run_dir),
            reason,
            args.dry_run,
        )
        abort_contract(reason)

    if args.real_v0_events_json:
        real_v0_events_path = Path(args.real_v0_events_json).expanduser().resolve()
    else:
        real_v0_events_path = _resolve_real_v0_events_path(run_dir)

    synth_events_path = _resolve_synth_events_path(run_dir)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    failure_catalog_path = outputs_dir / "failure_catalog.jsonl"
    report_path = outputs_dir / "real_v0_smoke_report.json"
    contract_path = outputs_dir / "contract_verdict.json"

    real_v0_events: List[Dict[str, Any]] = []
    real_v0_base_dir: Optional[Path] = None

    if real_v0_events_path is not None and real_v0_events_path.exists():
        try:
            real_v0_events = _read_json(real_v0_events_path)
            if not isinstance(real_v0_events, list):
                real_v0_events = []
            else:
                real_v0_events = sorted(
                    real_v0_events,
                    key=lambda ev: (
                        str(ev.get("event_id", "")),
                        str(ev.get("strain_npz", "")),
                    ),
                )
            real_v0_base_dir = real_v0_events_path.parent
        except Exception:
            real_v0_events = []

    io_compat_pass = False
    io_compat_violations: List[str] = []
    validation_results: List[Dict[str, Any]] = []

    if not real_v0_events:
        io_compat_violations.append("missing_real_v0_inputs")
    else:
        n_valid = 0
        n_invalid = 0

        for event in real_v0_events:
            vr = _validate_real_v0_event(event, real_v0_base_dir)
            validation_results.append(vr)
            if vr["valid"]:
                n_valid += 1
            else:
                n_invalid += 1
                if len(io_compat_violations) < 5:
                    io_compat_violations.append(
                        f"{vr['event_id']}:{vr.get('error', 'unknown')}"
                    )

        if n_valid == 0:
            io_compat_violations.insert(0, "no_valid_events")
        else:
            io_compat_pass = True

    smoke_results: List[Dict[str, Any]] = []
    n_smoke_ok = 0
    n_smoke_fail = 0
    n_smoke_error = 0

    with open(failure_catalog_path, "w", encoding="utf-8") as catalog_f:
        valid_events = [
            (ev, vr)
            for ev, vr in zip(real_v0_events, validation_results)
            if vr["valid"]
        ]
        invalid_events = [
            (ev, vr)
            for ev, vr in zip(real_v0_events, validation_results)
            if not vr["valid"]
        ]

        if io_compat_pass and not args.dry_run:
            for idx, (event, _) in enumerate(valid_events[: args.n_max_cases]):
                event_id = event.get("event_id", f"event_{idx:03d}")
                strain_rel = event.get("strain_npz", "")
                strain_path = real_v0_base_dir / strain_rel

                row: Dict[str, Any] = {
                    "event_id": event_id,
                    "strain_path": str(strain_path),
                    "status": "SKIP",
                    "fail_reason_code": "",
                    "notes": "",
                    "metrics": {},
                }

                try:
                    strain, t, dt = _load_strain(strain_path)
                    smoke = _smoke_inference(
                        strain, t, dt, timeout_s=args.timeout_per_case
                    )

                    row["status"] = smoke["status"]
                    row["fail_reason_code"] = smoke["fail_reason_code"]
                    row["notes"] = smoke["notes"]
                    row["metrics"] = smoke["metrics"]

                    if smoke["status"] == "OK":
                        n_smoke_ok += 1
                    elif smoke["status"] == "FAIL":
                        n_smoke_fail += 1
                    else:
                        n_smoke_error += 1

                except Exception as exc:
                    row["status"] = "ERROR"
                    row["fail_reason_code"] = _classify_failure(str(exc))
                    row["notes"] = f"{type(exc).__name__}: {exc}"
                    n_smoke_error += 1

                smoke_results.append(row)
                catalog_f.write(json.dumps(row, sort_keys=True) + "\n")

        elif args.dry_run and io_compat_pass:
            for idx, (event, _) in enumerate(valid_events[: args.n_max_cases]):
                row = {
                    "event_id": event.get("event_id", f"event_{idx:03d}"),
                    "strain_path": str(real_v0_base_dir / event.get("strain_npz", "")),
                    "status": "SKIP",
                    "fail_reason_code": "",
                    "notes": "dry_run: IO validated, inference skipped",
                    "metrics": {},
                }
                smoke_results.append(row)
                catalog_f.write(json.dumps(row, sort_keys=True) + "\n")

        if not io_compat_pass:
            if len(real_v0_events) == 0:
                row = {
                    "event_id": "<none>",
                    "strain_path": "",
                    "status": "FAIL",
                    "fail_reason_code": "missing_real_v0_inputs",
                    "notes": "no real v0 events available",
                    "metrics": {},
                }
                smoke_results.append(row)
                n_smoke_fail += 1
                catalog_f.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                for event, vr in invalid_events:
                    row = {
                        "event_id": event.get("event_id", "<unknown>"),
                        "strain_path": str(real_v0_base_dir / event.get("strain_npz", "")),
                        "status": "FAIL",
                        "fail_reason_code": "invalid_input",
                        "notes": vr.get("error") or "invalid_real_v0_event",
                        "metrics": {},
                    }
                    smoke_results.append(row)
                    n_smoke_fail += 1
                    catalog_f.write(json.dumps(row, sort_keys=True) + "\n")

    categorized_pass = True
    categorized_violations: List[str] = []

    for row in smoke_results:
        status = row.get("status", "")
        if status not in {"FAIL", "ERROR"}:
            continue
        code = row.get("fail_reason_code", "")
        if code not in ALLOWED_FAIL_CODES:
            categorized_pass = False
            categorized_violations.append(f"invalid_code:{code or '<empty>'}")

    if not io_compat_pass:
        code_in_catalog = "missing_real_v0_inputs"
        if code_in_catalog in ALLOWED_FAIL_CODES:
            pass
        else:
            categorized_pass = False
            categorized_violations.append(f"invalid_code:{code_in_catalog}")

    smoke_pass = io_compat_pass and (args.dry_run or n_smoke_ok > 0) and n_smoke_error == 0

    contracts = [
        {
            "id": "R08_REAL_IO_COMPAT",
            "verdict": "PASS" if io_compat_pass else "FAIL",
            "violations": io_compat_violations[:5],
            "metrics": {
                "real_v0_events_path": str(real_v0_events_path) if real_v0_events_path else None,
                "n_events_total": len(real_v0_events),
                "n_valid": sum(1 for vr in validation_results if vr.get("valid")),
                "n_invalid": sum(1 for vr in validation_results if not vr.get("valid")),
            },
        },
        {
            "id": "R08_PIPELINE_SMOKE",
            "verdict": "PASS" if (args.dry_run or smoke_pass) else "FAIL",
            "violations": [] if (args.dry_run or smoke_pass) else ["smoke_inference_failed"],
            "metrics": {
                "n_max_cases": args.n_max_cases,
                "n_smoke_ok": n_smoke_ok,
                "n_smoke_fail": n_smoke_fail,
                "n_smoke_error": n_smoke_error,
                "dry_run": args.dry_run,
            },
        },
        {
            "id": "R08_FAIL_CATEGORIZED",
            "verdict": "PASS" if categorized_pass else "FAIL",
            "violations": categorized_violations[:5],
            "metrics": {
                "allowed_codes": sorted(ALLOWED_FAIL_CODES),
            },
        },
    ]

    if args.dry_run:
        overall_verdict = "PASS" if (io_compat_pass and categorized_pass) else "FAIL"
    else:
        overall_verdict = "PASS" if all(c["verdict"] == "PASS" for c in contracts) else "FAIL"

    contract_payload = {
        "overall_verdict": overall_verdict,
        "contracts": contracts,
        "timestamp": utc_now_iso(),
        "schema_version": "exp_ringdown_08_v1",
    }

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    report_payload = {
        "run_id": args.run,
        "real_v0_events_path": str(real_v0_events_path) if real_v0_events_path else None,
        "synth_events_path": str(synth_events_path) if synth_events_path else None,
        "n_max_cases": args.n_max_cases,
        "dry_run": args.dry_run,
        "sentinel_failure_emitted": len(real_v0_events) == 0,
        "io_validation": {
            "n_events_total": len(real_v0_events),
            "n_valid": sum(1 for vr in validation_results if vr.get("valid")),
            "n_invalid": sum(1 for vr in validation_results if not vr.get("valid")),
            "validation_results": validation_results[:10],
        },
        "smoke_inference": {
            "n_smoke_ok": n_smoke_ok,
            "n_smoke_fail": n_smoke_fail,
            "n_smoke_error": n_smoke_error,
            "results": smoke_results,
        },
        "overall_verdict": overall_verdict,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    inputs: Dict[str, Any] = {
        "run_valid": {
            "path": str(_resolve_run_valid_path(run_dir).relative_to(run_dir)),
            "sha256": sha256_file(_resolve_run_valid_path(run_dir)),
        },
    }

    if real_v0_events_path is not None and real_v0_events_path.exists():
        try:
            rel = real_v0_events_path.relative_to(run_dir)
            inputs["real_v0_events_list"] = {
                "path": str(rel),
                "sha256": sha256_file(real_v0_events_path),
            }
        except ValueError:
            inputs["real_v0_events_list"] = {
                "path": str(real_v0_events_path),
                "sha256": sha256_file(real_v0_events_path),
            }

    if synth_events_path is not None and synth_events_path.exists():
        inputs["synthetic_events_list"] = {
            "path": str(synth_events_path.relative_to(run_dir)),
            "sha256": sha256_file(synth_events_path),
        }

    summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "created": utc_now_iso(),
        "inputs": inputs,
        "parameters": {
            "real_v0_events_json": str(args.real_v0_events_json) if args.real_v0_events_json else None,
            "n_max_cases": args.n_max_cases,
            "timeout_per_case": args.timeout_per_case,
            "dry_run": args.dry_run,
        },
        "outputs": {
            "real_v0_smoke_report": "outputs/real_v0_smoke_report.json",
            "failure_catalog": "outputs/failure_catalog.jsonl",
            "contract_verdict": "outputs/contract_verdict.json",
        },
        "results": {
            "overall_verdict": overall_verdict,
            "contracts": contracts,
        },
        "version": {"git_sha": _maybe_git_sha()},
    }

    summary_written = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "real_v0_smoke_report": report_path,
            "failure_catalog": failure_catalog_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    if overall_verdict != "PASS":
        abort_contract(
            f"EXP_RINGDOWN_08 FAIL: real v0 smoke gate not satisfied "
            f"(io_compat={io_compat_pass}, smoke={smoke_pass}, categorized={categorized_pass})"
        )

    print(f"OK: EXP_RINGDOWN_08 PASS (n_events={len(real_v0_events)}, n_smoke_ok={n_smoke_ok})")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
