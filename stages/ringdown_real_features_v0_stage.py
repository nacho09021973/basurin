#!/usr/bin/env python3
"""
stages/ringdown_real_features_v0_stage.py
---------------------------------------
Canonical stage: derive minimal features from real ringdown observables.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1], _here.parents[2]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

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

EXIT_CONTRACT_FAIL = 2
STAGE_NAME_DEFAULT = "ringdown_real_features_v0"


def _abort(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_single_jsonl(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path} debe tener exactamente 1 línea (tiene {len(lines)})")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido en {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} debe contener un objeto JSON")
    return payload


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} no es finito: {value}")
    return value


def _require_positive(value: float, label: str) -> float:
    value = _require_finite(value, label)
    if value <= 0:
        raise ValueError(f"{label} debe ser > 0 (got {value})")
    return value


def _coerce_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} debe ser entero (>0)")
    if isinstance(value, int):
        ivalue = value
    elif isinstance(value, float) and value.is_integer():
        ivalue = int(value)
    else:
        raise ValueError(f"{label} debe ser entero (>0)")
    if ivalue <= 0:
        raise ValueError(f"{label} debe ser > 0 (got {ivalue})")
    return ivalue


def _write_failure(
    stage_dir: Path,
    stage_name: str,
    run_id: str,
    params: dict[str, Any],
    inputs: list[dict[str, str]],
    reason: str,
) -> None:
    summary = {
        "stage": stage_name,
        "run": run_id,
        "created": utc_now_iso(),
        "version": "v0",
        "parameters": params,
        "inputs": inputs,
        "outputs": [],
        "verdict": "FAIL",
        "error": reason,
    }
    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {"stage_summary": summary_path},
        extra={"inputs": inputs, "verdict": "FAIL", "error": reason},
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Canonical stage: minimal features from real ringdown observables"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument(
        "--stage-name",
        default=STAGE_NAME_DEFAULT,
        help=f"stage name (default: {STAGE_NAME_DEFAULT})",
    )
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        _abort(str(exc))

    stage_name = args.stage_name
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name, base_dir=out_root)

    observables_path = (
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl"
    )
    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    input_paths = {
        "observables": observables_path,
        "H1_rd": inputs_dir / "H1_rd.npz",
        "L1_rd": inputs_dir / "L1_rd.npz",
        "segments": inputs_dir / "segments_rd.json",
    }

    params = {"run": args.run, "stage_name": stage_name}
    inputs_list: list[dict[str, str]] = []
    missing = []
    for path in input_paths.values():
        if not path.exists():
            missing.append(str(path))
            inputs_list.append({"path": str(path.relative_to(run_dir)), "sha256": ""})
        else:
            inputs_list.append(
                {"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)}
            )

    if missing:
        reason = f"missing inputs: {', '.join(missing)}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    try:
        observables = _read_single_jsonl(observables_path)
    except Exception as exc:
        reason = f"no se pudo leer observables.jsonl {observables_path}: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    detectors = observables.get("detectors")
    if not isinstance(detectors, list):
        reason = "observables missing detectors list"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)
    if "H1" not in detectors or "L1" not in detectors:
        reason = "observables must include detectors H1 and L1"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    fs_hz = observables.get("fs_hz")
    if fs_hz is None:
        reason = "observables.fs_hz missing"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)
    try:
        fs_hz = _require_positive(float(fs_hz), "fs_hz")
    except Exception as exc:
        reason = f"fs_hz inválido: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    t0_gps = observables.get("t0_gps")
    if t0_gps is not None:
        try:
            t0_gps = _require_finite(float(t0_gps), "t0_gps")
        except Exception as exc:
            reason = f"t0_gps inválido: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

    def _det_field(label: str) -> dict[str, Any]:
        payload = observables.get(label)
        if not isinstance(payload, dict):
            raise ValueError(f"observables.{label} missing or not a dict")
        if "H1" not in payload or "L1" not in payload:
            raise ValueError(f"observables.{label} missing H1/L1")
        return payload

    try:
        n_samples_raw = _det_field("n_samples")
        rms_raw = _det_field("rms")
        peak_raw = _det_field("peak_abs")
    except Exception as exc:
        reason = f"observables inválido: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    n_samples: dict[str, int] = {}
    rms: dict[str, float] = {}
    peak_abs: dict[str, float] = {}
    duration_s: dict[str, float] = {}
    snr_proxy: dict[str, float] = {}
    log10_rms: dict[str, float] = {}
    log10_peak_abs: dict[str, float] = {}

    for det in ["H1", "L1"]:
        try:
            n_samples[det] = _coerce_int(n_samples_raw[det], f"n_samples[{det}]")
            rms_val = _require_positive(float(rms_raw[det]), f"rms[{det}]")
            peak_val = _require_finite(float(peak_raw[det]), f"peak_abs[{det}]")
            if peak_val < 0:
                raise ValueError(f"peak_abs[{det}] debe ser >= 0 (got {peak_val})")
            rms[det] = rms_val
            peak_abs[det] = peak_val
            duration = n_samples[det] / fs_hz
            duration_s[det] = _require_positive(float(duration), f"duration_s[{det}]")
            snr = peak_val / rms_val
            snr_proxy[det] = _require_finite(float(snr), f"snr_proxy[{det}]")
            if rms_val <= 0:
                raise ValueError(f"log10_rms[{det}] requiere rms > 0")
            if peak_val <= 0:
                raise ValueError(f"log10_peak_abs[{det}] requiere peak_abs > 0")
            log10_rms[det] = _require_finite(math.log10(rms_val), f"log10_rms[{det}]")
            log10_peak_abs[det] = _require_finite(
                math.log10(peak_val), f"log10_peak_abs[{det}]"
            )
        except Exception as exc:
            reason = f"observables inválido para {det}: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

    features = {
        "run_id": args.run,
        "t0_gps": t0_gps,
        "fs_hz": fs_hz,
        "n_samples": n_samples,
        "duration_s": duration_s,
        "rms": rms,
        "peak_abs": peak_abs,
        "snr_proxy": snr_proxy,
        "log10_rms": log10_rms,
        "log10_peak_abs": log10_peak_abs,
    }

    features_path = outputs_dir / "features.jsonl"
    with open(features_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(features, sort_keys=True))
        f.write("\n")

    outputs_list = [
        {"path": str(features_path.relative_to(run_dir)), "sha256": sha256_file(features_path)}
    ]

    summary = {
        "stage": stage_name,
        "run": args.run,
        "created": utc_now_iso(),
        "version": "v0",
        "parameters": params,
        "inputs": inputs_list,
        "outputs": outputs_list,
        "verdict": "PASS",
        "format": {"features_jsonl": "single_record_per_run", "detectors": ["H1", "L1"]},
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {"features": features_path, "stage_summary": summary_path},
        extra={"inputs": inputs_list},
    )

    print(f"OK: {stage_name} PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
