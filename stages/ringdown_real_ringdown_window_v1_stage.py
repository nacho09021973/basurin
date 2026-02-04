#!/usr/bin/env python3
"""
stages/ringdown_real_ringdown_window_v1_stage.py
------------------------------------------------
Canonical stage: crop ringdown window anchored to t0_gps for real data.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
STAGE_NAME_DEFAULT = "ringdown_real_ringdown_window_v1"


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
    if not np.isfinite(value):
        raise ValueError(f"{label} no es finito: {value}")
    return value


def _require_positive(value: float, label: str) -> float:
    value = _require_finite(value, label)
    if value <= 0:
        raise ValueError(f"{label} debe ser > 0 (got {value})")
    return value


def _load_window_npz(path: Path) -> tuple[np.ndarray, float, float, float]:
    try:
        data = np.load(path)
    except Exception as exc:
        raise RuntimeError(f"no se pudo leer npz {path}: {exc}") from exc

    if "strain" in data:
        strain = np.asarray(data["strain"], dtype=float)
    else:
        raise RuntimeError(f"npz {path} no contiene 'strain'")

    if strain.ndim != 1:
        raise RuntimeError(f"strain debe ser 1D en {path}")
    if not np.all(np.isfinite(strain)):
        raise RuntimeError(f"strain contiene NaN/Inf en {path}")

    if "gps_start" not in data:
        raise RuntimeError(f"npz {path} no contiene 'gps_start'")
    gps_start = float(np.asarray(data["gps_start"]).reshape(-1)[0])

    if "duration_s" not in data:
        raise RuntimeError(f"npz {path} no contiene 'duration_s'")
    duration_s = float(np.asarray(data["duration_s"]).reshape(-1)[0])

    fs = None
    if "sample_rate_hz" in data:
        fs = float(np.asarray(data["sample_rate_hz"]).reshape(-1)[0])
    if fs is None:
        duration_s = _require_positive(duration_s, "duration_s")
        fs = float(strain.size / duration_s)

    fs = _require_positive(fs, "sample_rate_hz")
    gps_start = _require_finite(gps_start, "gps_start")
    duration_s = _require_positive(duration_s, "duration_s")

    return strain, gps_start, duration_s, fs


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
        "version": "v1",
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
        description="Canonical stage: crop ringdown window anchored to t0_gps"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument(
        "--stage-name",
        default=STAGE_NAME_DEFAULT,
        help=f"stage name (default: {STAGE_NAME_DEFAULT})",
    )
    ap.add_argument("--dt-start-s", type=float, default=0.0, help="offset from t0_gps")
    ap.add_argument("--duration-s", type=float, default=0.25, help="window duration in seconds")
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

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    input_paths = {
        "H1": inputs_dir / "H1_rd.npz",
        "L1": inputs_dir / "L1_rd.npz",
        "segments": inputs_dir / "segments_rd.json",
        "observables": run_dir
        / "ringdown_real_observables_v0"
        / "outputs"
        / "observables.jsonl",
    }

    params = {
        "run": args.run,
        "stage_name": stage_name,
        "dt_start_s": float(args.dt_start_s),
        "duration_s": float(args.duration_s),
    }
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
        observables = _read_single_jsonl(input_paths["observables"])
    except Exception as exc:
        reason = f"no se pudo leer observables.jsonl {input_paths['observables']}: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    if "t0_gps" not in observables:
        reason = "observables.jsonl no contiene t0_gps"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)
    try:
        t0_gps = _require_finite(float(observables["t0_gps"]), "t0_gps")
    except Exception as exc:
        reason = f"t0_gps inválido: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    duration_s = _require_positive(float(args.duration_s), "duration_s")
    dt_start_s = float(args.dt_start_s)
    window_start = t0_gps + dt_start_s

    outputs_map: dict[str, Path] = {}
    for det in ["H1", "L1"]:
        try:
            strain, gps_start, _, fs = _load_window_npz(input_paths[det])
        except Exception as exc:
            reason = f"error leyendo {input_paths[det]}: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

        i_start = int(round((window_start - gps_start) * fs))
        n_out = int(round(duration_s * fs))
        if n_out <= 0:
            reason = f"duration_s demasiado corta para {det}: n_out={n_out}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)
        i_end = i_start + n_out
        if i_start < 0 or i_end > strain.size:
            reason = (
                f"window fuera de rango para {det}: "
                f"i_start={i_start}, i_end={i_end}, n={strain.size}"
            )
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

        strain_out = np.asarray(strain[i_start:i_end], dtype=float)
        out_path = outputs_dir / f"{det}_rd.npz"
        np.savez(
            out_path,
            strain=strain_out,
            gps_start=float(window_start),
            duration_s=float(duration_s),
            sample_rate_hz=float(fs),
        )
        outputs_map[det] = out_path

    segments_out = outputs_dir / "segments_rd.json"
    shutil.copy2(input_paths["segments"], segments_out)
    outputs_map["segments"] = segments_out

    outputs_list = [
        {"path": str(outputs_map["H1"].relative_to(run_dir)), "sha256": sha256_file(outputs_map["H1"])},
        {"path": str(outputs_map["L1"].relative_to(run_dir)), "sha256": sha256_file(outputs_map["L1"])},
        {
            "path": str(segments_out.relative_to(run_dir)),
            "sha256": sha256_file(segments_out),
        },
    ]

    summary = {
        "stage": stage_name,
        "run": args.run,
        "created": utc_now_iso(),
        "version": "v1",
        "parameters": params,
        "inputs": inputs_list,
        "outputs": outputs_list,
        "verdict": "PASS",
        "window": {
            "t0_gps": t0_gps,
            "dt_start_s": dt_start_s,
            "duration_s": duration_s,
        },
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "H1_rd": outputs_map["H1"],
            "L1_rd": outputs_map["L1"],
            "segments": segments_out,
            "stage_summary": summary_path,
        },
        extra={"inputs": inputs_list},
    )

    print(f"OK: {stage_name} PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
