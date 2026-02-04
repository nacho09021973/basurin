#!/usr/bin/env python3
"""
stages/ringdown_real_inference_v0_stage.py
---------------------------------------
Canonical stage: minimal physical inference on real ringdown window.
"""
from __future__ import annotations

import argparse
import json
import math
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
STAGE_NAME_DEFAULT = "ringdown_real_inference_v0"
STRAIN_KEYS = ["strain", "h", "data", "x", "rd_strain"]
FS_KEYS = ["sample_rate_hz", "fs_hz", "fs", "sample_rate", "sr"]
MIN_BLOCKS_TAU = 5


def _parse_band_hz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("band-hz debe tener formato 'low,high'")
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band-hz debe ser numérico") from exc


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


def _load_strain_and_fs(
    path: Path, fallback_fs: float | None
) -> tuple[np.ndarray, float, str]:
    try:
        data = np.load(path)
    except Exception as exc:
        raise RuntimeError(f"no se pudo leer npz {path}: {exc}") from exc

    strain = None
    for key in STRAIN_KEYS:
        if key in data:
            strain = np.asarray(data[key], dtype=float)
            break
    if strain is None:
        available = ", ".join(list(data.files))
        raise RuntimeError(
            f"strain no encontrado en {path}; claves disponibles: [{available}]"
        )
    if strain.ndim != 1:
        raise RuntimeError(f"strain debe ser 1D en {path}")
    if not np.all(np.isfinite(strain)):
        raise RuntimeError(f"strain contiene NaN/Inf en {path}")

    fs = None
    fs_source = "fallback"
    for key in FS_KEYS:
        if key in data:
            fs = float(np.asarray(data[key]).reshape(-1)[0])
            fs_source = "npz"
            break
    if fs is None:
        fs = fallback_fs
    if fs is None:
        raise RuntimeError(f"fs_hz no encontrado en {path} ni en observables/features")
    fs = _require_positive(float(fs), f"fs_hz ({path})")

    return strain, fs, fs_source


def _estimate_f_peak(
    strain: np.ndarray, fs_hz: float, band_hz: list[float]
) -> tuple[float, float, float]:
    n = int(strain.size)
    if n <= 0:
        raise RuntimeError("strain vacío")
    window = np.hanning(n)
    spectrum = np.fft.rfft(strain * window)
    if not np.all(np.isfinite(spectrum)):
        raise RuntimeError("FFT contiene NaN/Inf")
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        raise RuntimeError(
            f"no hay bins en la banda {band_hz[0]}-{band_hz[1]} Hz"
        )

    band_freqs = freqs[mask]
    band_mag = mag[mask]
    idx = int(np.argmax(band_mag))
    f_peak = float(band_freqs[idx])
    peak_mag = float(band_mag[idx])
    df_hz = float(fs_hz / n)

    _require_finite(f_peak, "f_peak_hz")
    _require_finite(peak_mag, "peak_mag")
    _require_positive(df_hz, "df_hz")

    return f_peak, peak_mag, df_hz


def _estimate_tau(
    strain: np.ndarray, fs_hz: float,
) -> tuple[float | None, list[str], dict[str, Any]]:
    """Estimate exponential decay time.  Returns (tau_s, notes, metrics)."""
    notes: list[str] = []
    metrics: dict[str, Any] = {
        "block_len_s": None,
        "n_blocks_total": 0,
        "n_blocks_valid": 0,
        "reject_reasons": {},
    }
    n = int(strain.size)
    if n <= 0:
        notes.append("WARN: strain vacío para tau")
        return None, notes, metrics

    i0 = int(np.argmax(np.abs(strain)))
    block = max(16, int(0.01 * fs_hz))
    metrics["block_len_s"] = block / fs_hz

    amplitudes: list[float] = []
    times: list[float] = []
    start = i0
    while start < n:
        end = min(start + block, n)
        block_data = strain[start:end]
        if block_data.size == 0:
            break
        amp = float(np.sqrt(np.mean(np.square(block_data))))
        center = start + (block_data.size - 1) / 2.0
        t = (center - i0) / fs_hz
        amplitudes.append(amp)
        times.append(float(t))
        start = end

    metrics["n_blocks_total"] = len(amplitudes)

    if not amplitudes:
        notes.append("WARN: no se pudo construir envolvente para tau")
        return None, notes, metrics

    amp0 = amplitudes[0]
    if amp0 <= 0:
        notes.append("WARN: amplitude_0 no positiva para tau")
        return None, notes, metrics

    frac = 0.05
    reject_reasons: dict[str, int] = {}
    valid: list[tuple[float, float]] = []
    for t, a in zip(times, amplitudes):
        if a <= 0:
            reject_reasons["zero_amplitude"] = reject_reasons.get("zero_amplitude", 0) + 1
        elif a < frac * amp0:
            reject_reasons["low_power"] = reject_reasons.get("low_power", 0) + 1
        else:
            valid.append((t, a))

    metrics["n_blocks_valid"] = len(valid)
    metrics["reject_reasons"] = reject_reasons

    if len(valid) < MIN_BLOCKS_TAU:
        notes.append("WARN: menos de 5 bloques válidos para tau")
        return None, notes, metrics

    t_vals = np.array([item[0] for item in valid], dtype=float)
    a_vals = np.array([item[1] for item in valid], dtype=float)
    y = np.log(a_vals)
    A = np.vstack([np.ones_like(t_vals), t_vals]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    b = float(coeffs[1])

    if not math.isfinite(b):
        notes.append("WARN: ajuste lineal inválido para tau")
        return None, notes, metrics
    if b >= 0:
        notes.append("WARN: pendiente no negativa en ajuste de tau")
        return None, notes, metrics

    tau_s = -1.0 / b
    if not math.isfinite(tau_s) or tau_s <= 0:
        notes.append("WARN: tau no finito o no positivo")
        return None, notes, metrics

    return float(tau_s), notes, metrics


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
        description="Canonical stage: minimal physical inference on real ringdown window"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument(
        "--stage-name",
        default=STAGE_NAME_DEFAULT,
        help=f"stage name (default: {STAGE_NAME_DEFAULT})",
    )
    ap.add_argument(
        "--window-stage",
        default="ringdown_real_ringdown_window",
        help="stage name for ringdown window inputs",
    )
    ap.add_argument("--band-hz", default="150,400", type=_parse_band_hz)
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

    inputs_dir = run_dir / args.window_stage / "outputs"
    input_paths = {
        "features": run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        "observables": run_dir
        / "ringdown_real_observables_v0"
        / "outputs"
        / "observables.jsonl",
        "H1": inputs_dir / "H1_rd.npz",
        "L1": inputs_dir / "L1_rd.npz",
        "segments": inputs_dir / "segments_rd.json",
    }

    params = {
        "run": args.run,
        "stage_name": stage_name,
        "window_stage": args.window_stage,
        "band_hz": args.band_hz,
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
        features = _read_single_jsonl(input_paths["features"])
        observables = _read_single_jsonl(input_paths["observables"])
    except Exception as exc:
        reason = f"no se pudo leer features/observables: {exc}"
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

    fs_hz_fallback = observables.get("fs_hz")
    if fs_hz_fallback is None:
        fs_hz_fallback = features.get("fs_hz") if isinstance(features, dict) else None
    if fs_hz_fallback is None:
        reason = "fs_hz missing in observables/features"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)
    try:
        fs_hz_fallback = _require_positive(float(fs_hz_fallback), "fs_hz")
    except Exception as exc:
        reason = f"fs_hz inválido: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    n_samples = observables.get("n_samples")
    if not isinstance(n_samples, dict) or "H1" not in n_samples or "L1" not in n_samples:
        reason = "observables.n_samples missing H1/L1"
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

    fit: dict[str, dict[str, Any]] = {}
    window: dict[str, Any] = {
        "stage": args.window_stage,
        "fs_hz": None,
        "duration_s": {},
        "n_samples": {},
    }
    decision_reasons: list[str] = []
    decision_verdict = "PASS"
    contract_verdict = "PASS"
    contract_reasons: list[str] = []
    tau_estimator_det: dict[str, dict[str, Any]] = {}

    for det in ["H1", "L1"]:
        try:
            strain, fs_det, fs_source = _load_strain_and_fs(
                input_paths[det], fs_hz_fallback
            )
            f_peak_hz, peak_mag, df_hz = _estimate_f_peak(strain, fs_det, args.band_hz)
        except Exception as exc:
            reason = f"no se pudo estimar f_peak para {det}: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

        notes: list[str] = []
        if window["fs_hz"] is None:
            window["fs_hz"] = fs_det
        elif abs(fs_det - float(window["fs_hz"])) > 1e-12:
            contract_verdict = "INSPECT"
            contract_reasons.append(
                f"{det}: fs_hz mismatch (npz={fs_det}, window={window['fs_hz']})"
            )
            notes.append(f"WARN: fs_hz distinto ({fs_det} vs {window['fs_hz']})")
        if fs_source != "npz":
            contract_verdict = "INSPECT"
            contract_reasons.append(f"{det}: fs_hz inferred from observables/features")

        tau_s, tau_notes, tau_metrics = _estimate_tau(strain, fs_det)
        notes.extend(tau_notes)
        tau_estimator_det[det] = tau_metrics

        q_val = None
        if tau_s is not None:
            q_val = float(math.pi * f_peak_hz * tau_s)
            if not math.isfinite(q_val):
                notes.append("WARN: Q no finito")
                q_val = None

        if tau_s is None:
            decision_verdict = "INSPECT"
            nv = tau_metrics["n_blocks_valid"]
            if nv < MIN_BLOCKS_TAU:
                decision_reasons.append(
                    f"{det}: tau_s no estimado (n_blocks_valid={nv} < {MIN_BLOCKS_TAU})"
                )
            else:
                decision_reasons.append(f"{det}: tau_s no estimado")

        n_samples_det = int(strain.size)
        window["n_samples"][det] = n_samples_det
        fs_ref = float(window["fs_hz"])
        window["duration_s"][det] = n_samples_det / fs_ref

        fit[det] = {
            "n_samples": int(strain.size),
            "df_hz": df_hz,
            "f_peak_hz": f_peak_hz,
            "peak_mag": peak_mag,
            "tau_s": tau_s,
            "Q": q_val,
            "notes": notes,
        }

        expected_df = fs_ref / n_samples_det
        if abs(df_hz - expected_df) > 1e-12:
            contract_verdict = "INSPECT"
            contract_reasons.append(
                f"{det}: df_hz inconsistente (df_hz={df_hz}, esperado={expected_df})"
            )

        expected_duration = n_samples_det / fs_ref
        if abs(window["duration_s"][det] - expected_duration) > 1e-12:
            contract_verdict = "INSPECT"
            contract_reasons.append(
                f"{det}: duration_s inconsistente (duration_s={window['duration_s'][det]}, esperado={expected_duration})"
            )

    if window["fs_hz"] is None:
        contract_verdict = "INSPECT"
        contract_reasons.append("fs_hz missing from window npz inputs")
        window["fs_hz"] = float(fs_hz_fallback)

    features_payload = {
        key: features.get(key)
        for key in ["snr_proxy", "rms", "peak_abs"]
        if isinstance(features, dict) and key in features
    }
    if isinstance(features, dict) and "duration_s" in features:
        features_payload["features_duration_s"] = features.get("duration_s")

    # --- tau_estimator audit block ---
    fs_report = float(window["fs_hz"])
    block_samples = max(16, int(0.01 * fs_report))
    tau_estimator: dict[str, Any] = {
        "block_len_s": block_samples / fs_report,
        "min_blocks_required": MIN_BLOCKS_TAU,
    }
    for det_key, det_metrics in tau_estimator_det.items():
        tau_estimator[det_key] = {
            "n_blocks_total": det_metrics["n_blocks_total"],
            "n_blocks_valid": det_metrics["n_blocks_valid"],
            "reject_reasons": det_metrics["reject_reasons"],
        }

    # Contract: if fit.<IFO>.tau_s is null, tau_estimator.<IFO> must exist
    for det in ["H1", "L1"]:
        if fit[det]["tau_s"] is None:
            te_det = tau_estimator.get(det)
            if (
                te_det is None
                or "n_blocks_valid" not in te_det
            ):
                contract_verdict = "INSPECT"
                contract_reasons.append(
                    f"{det}: tau_estimator metrics missing for null tau_s"
                )

    report = {
        "run_id": args.run,
        "t0_gps": t0_gps,
        "fs_hz": fs_report,
        "band_hz": [float(args.band_hz[0]), float(args.band_hz[1])],
        "features": features_payload,
        "window": window,
        "fit": fit,
        "tau_estimator": tau_estimator,
        "decision": {
            "verdict": decision_verdict,
            "reasons": decision_reasons,
        },
    }

    report_path = outputs_dir / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    verdict_path = outputs_dir / "contract_verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(
            {"verdict": contract_verdict, "reasons": contract_reasons},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    outputs_list = [
        {"path": str(report_path.relative_to(run_dir)), "sha256": sha256_file(report_path)},
        {"path": str(verdict_path.relative_to(run_dir)), "sha256": sha256_file(verdict_path)},
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
        "format": {"inference_report": "single_record_per_run", "detectors": ["H1", "L1"]},
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "inference_report": report_path,
            "contract_verdict": verdict_path,
            "stage_summary": summary_path,
        },
        extra={"inputs": inputs_list},
    )

    print(f"OK: {stage_name} PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
