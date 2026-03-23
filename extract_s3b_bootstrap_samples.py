#!/usr/bin/env python3
"""Extract raw s3b bootstrap samples for mode 221 from existing run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from mvp import s3b_multimode_estimates as s3b
from mvp.kerr_qnm_fits import kerr_qnm


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _load_catalog(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if all(isinstance(v, dict) for v in payload.values()):
        return payload
    raise RuntimeError(f"unsupported catalog structure: {path}")


def _event_catalog_entry(catalog: dict[str, Any], event_id: str) -> dict[str, Any] | None:
    raw = catalog.get(event_id)
    return raw if isinstance(raw, dict) else None


def _kerr_reference(catalog_entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(catalog_entry, dict):
        return None
    mf = catalog_entry.get("Mf_source")
    af = catalog_entry.get("af")
    if not isinstance(mf, (int, float)) or not isinstance(af, (int, float)):
        return None
    if not (math.isfinite(float(mf)) and math.isfinite(float(af))):
        return None
    qnm = kerr_qnm(float(mf), float(af), (2, 2, 1))
    return {
        "Mf_source": float(mf),
        "af": float(af),
        "f_221_hz": float(qnm.f_hz),
        "Q_221": float(qnm.Q),
        "tau_221_s": float(qnm.tau_s),
    }


def _extract_run(run_id: str, catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    out_root = Path(os.environ.get("BASURIN_RUNS_ROOT", Path.cwd() / "runs")).resolve()
    run_dir = out_root / run_id

    stage_summary = _load_json(run_dir / "s3b_multimode_estimates" / "stage_summary.json")
    params = stage_summary.get("parameters")
    if not isinstance(params, dict):
        raise RuntimeError(f"missing s3b parameters in {run_id}")

    window_meta_path = s3b._discover_s2_window_meta(run_dir)
    window_meta = _load_json(window_meta_path) if window_meta_path is not None else None

    s3_estimates_path = s3b._resolve_s3_estimates(run_dir, None)
    s3_payload = _load_json(s3_estimates_path)
    band_low, band_high = s3b._load_s3_band(s3_payload, window_meta=window_meta)
    event_id = s3b._resolve_event_id(window_meta=window_meta, s3_payload=s3_payload)
    band_220, band_221, band_strategy = s3b._resolve_mode_bands(
        band_low=band_low,
        band_high=band_high,
        event_id=event_id,
        band_strategy=str(params.get("band_strategy", "kerr_centered_overlap")),
    )

    npz_path = s3b._discover_s2_npz(run_dir)
    signal, fs = s3b._load_signal_from_npz(npz_path, window_meta=window_meta)

    psd_path = params.get("psd_path")
    psd_source = "internal_welch"
    if isinstance(psd_path, str) and psd_path.strip():
        payload = s3b._load_measured_psd(Path(psd_path))
        detector = s3b._detector_from_npz_path(npz_path)
        if detector is not None:
            signal = s3b._whiten_with_measured_psd(signal, fs, detector=detector, psd_payload=payload)
            psd_source = "external_measured_psd"

    selected_t0_offset_ms = s3b._load_s3_selected_t0_offset_ms(s3_estimates_path)
    applied_t0_offset_ms = 0.0
    if selected_t0_offset_ms > 0.0:
        shift = int(round((selected_t0_offset_ms / 1000.0) * fs))
        if 0 < shift < signal.size - 16:
            signal = signal[shift:]
            applied_t0_offset_ms = selected_t0_offset_ms

    mode_221_signal, estimator, mode_221_residual, _ = s3b._prepare_mode_221_bootstrap_inputs(
        signal,
        fs,
        method=str(params.get("method", "hilbert_peakband")),
        band_220=band_220,
        band_221=band_221,
        residual_strategy=str(params.get("bootstrap_221_residual_strategy", "refit_220_each_iter")),
        topology=str(params.get("mode_221_topology", "rigid_spectral_split")),
    )

    seed = int(params.get("seed", 12345)) + 1
    n_bootstrap = int(params.get("n_bootstrap", 200))
    rng = s3b.np.random.default_rng(seed)
    samples, n_failed = s3b._bootstrap_mode_log_samples(
        mode_221_signal,
        fs,
        estimator,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    valid_mask = s3b.np.all(s3b.np.isfinite(samples), axis=1) if samples.size else s3b.np.asarray([], dtype=bool)
    samples = samples[valid_mask] if samples.size else samples
    n_valid = int(samples.shape[0])
    stability = s3b.compute_robust_stability([tuple(row) for row in samples.tolist()])

    event_entry = _event_catalog_entry(catalog or {}, event_id) if event_id else None
    f_vals = [float(math.exp(row[0])) for row in samples.tolist()]
    q_vals = [float(math.exp(row[1])) for row in samples.tolist()]

    return {
        "run_id": run_id,
        "event": event_id,
        "config": {
            "topology": params.get("mode_221_topology"),
            "residual_strategy": params.get("bootstrap_221_residual_strategy"),
            "method": params.get("method"),
            "band_strategy": params.get("band_strategy"),
            "n_bootstrap": n_bootstrap,
            "seed_mode_221": seed,
            "psd_source": psd_source,
            "t0_offset_ms_from_s3": applied_t0_offset_ms,
        },
        "source": {
            "run_dir": str(run_dir),
            "s2_npz": str(npz_path),
            "s3_estimates": str(s3_estimates_path),
            "mode_220_band_hz": [float(band_220[0]), float(band_220[1])],
            "mode_221_band_hz": [float(band_221[0]), float(band_221[1])],
            "mode_221_residual": mode_221_residual,
            "band_strategy_resolved": band_strategy,
        },
        "samples_221": {
            "ln_f": [float(row[0]) for row in samples.tolist()],
            "ln_Q": [float(row[1]) for row in samples.tolist()],
            "f_hz": f_vals,
            "Q": q_vals,
            "n_valid": n_valid,
            "n_total": n_bootstrap,
            "n_failed": int(n_failed + (0 if samples.size == 0 else int(s3b.np.size(valid_mask) - s3b.np.count_nonzero(valid_mask)))),
        },
        "stability_recomputed": stability,
        "kerr_reference_221": _kerr_reference(event_entry),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT")
    ap.add_argument("--catalog", default=None, help="Optional path to gwtc_events_t0.json for Kerr reference")
    ap.add_argument("--run-id", action="append", required=True, help="Run id to extract (repeatable)")
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).resolve())

    catalog = None
    if args.catalog:
        catalog = _load_catalog(Path(args.catalog))

    payload = {
        "schema_version": "s3b_bootstrap_samples_v1",
        "runs_root": str(Path(os.environ.get("BASURIN_RUNS_ROOT", Path.cwd() / "runs")).resolve()),
        "runs": [_extract_run(run_id, catalog=catalog) for run_id in args.run_id],
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
