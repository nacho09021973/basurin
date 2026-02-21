#!/usr/bin/env python3
"""Canonical stage s5_event_row: build a traceable row for one event + one t0 point."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE = "s5_event_row"


def _fatal(msg: str) -> None:
    print(f"ERROR: [{STAGE}] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _safe_float(val: Any) -> float | None:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _collect_s3b_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    modes = payload.get("modes") if isinstance(payload.get("modes"), list) else []
    by_label: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if isinstance(mode, dict):
            by_label[str(mode.get("label", ""))] = mode

    def _mode(label: str) -> dict[str, Any]:
        mode = by_label.get(label, {})
        fit = mode.get("fit") if isinstance(mode.get("fit"), dict) else {}
        stability = fit.get("stability") if isinstance(fit.get("stability"), dict) else {}
        return {
            "ln_f": _safe_float(mode.get("ln_f")),
            "ln_Q": _safe_float(mode.get("ln_Q")),
            "valid_fraction": _safe_float(stability.get("valid_fraction")),
            "spans": {
                "lnf_span": _safe_float(stability.get("lnf_span")),
                "lnQ_span": _safe_float(stability.get("lnQ_span")),
            },
            "n_failed": int(stability.get("n_failed", 0)) if str(stability.get("n_failed", "")).isdigit() else stability.get("n_failed"),
        }

    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    return {
        "modes": {
            "220": _mode("220"),
            "221": _mode("221"),
        },
        "verdict": results.get("verdict"),
        "quality_flags": list(results.get("quality_flags", [])) if isinstance(results.get("quality_flags"), list) else [],
    }


def _collect_s4c_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    deltas = payload.get("deltas") if isinstance(payload.get("deltas"), dict) else {}
    verdict = payload.get("verdict")
    if verdict is None:
        verdict = payload.get("kerr_consistent")
    return {
        "chi_best": _safe_float(payload.get("chi_best")),
        "d2_min": _safe_float(payload.get("d2_min")),
        "deltas": {
            "delta_logfreq": _safe_float(deltas.get("delta_logfreq", payload.get("delta_logfreq"))),
            "delta_logQ": _safe_float(deltas.get("delta_logQ", payload.get("delta_logQ"))),
        },
        "verdict": verdict,
    }


def _find_geometry_h5(subrun_dir: Path, *payloads: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []

    def _scan(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _scan(v)
        elif isinstance(obj, list):
            for v in obj:
                _scan(v)
        elif isinstance(obj, str) and obj.lower().endswith((".h5", ".hdf5")):
            p = Path(obj)
            if not p.is_absolute():
                p = subrun_dir / p
            candidates.append(p.resolve())

    for payload in payloads:
        _scan(payload)

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _collect_geometry(subrun_dir: Path, *payloads: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    path = _find_geometry_h5(subrun_dir, *payloads)
    if path is None:
        return None, None, "geometry_hdf5_not_found"

    try:
        import h5py  # type: ignore
    except Exception:
        return None, path, "h5py_not_available"

    with h5py.File(path, "r") as h5:
        z_values = [float(x) for x in h5["z_grid"][...].reshape(-1)] if "z_grid" in h5 else []
        a_values = [float(x) for x in h5["A_of_z"][...].reshape(-1)] if "A_of_z" in h5 else []
        f_values = [float(x) for x in h5["f_of_z"][...].reshape(-1)] if "f_of_z" in h5 else []

    return {
        "n_points": len(z_values),
        "z_grid_stats": _stats(z_values),
        "A_of_z_stats": _stats(a_values),
        "f_of_z_stats": _stats(f_values),
    }, path, None


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s5 event row")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--t0-ms", required=True, type=int)
    args = ap.parse_args()

    os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())
    out_root = resolve_out_root("runs")

    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run_id, STAGE, base_dir=out_root)

    seed_dir = out_root / args.run_id / "experiment" / f"t0_sweep_full_seed{int(args.seed)}"
    subrun_id = f"{args.run_id}__t0ms{int(args.t0_ms):04d}"
    subrun_dir = seed_dir / "runs" / subrun_id

    window_meta_path = subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
    s3b_path = subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    s4c_path = subrun_dir / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json"

    required = {
        "window_meta": window_meta_path,
        "multimode_estimates": s3b_path,
        "kerr_consistency": s4c_path,
    }
    missing = [f"{k}: {p}" for k, p in required.items() if not p.exists()]
    if missing:
        _fatal("Missing required inputs: " + "; ".join(missing))

    window_meta = _read_json(window_meta_path)
    s3b = _read_json(s3b_path)
    s4c = _read_json(s4c_path)

    inputs = []
    for label, path in required.items():
        inputs.append(
            {
                "label": label,
                "path": str(path.relative_to(out_root / args.run_id)),
                "sha256": sha256_file(path),
            }
        )

    geometry, geometry_path, geometry_reason = _collect_geometry(subrun_dir, window_meta, s3b, s4c)
    if geometry_path is not None and geometry_path.exists():
        inputs.append(
            {
                "label": "geometry_hdf5",
                "path": str(geometry_path.relative_to(out_root / args.run_id)) if geometry_path.is_relative_to(out_root / args.run_id) else str(geometry_path),
                "sha256": sha256_file(geometry_path),
            }
        )

    row = {
        "schema_version": "s5_event_row_v1",
        "run_id": args.run_id,
        "subrun_id": subrun_id,
        "seed": int(args.seed),
        "t0_ms": int(args.t0_ms),
        "event_window": window_meta,
        "s3b": _collect_s3b_metrics(s3b),
        "s4c": _collect_s4c_metrics(s4c),
        "geometry": geometry if geometry is not None else {"value": None, "reason": geometry_reason},
        "artifacts": {
            "window_meta": {"path": str(window_meta_path.relative_to(out_root / args.run_id)), "sha256": sha256_file(window_meta_path)},
            "multimode_estimates": {"path": str(s3b_path.relative_to(out_root / args.run_id)), "sha256": sha256_file(s3b_path)},
            "kerr_consistency": {"path": str(s4c_path.relative_to(out_root / args.run_id)), "sha256": sha256_file(s4c_path)},
        },
    }
    if geometry_path is not None and geometry_path.exists():
        row["artifacts"]["geometry_hdf5"] = {
            "path": str(geometry_path.relative_to(out_root / args.run_id)) if geometry_path.is_relative_to(out_root / args.run_id) else str(geometry_path),
            "sha256": sha256_file(geometry_path),
        }

    row_path = outputs_dir / "event_row.json"
    write_json_atomic(row_path, row)

    outputs = [{"path": str(row_path.relative_to(out_root / args.run_id)), "sha256": sha256_file(row_path)}]
    summary = {
        "stage": STAGE,
        "run": args.run_id,
        "runs_root": str(out_root),
        "created": utc_now_iso(),
        "version": "v1",
        "parameters": {
            "seed": int(args.seed),
            "t0_ms": int(args.t0_ms),
            "seed_dir": str(seed_dir),
            "subrun_id": subrun_id,
        },
        "inputs": inputs,
        "outputs": outputs,
        "verdict": "PASS",
        "results": {
            "subrun_id": subrun_id,
            "geometry_present": geometry is not None,
        },
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(stage_dir, {"event_row": row_path, "stage_summary": summary_path}, extra={"inputs": inputs})
    print("OK: s5_event_row PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
