#!/usr/bin/env python3
"""MVP Stage 6b: 3D information geometry with explicit censoring audit."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import sha256_file, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage
from mvp.path_utils import resolve_run_scoped_input

STAGE = "s6b_information_geometry_3d"


def _resolve_estimates_path(run_dir: Path, estimates_path_override: str | None) -> Path:
    return resolve_run_scoped_input(
        run_dir,
        estimates_path_override,
        default_rel="s3_ringdown_estimates/outputs/estimates.json",
        arg_name="--estimates-path",
    )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _mode_row(multimode: dict[str, Any], label: str) -> dict[str, Any] | None:
    for row in multimode.get("modes", []):
        if isinstance(row, dict) and str(row.get("label")) == label:
            return row
    return None


def _infer_censoring(multimode: dict[str, Any], mode_221: dict[str, Any] | None) -> tuple[bool, str | None, float]:
    flags = multimode.get("results", {}).get("quality_flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(x) for x in flags]

    has_mode = mode_221 is not None
    has_coords = has_mode and isinstance(mode_221.get("ln_f"), (int, float)) and isinstance(mode_221.get("ln_Q"), (int, float))
    flagged_221 = [f for f in flags if "221" in f]

    if has_coords and not flagged_221:
        return True, None, 1.0

    reason = flagged_221[0] if flagged_221 else (flags[0] if flags else "mode_221_missing_or_degraded")
    return False, reason, 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s6b information geometry 3D")
    ap.add_argument("--run", required=True)
    ap.add_argument("--estimates-path", default=None)
    args = ap.parse_args()

    ctx = init_stage(args.run, STAGE, params={"estimates_path_override": args.estimates_path})

    try:
        estimates_path = _resolve_estimates_path(ctx.run_dir, args.estimates_path)
    except ValueError as exc:
        abort(ctx, str(exc))

    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    compatible_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"

    check_inputs(
        ctx,
        {
            "s3_estimates": estimates_path,
            "s3b_multimode": multimode_path,
            "s4_compatible_set": compatible_path,
        },
    )

    try:
        estimates = _read_json(estimates_path)
        multimode = _read_json(multimode_path)
        compatible = _read_json(compatible_path)

        ranked = compatible.get("ranked_all", [])
        if not ranked:
            abort(ctx, "joint_3d not available: compatible_set has no ranked_all entries")

        combined = estimates.get("combined", {})
        mode_220 = _mode_row(multimode, "220")
        mode_221 = _mode_row(multimode, "221")
        if mode_220 is None:
            abort(ctx, "joint_3d not available: mode 220 missing in multimode_estimates")

        ln_f_220 = float(mode_220.get("ln_f")) if isinstance(mode_220.get("ln_f"), (int, float)) else math.log(float(combined.get("f_hz")))
        ln_q_220 = float(mode_220.get("ln_Q")) if isinstance(mode_220.get("ln_Q"), (int, float)) else math.log(float(combined.get("Q")))

        has_221, reason, weight = _infer_censoring(multimode, mode_221)
        ln_f_ratio = None
        if has_221 and mode_221 is not None:
            ln_f_ratio = float(mode_221["ln_f"]) - float(ln_f_220)

        coords_payload = {
            "event_id": estimates.get("event_id", "unknown"),
            "coords": {
                "ln_f_220": ln_f_220,
                "ln_Q_220": ln_q_220,
                "ln_f_ratio_221_220": ln_f_ratio,
            },
            "censoring": {
                "has_221": has_221,
                "reason": reason,
                "weight": weight,
            },
            "source": {
                "s3_estimates_sha256": sha256_file(estimates_path),
                "s3b_multimode_sha256": sha256_file(multimode_path),
                "s4_compatible_set_sha256": sha256_file(compatible_path),
            },
        }

        # Preserve expected outputs for downstream consumers.
        curvature_rows = []
        for row in ranked:
            if not isinstance(row, dict):
                continue
            d2 = row.get("d2")
            distance = row.get("distance")
            d_base = float(distance) if isinstance(distance, (int, float)) else (math.sqrt(float(d2)) if isinstance(d2, (int, float)) and d2 >= 0 else 0.0)
            d3 = d_base if not has_221 else math.sqrt(d_base**2 + float(ln_f_ratio or 0.0) ** 2)
            curvature_rows.append({
                "geometry_id": row.get("geometry_id"),
                "d_conformal": d_base,
                "d_conformal_3d": d3,
                "rank_flat": row.get("rank"),
                "metadata": row.get("metadata"),
            })

        curvature_rows.sort(key=lambda x: (float(x.get("d_conformal_3d", 0.0)), str(x.get("geometry_id"))))
        for idx, row in enumerate(curvature_rows):
            row["rank_3d"] = idx

        curvature_payload = {
            "schema_version": "curvature_3d_v1",
            "run_id": args.run,
            "event_id": estimates.get("event_id", "unknown"),
            "censoring": coords_payload["censoring"],
            "coords": coords_payload["coords"],
            "n_geometries_reranked": len(curvature_rows),
            "reranked_geometries": curvature_rows,
        }
        diagnostics_payload = {
            "schema_version": "metric_diag_3d_v1",
            "run_id": args.run,
            "event_id": estimates.get("event_id", "unknown"),
            "joint_3d_available": True,
            "censoring": coords_payload["censoring"],
            "n_ranked_input": len(ranked),
            "n_ranked_output": len(curvature_rows),
        }

        curv_path = ctx.outputs_dir / "curvature_3d.json"
        diag_path = ctx.outputs_dir / "metric_diagnostics_3d.json"
        coords_path = ctx.outputs_dir / "coords_3d.json"
        write_json_atomic(curv_path, curvature_payload)
        write_json_atomic(diag_path, diagnostics_payload)
        write_json_atomic(coords_path, coords_payload)

        finalize(
            ctx,
            artifacts={
                "curvature_3d": curv_path,
                "metric_diagnostics_3d": diag_path,
                "coords_3d": coords_path,
            },
            results={
                "has_221": has_221,
                "weight": weight,
                "n_geometries_reranked": len(curvature_rows),
            },
        )

        print(f"OUT_ROOT={ctx.out_root}")
        print(f"STAGE_DIR={ctx.stage_dir}")
        print(f"OUTPUTS_DIR={ctx.outputs_dir}")
        print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
        print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
