#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[1], _here.parents[2]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import utc_now_iso, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage

STAGE = "experiment_geometry_evidence_vs_gr"
SCHEMA_VERSION = "experiment_geometry_evidence_vs_gr_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic evidence proxy vs GR from geometry compatibility")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--prefer-3d",
        dest="prefer_3d",
        action="store_true",
        default=True,
        help="Prefer 3D conformal distance when available (default: true)",
    )
    parser.add_argument(
        "--no-prefer-3d",
        dest="prefer_3d",
        action="store_false",
        help="Disable 3D preference and always use 2D baseline",
    )
    parser.add_argument("--min-required-compatible", type=int, default=1)
    return parser.parse_args(argv)


def _load_json(path: Path, label: str, ctx: Any) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt input at {path} ({label}): {exc}")
        raise AssertionError("unreachable")


def _extract_rows(compatible_set: dict[str, Any]) -> list[dict[str, Any]]:
    rows = compatible_set.get("compatible_geometries")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    ranked = compatible_set.get("ranked_all")
    if isinstance(ranked, list):
        return [row for row in ranked if isinstance(row, dict)]
    return []


def _num(row: dict[str, Any], key: str) -> float | None:
    val = row.get(key)
    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val)
    return None


def _best_distance(rows: list[dict[str, Any]], keys: list[str]) -> tuple[float | None, str | None]:
    best: tuple[float, str | None] | None = None
    for row in rows:
        for key in keys:
            dist = _num(row, key)
            if dist is None:
                continue
            gid = row.get("geometry_id") or row.get("id") or row.get("name")
            gid_str = str(gid) if gid is not None else None
            if best is None or dist < best[0]:
                best = (dist, gid_str)
            break
    return (None, None) if best is None else best


def _score_lnz_proxy(distance: float | None) -> float:
    if distance is None:
        return -1.0e12
    return -0.5 * (distance ** 2)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "prefer_3d": args.prefer_3d,
            "min_required_compatible": args.min_required_compatible,
        },
    )

    compatible_set_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    curvature_2d_path = ctx.run_dir / "s6_information_geometry" / "outputs" / "curvature.json"
    curvature_3d_path = ctx.run_dir / "s6b_information_geometry_3d" / "outputs" / "curvature_3d.json"
    metric_diag_3d_path = ctx.run_dir / "s6b_information_geometry_3d" / "outputs" / "metric_diagnostics_3d.json"

    check_inputs(
        ctx,
        {
            "compatible_set": compatible_set_path,
            "curvature_2d": curvature_2d_path,
        },
        optional={
            "curvature_3d": curvature_3d_path,
            "metric_diagnostics_3d": metric_diag_3d_path,
        },
    )

    compatible_set = _load_json(compatible_set_path, "compatible_set", ctx)
    _ = _load_json(curvature_2d_path, "curvature_2d", ctx)
    if curvature_3d_path.exists():
        _ = _load_json(curvature_3d_path, "curvature_3d", ctx)

    rows = _extract_rows(compatible_set)
    n_compatible = len(rows)

    if n_compatible < args.min_required_compatible:
        abort(
            ctx,
            f"Insufficient compatible geometries in {compatible_set_path}: "
            f"n_compatible={n_compatible} < min_required_compatible={args.min_required_compatible}",
        )

    best_2d, best_2d_gid = _best_distance(rows, ["d_conformal", "distance"])
    best_3d, best_3d_gid = _best_distance(rows, ["d_conformal_3d", "d_conformal", "distance"])

    score_gr = _score_lnz_proxy(best_2d)
    use_3d = args.prefer_3d and curvature_3d_path.exists()
    score_alt = _score_lnz_proxy(best_3d if use_3d else best_2d)
    best_geometry_id = best_3d_gid if use_3d else best_2d_gid

    output_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": args.run_id,
        "created": utc_now_iso(),
        "inputs": {
            "compatible_set_path": str(compatible_set_path.relative_to(ctx.run_dir)),
            "curvature_2d_path": str(curvature_2d_path.relative_to(ctx.run_dir)),
            "curvature_3d_path": str(curvature_3d_path.relative_to(ctx.run_dir)) if curvature_3d_path.exists() else None,
        },
        "evidence_model": {
            "definition": "heuristic_v1",
            "notes": "Deterministic proxy; not a Bayesian evidence unless stated.",
        },
        "metrics": {
            "n_compatible": n_compatible,
            "best_geometry_id": best_geometry_id,
            "best_distance_2d": best_2d,
            "best_distance_3d": best_3d if use_3d else None,
            "score_lnZ_proxy_GR": score_gr,
            "score_lnZ_proxy_alt": score_alt,
            "logB_proxy_alt_vs_GR": score_alt - score_gr,
        },
        "verdict": "PASS",
    }

    evidence_path = ctx.outputs_dir / "evidence_vs_gr.json"
    write_json_atomic(evidence_path, output_payload)

    finalize(
        ctx,
        artifacts={"evidence_vs_gr": evidence_path},
        results={
            "logB_proxy_alt_vs_GR": output_payload["metrics"]["logB_proxy_alt_vs_GR"],
            "n_compatible": n_compatible,
            "best_geometry_id": best_geometry_id,
        },
        extra_summary={
            "evidence_model": output_payload["evidence_model"],
        },
    )

    print(f"OUT_ROOT={ctx.out_root}")
    print(f"STAGE_DIR={ctx.stage_dir}")
    print(f"OUTPUTS_DIR={ctx.outputs_dir}")
    print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
    print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
