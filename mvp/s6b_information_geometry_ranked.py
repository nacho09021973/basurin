#!/usr/bin/env python3
"""MVP Stage 6b: export compact ranked/compatible geometry indices for aggregation."""
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

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "s6b_information_geometry_ranked"


def _load_atlas_index_map(ctx: Any, atlas_path_arg: str | None) -> tuple[dict[str, int], str]:
    if atlas_path_arg:
        atlas_path = Path(atlas_path_arg).expanduser().resolve()
        if not atlas_path.exists():
            abort(
                ctx,
                "Missing required inputs: "
                f"atlas_path: {atlas_path}; "
                "to regenerate upstream run: python -m mvp.s4_geometry_filter --run "
                f"{ctx.run_id}",
            )
    else:
        s4_summary_path = ctx.run_dir / "s4_geometry_filter" / "stage_summary.json"
        if not s4_summary_path.exists():
            abort(
                ctx,
                "Missing required inputs: "
                f"s4_stage_summary: {s4_summary_path}; "
                f"to regenerate upstream run: python -m mvp.s4_geometry_filter --run {ctx.run_id}",
            )

        s4_summary = json.loads(s4_summary_path.read_text(encoding="utf-8"))
        atlas_path_raw = s4_summary.get("parameters", {}).get("atlas_path")
        if not isinstance(atlas_path_raw, str) or not atlas_path_raw.strip():
            abort(
                ctx,
                f"Missing atlas_path in {s4_summary_path} at parameters.atlas_path. "
                f"to regenerate upstream run: python -m mvp.s4_geometry_filter --run {ctx.run_id}",
            )

        atlas_path = Path(atlas_path_raw)
        if not atlas_path.is_absolute():
            atlas_path = (ctx.run_dir / atlas_path).resolve()
        if not atlas_path.exists():
            abort(
                ctx,
                "Missing required inputs: "
                f"atlas_from_s4: {atlas_path}; "
                f"from {s4_summary_path}; "
                f"to regenerate upstream run: python -m mvp.s4_geometry_filter --run {ctx.run_id}",
            )

    atlas_payload = json.loads(atlas_path.read_text(encoding="utf-8"))
    if isinstance(atlas_payload, dict):
        entries = atlas_payload.get("entries")
    else:
        entries = atlas_payload
    if not isinstance(entries, list):
        abort(
            ctx,
            f"Invalid atlas format at {atlas_path}: expected list or dict with list at 'entries'.",
        )

    atlas_index_map: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        gid = entry.get("id") or entry.get("geometry_id") or entry.get("name")
        if isinstance(gid, str) and gid.strip():
            atlas_index_map[gid] = idx

    return atlas_index_map, str(atlas_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: export ranked geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--compat-score-threshold", type=float, default=-1.0e18)
    args = ap.parse_args()

    ctx = init_stage(args.run, STAGE, params={
        "top_k": args.top_k,
        "compat_score_threshold": args.compat_score_threshold,
    })

    compatible_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    curvature_path = ctx.run_dir / "s6_information_geometry" / "outputs" / "curvature.json"
    check_inputs(ctx, {"compatible_set": compatible_path, "curvature": curvature_path})

    try:
        compat = json.loads(compatible_path.read_text(encoding="utf-8"))
        curvature = json.loads(curvature_path.read_text(encoding="utf-8"))

        atlas_index_map, atlas_path_used = _load_atlas_index_map(ctx, args.atlas_path)
        ctx.params["atlas_path"] = atlas_path_used

        reranked = curvature.get("reranked_geometries", [])
        if not isinstance(reranked, list):
            reranked = []

        ranked_rows: list[dict[str, Any]] = []
        top_k = max(0, int(args.top_k))
        for idx, row in enumerate(reranked):
            if not isinstance(row, dict):
                continue
            d_conf = row.get("d_conformal")
            if not isinstance(d_conf, (int, float)) or not math.isfinite(float(d_conf)):
                continue
            gid = row.get("geometry_id")
            if not isinstance(gid, str) or not gid.strip():
                abort(ctx, f"Invalid geometry_id in reranked_geometries[{idx}]: expected non-empty string")
            atlas_index = atlas_index_map.get(gid)
            if atlas_index is None:
                abort(
                    ctx,
                    f"geometry_id={gid!r} not found in atlas loaded from s4 parameters.atlas_path={atlas_path_used}",
                )
            score = -float(d_conf)
            ranked_rows.append({
                "atlas_index": atlas_index,
                "score": score,
                "geometry_id": gid,
            })

        ranked_rows.sort(key=lambda r: (-r["score"], r["atlas_index"]))
        ranked_rows = ranked_rows[:top_k]

        compatible_ids = {
            str(row.get("geometry_id"))
            for row in compat.get("compatible_geometries", [])
            if isinstance(row, dict) and row.get("geometry_id") is not None
        }
        if not compatible_ids:
            compatible_ids = {
                str(row.get("geometry_id"))
                for row in compat.get("ranked_all", [])
                if isinstance(row, dict) and row.get("compatible") is True and row.get("geometry_id") is not None
            }

        compatible_rows = [
            {"atlas_index": row["atlas_index"], "geometry_id": row["geometry_id"], "score": row["score"]}
            for row in ranked_rows
            if str(row.get("geometry_id")) in compatible_ids and row["score"] >= float(args.compat_score_threshold)
        ]

        result = {
            "schema_version": "mvp_s6b_ranked_v1",
            "event_id": curvature.get("event_id", compat.get("event_id", "unknown")),
            "atlas_id": compat.get("atlas_id", "unknown"),
            "n_atlas": int(compat.get("n_atlas", len(reranked))),
            "ranked": [
                {"atlas_index": r["atlas_index"], "geometry_id": r["geometry_id"], "score": r["score"]}
                for r in ranked_rows
            ],
            "compatible": compatible_rows,
            "compatibility_criterion": {
                "name": "s4_geometry_filter_membership_and_score",
                "params": {
                    "source": "compatible_geometries",
                    "compat_score_threshold": float(args.compat_score_threshold),
                    "top_k": top_k,
                },
            },
        }

        out_path = ctx.outputs_dir / "ranked_geometries.json"
        write_json_atomic(out_path, result)
        finalize(
            ctx,
            artifacts={"ranked_geometries": out_path},
            results={"n_ranked": len(result["ranked"]), "n_compatible": len(result["compatible"])},
        )
        log_stage_paths(ctx)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
