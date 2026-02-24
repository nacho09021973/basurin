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
from mvp.contracts import abort, check_inputs, finalize, init_stage

STAGE = "s6b_information_geometry_ranked"


def _atlas_index(row: dict[str, Any], fallback: int) -> int:
    val = row.get("atlas_index")
    if isinstance(val, int):
        return val
    meta = row.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("atlas_index"), int):
        return int(meta["atlas_index"])
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: export ranked geometries")
    ap.add_argument("--run", required=True)
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
            score = -float(d_conf)
            ranked_rows.append({
                "atlas_index": _atlas_index(row, idx),
                "score": score,
                "geometry_id": row.get("geometry_id"),
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
            {"atlas_index": row["atlas_index"], "score": row["score"]}
            for row in ranked_rows
            if str(row.get("geometry_id")) in compatible_ids and row["score"] >= float(args.compat_score_threshold)
        ]

        result = {
            "schema_version": "mvp_s6b_ranked_v1",
            "event_id": curvature.get("event_id", compat.get("event_id", "unknown")),
            "atlas_id": compat.get("atlas_id", "unknown"),
            "n_atlas": int(compat.get("n_atlas", len(reranked))),
            "ranked": [{"atlas_index": r["atlas_index"], "score": r["score"]} for r in ranked_rows],
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
