"""s4j_hawking_area_filter — Canonical stage: apply the Hawking area theorem to
filter the common geometry intersection down to golden geometries.
"""
from __future__ import annotations

import argparse
import json
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
from mvp.golden_geometry_spec import (
    DEFAULT_AREA_TOLERANCE,
    VERDICT_NO_GOLDEN_GEOMETRIES,
    VERDICT_PASS,
    _utc_now_iso,
    delta_area,
    passes_area_law,
)

STAGE = "s4j_hawking_area_filter"
S4I_OUTPUT_REL = "s4i_common_geometry_intersection/outputs/common_intersection.json"
AREA_OBS_FILE_REL = "s4j_hawking_area_filter/inputs/area_obs.json"
OUTPUT_FILE = "hawking_area_filter.json"


def filter_area_law(
    *,
    common_geometry_ids: list[str],
    area_data: dict[str, dict[str, float]],
    area_tolerance: float,
) -> list[str]:
    passed: list[str] = []
    for gid in common_geometry_ids:
        geo_areas = area_data.get(gid)
        if geo_areas is None:
            passed.append(gid)
            continue
        area_f = float(geo_areas["area_final"])
        area_i = float(geo_areas["area_initial"])
        da = delta_area(area_f, area_i)
        if passes_area_law(da, area_tolerance):
            passed.append(gid)
    return sorted(passed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: Hawking area filter")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--area-tolerance", type=float, default=DEFAULT_AREA_TOLERANCE)
    args = ap.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE, params={"area_tolerance": float(args.area_tolerance)})

    s4i_path = ctx.run_dir / S4I_OUTPUT_REL
    area_obs_path = ctx.run_dir / AREA_OBS_FILE_REL

    if not s4i_path.exists():
        abort(
            ctx,
            "common intersection output not found. "
            f"expected: {s4i_path}. "
            "Command to regenerate upstream: python -m mvp.s4i_common_geometry_intersection --run-id <RUN_ID>.",
        )

    try:
        check_inputs(ctx, {"s4i_common": s4i_path}, optional={"area_obs": area_obs_path})

        s4i_data = json.loads(s4i_path.read_text(encoding="utf-8"))
        common_ids: list[str] = s4i_data.get("common_geometry_ids", [])

        area_data: dict[str, dict[str, float]] = {}
        if area_obs_path.exists():
            area_obs = json.loads(area_obs_path.read_text(encoding="utf-8"))
            area_data = area_obs.get("area_data", {})

        golden_ids = filter_area_law(
            common_geometry_ids=common_ids,
            area_data=area_data,
            area_tolerance=args.area_tolerance,
        )

        verdict = VERDICT_PASS if golden_ids else VERDICT_NO_GOLDEN_GEOMETRIES

        payload: dict[str, Any] = {
            "schema_name": "golden_geometry_per_event",
            "schema_version": "v1",
            "created_utc": _utc_now_iso(),
            "run_id": args.run_id,
            "stage": STAGE,
            "area_tolerance": args.area_tolerance,
            "n_common_input": len(common_ids),
            "area_data": area_data,
            "golden_geometry_ids": golden_ids,
            "n_golden": len(golden_ids),
            "verdict": verdict,
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"hawking_area_filter": out_path},
            verdict="PASS",
            results={
                "area_tolerance": float(args.area_tolerance),
                "n_common_input": len(common_ids),
                "n_golden": len(golden_ids),
                "verdict": verdict,
            },
        )
        log_stage_paths(ctx)
        print(f"[{STAGE}] n_golden={len(golden_ids)} verdict={verdict}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
