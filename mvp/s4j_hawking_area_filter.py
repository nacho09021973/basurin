"""s4j_hawking_area_filter — Canonical stage: apply the Hawking area theorem to
filter the common geometry intersection down to "golden geometries".

A geometry passes the area filter if:
    A_final - A_initial >= -area_tolerance

where A_final and A_initial are provided per geometry in the area data file.

Reads:
    <run_dir>/s4i_common_geometry_intersection/outputs/common_intersection.json
    <run_dir>/s4j_hawking_area_filter/inputs/area_obs.json
        {
            "area_data": {
                "<geometry_id>": {
                    "area_final":   <float>,   # post-merger area (any consistent units)
                    "area_initial": <float>    # pre-merger total area (same units)
                },
                ...
            }
        }

Output:
    <run_dir>/s4j_hawking_area_filter/outputs/hawking_area_filter.json
    <run_dir>/s4j_hawking_area_filter/stage_summary.json
    <run_dir>/s4j_hawking_area_filter/manifest.json

Geometries absent from area_data are passed through (no area constraint applied).
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

from basurin_io import (
    resolve_out_root,
    require_run_valid,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)
from mvp.golden_geometry_spec import (
    DEFAULT_AREA_TOLERANCE,
    VERDICT_NO_GOLDEN_GEOMETRIES,
    VERDICT_PASS,
    delta_area,
    passes_area_law,
    _utc_now_iso,
)

STAGE = "s4j_hawking_area_filter"
S4I_OUTPUT_REL = "s4i_common_geometry_intersection/outputs/common_intersection.json"
AREA_OBS_FILE_REL = "s4j_hawking_area_filter/inputs/area_obs.json"
OUTPUT_FILE = "hawking_area_filter.json"


# ---------------------------------------------------------------------------
# Pure helpers (reusable by experiment)
# ---------------------------------------------------------------------------


def filter_area_law(
    *,
    common_geometry_ids: list[str],
    area_data: dict[str, dict[str, float]],
    area_tolerance: float,
) -> list[str]:
    """Return sorted geometry_ids that satisfy the Hawking area law.

    Parameters
    ----------
    common_geometry_ids : candidate geometry_ids (already filtered by mode compatibility).
    area_data           : mapping geometry_id → {"area_final": ..., "area_initial": ...}.
                          Geometries absent from area_data pass through unconditionally.
    area_tolerance      : non-negative tolerance; passes_area_law requires
                          delta_area >= -area_tolerance.

    Returns
    -------
    Sorted list of geometry_ids that pass the area constraint.
    """
    passed: list[str] = []
    for gid in common_geometry_ids:
        geo_areas = area_data.get(gid)
        if geo_areas is None:
            # No area data → pass through (conservative: no constraint)
            passed.append(gid)
            continue
        area_f = float(geo_areas["area_final"])
        area_i = float(geo_areas["area_initial"])
        da = delta_area(area_f, area_i)
        if passes_area_law(da, area_tolerance):
            passed.append(gid)
    return sorted(passed)


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: Hawking area filter")
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--area-tolerance",
        type=float,
        default=DEFAULT_AREA_TOLERANCE,
        help=f"Non-negative deficit allowed by area law (default: {DEFAULT_AREA_TOLERANCE})",
    )
    args = ap.parse_args(argv)

    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    run_dir = out_root / args.run_id
    stage_dir = run_dir / STAGE
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    s4i_path = run_dir / S4I_OUTPUT_REL
    area_obs_path = run_dir / AREA_OBS_FILE_REL

    if not s4i_path.exists():
        print(
            f"ERROR: common intersection output not found.\n"
            f"  expected: {s4i_path}\n"
            f"  Run s4i_common_geometry_intersection first.",
            file=sys.stderr,
        )
        return 2

    try:
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

        out_path = outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)

        summary = {
            "stage": STAGE,
            "run_id": args.run_id,
            "area_tolerance": args.area_tolerance,
            "n_common_input": len(common_ids),
            "n_golden": len(golden_ids),
            "verdict": verdict,
        }
        stage_summary = write_stage_summary(stage_dir, summary)
        manifest = write_manifest(
            stage_dir,
            {"hawking_area_filter": out_path, "stage_summary": stage_summary},
        )

        print(f"OUT_ROOT={out_root}")
        print(f"STAGE_DIR={stage_dir}")
        print(f"OUTPUTS_DIR={outputs_dir}")
        print(f"STAGE_SUMMARY={stage_summary}")
        print(f"MANIFEST={manifest}")
        print(f"[{STAGE}] n_golden={len(golden_ids)} verdict={verdict}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
