"""s4i_common_geometry_intersection — Canonical stage: compute the intersection of
geometry_ids that passed both the mode-220 and mode-221 filters.

Reads:
    <run_dir>/s4g_mode220_geometry_filter/outputs/geometries_220.json
    <run_dir>/s4h_mode221_geometry_filter/outputs/mode221_filter.json

Output:
    <run_dir>/s4i_common_geometry_intersection/outputs/common_intersection.json
    <run_dir>/s4i_common_geometry_intersection/stage_summary.json
    <run_dir>/s4i_common_geometry_intersection/manifest.json

If mode-221 stage was SKIPPED (mode221 unavailable), the intersection falls back
to the mode-220 set (no 221 constraint applied).
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
    write_json_atomic,
)
from mvp.contracts import (
    init_stage,
    check_inputs,
    finalize,
    abort,
    log_stage_paths,
)
from mvp.golden_geometry_spec import (
    VERDICT_NO_COMMON_GEOMETRIES,
    VERDICT_PASS,
    VERDICT_SKIPPED_221_UNAVAILABLE,
    _utc_now_iso,
)

STAGE = "s4i_common_geometry_intersection"
S4G_OUTPUT_REL = "s4g_mode220_geometry_filter/outputs/geometries_220.json"
S4H_OUTPUT_REL = "s4h_mode221_geometry_filter/outputs/mode221_filter.json"
OUTPUT_FILE = "common_intersection.json"


# ---------------------------------------------------------------------------
# Pure helpers (reusable by experiment)
# ---------------------------------------------------------------------------


def compute_intersection(
    ids_220: list[str],
    ids_221: "list[str] | None",
) -> list[str]:
    """Return the sorted intersection of geometry_ids from mode 220 and mode 221.

    If ids_221 is None (mode 221 unavailable), returns the mode-220 set as-is.
    """
    if ids_221 is None:
        return sorted(ids_220)
    common = set(ids_220) & set(ids_221)
    return sorted(common)


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: geometry intersection")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)

    s4g_path = ctx.run_dir / S4G_OUTPUT_REL
    s4h_path = ctx.run_dir / S4H_OUTPUT_REL

    if not s4g_path.exists():
        abort(ctx, f"mode-220 filter output not found: {s4g_path}. Run s4g_mode220_geometry_filter first.")

    inputs = check_inputs(ctx, {"s4g_geometries_220": s4g_path}, optional={"s4h_mode221_filter": s4h_path})

    try:
        s4g_data = json.loads(s4g_path.read_text(encoding="utf-8"))
        ids_220: list[str] = s4g_data.get("accepted_geometry_ids", s4g_data.get("geometry_ids", []))

        ids_221: list[str] | None = None
        mode221_skipped = False
        if s4h_path.exists():
            s4h_data = json.loads(s4h_path.read_text(encoding="utf-8"))
            if s4h_data.get("verdict") == VERDICT_SKIPPED_221_UNAVAILABLE:
                mode221_skipped = True
            else:
                ids_221 = s4h_data.get("geometry_ids", [])
        else:
            mode221_skipped = True

        common_ids = compute_intersection(ids_220, ids_221)

        if not common_ids:
            verdict = VERDICT_NO_COMMON_GEOMETRIES
        else:
            verdict = VERDICT_PASS

        payload: dict[str, Any] = {
            "schema_name": "golden_geometry_common",
            "schema_version": "v1",
            "created_utc": _utc_now_iso(),
            "run_id": args.run_id,
            "stage": STAGE,
            "n_geometries_220": len(ids_220),
            "n_geometries_221": len(ids_221) if ids_221 is not None else None,
            "mode221_skipped": mode221_skipped,
            "common_geometry_ids": common_ids,
            "n_common": len(common_ids),
            "verdict": verdict,
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)

        finalize(ctx, artifacts={"common_intersection": out_path}, verdict=verdict, results={
            "n_geometries_220": len(ids_220),
            "n_geometries_221": len(ids_221) if ids_221 is not None else None,
            "mode221_skipped": mode221_skipped,
            "n_common": len(common_ids),
        })
        log_stage_paths(ctx)
        print(f"[{STAGE}] n_common={len(common_ids)} verdict={verdict}")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
