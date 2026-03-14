"""s4i_common_geometry_intersection — Canonical stage: compute the intersection of
geometry_ids that passed both the mode-220 and mode-221 filters.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.golden_geometry_spec import (
    VERDICT_NO_COMMON_GEOMETRIES,
    VERDICT_PASS,
    VERDICT_SKIPPED_221_UNAVAILABLE,
    _utc_now_iso,
)

STAGE = "s4i_common_geometry_intersection"
S4G_OUTPUT_PRIMARY_REL = "s4g_mode220_geometry_filter/outputs/mode220_filter.json"
S4G_OUTPUT_LEGACY_REL = "s4g_mode220_geometry_filter/outputs/geometries_220.json"
# Legacy alias kept for older tests/imports.
S4G_OUTPUT_REL = S4G_OUTPUT_PRIMARY_REL
S4H_OUTPUT_REL = "s4h_mode221_geometry_filter/outputs/mode221_filter.json"
OUTPUT_FILE = "common_intersection.json"
_MODE_SUFFIX_RE = re.compile(r"_l\d+m\d+n\d+$")


def canonical_geometry_id(geometry_id: str) -> str:
    """Return a mode-agnostic geometry id.

    Atlas ids often encode the QNM mode suffix (e.g. ``..._l2m2n0`` vs
    ``..._l2m2n1``). For the physical common-geometry intersection, we remove
    that suffix so 220/221 predictions from the same remnant can match.
    """
    return _MODE_SUFFIX_RE.sub("", geometry_id)


def compute_intersection(ids_220: list[str], ids_221: "list[str] | None") -> list[str]:
    canonical_220 = {canonical_geometry_id(str(gid)) for gid in ids_220}
    if ids_221 is None:
        return sorted(canonical_220)
    canonical_221 = {canonical_geometry_id(str(gid)) for gid in ids_221}
    return sorted(canonical_220 & canonical_221)


def _read_ids(payload: dict) -> list[str]:
    ids = payload.get("geometry_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids]
    ids = payload.get("accepted_geometry_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: geometry intersection")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)

    s4g_primary = ctx.run_dir / S4G_OUTPUT_PRIMARY_REL
    s4g_legacy = ctx.run_dir / S4G_OUTPUT_LEGACY_REL
    s4g_path = s4g_primary if s4g_primary.exists() else s4g_legacy
    s4h_path = ctx.run_dir / S4H_OUTPUT_REL

    if not s4g_path.exists():
        abort(
            ctx,
            "mode-220 filter output not found. "
            f"expected: {s4g_primary} (or legacy {s4g_legacy}). "
            "Command to regenerate upstream: python -m mvp.s4g_mode220_geometry_filter --run-id <RUN_ID> --atlas-path <ATLAS_PATH>.",
        )

    try:
        check_inputs(ctx, {"s4g_mode220": s4g_path}, optional={"s4h_mode221": s4h_path})
        s4g_data = json.loads(s4g_path.read_text(encoding="utf-8"))
        ids_220 = _read_ids(s4g_data)

        ids_221: list[str] | None = None
        mode221_skipped = False
        if s4h_path.exists():
            s4h_data = json.loads(s4h_path.read_text(encoding="utf-8"))
            if s4h_data.get("verdict") == VERDICT_SKIPPED_221_UNAVAILABLE:
                mode221_skipped = True
            else:
                ids_221 = _read_ids(s4h_data)
        else:
            mode221_skipped = True

        common_ids = compute_intersection(ids_220, ids_221)
        verdict = VERDICT_PASS if common_ids else VERDICT_NO_COMMON_GEOMETRIES

        payload = {
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
        finalize(
            ctx,
            artifacts={"common_intersection": out_path},
            verdict="PASS",
            results={
                "n_geometries_220": len(ids_220),
                "n_geometries_221": len(ids_221) if ids_221 is not None else None,
                "mode221_skipped": mode221_skipped,
                "n_common": len(common_ids),
                "verdict": verdict,
            },
        )
        log_stage_paths(ctx)
        print(f"[{STAGE}] n_common={len(common_ids)} verdict={verdict}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
