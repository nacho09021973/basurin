"""s4k_event_support_region — canonical per-event support-region artifact.

Consolidates the explicit golden-geometry branch:

  s4g mode-220 compatibility
  s4h mode-221 compatibility
  s4i common intersection
  s4j Hawking area filter

plus the multimode viability metadata from s3b and, when available, the
domain-status metadata from s4d. This produces a single downstream artifact
that describes the event-level support region without re-running any physics.
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
from mvp.golden_geometry_spec import VERDICT_PASS, VERDICT_SKIPPED_221_UNAVAILABLE, _utc_now_iso

STAGE = "s4k_event_support_region"
S3B_SUMMARY_REL = "s3b_multimode_estimates/stage_summary.json"
S4G_OUTPUT_REL = "s4g_mode220_geometry_filter/outputs/mode220_filter.json"
S4H_OUTPUT_REL = "s4h_mode221_geometry_filter/outputs/mode221_filter.json"
S4I_OUTPUT_REL = "s4i_common_geometry_intersection/outputs/common_intersection.json"
S4J_OUTPUT_REL = "s4j_hawking_area_filter/outputs/hawking_area_filter.json"
S4D_OUTPUT_REL = "s4d_kerr_from_multimode/outputs/kerr_extraction.json"
RUN_PROVENANCE_REL = "run_provenance.json"
OUTPUT_FILE = "event_support_region.json"

DOMAIN_STATUS_UNKNOWN = "UNKNOWN"
ANALYSIS_PATH_MULTIMODE = "MULTIMODE_INTERSECTION"
ANALYSIS_PATH_MODE220_FALLBACK = "MODE220_PLUS_HAWKING"
SUPPORT_REGION_AVAILABLE = "SUPPORT_REGION_AVAILABLE"
HAWKING_FILTER_EMPTY = "HAWKING_FILTER_EMPTY"
NO_COMMON_REGION = "NO_COMMON_REGION"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _extract_ids(payload: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _relative_to_run(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _mode_block(
    *,
    payload: dict[str, Any],
    ids: list[str],
    source_relpath: str,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "available": available,
        "verdict": payload.get("verdict", "UNKNOWN"),
        "n_geometries": len(ids),
        "geometry_ids": ids,
        "source_artifact": source_relpath,
    }


def _derive_support_region_status(*, final_ids: list[str], common_ids: list[str]) -> str:
    if final_ids:
        return SUPPORT_REGION_AVAILABLE
    if common_ids:
        return HAWKING_FILTER_EMPTY
    return NO_COMMON_REGION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: consolidate explicit per-event support region")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)

    s3b_summary_path = ctx.run_dir / S3B_SUMMARY_REL
    s4g_path = ctx.run_dir / S4G_OUTPUT_REL
    s4h_path = ctx.run_dir / S4H_OUTPUT_REL
    s4i_path = ctx.run_dir / S4I_OUTPUT_REL
    s4j_path = ctx.run_dir / S4J_OUTPUT_REL
    s4d_path = ctx.run_dir / S4D_OUTPUT_REL
    run_provenance_path = ctx.run_dir / RUN_PROVENANCE_REL

    try:
        check_inputs(
            ctx,
            {
                "s3b_stage_summary": s3b_summary_path,
                "s4g_mode220": s4g_path,
                "s4h_mode221": s4h_path,
                "s4i_common_intersection": s4i_path,
                "s4j_hawking_filter": s4j_path,
            },
            optional={
                "s4d_kerr_extraction": s4d_path,
                "run_provenance": run_provenance_path,
            },
        )

        s3b_summary = _load_json(s3b_summary_path)
        s4g_payload = _load_json(s4g_path)
        s4h_payload = _load_json(s4h_path)
        s4i_payload = _load_json(s4i_path)
        s4j_payload = _load_json(s4j_path)
        s4d_payload = _load_json(s4d_path) if s4d_path.exists() else {}
        run_provenance = _load_json(run_provenance_path) if run_provenance_path.exists() else {}

        geometry_ids_220 = _extract_ids(s4g_payload, "accepted_geometry_ids", "geometry_ids")
        geometry_ids_221 = _extract_ids(s4h_payload, "geometry_ids")
        common_geometry_ids = _extract_ids(s4i_payload, "common_geometry_ids")
        golden_geometry_ids = _extract_ids(s4j_payload, "golden_geometry_ids")

        mode221_skipped = bool(s4i_payload.get("mode221_skipped")) or (
            s4h_payload.get("verdict") == VERDICT_SKIPPED_221_UNAVAILABLE
        )
        analysis_path = ANALYSIS_PATH_MODE220_FALLBACK if mode221_skipped else ANALYSIS_PATH_MULTIMODE
        support_region_status = _derive_support_region_status(
            final_ids=golden_geometry_ids,
            common_ids=common_geometry_ids,
        )

        domain_status = str(s4d_payload.get("domain_status", DOMAIN_STATUS_UNKNOWN))
        domain_status_source = (
            _relative_to_run(ctx.run_dir, s4d_path) if s4d_path.exists() else "unknown"
        )

        event_id: str | None = None
        invocation = run_provenance.get("invocation")
        if isinstance(invocation, dict):
            raw_event_id = invocation.get("event_id")
            if isinstance(raw_event_id, str) and raw_event_id.strip():
                event_id = raw_event_id

        payload: dict[str, Any] = {
            "schema_name": "golden_geometry_event_support",
            "schema_version": "v1",
            "created_utc": _utc_now_iso(),
            "run_id": args.run_id,
            "event_id": event_id,
            "stage": STAGE,
            "analysis_path": analysis_path,
            "support_region_status": support_region_status,
            "final_geometry_ids": golden_geometry_ids,
            "n_final_geometries": len(golden_geometry_ids),
            "domain_status": domain_status,
            "domain_status_source": domain_status_source,
            "multimode_viability": s3b_summary.get("multimode_viability"),
            "systematics_gate": s3b_summary.get("systematics_gate"),
            "science_evidence": s3b_summary.get("science_evidence"),
            "annotations": s3b_summary.get("annotations"),
            "mode_220_region": _mode_block(
                payload=s4g_payload,
                ids=geometry_ids_220,
                source_relpath=_relative_to_run(ctx.run_dir, s4g_path),
                available=True,
            ),
            "mode_221_region": _mode_block(
                payload=s4h_payload,
                ids=geometry_ids_221,
                source_relpath=_relative_to_run(ctx.run_dir, s4h_path),
                available=not mode221_skipped,
            ),
            "common_intersection": {
                "mode221_skipped": mode221_skipped,
                "verdict": s4i_payload.get("verdict", "UNKNOWN"),
                "n_common": len(common_geometry_ids),
                "geometry_ids": common_geometry_ids,
                "source_artifact": _relative_to_run(ctx.run_dir, s4i_path),
            },
            "hawking_filtered_region": {
                "verdict": s4j_payload.get("verdict", "UNKNOWN"),
                "n_golden": len(golden_geometry_ids),
                "golden_geometry_ids": golden_geometry_ids,
                "source_artifact": _relative_to_run(ctx.run_dir, s4j_path),
            },
            "sources": {
                "s3b_stage_summary": _relative_to_run(ctx.run_dir, s3b_summary_path),
                "s4g_mode220": _relative_to_run(ctx.run_dir, s4g_path),
                "s4h_mode221": _relative_to_run(ctx.run_dir, s4h_path),
                "s4i_common_intersection": _relative_to_run(ctx.run_dir, s4i_path),
                "s4j_hawking_filter": _relative_to_run(ctx.run_dir, s4j_path),
                "s4d_kerr_extraction": _relative_to_run(ctx.run_dir, s4d_path) if s4d_path.exists() else None,
                "run_provenance": _relative_to_run(ctx.run_dir, run_provenance_path) if run_provenance_path.exists() else None,
            },
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"event_support_region": out_path},
            verdict=VERDICT_PASS,
            results={
                "analysis_path": analysis_path,
                "support_region_status": support_region_status,
                "multimode_viability_class": (
                    payload["multimode_viability"].get("class")
                    if isinstance(payload.get("multimode_viability"), dict)
                    else None
                ),
                "domain_status": domain_status,
                "mode221_skipped": mode221_skipped,
                "n_mode220": len(geometry_ids_220),
                "n_mode221": len(geometry_ids_221),
                "n_common": len(common_geometry_ids),
                "n_final": len(golden_geometry_ids),
            },
        )
        log_stage_paths(ctx)
        print(
            f"[{STAGE}] analysis_path={analysis_path} support_region_status={support_region_status} "
            f"n_final={len(golden_geometry_ids)}"
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
