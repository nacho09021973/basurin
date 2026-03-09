#!/usr/bin/env python3
"""Canonical family handler for GR Kerr BH interpretation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.s8_family_router import FAMILY_GR_KERR

STAGE = "s8a_family_gr_kerr"
OUTPUT_FILE = "gr_kerr_family.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def assess_gr_kerr_family(
    *,
    router_payload: dict[str, Any],
    kerr_extraction: dict[str, Any],
    beyond_kerr_score: dict[str, Any],
) -> dict[str, Any]:
    families = router_payload.get("families_to_run")
    selected = isinstance(families, list) and FAMILY_GR_KERR in {str(v) for v in families}
    if not selected:
        return {
            "status": "SKIPPED_BY_ROUTER",
            "assessment": "NOT_SELECTED",
            "reason": "family router did not select GR_KERR_BH for this run",
        }

    score_verdict = str(beyond_kerr_score.get("verdict"))
    kerr_verdict = str(kerr_extraction.get("verdict"))

    if score_verdict == "GR_CONSISTENT" and kerr_extraction.get("M_final_Msun") is not None:
        assessment = "SUPPORTED"
        reason = "multimode Kerr inversion succeeded and the 221 residual is GR-consistent"
    elif score_verdict == "GR_TENSION":
        assessment = "TENSION"
        reason = "Kerr inversion exists but the 221 residual shows moderate tension with GR Kerr"
    elif score_verdict == "GR_INCONSISTENT":
        assessment = "DISFAVORED"
        reason = "the inferred Kerr remnant is inconsistent with the observed 221 mode"
    else:
        assessment = "INCONCLUSIVE"
        reason = "the multimode Kerr gate or deviation score is non-informative for this run"

    return {
        "status": "EVALUATED",
        "assessment": assessment,
        "reason": reason,
        "router_primary_family": router_payload.get("primary_family"),
        "kerr_extraction_verdict": kerr_verdict,
        "beyond_kerr_verdict": score_verdict,
        "m_final_msun": kerr_extraction.get("M_final_Msun"),
        "chi_final": kerr_extraction.get("chi_final"),
        "chi2_kerr_2dof": beyond_kerr_score.get("chi2_kerr_2dof"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"MVP {STAGE}: assess GR Kerr family support")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)
    router_path = ctx.run_dir / "s8_family_router" / "outputs" / "family_router.json"
    kerr_path = ctx.run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
    score_path = ctx.run_dir / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json"

    try:
        check_inputs(
            ctx,
            {
                "family_router": router_path,
                "kerr_extraction": kerr_path,
                "beyond_kerr_score": score_path,
            },
        )
        router = _load_json_object(router_path)
        kerr = _load_json_object(kerr_path)
        score = _load_json_object(score_path)
        assessment = assess_gr_kerr_family(
            router_payload=router,
            kerr_extraction=kerr,
            beyond_kerr_score=score,
        )

        payload = {
            "schema_name": "family_assessment",
            "schema_version": "v1",
            "run_id": args.run_id,
            "stage": STAGE,
            "family": FAMILY_GR_KERR,
            **assessment,
            "verdict": "PASS",
        }
        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"gr_kerr_family": out_path},
            verdict="PASS",
            results={
                "family": FAMILY_GR_KERR,
                "status": assessment["status"],
                "assessment": assessment["assessment"],
            },
        )
        log_stage_paths(ctx)
        print(f"[{STAGE}] status={assessment['status']} assessment={assessment['assessment']}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
