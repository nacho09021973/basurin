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
    ratio_filter: dict[str, Any],
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
    kerr_ratio = ratio_filter.get("kerr_consistency") if isinstance(ratio_filter.get("kerr_consistency"), dict) else {}
    diagnostics = ratio_filter.get("diagnostics") if isinstance(ratio_filter.get("diagnostics"), dict) else {}
    filtering = ratio_filter.get("filtering") if isinstance(ratio_filter.get("filtering"), dict) else {}
    ratio_consistent = kerr_ratio.get("Rf_consistent")
    informativity_class = diagnostics.get("informativity_class")
    n_input_geometries = filtering.get("n_input_geometries")
    n_ratio_compatible = filtering.get("n_ratio_compatible")
    n_ratio_excluded = filtering.get("n_ratio_excluded")
    n_ratio_not_applicable = filtering.get("n_ratio_not_applicable")

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

    if ratio_consistent is False and assessment == "SUPPORTED":
        assessment = "TENSION"
        reason = "Kerr inversion is viable, but the observed 221/220 ratio falls outside the Kerr reference band"
    elif ratio_consistent is False and assessment == "TENSION":
        reason = "Kerr inversion exists with tension and the observed 221/220 ratio is also Kerr-inconsistent"

    has_geometric_support = isinstance(n_input_geometries, int) and n_input_geometries > 0
    if assessment in {"SUPPORTED", "TENSION"} and not has_geometric_support:
        assessment = "INCONCLUSIVE"
        reason = "Kerr inversion exists, but s4 geometry filtering produced no surviving geometries; cannot claim GR Kerr support without geometric support"

    ratio_is_informative = isinstance(informativity_class, str) and informativity_class in {"HIGH", "MODERATE", "LOW"}
    ratio_fully_excludes_supported_geometries = (
        has_geometric_support
        and isinstance(n_ratio_compatible, int)
        and n_ratio_compatible <= 0
        and isinstance(n_ratio_not_applicable, int)
        and isinstance(n_input_geometries, int)
        and n_ratio_not_applicable < n_input_geometries
        and ratio_is_informative
    )
    if ratio_fully_excludes_supported_geometries:
        assessment = "DISFAVORED"
        reason = "the Kerr ratio filter is informative and excludes all surviving spin-bearing geometries"

    return {
        "status": "EVALUATED",
        "assessment": assessment,
        "reason": reason,
        "router_primary_family": router_payload.get("primary_family"),
        "kerr_extraction_verdict": kerr_verdict,
        "beyond_kerr_verdict": score_verdict,
        "ratio_filter_verdict": ratio_filter.get("verdict"),
        "ratio_rf_consistent": ratio_consistent,
        "ratio_informativity_class": informativity_class,
        "n_input_geometries": n_input_geometries,
        "n_ratio_compatible": n_ratio_compatible,
        "n_ratio_excluded": n_ratio_excluded,
        "n_ratio_not_applicable": n_ratio_not_applicable,
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
    ratio_path = ctx.run_dir / "s4e_kerr_ratio_filter" / "outputs" / "ratio_filter_result.json"
    kerr_path = ctx.run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
    score_path = ctx.run_dir / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json"

    try:
        check_inputs(
            ctx,
            {
                "family_router": router_path,
                "ratio_filter": ratio_path,
                "kerr_extraction": kerr_path,
                "beyond_kerr_score": score_path,
            },
        )
        router = _load_json_object(router_path)
        ratio = _load_json_object(ratio_path)
        kerr = _load_json_object(kerr_path)
        score = _load_json_object(score_path)
        assessment = assess_gr_kerr_family(
            router_payload=router,
            ratio_filter=ratio,
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
