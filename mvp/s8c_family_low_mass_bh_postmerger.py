#!/usr/bin/env python3
"""Canonical family handler for low-mass Kerr BH post-merger remnants."""
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
from mvp.s8_family_router import FAMILY_LOW_MASS_BH

STAGE = "s8c_family_low_mass_bh_postmerger"
OUTPUT_FILE = "low_mass_bh_family.json"
MULTIMODE_OK = "MULTIMODE_OK"
DOMAIN_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
DEFAULT_PRIOR = {
    "mass_msun_range": [2.3, 3.2],
    "chi_range": [0.55, 0.98],
    "allow_gr_tension": True,
}


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _event_metadata_path(event_id: str) -> Path:
    return _here.parents[1] / "docs" / "ringdown" / "event_metadata" / f"{event_id}_metadata.json"


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _coerce_range(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    low = _to_float(raw[0])
    high = _to_float(raw[1])
    if low is None or high is None or high <= low:
        return None
    return (float(low), float(high))


def _extract_analysis_band_hz(
    *,
    run_provenance: dict[str, Any],
) -> tuple[float, float] | None:
    invocation = run_provenance.get("invocation")
    if not isinstance(invocation, dict):
        return None
    key_params = invocation.get("key_params")
    if not isinstance(key_params, dict):
        return None
    low = _to_float(key_params.get("band_low"))
    high = _to_float(key_params.get("band_high"))
    if low is None or high is None or high <= low:
        return None
    return (float(low), float(high))


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or hi <= lo:
        return [float(lo)]
    step = (hi - lo) / float(n - 1)
    return [float(lo + step * i) for i in range(n)]


def _low_mass_kerr_frequency_envelope(prior: dict[str, Any]) -> tuple[float, float] | None:
    mass_range = _coerce_range(prior.get("mass_msun_range"))
    chi_range = _coerce_range(prior.get("chi_range"))
    if mass_range is None or chi_range is None:
        return None

    from mvp.kerr_qnm_fits import kerr_qnm

    freqs: list[float] = []
    for mass_msun in _linspace(float(mass_range[0]), float(mass_range[1]), 9):
        for chi in _linspace(float(chi_range[0]), float(chi_range[1]), 9):
            for mode in ((2, 2, 0), (2, 2, 1)):
                qnm = kerr_qnm(mass_msun, chi, mode)
                freq = _to_float(qnm.f_hz)
                if freq is not None and freq > 0.0:
                    freqs.append(float(freq))
    if not freqs:
        return None
    return (float(min(freqs)), float(max(freqs)))


def _merge_prior(event_metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prior = json.loads(json.dumps(DEFAULT_PRIOR))
    family_priors = event_metadata.get("family_priors")
    source = "default"
    if isinstance(family_priors, dict):
        block = family_priors.get(FAMILY_LOW_MASS_BH)
        if isinstance(block, dict):
            source = "event_metadata"
            for key in ("mass_msun_range", "chi_range"):
                value = block.get(key)
                if isinstance(value, list) and len(value) == 2:
                    prior[key] = [float(value[0]), float(value[1])]
            allow = block.get("allow_gr_tension")
            if isinstance(allow, bool):
                prior["allow_gr_tension"] = allow
    return prior, source


def assess_low_mass_bh_family(
    *,
    router_payload: dict[str, Any],
    run_provenance: dict[str, Any],
    s3b_stage_summary: dict[str, Any],
    ratio_filter: dict[str, Any],
    kerr_extraction: dict[str, Any],
    beyond_kerr_score: dict[str, Any],
    event_metadata: dict[str, Any],
) -> dict[str, Any]:
    families = router_payload.get("families_to_run")
    selected = isinstance(families, list) and FAMILY_LOW_MASS_BH in {str(v) for v in families}
    if not selected:
        return {
            "status": "SKIPPED_BY_ROUTER",
            "assessment": "NOT_SELECTED",
            "reason": "family router did not select LOW_MASS_BH_POSTMERGER for this run",
            "model_status": "NOT_REQUESTED",
        }

    viability = s3b_stage_summary.get("multimode_viability") if isinstance(s3b_stage_summary.get("multimode_viability"), dict) else {}
    viability_class = viability.get("class")
    prior, prior_source = _merge_prior(event_metadata)
    m_final = _to_float(kerr_extraction.get("M_final_Msun"))
    chi_final = _to_float(kerr_extraction.get("chi_final"))
    score_verdict = str(beyond_kerr_score.get("verdict"))
    kerr_verdict = str(kerr_extraction.get("verdict"))
    ratio_consistency = ratio_filter.get("kerr_consistency") if isinstance(ratio_filter.get("kerr_consistency"), dict) else {}
    filtering = ratio_filter.get("filtering") if isinstance(ratio_filter.get("filtering"), dict) else {}
    diagnostics = ratio_filter.get("diagnostics") if isinstance(ratio_filter.get("diagnostics"), dict) else {}
    analysis_band = _extract_analysis_band_hz(run_provenance=run_provenance)
    local_envelope = _low_mass_kerr_frequency_envelope(prior)

    mass_ok = (
        m_final is not None and
        float(prior["mass_msun_range"][0]) <= m_final <= float(prior["mass_msun_range"][1])
    )
    chi_ok = (
        chi_final is not None and
        float(prior["chi_range"][0]) <= chi_final <= float(prior["chi_range"][1])
    )
    ratio_rf_consistent = ratio_consistency.get("Rf_consistent")
    ratio_informativity = diagnostics.get("informativity_class")
    n_ratio_compatible = int(filtering.get("n_ratio_compatible") or 0)
    out_of_domain_reason: str | None = None
    if analysis_band is not None and local_envelope is not None:
        overlap_low = max(float(analysis_band[0]), float(local_envelope[0]))
        overlap_high = min(float(analysis_band[1]), float(local_envelope[1]))
        overlap_hz = max(0.0, overlap_high - overlap_low)
        if overlap_hz <= 0.0:
            out_of_domain_reason = (
                "analysis band has no physically useful overlap with the low-mass Kerr modal envelope "
                f"for this event domain: band_hz=[{analysis_band[0]:.3f}, {analysis_band[1]:.3f}], "
                f"kerr_envelope_hz=[{local_envelope[0]:.3f}, {local_envelope[1]:.3f}]"
            )

    if out_of_domain_reason is not None:
        assessment = "INCONCLUSIVE"
        reason = out_of_domain_reason
    elif viability_class != MULTIMODE_OK or kerr_verdict != "PASS" or m_final is None or chi_final is None:
        assessment = "INCONCLUSIVE"
        reason = "low-mass BH branch requires a valid multimode Kerr inversion"
    elif not mass_ok or not chi_ok:
        assessment = "DISFAVORED"
        reason = "the inferred Kerr remnant lies outside the configured low-mass post-merger prior volume"
    elif ratio_rf_consistent is False and ratio_informativity != "UNINFORMATIVE":
        assessment = "DISFAVORED"
        reason = "the observed 221/220 ratio is informative and inconsistent with the Kerr low-mass branch"
    elif n_ratio_compatible == 0 and ratio_informativity in {"HIGH", "MODERATE", "LOW"}:
        assessment = "DISFAVORED"
        reason = "the ratio filter excludes all surviving Kerr-compatible geometries in the low-mass branch"
    elif score_verdict == "GR_CONSISTENT":
        assessment = "SUPPORTED"
        reason = "the inferred Kerr remnant satisfies the low-mass prior and remains consistent with the multimode ratio evidence"
    elif score_verdict == "GR_TENSION" and bool(prior.get("allow_gr_tension", True)):
        assessment = "TENSION"
        reason = "the remnant remains inside the low-mass prior, but the multimode residuals show moderate Kerr tension"
    else:
        assessment = "DISFAVORED"
        reason = "the low-mass Kerr post-merger branch is not supported by the current multimode evidence"

    invocation = run_provenance.get("invocation") if isinstance(run_provenance.get("invocation"), dict) else {}
    payload = {
        "status": "EVALUATED",
        "assessment": assessment,
        "reason": reason,
        "model_status": "LOW_MASS_KERR_PRIOR_V1",
        "event_id": invocation.get("event_id"),
        "router_primary_family": router_payload.get("primary_family"),
        "multimode_viability_class": viability_class,
        "prior_source": prior_source,
        "priors": prior,
        "m_final_msun": m_final,
        "chi_final": chi_final,
        "mass_in_range": mass_ok,
        "chi_in_range": chi_ok,
        "beyond_kerr_verdict": score_verdict,
        "ratio_rf_consistent": ratio_rf_consistent,
        "ratio_informativity_class": ratio_informativity,
        "n_ratio_compatible": n_ratio_compatible,
    }
    if out_of_domain_reason is not None:
        payload["domain_status"] = DOMAIN_OUT_OF_DOMAIN
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"MVP {STAGE}: assess low-mass post-merger Kerr BH branch")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)
    router_path = ctx.run_dir / "s8_family_router" / "outputs" / "family_router.json"
    run_provenance_path = ctx.run_dir / "run_provenance.json"
    s3b_summary_path = ctx.run_dir / "s3b_multimode_estimates" / "stage_summary.json"
    ratio_path = ctx.run_dir / "s4e_kerr_ratio_filter" / "outputs" / "ratio_filter_result.json"
    kerr_path = ctx.run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
    score_path = ctx.run_dir / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json"

    try:
        run_provenance = _load_json_object(run_provenance_path)
        invocation = run_provenance.get("invocation") if isinstance(run_provenance.get("invocation"), dict) else {}
        event_id = invocation.get("event_id")
        metadata_path = _event_metadata_path(str(event_id)) if isinstance(event_id, str) else None
        check_inputs(
            ctx,
            {
                "family_router": router_path,
                "run_provenance": run_provenance_path,
                "s3b_stage_summary": s3b_summary_path,
                "ratio_filter": ratio_path,
                "kerr_extraction": kerr_path,
                "beyond_kerr_score": score_path,
            },
            optional={"event_metadata": metadata_path} if metadata_path is not None else None,
        )
        router = _load_json_object(router_path)
        s3b_summary = _load_json_object(s3b_summary_path)
        ratio = _load_json_object(ratio_path)
        kerr = _load_json_object(kerr_path)
        score = _load_json_object(score_path)
        event_metadata = _load_json_object(metadata_path) if metadata_path is not None and metadata_path.exists() else {}
        assessment = assess_low_mass_bh_family(
            router_payload=router,
            run_provenance=run_provenance,
            s3b_stage_summary=s3b_summary,
            ratio_filter=ratio,
            kerr_extraction=kerr,
            beyond_kerr_score=score,
            event_metadata=event_metadata,
        )

        payload = {
            "schema_name": "family_assessment",
            "schema_version": "v1",
            "run_id": args.run_id,
            "stage": STAGE,
            "family": FAMILY_LOW_MASS_BH,
            "metadata_path": str(metadata_path) if metadata_path is not None and metadata_path.exists() else None,
            **assessment,
            "verdict": "PASS",
        }
        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        results = {
            "family": FAMILY_LOW_MASS_BH,
            "status": assessment["status"],
            "assessment": assessment["assessment"],
            "mass_in_range": assessment.get("mass_in_range"),
            "chi_in_range": assessment.get("chi_in_range"),
            "ratio_rf_consistent": assessment.get("ratio_rf_consistent"),
        }
        if assessment.get("domain_status") is not None:
            results["domain_status"] = assessment.get("domain_status")
        finalize(
            ctx,
            artifacts={"low_mass_bh_family": out_path},
            verdict="PASS",
            results=results,
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
