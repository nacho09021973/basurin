#!/usr/bin/env python3
"""Canonical stage s7: beyond-Kerr deviation score from s4d + s3b."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import write_json_atomic
from mvp.contracts import StageContext, abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "s7_beyond_kerr_deviation_score"
GR_THRESHOLD_90 = 4.605
GR_THRESHOLD_99 = 9.210
BNS_MAX_REMNANT_MASS_MSUN = 10.0
BNS_MAX_REMNANT_MASS_MSUN_MIN = 5.0
BNS_MAX_REMNANT_MASS_MSUN_MAX = 15.0

logger = logging.getLogger(__name__)


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _chi2_cdf_2dof(x: float) -> float:
    """CDF acumulada chi2 con 2 grados de libertad: F(x) = 1 - exp(-x/2)."""
    if x < 0.0:
        return 0.0
    return 1.0 - math.exp(-x / 2.0)


def _beyond_kerr_verdict(chi2: float) -> str:
    # chi2 2dof: 90% CL -> 4.605, 99% CL -> 9.210
    if chi2 < GR_THRESHOLD_90:
        return "GR_CONSISTENT"
    if chi2 < GR_THRESHOLD_99:
        return "GR_TENSION"
    return "GR_INCONSISTENT"


def _compute_score(
    M_final: float,
    chi_final: float,
    f221_obs: float,
    tau221_obs: float,
    sigma_f221: float,
    sigma_tau221: float,
) -> dict[str, Any]:
    from mvp.kerr_qnm_fits import kerr_qnm

    pred = kerr_qnm(M_final, chi_final, (2, 2, 1))
    delta_f_norm = (f221_obs - pred.f_hz) / max(sigma_f221, pred.f_hz * 1e-6)
    delta_tau_norm = (tau221_obs - pred.tau_s) / max(sigma_tau221, pred.tau_s * 1e-6)
    chi2 = delta_f_norm**2 + delta_tau_norm**2
    epsilon_f = (f221_obs - pred.f_hz) / pred.f_hz
    epsilon_tau = (tau221_obs - pred.tau_s) / pred.tau_s
    return {
        "chi2_kerr_2dof": chi2,
        "chi2_cdf_proxy": _chi2_cdf_2dof(chi2),
        "verdict": _beyond_kerr_verdict(chi2),
        "epsilon_f": epsilon_f,
        "epsilon_tau": epsilon_tau,
        "delta_f_norm": delta_f_norm,
        "delta_tau_norm": delta_tau_norm,
        "predicted_f221_hz": pred.f_hz,
        "predicted_tau221_s": pred.tau_s,
        "gr_threshold_90pct": GR_THRESHOLD_90,
        "gr_threshold_99pct": GR_THRESHOLD_99,
    }


def _empty_score_payload(verdict: str) -> dict[str, Any]:
    return {
        "chi2_kerr_2dof": None,
        "chi2_cdf_proxy": None,
        "verdict": verdict,
        "epsilon_f": None,
        "epsilon_tau": None,
        "delta_f_norm": None,
        "delta_tau_norm": None,
        "predicted_f221_hz": None,
        "predicted_tau221_s": None,
        "gr_threshold_90pct": GR_THRESHOLD_90,
        "gr_threshold_99pct": GR_THRESHOLD_99,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _extract_source_class(metadata: dict[str, Any]) -> str | None:
    for key in ("source_class", "event_class", "event_type", "source_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    classification = metadata.get("classification")
    if isinstance(classification, dict):
        for key in ("source_class", "event_class", "event_type"):
            value = classification.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _classify_source_kind(source_class: str | None) -> str | None:
    if source_class is None:
        return None
    token = source_class.strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"bns", "binary_neutron_star", "nsns"}:
        return "BNS"
    if token in {"bbh", "binary_black_hole"}:
        return "BBH"
    if token in {"nsbh", "neutron_star_black_hole"}:
        return "NSBH"
    return token.upper()


def _resolve_bns_mass_upper_bound_msun() -> float:
    """Resolve BNS remnant-mass upper bound from env with conservative default.

    BASURIN_BNS_MAX_REMNANT_MASS_MSUN can tune this threshold without changing code.
    Recommended range: [5, 15] Msun. Values outside this range are clamped.
    """
    raw = os.environ.get("BASURIN_BNS_MAX_REMNANT_MASS_MSUN")
    if raw is None:
        return BNS_MAX_REMNANT_MASS_MSUN
    try:
        bound = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid BASURIN_BNS_MAX_REMNANT_MASS_MSUN=%r; using default %.1f",
            raw,
            BNS_MAX_REMNANT_MASS_MSUN,
        )
        return BNS_MAX_REMNANT_MASS_MSUN
    if not math.isfinite(bound) or bound <= 0.0:
        logger.warning(
            "Ignoring non-finite/out-of-domain BASURIN_BNS_MAX_REMNANT_MASS_MSUN=%r; using default %.1f",
            raw,
            BNS_MAX_REMNANT_MASS_MSUN,
        )
        return BNS_MAX_REMNANT_MASS_MSUN
    clamped = min(BNS_MAX_REMNANT_MASS_MSUN_MAX, max(BNS_MAX_REMNANT_MASS_MSUN_MIN, bound))
    if clamped != bound:
        logger.warning(
            "Clamping BASURIN_BNS_MAX_REMNANT_MASS_MSUN from %.3f to %.3f (allowed range %.1f-%.1f)",
            bound,
            clamped,
            BNS_MAX_REMNANT_MASS_MSUN_MIN,
            BNS_MAX_REMNANT_MASS_MSUN_MAX,
        )
    return clamped


def _validate_metadata_for_source_inference(metadata: dict[str, Any]) -> str | None:
    preferred_families = metadata.get("preferred_families")
    if preferred_families is not None and not isinstance(preferred_families, list):
        return "preferred_families must be a list when present"
    family_priors = metadata.get("family_priors")
    if family_priors is not None and not isinstance(family_priors, dict):
        return "family_priors must be an object when present"
    return None


def _infer_source_kind_from_metadata(metadata: dict[str, Any]) -> str | None:
    preferred_families = metadata.get("preferred_families")
    if isinstance(preferred_families, list) and any(str(f) == "BNS_REMNANT" for f in preferred_families):
        return "BNS"
    family_priors = metadata.get("family_priors")
    if isinstance(family_priors, dict) and "BNS_REMNANT" in family_priors:
        return "BNS"
    return None


def _load_event_metadata_for_run(run_dir: Path) -> dict[str, Any]:
    run_provenance_path = run_dir / "run_provenance.json"
    if not run_provenance_path.exists():
        return {"_metadata_lookup": "not_requested"}
    provenance = _load_json_object(run_provenance_path)
    invocation = provenance.get("invocation")
    if not isinstance(invocation, dict):
        return {"_metadata_lookup": "not_requested"}
    event_id = invocation.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        return {"_metadata_lookup": "not_requested"}
    metadata_path = _here.parents[1] / "docs" / "ringdown" / "event_metadata" / f"{event_id}_metadata.json"
    if not metadata_path.exists():
        return {"_metadata_lookup": "missing", "event_id": event_id}
    metadata = _load_json_object(metadata_path)
    metadata["_metadata_lookup"] = "found"
    return metadata


def _astrophysical_consistency(*, M_final: float, metadata: dict[str, Any]) -> dict[str, Any]:
    """Evaluate lightweight astrophysical priors.

    Output schema example:
    {
      "source_kind": "BNS",
      "status": "INCONSISTENT",
      "reason": "BNS prior violated: inferred Kerr remnant mass ...",
      "mass_upper_bound_msun": 10.0
    }

    TODO: migrate to probabilistic consistency with upstream uncertainties
    (e.g. if s4d provides mu/sigma, evaluate P[M_final > bound] via Normal CDF
    and compare against a confidence threshold) instead of a hard cut.
    """
    metadata_validation_error = _validate_metadata_for_source_inference(metadata)
    if metadata_validation_error is not None:
        return {
            "source_kind": None,
            "status": "METADATA_INSUFFICIENT",
            "reason": f"event metadata malformed for source inference: {metadata_validation_error}",
            "mass_upper_bound_msun": None,
        }
    source_kind = _classify_source_kind(_extract_source_class(metadata)) or _infer_source_kind_from_metadata(metadata)
    mass_upper_bound_msun = _resolve_bns_mass_upper_bound_msun()
    lookup_state = str(metadata.get("_metadata_lookup", "missing"))
    result = {
        "source_kind": source_kind,
        "status": "NOT_APPLICABLE" if lookup_state == "not_requested" else "METADATA_INSUFFICIENT",
        "reason": (
            "event metadata lookup was not requested for this run"
            if lookup_state == "not_requested"
            else "event metadata is insufficient to classify source kind for astrophysical checks"
        ),
        "mass_upper_bound_msun": None,
    }
    if source_kind != "BNS":
        if source_kind is None:
            return result
        return {
            "source_kind": source_kind,
            "status": "NOT_APPLICABLE",
            "reason": "astrophysical remnant-mass sanity prior currently applied only to BNS events",
            "mass_upper_bound_msun": None,
        }

    if M_final > mass_upper_bound_msun:
        return {
            "source_kind": source_kind,
            "status": "INCONSISTENT",
            "reason": (
                f"BNS prior violated: inferred Kerr remnant mass {M_final:.3f} Msun exceeds "
                f"BNS_MAX_REMNANT_MASS_MSUN={mass_upper_bound_msun:.1f}"
            ),
            "mass_upper_bound_msun": mass_upper_bound_msun,
        }

    return {
        "source_kind": source_kind,
        "status": "CONSISTENT",
        "reason": "BNS prior satisfied by Kerr remnant mass",
        "mass_upper_bound_msun": mass_upper_bound_msun,
    }


def _as_finite_float(value: Any, json_path: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Invalid numeric value at JSON path: {json_path}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite numeric value at JSON path: {json_path}")
    return out


def _validate_upstream_governance(ctx: StageContext) -> None:
    regen_cmd = {
        "s4d_kerr_from_multimode": f"python -m mvp.s4d_kerr_from_multimode --run-id {ctx.run_id}",
        "s3b_multimode_estimates": f"python -m mvp.s3b_multimode_estimates --run-id {ctx.run_id}",
    }
    for upstream in ctx.contract.upstream_stages:
        summary_path = ctx.run_dir / upstream / "stage_summary.json"
        if not summary_path.exists():
            abort(
                ctx,
                (
                    f"Missing required inputs: {summary_path}. "
                    f"Comando exacto para regenerar upstream: {regen_cmd.get(upstream, '<unknown>')}"
                ),
            )
        summary = _load_json_object(summary_path)
        if summary.get("verdict") != "PASS":
            abort(
                ctx,
                (
                    f"Upstream governance violation: {summary_path} has verdict="
                    f"{summary.get('verdict')!r} (expected 'PASS'). "
                    f"Comando exacto para regenerar upstream: {regen_cmd.get(upstream, '<unknown>')}"
                ),
            )


def _extract_mode_221_observed(multimode: dict[str, Any]) -> dict[str, float]:
    modes = multimode.get("modes")
    if not isinstance(modes, list):
        raise ValueError("Missing required JSON path: modes")

    node_221: dict[str, Any] | None = None
    idx_221 = -1
    labels: list[str] = []
    for idx, node in enumerate(modes):
        if not isinstance(node, dict):
            continue
        label = node.get("label")
        if isinstance(label, str):
            labels.append(label)
        if str(label) == "221":
            node_221 = node
            idx_221 = idx
            break

    if node_221 is None:
        raise ValueError(
            "Missing required JSON path: modes[*].label == '221'; "
            f"candidates={sorted(set(labels))}"
        )

    fit = node_221.get("fit")
    if not isinstance(fit, dict):
        raise ValueError(f"Missing required JSON path: modes[{idx_221}].fit")

    stability = fit.get("stability")
    if not isinstance(stability, dict):
        raise ValueError(f"Missing required JSON path: modes[{idx_221}].fit.stability")

    lnf_p10 = _as_finite_float(stability.get("lnf_p10"), f"modes[{idx_221}].fit.stability.lnf_p10")
    lnf_p50 = _as_finite_float(stability.get("lnf_p50"), f"modes[{idx_221}].fit.stability.lnf_p50")
    lnf_p90 = _as_finite_float(stability.get("lnf_p90"), f"modes[{idx_221}].fit.stability.lnf_p90")
    lnq_p10 = _as_finite_float(stability.get("lnQ_p10"), f"modes[{idx_221}].fit.stability.lnQ_p10")
    lnq_p50 = _as_finite_float(stability.get("lnQ_p50"), f"modes[{idx_221}].fit.stability.lnQ_p50")
    lnq_p90 = _as_finite_float(stability.get("lnQ_p90"), f"modes[{idx_221}].fit.stability.lnQ_p90")

    f_p10 = math.exp(lnf_p10)
    f_p50 = math.exp(lnf_p50)
    f_p90 = math.exp(lnf_p90)

    tau_p10 = math.exp(lnq_p10 - lnf_p10) / math.pi
    tau_p50 = math.exp(lnq_p50 - lnf_p50) / math.pi
    tau_p90 = math.exp(lnq_p90 - lnf_p90) / math.pi

    return {
        "f221_obs": f_p50,
        "tau221_obs": tau_p50,
        "sigma_f221": max((f_p90 - f_p10) / 2.0, 0.0),
        "sigma_tau221": max((tau_p90 - tau_p10) / 2.0, 0.0),
    }


def _execute(ctx: StageContext) -> dict[str, Path]:
    kerr_path = ctx.run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"

    check_inputs(ctx, {"kerr_extraction": kerr_path, "multimode_estimates": multimode_path})
    _validate_upstream_governance(ctx)

    kerr = _load_json_object(kerr_path)

    if (
        kerr.get("verdict") == "SKIPPED_MULTIMODE_GATE"
        or kerr.get("M_final_Msun") is None
        or kerr.get("chi_final") is None
    ):
        score = _empty_score_payload("SKIPPED_S4D_GATE")
        score["astrophysical_consistency"] = {
            "source_kind": None,
            "status": "NOT_APPLICABLE",
            "reason": "Kerr extraction unavailable; astrophysical consistency was not evaluated",
            "mass_upper_bound_msun": None,
        }
    else:
        multimode = _load_json_object(multimode_path)
        try:
            observed = _extract_mode_221_observed(multimode)
        except ValueError as exc:
            abort(
                ctx,
                (
                    f"{exc}. Expected input: {multimode_path}. "
                    f"Comando exacto para regenerar upstream: "
                    f"python -m mvp.s3b_multimode_estimates --run-id {ctx.run_id}"
                ),
            )
        try:
            M_final = _as_finite_float(kerr.get("M_final_Msun"), "M_final_Msun")
            chi_final = _as_finite_float(kerr.get("chi_final"), "chi_final")
        except ValueError as exc:
            abort(
                ctx,
                (
                    f"{exc}. Expected input: {kerr_path}. "
                    f"Comando exacto para regenerar upstream: "
                    f"python -m mvp.s4d_kerr_from_multimode --run-id {ctx.run_id}"
                ),
            )

        score = _compute_score(
            M_final=M_final,
            chi_final=chi_final,
            f221_obs=observed["f221_obs"],
            tau221_obs=observed["tau221_obs"],
            sigma_f221=observed["sigma_f221"],
            sigma_tau221=observed["sigma_tau221"],
        )
        event_metadata = _load_event_metadata_for_run(ctx.run_dir)
        astro_consistency = _astrophysical_consistency(M_final=M_final, metadata=event_metadata)
        score["astrophysical_consistency"] = astro_consistency
        if astro_consistency.get("status") == "INCONSISTENT":
            score["verdict"] = "ASTRO_INCONSISTENT"
        elif astro_consistency.get("status") == "METADATA_INSUFFICIENT":
            score["verdict"] = "INCONCLUSIVE"

    payload = {
        "schema_name": "beyond_kerr_score",
        "schema_version": "v1",
        "created_utc": _utc_now_z(),
        "run_id": ctx.run_id,
        "stage": STAGE,
        **score,
    }
    out_path = ctx.outputs_dir / "beyond_kerr_score.json"
    write_json_atomic(out_path, payload)
    return {"beyond_kerr_score": out_path}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=STAGE)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    ctx = init_stage(args.run_id, STAGE, params={})
    try:
        artifacts = _execute(ctx)
        output_payload = _load_json_object(artifacts["beyond_kerr_score"])
        finalize(
            ctx,
            artifacts=artifacts,
            verdict="PASS",
            results={
                "verdict": output_payload.get("verdict"),
                "chi2_kerr_2dof": output_payload.get("chi2_kerr_2dof"),
            },
        )
        log_stage_paths(ctx)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, f"{STAGE} failed: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
