#!/usr/bin/env python3
"""Canonical family handler for BNS/HMNS/SMNS remnant follow-up.

This stage evaluates a phenomenological post-merger atlas for binary-neutron-star
remnants. It is not an EOS solver; it is a compact, documented forward model
that maps remnant mass/radius/class priors into two-mode predictions that can
be compared against the observed multimode estimates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.s8_family_router import FAMILY_BNS

STAGE = "s8b_family_bns"
OUTPUT_FILE = "bns_family.json"
MODEL_STATUS = "PHENOMENOLOGICAL_V1"
MULTIMODE_OK = "MULTIMODE_OK"
DEFAULT_PRIOR = {
    "remnant_mass_msun_range": [2.3, 3.0],
    "radius_1p6_km_range": [10.5, 13.5],
    "classes": ["HMNS", "SMNS", "STABLE_NS"],
    "n_mass_points": 8,
    "n_radius_points": 7,
    "collapse_time_ms_values": {
        "HMNS": [5.0, 10.0, 20.0, 40.0, 80.0],
        "SMNS": [100.0, 300.0, 1000.0],
        "STABLE_NS": [3000.0, 10000.0],
    },
}


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or math.isclose(lo, hi):
        return [float(lo)]
    step = (hi - lo) / float(n - 1)
    return [float(lo + step * i) for i in range(n)]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract_quantile_block(mode_obj: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    direct = mode_obj.get("estimates")
    if isinstance(direct, dict):
        f_hz = direct.get("f_hz")
        tau_s = direct.get("tau_s")
        if isinstance(f_hz, dict) and isinstance(tau_s, dict):
            vals = {
                "f_hz": {q: _to_float(f_hz.get(q)) for q in ("p10", "p50", "p90")},
                "tau_s": {q: _to_float(tau_s.get(q)) for q in ("p10", "p50", "p90")},
            }
            if all(vals["f_hz"][q] is not None and vals["tau_s"][q] is not None for q in ("p10", "p50", "p90")):
                return vals  # type: ignore[return-value]

    stability = (((mode_obj.get("fit") or {}).get("stability") or {}))
    if isinstance(stability, dict):
        lnf = {q: _to_float(stability.get(f"lnf_{q}")) for q in ("p10", "p50", "p90")}
        lnq = {q: _to_float(stability.get(f"lnQ_{q}")) for q in ("p10", "p50", "p90")}
        if all(lnf[q] is not None and lnq[q] is not None for q in ("p10", "p50", "p90")):
            f_vals = {q: float(math.exp(float(lnf[q]))) for q in ("p10", "p50", "p90")}
            tau_vals = {
                q: float(math.exp(float(lnq[q]) - float(lnf[q])) / math.pi)
                for q in ("p10", "p50", "p90")
            }
            return {"f_hz": f_vals, "tau_s": tau_vals}
    return None


def _extract_mode_quantiles(multimode: dict[str, Any], label: str) -> dict[str, dict[str, float]]:
    estimates = multimode.get("estimates")
    if isinstance(estimates, dict):
        per_mode = estimates.get("per_mode")
        if isinstance(per_mode, dict):
            node = per_mode.get(label)
            if isinstance(node, dict):
                f_hz = node.get("f_hz")
                tau_s = node.get("tau_s")
                if isinstance(f_hz, dict) and isinstance(tau_s, dict):
                    vals = {
                        "f_hz": {q: _to_float(f_hz.get(q)) for q in ("p10", "p50", "p90")},
                        "tau_s": {q: _to_float(tau_s.get(q)) for q in ("p10", "p50", "p90")},
                    }
                    if all(vals["f_hz"][q] is not None and vals["tau_s"][q] is not None for q in ("p10", "p50", "p90")):
                        return vals  # type: ignore[return-value]
                vals = _extract_quantile_block(node)
                if vals is not None:
                    return vals

    modes = multimode.get("modes")
    if isinstance(modes, list):
        for node in modes:
            if not isinstance(node, dict):
                continue
            if str(node.get("label")) != label:
                continue
            vals = _extract_quantile_block(node)
            if vals is not None:
                return vals

    raise ValueError(f"Missing required fields in multimode_estimates for mode {label}: f_hz/tau_s p10/p50/p90")


def _summarize_mode(multimode: dict[str, Any], label: str) -> dict[str, float]:
    quantiles = _extract_mode_quantiles(multimode, label)
    f_p10 = float(quantiles["f_hz"]["p10"])
    f_p50 = float(quantiles["f_hz"]["p50"])
    f_p90 = float(quantiles["f_hz"]["p90"])
    tau_p10 = float(quantiles["tau_s"]["p10"])
    tau_p50 = float(quantiles["tau_s"]["p50"])
    tau_p90 = float(quantiles["tau_s"]["p90"])
    return {
        "f_hz": f_p50,
        "tau_s": tau_p50,
        "sigma_f_hz": max((f_p90 - f_p10) / 2.0, max(50.0, 0.02 * f_p50)),
        "sigma_tau_s": max((tau_p90 - tau_p10) / 2.0, max(0.001, 0.15 * tau_p50)),
    }


def _event_metadata_path(event_id: str) -> Path:
    return _here.parents[1] / "docs" / "ringdown" / "event_metadata" / f"{event_id}_metadata.json"


def _merge_bns_prior(event_metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prior = json.loads(json.dumps(DEFAULT_PRIOR))
    family_priors = event_metadata.get("family_priors")
    source = "default"
    if isinstance(family_priors, dict):
        bns_prior = family_priors.get(FAMILY_BNS)
        if isinstance(bns_prior, dict):
            source = "event_metadata"
            for key in ("remnant_mass_msun_range", "radius_1p6_km_range", "classes"):
                value = bns_prior.get(key)
                if isinstance(value, list) and value:
                    prior[key] = value
            for key in ("n_mass_points", "n_radius_points"):
                value = bns_prior.get(key)
                if isinstance(value, int) and value > 0:
                    prior[key] = value
            collapse = bns_prior.get("collapse_time_ms_values")
            if isinstance(collapse, dict):
                prior["collapse_time_ms_values"] = collapse
    return prior, source


def _predict_bns_modes(remnant_class: str, remnant_mass_msun: float, radius_1p6_km: float, collapse_time_ms: float) -> dict[str, dict[str, float]]:
    mass_term = remnant_mass_msun - 2.6
    radius_term = radius_1p6_km - 12.0

    dominant_khz = 3.2 + (1.15 * mass_term) - (0.18 * radius_term)
    dominant_khz += {"HMNS": 0.15, "SMNS": 0.0, "STABLE_NS": -0.12}[remnant_class]
    dominant_khz = _clamp(dominant_khz, 1.6, 4.5)

    secondary_ratio = {"HMNS": 0.82, "SMNS": 0.74, "STABLE_NS": 0.66}[remnant_class]
    secondary_ratio += (-0.03 * mass_term) + (0.015 * radius_term)
    secondary_ratio = _clamp(secondary_ratio, 0.55, 0.92)
    secondary_khz = dominant_khz * secondary_ratio

    tau_dom_ms = {"HMNS": 8.0, "SMNS": 20.0, "STABLE_NS": 60.0}[remnant_class]
    tau_dom_ms += (0.08 * collapse_time_ms) + (1.5 * radius_term) - (2.0 * mass_term)
    tau_dom_ms = _clamp(tau_dom_ms, 2.0, 200.0)
    tau_sec_ms = _clamp(0.55 * tau_dom_ms, 1.0, 150.0)

    return {
        "mode_220": {"f_hz": 1000.0 * dominant_khz, "tau_s": tau_dom_ms / 1000.0},
        "mode_221": {"f_hz": 1000.0 * secondary_khz, "tau_s": tau_sec_ms / 1000.0},
    }


def build_bns_candidate_atlas(prior: dict[str, Any]) -> list[dict[str, Any]]:
    mass_range = prior["remnant_mass_msun_range"]
    radius_range = prior["radius_1p6_km_range"]
    classes = [str(v) for v in prior["classes"]]
    masses = _linspace(float(mass_range[0]), float(mass_range[1]), int(prior["n_mass_points"]))
    radii = _linspace(float(radius_range[0]), float(radius_range[1]), int(prior["n_radius_points"]))

    candidates: list[dict[str, Any]] = []
    for remnant_class in classes:
        collapse_values = prior["collapse_time_ms_values"].get(remnant_class, [50.0])
        for remnant_mass_msun in masses:
            for radius_1p6_km in radii:
                for collapse_time_ms in collapse_values:
                    modes = _predict_bns_modes(remnant_class, remnant_mass_msun, radius_1p6_km, float(collapse_time_ms))
                    candidates.append(
                        {
                            "geometry_id": (
                                f"BNS_{remnant_class}_M{remnant_mass_msun:.3f}"
                                f"_R{radius_1p6_km:.2f}_tc{float(collapse_time_ms):.1f}"
                            ),
                            "remnant_class": remnant_class,
                            "remnant_mass_msun": round(remnant_mass_msun, 6),
                            "radius_1p6_km": round(radius_1p6_km, 6),
                            "collapse_time_ms": float(collapse_time_ms),
                            **modes,
                        }
                    )
    return candidates


def _score_candidate(candidate: dict[str, Any], observed_modes: dict[str, dict[str, float]]) -> dict[str, float | bool]:
    pred220 = candidate["mode_220"]
    pred221 = candidate["mode_221"]
    obs220 = observed_modes["mode_220"]
    obs221 = observed_modes["mode_221"]

    z_f220 = (obs220["f_hz"] - pred220["f_hz"]) / obs220["sigma_f_hz"]
    z_f221 = (obs221["f_hz"] - pred221["f_hz"]) / obs221["sigma_f_hz"]
    z_tau220 = (obs220["tau_s"] - pred220["tau_s"]) / obs220["sigma_tau_s"]
    z_tau221 = (obs221["tau_s"] - pred221["tau_s"]) / obs221["sigma_tau_s"]

    freq_chi2 = (z_f220 * z_f220) + (z_f221 * z_f221)
    tau_chi2 = (z_tau220 * z_tau220) + (z_tau221 * z_tau221)
    score = freq_chi2 + (0.25 * tau_chi2)
    compatible = bool(freq_chi2 <= 9.21 and score <= 13.28)
    return {
        "freq_chi2": freq_chi2,
        "tau_chi2": tau_chi2,
        "score": score,
        "compatible": compatible,
    }


def assess_bns_family(
    *,
    router_payload: dict[str, Any],
    run_provenance: dict[str, Any],
    s3b_stage_summary: dict[str, Any],
    multimode_estimates: dict[str, Any],
    event_metadata: dict[str, Any],
) -> dict[str, Any]:
    families = router_payload.get("families_to_run")
    selected = isinstance(families, list) and FAMILY_BNS in {str(v) for v in families}
    if not selected:
        return {
            "status": "SKIPPED_BY_ROUTER",
            "assessment": "NOT_SELECTED",
            "reason": "family router did not select BNS_REMNANT for this run",
            "model_status": "NOT_REQUESTED",
        }

    invocation = run_provenance.get("invocation") if isinstance(run_provenance.get("invocation"), dict) else {}
    event_id = invocation.get("event_id")
    viability = s3b_stage_summary.get("multimode_viability") if isinstance(s3b_stage_summary.get("multimode_viability"), dict) else {}
    viability_class = viability.get("class")

    observed_modes = {
        "mode_220": _summarize_mode(multimode_estimates, "220"),
        "mode_221": _summarize_mode(multimode_estimates, "221"),
    }
    prior, prior_source = _merge_bns_prior(event_metadata)
    candidates = build_bns_candidate_atlas(prior)

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = _score_candidate(candidate, observed_modes)
        scored.append({**candidate, **metrics})
    scored.sort(key=lambda row: float(row["score"]))

    compatible = [row for row in scored if bool(row["compatible"])]
    compatible_classes = sorted({str(row["remnant_class"]) for row in compatible})
    best_candidate = compatible[0] if compatible else (scored[0] if scored else None)

    if viability_class != MULTIMODE_OK:
        assessment = "INCONCLUSIVE"
        reason = "BNS family evaluation ran, but multimode viability is not MULTIMODE_OK"
    elif compatible:
        assessment = "SUPPORTED"
        reason = "the BNS post-merger atlas contains compatible remnant candidates for the observed two-mode summary"
    else:
        assessment = "DISFAVORED"
        reason = "no BNS/HMNS/SMNS candidate in the configured prior volume matches the observed two-mode summary"

    compatible_view = []
    for row in compatible[:25]:
        compatible_view.append(
            {
                "geometry_id": row["geometry_id"],
                "remnant_class": row["remnant_class"],
                "remnant_mass_msun": row["remnant_mass_msun"],
                "radius_1p6_km": row["radius_1p6_km"],
                "collapse_time_ms": row["collapse_time_ms"],
                "score": row["score"],
                "freq_chi2": row["freq_chi2"],
                "tau_chi2": row["tau_chi2"],
            }
        )

    best_view = None
    if isinstance(best_candidate, dict):
        best_view = {
            "geometry_id": best_candidate["geometry_id"],
            "remnant_class": best_candidate["remnant_class"],
            "remnant_mass_msun": best_candidate["remnant_mass_msun"],
            "radius_1p6_km": best_candidate["radius_1p6_km"],
            "collapse_time_ms": best_candidate["collapse_time_ms"],
            "score": best_candidate["score"],
            "freq_chi2": best_candidate["freq_chi2"],
            "tau_chi2": best_candidate["tau_chi2"],
            "mode_220": best_candidate["mode_220"],
            "mode_221": best_candidate["mode_221"],
        }

    return {
        "status": "EVALUATED",
        "assessment": assessment,
        "reason": reason,
        "model_status": MODEL_STATUS,
        "event_id": event_id,
        "router_primary_family": router_payload.get("primary_family"),
        "multimode_viability_class": viability_class,
        "prior_source": prior_source,
        "priors": prior,
        "observed_modes": observed_modes,
        "atlas_summary": {
            "n_candidates_total": len(scored),
            "n_candidates_compatible": len(compatible),
            "compatible_classes": compatible_classes,
        },
        "best_candidate": best_view,
        "compatible_candidates": compatible_view,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"MVP {STAGE}: BNS family follow-up handler")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)
    router_path = ctx.run_dir / "s8_family_router" / "outputs" / "family_router.json"
    run_provenance_path = ctx.run_dir / "run_provenance.json"
    s3b_summary_path = ctx.run_dir / "s3b_multimode_estimates" / "stage_summary.json"
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"

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
                "multimode_estimates": multimode_path,
            },
            optional={"event_metadata": metadata_path} if metadata_path is not None else None,
        )
        router = _load_json_object(router_path)
        s3b_summary = _load_json_object(s3b_summary_path)
        multimode_estimates = _load_json_object(multimode_path)
        event_metadata = _load_json_object(metadata_path) if metadata_path is not None and metadata_path.exists() else {}
        assessment = assess_bns_family(
            router_payload=router,
            run_provenance=run_provenance,
            s3b_stage_summary=s3b_summary,
            multimode_estimates=multimode_estimates,
            event_metadata=event_metadata,
        )

        payload = {
            "schema_name": "family_assessment",
            "schema_version": "v1",
            "run_id": args.run_id,
            "stage": STAGE,
            "family": FAMILY_BNS,
            "metadata_path": str(metadata_path) if metadata_path is not None and metadata_path.exists() else None,
            **assessment,
            "verdict": "PASS",
        }
        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"bns_family": out_path},
            verdict="PASS",
            results={
                "family": FAMILY_BNS,
                "status": assessment["status"],
                "assessment": assessment["assessment"],
                "model_status": assessment["model_status"],
                "n_candidates_compatible": assessment.get("atlas_summary", {}).get("n_candidates_compatible"),
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
