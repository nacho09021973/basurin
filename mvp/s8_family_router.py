#!/usr/bin/env python3
"""Canonical family router for the multimode pipeline.

This stage decides which physical family handlers should run next. The router
does not force a single interpretation when the available metadata is weak; it
can emit multiple candidate families in priority order.
"""
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
from mvp.gwtc_events import get_event

STAGE = "s8_family_router"
FAMILY_GR_KERR = "GR_KERR_BH"
FAMILY_BNS = "BNS_REMNANT"
FAMILY_LOW_MASS_BH = "LOW_MASS_BH_POSTMERGER"
VALID_FAMILIES = {FAMILY_GR_KERR, FAMILY_BNS, FAMILY_LOW_MASS_BH}
OUTPUT_FILE = "family_router.json"
ROUTE_STAGE_BY_FAMILY = {
    FAMILY_GR_KERR: "s8a_family_gr_kerr",
    FAMILY_BNS: "s8b_family_bns",
    FAMILY_LOW_MASS_BH: "s8c_family_low_mass_bh_postmerger",
}
PROGRAM_MULTIMODE = "MULTIMODE_PROGRAM"
PROGRAM_SINGLEMODE_CONSTRAINED = "SINGLE_MODE_CONSTRAINED_PROGRAM"


def _repo_root() -> Path:
    return _here.parents[1]


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _event_metadata_path(event_id: str) -> Path:
    return _repo_root() / "docs" / "ringdown" / "event_metadata" / f"{event_id}_metadata.json"


def _normalize_family_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        family = str(item).strip().upper()
        if family in VALID_FAMILIES and family not in out:
            out.append(family)
    return out


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


def _is_truthy_hint(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "present", "detected"}
    return False


def _has_multimessenger_hint(metadata: dict[str, Any]) -> bool:
    for key in (
        "multimessenger",
        "em_counterpart",
        "has_em_counterpart",
        "kilonova",
        "gamma_ray_burst",
        "grb_counterpart",
    ):
        if _is_truthy_hint(metadata.get(key)):
            return True
    counterparts = metadata.get("counterparts")
    if isinstance(counterparts, list) and counterparts:
        return True
    return False


def _classify_source(source_class: str | None) -> str | None:
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


def route_family_candidates(
    *,
    event_id: str,
    metadata: dict[str, Any],
    known_bbh_catalog_entry: dict[str, float] | None,
    multimode_viability_class: str | None,
    multimode_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preferred = _normalize_family_list(metadata.get("preferred_families"))
    source_class = _extract_source_class(metadata)
    source_kind = _classify_source(source_class)
    has_multimessenger = _has_multimessenger_hint(metadata)

    if preferred:
        families = preferred
        routing_mode = "metadata_preferred"
        routing_reason = "event metadata defines preferred_families"
    elif source_kind == "BBH" or known_bbh_catalog_entry is not None:
        families = [FAMILY_GR_KERR]
        routing_mode = "catalog_known_bbh"
        routing_reason = "event matches known BBH metadata/catalog"
    elif source_kind == "BNS" or has_multimessenger:
        families = [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]
        routing_mode = "metadata_bns_or_multimessenger"
        routing_reason = "event metadata indicates BNS or multimessenger counterpart"
    else:
        families = [FAMILY_GR_KERR, FAMILY_LOW_MASS_BH, FAMILY_BNS]
        routing_mode = "fallback_multi_family"
        routing_reason = "insufficient source-class metadata; evaluate multiple families"

    program_classification = (
        PROGRAM_MULTIMODE if multimode_viability_class == "MULTIMODE_OK" else "MULTIMODE_PROGRAM_UNAVAILABLE"
    )
    fallback_classification = None
    fallback_path = None
    fallback_reason = None
    if isinstance(multimode_fallback, dict):
        fallback_classification = multimode_fallback.get("classification")
        fallback_path = multimode_fallback.get("fallback_path")
        fallback_reason = multimode_fallback.get("reason")
        if fallback_classification is not None:
            routing_mode = "single_mode_constrained_program"
            routing_reason = str(
                fallback_reason
                or "mode 221 unavailable; route families conservatively on the single-mode fallback program"
            )
            program_classification = str(
                multimode_fallback.get("program_classification") or PROGRAM_SINGLEMODE_CONSTRAINED
            )

    family_routes = [
        {
            "family": family,
            "priority": idx + 1,
            "stage": ROUTE_STAGE_BY_FAMILY[family],
            "reason": routing_reason,
        }
        for idx, family in enumerate(families)
    ]

    return {
        "event_id": event_id,
        "source_class_hint": source_class,
        "source_kind": source_kind,
        "has_multimessenger_hint": has_multimessenger,
        "known_bbh_catalog_match": known_bbh_catalog_entry is not None,
        "multimode_viability_class": multimode_viability_class,
        "program_classification": program_classification,
        "fallback_classification": fallback_classification,
        "fallback_path": fallback_path,
        "fallback_reason": fallback_reason,
        "routing_mode": routing_mode,
        "primary_family": families[0],
        "families_to_run": families,
        "family_routes": family_routes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"MVP {STAGE}: route run to physical family handlers")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE)
    run_provenance_path = ctx.run_dir / "run_provenance.json"
    s3b_summary_path = ctx.run_dir / "s3b_multimode_estimates" / "stage_summary.json"
    s4d_path = ctx.run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_from_multimode.json"

    try:
        run_provenance = _load_json_object(run_provenance_path)
        invocation = run_provenance.get("invocation")
        if not isinstance(invocation, dict):
            abort(ctx, f"Invalid run_provenance schema in {run_provenance_path}: missing invocation object")
        event_id = invocation.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            abort(ctx, f"Invalid run_provenance schema in {run_provenance_path}: missing invocation.event_id")

        metadata_path = _event_metadata_path(event_id)
        check_inputs(
            ctx,
            {
                "run_provenance": run_provenance_path,
                "s3b_stage_summary": s3b_summary_path,
                "s4d_kerr_from_multimode": s4d_path,
            },
            optional={"event_metadata": metadata_path},
        )

        s3b_summary = _load_json_object(s3b_summary_path)
        s4d_payload = _load_json_object(s4d_path)
        multimode_viability = s3b_summary.get("multimode_viability")
        multimode_viability_class = None
        if isinstance(multimode_viability, dict):
            value = multimode_viability.get("class")
            if isinstance(value, str):
                multimode_viability_class = value
        multimode_fallback = s4d_payload.get("multimode_fallback")
        if not isinstance(multimode_fallback, dict):
            multimode_fallback = None

        metadata = _load_json_object(metadata_path) if metadata_path.exists() else {}
        known_bbh = get_event(event_id)
        routing = route_family_candidates(
            event_id=event_id,
            metadata=metadata,
            known_bbh_catalog_entry=known_bbh,
            multimode_viability_class=multimode_viability_class,
            multimode_fallback=multimode_fallback,
        )

        payload = {
            "schema_name": "family_router",
            "schema_version": "v1",
            "run_id": args.run_id,
            "stage": STAGE,
            "event_id": event_id,
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
            **routing,
            "verdict": "PASS",
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"family_router": out_path},
            verdict="PASS",
            results={
                "event_id": event_id,
                "routing_mode": routing["routing_mode"],
                "primary_family": routing["primary_family"],
                "n_families": len(routing["families_to_run"]),
                "program_classification": routing["program_classification"],
                "fallback_classification": routing["fallback_classification"],
                "fallback_path": routing["fallback_path"],
                "fallback_reason": routing["fallback_reason"],
            },
        )
        log_stage_paths(ctx)
        print(
            f"[{STAGE}] primary_family={routing['primary_family']} "
            f"families={','.join(routing['families_to_run'])}"
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
