#!/usr/bin/env python3
"""E5-D — Bridge to MALDA (or external Bayesian framework).

Translates compatible_set.json + estimates.json to the input format of an
external inference framework (MALDA, pyRing, or analog).

GOVERNANCE WARNING: The target schema (malda_schema.json) is an external
dependency not controlled by BASURIN.  This file MUST exist before execution:
  experiment/e5d_bridge_malda/external_input/malda_schema.json

This bridge NEVER writes to any canonical directory.  It is a one-way
translation layer.

Governance:
  - Reads compatible_set.json + estimates.json (RUN_VALID=PASS).
  - Reads external_input/malda_schema.json (MUST pre-exist).
  - Writes only under runs/<run_id>/experiment/bridge_malda/.
  - Never promoted to core (external dependency fragility).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mvp.experiment.base_contract import (
    REQUIRED_CANONICAL_GATES,
    GovernanceViolation,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_and_load_run,
    write_manifest,
)

SCHEMA_VERSION = "e5d-0.1"
EXPERIMENT_NAME = "bridge_malda"

# External schema location (relative to experiment/)
EXTERNAL_SCHEMA_PATH = Path(__file__).parent / "e5d_bridge_malda" / "external_input" / "malda_schema.json"


def _check_external_schema() -> dict | None:
    """Check if external schema exists.  Returns schema or None."""
    if EXTERNAL_SCHEMA_PATH.exists():
        return load_json(EXTERNAL_SCHEMA_PATH)
    return None


def translate_to_malda(
    run_id: str,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Translate BASURIN artifacts to MALDA input format."""
    external_schema = _check_external_schema()
    schema_status = "LOADED" if external_schema else "MISSING — bridge operates in DRAFT mode"

    run_dir, _ = validate_and_load_run(run_id, runs_root)
    cs_path = run_dir / REQUIRED_CANONICAL_GATES["compatible_set"]
    est_path = run_dir / REQUIRED_CANONICAL_GATES["estimates"]

    for p, name in [(cs_path, "compatible_set"), (est_path, "estimates")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} missing: {p}")

    cs = load_json(cs_path)
    estimates = load_json(est_path)

    # Standard BASURIN fields that can be translated
    translatable_fields = [
        "frequency_220", "quality_factor_220", "damping_time_220",
        "frequency_221", "quality_factor_221", "damping_time_221",
        "delta_lnL", "mahalanobis_d2",
    ]

    # Determine which fields exist in our data
    geometries = cs if isinstance(cs, list) else cs.get("geometries", cs.get("compatible", []))
    available_fields = set()
    for g in geometries:
        available_fields.update(g.keys())

    fields_translated = [f for f in translatable_fields if f in available_fields]
    fields_untranslatable = []

    # Check against external schema if available
    if external_schema:
        required_ext = set(external_schema.get("required_fields", []))
        fields_untranslatable = sorted(required_ext - available_fields)

    # Build payload in generic format (actual translation depends on schema)
    payload_entries = []
    for g in geometries:
        entry = {}
        for field in fields_translated:
            if field in g:
                entry[field] = g[field]
        entry["geometry_id"] = g.get("geometry_id", g.get("id"))
        payload_entries.append(entry)

    # Validation report
    validation = {
        "n_geometries": len(payload_entries),
        "fields_translated": fields_translated,
        "fields_untranslatable": fields_untranslatable,
        "schema_loaded": external_schema is not None,
    }

    input_hashes = {
        "compatible_set": sha256_file(cs_path),
        "estimates": sha256_file(est_path),
    }
    if EXTERNAL_SCHEMA_PATH.exists():
        input_hashes["malda_schema"] = sha256_file(EXTERNAL_SCHEMA_PATH)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": run_id,
        "target_framework": "MALDA",
        "target_schema_version": (
            external_schema.get("version", "LOADED") if external_schema
            else "UNKNOWN — declare before executing"
        ),
        "fields_translated": fields_translated,
        "fields_untranslatable": fields_untranslatable,
        "bridge_status": "READY" if external_schema else "DRAFT — requires malda_schema.json",
        "payload": payload_entries,
        "validation": validation,
        "input_hashes": input_hashes,
    }


def run_bridge(
    run_id: str,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full bridge: validate, translate, write."""
    result = translate_to_malda(run_id, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    run_dir, _ = validate_and_load_run(run_id, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    _write_json_atomic(out_dir / "malda_input_payload.json", result["payload"])
    _write_json_atomic(out_dir / "translation_manifest.json", {
        "fields_translated": result["fields_translated"],
        "fields_untranslatable": result["fields_untranslatable"],
        "input_hashes": result["input_hashes"],
    })
    _write_json_atomic(out_dir / "bridge_validation_report.json", result["validation"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E5-D: Bridge to MALDA/external framework")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not EXTERNAL_SCHEMA_PATH.exists():
        print(f"WARNING: {EXTERNAL_SCHEMA_PATH} does not exist — bridge in DRAFT mode")

    result = run_bridge(run_id=args.run_id, runs_root=args.runs_root, dry_run=args.dry_run)
    print(f"Bridge status: {result['bridge_status']}")
    print(f"Fields translated: {len(result['fields_translated'])}")
    print(f"Fields untranslatable: {len(result['fields_untranslatable'])}")


if __name__ == "__main__":
    main()
