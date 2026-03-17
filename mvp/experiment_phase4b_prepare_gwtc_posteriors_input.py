#!/usr/bin/env python3
"""Phase4B prepare: materialise/validate external IMR input (GWTC posteriors).

Chain context (why this script exists):
    Phase3  → K_common(phys_key)
    Phase4A (phase4_hawking_area_common_support) → A_final, S_final  [domain admissibility only]
    + external IMR input (THIS SCRIPT prepares/validates) → A_initial per event
    → Phase4B (phase4b_hawking_area_law_filter) → hawking_pass = (A_final >= A_initial)

A_initial is NOT produced by Phase3 or Phase4A.
It must come from external GWTC IMR posterior samples: m1_source, m2_source, chi1, chi2.

Modes:
  --write-placeholders  Create placeholder JSONs for missing events (no overwrite).
  --validate-only       Validate existing JSON files (schema + optional numeric check).

All writes to:
    runs/<host_run>/experiment/phase4b_prepare_gwtc_posteriors_input/
      manifest.json
      stage_summary.json
      outputs/
        required_event_ids.txt
        gwtc_posteriors_inventory.json
        gwtc_posteriors_validation_summary.json

Placeholder writes (--write-placeholders only) also go to:
    runs/<host_run>/external_inputs/gwtc_posteriors/<EVENT_ID>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
)
from mvp import contracts

EXPERIMENT_NAME = "phase4b_prepare_gwtc_posteriors_input"
PHASE4_UPSTREAM_NAME = "phase4_hawking_area_common_support"

SCHEMA_VERSION_INVENTORY = "gwtc_posteriors_inventory_v1"
SCHEMA_VERSION_VALIDATION = "gwtc_posteriors_validation_summary_v1"
SCHEMA_VERSION_POSTERIOR = "gwtc_posteriors_json_v1"

_REQUIRED_SAMPLE_FIELDS = ("m1_source", "m2_source", "chi1", "chi2")

_PHASE4_REQUIRED_ARTIFACTS = [
    "stage_summary.json",
    "manifest.json",
    "outputs/per_event_hawking_area.csv",
]


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------


def _require_phase4_upstream(phase4_dir: Path) -> dict[str, Any]:
    """Assert Phase4 upstream artifacts exist and verdict == PASS.

    Raises FileNotFoundError for missing artifacts.
    Raises RuntimeError if verdict != PASS.
    Returns parsed stage_summary dict.
    """
    for rel in _PHASE4_REQUIRED_ARTIFACTS:
        p = phase4_dir / rel
        if not p.exists():
            raise FileNotFoundError(
                f"[{EXPERIMENT_NAME}] Missing Phase4 upstream artifact: {p}"
            )
    ss_path = phase4_dir / "stage_summary.json"
    ss = json.loads(ss_path.read_text(encoding="utf-8"))
    if ss.get("verdict") != "PASS":
        raise RuntimeError(
            f"[{EXPERIMENT_NAME}] Phase4 upstream verdict is not PASS: "
            f"{ss.get('verdict')!r}"
        )
    return ss


# ---------------------------------------------------------------------------
# Event-id extraction from Phase4 upstream
# ---------------------------------------------------------------------------


def _read_required_event_ids_from_phase4(phase4_dir: Path) -> list[str]:
    """Return sorted list of unique event_ids from per_event_hawking_area.csv."""
    csv_path = phase4_dir / "outputs" / "per_event_hawking_area.csv"
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen: set[str] = set()
    for row in rows:
        eid = row.get("event_id", "").strip()
        if eid:
            seen.add(eid)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Posterior JSON helpers
# ---------------------------------------------------------------------------


def _placeholder_payload(event_id: str) -> dict[str, Any]:
    """Return a structurally valid placeholder posterior JSON (no numeric data)."""
    return {
        "event_id": event_id,
        "samples": [
            {
                "m1_source": "TO_FILL",
                "m2_source": "TO_FILL",
                "chi1": "TO_FILL",
                "chi2": "TO_FILL",
            }
        ],
    }


def _load_json_posterior(path: Path) -> dict[str, Any]:
    """Load a posterior JSON file. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"[{EXPERIMENT_NAME}] Posterior file not found: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_posterior_schema(
    data: dict[str, Any], path_name: str
) -> list[str]:
    """Validate minimum schema contract. Returns list of error strings (empty = OK)."""
    errors: list[str] = []
    if "event_id" not in data:
        errors.append(f"{path_name}: missing 'event_id'")
    if "samples" not in data:
        errors.append(f"{path_name}: missing 'samples'")
        return errors
    samples = data["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        errors.append(f"{path_name}: 'samples' must be non-empty list")
        return errors
    for i, s in enumerate(samples):
        for field in _REQUIRED_SAMPLE_FIELDS:
            if field not in s:
                errors.append(
                    f"{path_name} sample[{i}]: missing required field '{field}'"
                )
    return errors


def _validate_numeric_samples(
    data: dict[str, Any], path_name: str
) -> list[str]:
    """Check all sample fields are numeric and finite. Returns error strings."""
    errors: list[str] = []
    samples = data.get("samples", [])
    for i, s in enumerate(samples):
        for field in _REQUIRED_SAMPLE_FIELDS:
            val = s.get(field)
            if val == "TO_FILL":
                errors.append(
                    f"{path_name} sample[{i}]: field '{field}' is placeholder "
                    f"'TO_FILL'"
                )
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                errors.append(
                    f"{path_name} sample[{i}]: field '{field}'={val!r} is not "
                    f"numeric"
                )
                continue
            if not math.isfinite(fval):
                errors.append(
                    f"{path_name} sample[{i}]: field '{field}'={fval} is not "
                    f"finite"
                )
    return errors


# ---------------------------------------------------------------------------
# Inventory helper
# ---------------------------------------------------------------------------


def _inventory_posteriors(
    posteriors_dir: Path, required_event_ids: list[str]
) -> dict[str, Any]:
    """Compare required vs present posterior JSONs.

    Returns dict with standard inventory fields plus _present_required (internal).
    """
    required_set = set(required_event_ids)
    if posteriors_dir.is_dir():
        present_set = {
            p.stem
            for p in posteriors_dir.iterdir()
            if p.suffix == ".json" and p.is_file()
        }
    else:
        present_set = set()

    missing = sorted(required_set - present_set)
    extra = sorted(present_set - required_set)
    present_required = sorted(required_set & present_set)

    return {
        "required_event_ids": sorted(required_event_ids),
        "present_event_ids": sorted(present_set),
        "missing_event_ids": missing,
        "extra_event_ids": extra,
        "n_required": len(required_set),
        "n_present": len(present_set),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "_present_required": present_required,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_required_event_ids_txt(path: Path, event_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(event_ids) + ("\n" if event_ids else ""), encoding="utf-8"
    )


def _write_inventory_json(
    path: Path,
    host_run: str,
    inv: dict[str, Any],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION_INVENTORY,
        "host_run": host_run,
        "source_phase4_experiment": PHASE4_UPSTREAM_NAME,
        "required_event_ids": inv["required_event_ids"],
        "present_event_ids": inv["present_event_ids"],
        "missing_event_ids": inv["missing_event_ids"],
        "extra_event_ids": inv["extra_event_ids"],
        "n_required": inv["n_required"],
        "n_present": inv["n_present"],
        "n_missing": inv["n_missing"],
        "n_extra": inv["n_extra"],
    }
    write_json_atomic(path, payload)


def _write_validation_summary(
    path: Path,
    host_run: str,
    mode: str,
    require_numeric_samples: bool,
    coverage_complete: bool,
    schema_valid: bool,
    numeric_samples_valid: bool,
    files_created: int,
    files_already_present: int,
    files_invalid_schema: list[str],
    files_invalid_numeric: list[str],
    notes: list[str],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION_VALIDATION,
        "host_run": host_run,
        "mode": mode,
        "require_numeric_samples": require_numeric_samples,
        "coverage_complete": coverage_complete,
        "schema_valid": schema_valid,
        "numeric_samples_valid": numeric_samples_valid,
        "files_created": files_created,
        "files_already_present": files_already_present,
        "files_invalid_schema": files_invalid_schema,
        "files_invalid_numeric": files_invalid_numeric,
        "notes": notes,
    }
    write_json_atomic(path, payload)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    run_id: str,
    mode: str,  # "write_placeholders" | "validate_only"
    require_numeric_samples: bool = False,
    out_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Prepare/validate external IMR input for Phase4B under runs/<run_id>/."""
    if mode not in ("write_placeholders", "validate_only"):
        raise ValueError(f"[{EXPERIMENT_NAME}] Invalid mode: {mode!r}")

    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)

    run_dir = out_root / run_id
    phase4_dir = run_dir / "experiment" / PHASE4_UPSTREAM_NAME
    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"

    # --- Gating ---
    _require_phase4_upstream(phase4_dir)

    # --- Extract required event_ids from Phase4 upstream (sole source of truth) ---
    required_event_ids = _read_required_event_ids_from_phase4(phase4_dir)

    # --- Inventory before any writes ---
    inv = _inventory_posteriors(posteriors_dir, required_event_ids)

    notes: list[str] = [
        "This experiment prepares/validates the external IMR input required by "
        "phase4b_hawking_area_law_filter.",
        "Phase4b requires external IMR input because A_initial(event) is not "
        "produced by Phase3/Phase4A.",
    ]

    files_created = 0
    files_already_present = 0
    files_invalid_schema: list[str] = []
    files_invalid_numeric: list[str] = []

    if mode == "write_placeholders":
        # Create posteriors_dir if missing, write placeholders for absent events
        posteriors_dir.mkdir(parents=True, exist_ok=True)
        for eid in required_event_ids:
            json_path = posteriors_dir / f"{eid}.json"
            if json_path.exists():
                files_already_present += 1
            else:
                payload = _placeholder_payload(eid)
                write_json_atomic(json_path, payload)
                files_created += 1

        # Re-inventory after writes
        inv = _inventory_posteriors(posteriors_dir, required_event_ids)

        # Validate schema of all required present files
        for eid in inv["_present_required"]:
            json_path = posteriors_dir / f"{eid}.json"
            data = _load_json_posterior(json_path)
            errs = _validate_posterior_schema(data, f"{eid}.json")
            if errs:
                files_invalid_schema.append(eid)

    else:  # validate_only — no writes to external_inputs
        files_already_present = inv["n_present"]

        for eid in inv["_present_required"]:
            json_path = posteriors_dir / f"{eid}.json"
            data = _load_json_posterior(json_path)
            errs = _validate_posterior_schema(data, f"{eid}.json")
            if errs:
                files_invalid_schema.append(eid)
            elif require_numeric_samples:
                num_errs = _validate_numeric_samples(data, f"{eid}.json")
                if num_errs:
                    files_invalid_numeric.append(eid)

    # --- Determine final status flags ---
    coverage_complete = inv["n_missing"] == 0
    schema_valid = len(files_invalid_schema) == 0
    numeric_samples_valid = len(files_invalid_numeric) == 0

    # In validate_only mode, abort if contract is broken
    if mode == "validate_only":
        if not coverage_complete:
            raise RuntimeError(
                f"[{out_name}] Coverage incomplete: missing event posteriors: "
                f"{inv['missing_event_ids']}"
            )
        if not schema_valid:
            raise RuntimeError(
                f"[{out_name}] Schema invalid for: {files_invalid_schema}"
            )
        if require_numeric_samples and not numeric_samples_valid:
            raise RuntimeError(
                f"[{out_name}] Non-numeric/placeholder samples in: "
                f"{files_invalid_numeric}"
            )

    overall_pass = coverage_complete and schema_valid
    if require_numeric_samples:
        overall_pass = overall_pass and numeric_samples_valid
    verdict = "PASS" if overall_pass else "WARN"

    # --- Build output payloads ---
    inventory_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_INVENTORY,
        "host_run": run_id,
        "source_phase4_experiment": PHASE4_UPSTREAM_NAME,
        "required_event_ids": inv["required_event_ids"],
        "present_event_ids": inv["present_event_ids"],
        "missing_event_ids": inv["missing_event_ids"],
        "extra_event_ids": inv["extra_event_ids"],
        "n_required": inv["n_required"],
        "n_present": inv["n_present"],
        "n_missing": inv["n_missing"],
        "n_extra": inv["n_extra"],
    }

    validation_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_VALIDATION,
        "host_run": run_id,
        "mode": mode,
        "require_numeric_samples": require_numeric_samples,
        "coverage_complete": coverage_complete,
        "schema_valid": schema_valid,
        "numeric_samples_valid": numeric_samples_valid,
        "files_created": files_created,
        "files_already_present": files_already_present,
        "files_invalid_schema": files_invalid_schema,
        "files_invalid_numeric": files_invalid_numeric,
        "notes": notes,
    }

    stage_summary: dict[str, Any] = {
        "experiment_name": out_name,
        "verdict": verdict,
        "host_run": run_id,
        "created_utc": utc_now_iso(),
        "gating": {
            "host_run_valid": True,
            "phase4_upstream_present": True,
            "phase4_upstream_pass": True,
        },
        "mode": mode,
        "require_numeric_samples": require_numeric_samples,
        "external_input_definition": {
            "path": f"runs/{run_id}/external_inputs/gwtc_posteriors",
            "schema_version": SCHEMA_VERSION_POSTERIOR,
            "required_sample_fields": list(_REQUIRED_SAMPLE_FIELDS),
        },
        "inventory": {
            "n_required": inv["n_required"],
            "n_present": inv["n_present"],
            "n_missing": inv["n_missing"],
            "n_extra": inv["n_extra"],
        },
        "validation": {
            "coverage_complete": coverage_complete,
            "schema_valid": schema_valid,
            "numeric_samples_valid": numeric_samples_valid,
        },
        "notes": notes,
    }

    # --- Atomic write (tempdir + shutil.move) ---
    exp_dir = run_dir / "experiment" / out_name
    try:
        exp_dir.relative_to(run_dir)
    except ValueError:
        raise RuntimeError(
            f"[{EXPERIMENT_NAME}] Write-path safety violation: "
            f"{exp_dir} outside {run_dir}"
        )

    phase4_csv_path = phase4_dir / "outputs" / "per_event_hawking_area.csv"

    with tempfile.TemporaryDirectory(prefix=f"{out_name}_", dir=run_dir) as tmpdir:
        tmp_stage = Path(tmpdir) / "stage"
        tmp_outputs = tmp_stage / "outputs"
        tmp_outputs.mkdir(parents=True, exist_ok=True)

        # 1. required_event_ids.txt
        out_event_ids_txt = tmp_outputs / "required_event_ids.txt"
        _write_required_event_ids_txt(out_event_ids_txt, required_event_ids)

        # 2. gwtc_posteriors_inventory.json
        out_inventory = tmp_outputs / "gwtc_posteriors_inventory.json"
        write_json_atomic(out_inventory, inventory_payload)

        # 3. gwtc_posteriors_validation_summary.json
        out_validation = tmp_outputs / "gwtc_posteriors_validation_summary.json"
        write_json_atomic(out_validation, validation_payload)

        outputs = [out_event_ids_txt, out_inventory, out_validation]
        output_records = [
            {"path": str(p.relative_to(tmp_stage)), "sha256": sha256_file(p)}
            for p in outputs
        ]

        # 4. manifest.json
        input_records = [
            {
                "path": str(phase4_csv_path),
                "sha256": sha256_file(phase4_csv_path),
            },
            {
                "path": str(phase4_dir / "stage_summary.json"),
                "sha256": sha256_file(phase4_dir / "stage_summary.json"),
            },
        ]
        manifest_payload: dict[str, Any] = {
            "schema_version": "mvp_manifest_v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "artifacts": output_records,
            "inputs": input_records,
        }
        write_json_atomic(tmp_stage / "manifest.json", manifest_payload)

        # 5. stage_summary.json
        stage_summary["outputs"] = output_records
        stage_summary["inputs"] = input_records
        write_json_atomic(tmp_stage / "stage_summary.json", stage_summary)

        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_stage), str(exp_dir))

    contracts.log_stage_paths(
        SimpleNamespace(
            out_root=out_root,
            stage_dir=exp_dir,
            outputs_dir=exp_dir / "outputs",
        )
    )

    return {
        "inventory": inventory_payload,
        "validation_summary": validation_payload,
        "stage_summary": stage_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-4B prepare: materialise/validate external GWTC IMR posterior "
            "input required by phase4b_hawking_area_law_filter."
        )
    )
    ap.add_argument("--host-run", required=True, dest="host_run", help="Host run ID")
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--write-placeholders",
        action="store_true",
        dest="write_placeholders",
        help="Create placeholder JSONs for missing events (does not overwrite existing).",
    )
    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        dest="validate_only",
        help="Validate existing JSON files (no writes to external_inputs).",
    )
    ap.add_argument(
        "--require-numeric-samples",
        action="store_true",
        dest="require_numeric_samples",
        default=False,
        help=(
            "(validate-only) Abort if any sample field is 'TO_FILL' or non-numeric."
        ),
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    mode = "write_placeholders" if args.write_placeholders else "validate_only"
    run_experiment(
        run_id=args.host_run,
        mode=mode,
        require_numeric_samples=args.require_numeric_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
