#!/usr/bin/env python3
"""Phase4B convert: transform real IMR posteriors to canonical external input schema.

Chain context (why this script exists):
    Phase3  → K_common(phys_key)
    Phase4A (phase4_hawking_area_common_support) → A_final, S_final  [domain admissibility only]
    + external IMR input (THIS SCRIPT converts real posteriors) → A_initial per event
    → Phase4B (phase4b_hawking_area_law_filter) → hawking_pass = (A_final >= A_initial)

A_initial is NOT produced by Phase3 or Phase4A.
It must come from external GWTC IMR posterior samples: m1_source, m2_source, chi1, chi2.

This script converts real IMR posteriors from a user-supplied source directory
to the canonical JSON schema already consumed by Phase4B.

Modes:
  --validate-only  Inspect source, check coverage and field extractability. No writes to
                   external_inputs/gwtc_posteriors/.
  --write-output   Convert and write canonical JSONs to
                   runs/<host_run>/external_inputs/gwtc_posteriors/<EVENT_ID>.json

Supported input formats:
  --input-format json   JSON files with configurable field mapping (fully implemented).
  --input-format hdf5   HDF5 files via h5py; requires --hdf5-dataset <dataset_path>.
                        Reads compound (structured) datasets, e.g.:
                        --hdf5-dataset C01:Mixed/posterior_samples

Field mapping (required for both formats):
  --field-m1 <source_field>    Source field name mapping to m1_source
  --field-m2 <source_field>    Source field name mapping to m2_source
  --field-chi1 <source_field>  Source field name mapping to chi1
  --field-chi2 <source_field>  Source field name mapping to chi2

All writes to:
    runs/<host_run>/experiment/phase4b_convert_imr_posteriors/
      manifest.json
      stage_summary.json
      outputs/
        required_event_ids.txt
        source_inventory.json
        conversion_summary.json

Canonical outputs (--write-output only):
    runs/<host_run>/external_inputs/gwtc_posteriors/
      <EVENT_ID>.json
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

EXPERIMENT_NAME = "phase4b_convert_imr_posteriors"
PHASE4_UPSTREAM_NAME = "phase4_hawking_area_common_support"

SCHEMA_VERSION_INVENTORY = "phase4b_convert_imr_posteriors_inventory_v1"
SCHEMA_VERSION_SUMMARY = "phase4b_convert_imr_posteriors_summary_v1"

_CANONICAL_FIELDS = ("m1_source", "m2_source", "chi1", "chi2")

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
# Source file location
# ---------------------------------------------------------------------------


def _locate_source_file(
    source_dir: Path, event_id: str, input_format: str
) -> tuple[Path | None, str]:
    """Find source file for event_id in source_dir.

    Returns (path_or_None, location_note).
    Tries in order (multiple matches at each level: sorted alphabetically, first chosen):
      1. exact match:     <event_id>.<ext>
      2. prefix match:    <event_id>*.<ext>
      3. substring match: *<event_id>*.<ext>
    """
    ext = ".json" if input_format == "json" else ".h5"

    # 1. Exact match
    exact = source_dir / f"{event_id}{ext}"
    if exact.exists():
        return exact, "exact_filename_match"

    # 2. Prefix match: <event_id>*.<ext>
    matches = sorted(source_dir.glob(f"{event_id}*{ext}"))
    if matches:
        return (
            matches[0],
            f"prefix_match:{matches[0].name} (event_id is filename prefix)",
        )

    # 3. Substring match: *<event_id>*.<ext>
    matches = sorted(source_dir.glob(f"*{event_id}*{ext}"))
    if matches:
        return (
            matches[0],
            f"substring_match:{matches[0].name} (event_id found as substring in filename)",
        )

    return None, "not_found"


# ---------------------------------------------------------------------------
# JSON sample extraction with field mapping
# ---------------------------------------------------------------------------


def _extract_json_samples(
    path: Path,
    event_id: str,
    field_mapping: dict[str, str],
) -> tuple[list[dict[str, float]], list[str]]:
    """Extract and map samples from source JSON using field_mapping.

    field_mapping: {"m1_source": "<source_field>", "m2_source": ..., ...}
    Returns (canonical_samples, notes).

    Raises ValueError if:
    - required source fields are missing from any sample
    - any value is non-numeric or non-finite
    - source JSON structure is not list or dict with a list value
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    notes: list[str] = []

    # Locate the list of sample records
    if isinstance(data, list):
        raw_samples = data
        notes.append(f"{event_id}: source JSON is top-level array")
    elif isinstance(data, dict):
        # Prefer key "samples", else first list-valued key
        if "samples" in data and isinstance(data["samples"], list):
            raw_samples = data["samples"]
        else:
            list_keys = [k for k, v in data.items() if isinstance(v, list)]
            if not list_keys:
                raise ValueError(
                    f"[{EXPERIMENT_NAME}] No list found in source JSON for "
                    f"event {event_id}: {path}"
                )
            key = list_keys[0]
            raw_samples = data[key]
            notes.append(
                f"{event_id}: source JSON extracted from key '{key}' (not 'samples')"
            )
    else:
        raise ValueError(
            f"[{EXPERIMENT_NAME}] Unexpected JSON structure (not list or dict) "
            f"for event {event_id}: {path}"
        )

    if not raw_samples:
        raise ValueError(
            f"[{EXPERIMENT_NAME}] Empty samples list for event {event_id}: {path}"
        )

    # Validate required source fields exist in first sample
    first = raw_samples[0] if isinstance(raw_samples[0], dict) else {}
    for canonical, source_field in field_mapping.items():
        if source_field not in first:
            raise ValueError(
                f"[{EXPERIMENT_NAME}] Missing mapped field '{source_field}' "
                f"(-> canonical '{canonical}') in event {event_id} sample[0]: {path}"
            )

    # Extract and validate all samples
    canonical_samples: list[dict[str, float]] = []
    for i, s in enumerate(raw_samples):
        if not isinstance(s, dict):
            raise ValueError(
                f"[{EXPERIMENT_NAME}] sample[{i}] is not a dict in event "
                f"{event_id}: {path}"
            )
        csample: dict[str, float] = {}
        for canonical, source_field in field_mapping.items():
            if source_field not in s:
                raise ValueError(
                    f"[{EXPERIMENT_NAME}] Missing field '{source_field}' in event "
                    f"{event_id} sample[{i}]: {path}"
                )
            val = s[source_field]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"[{EXPERIMENT_NAME}] Non-numeric value {val!r} for field "
                    f"'{source_field}' (-> '{canonical}') in event {event_id} "
                    f"sample[{i}]: {path}"
                )
            if not math.isfinite(fval):
                raise ValueError(
                    f"[{EXPERIMENT_NAME}] Non-finite value {fval} for field "
                    f"'{source_field}' (-> '{canonical}') in event {event_id} "
                    f"sample[{i}]: {path}"
                )
            csample[canonical] = fval
        canonical_samples.append(csample)

    return canonical_samples, notes


# ---------------------------------------------------------------------------
# HDF5 sample extraction with field mapping
# ---------------------------------------------------------------------------


def _extract_hdf5_samples(
    path: Path,
    dataset_path: str,
    field_mapping: dict[str, str],
) -> tuple[list[dict[str, float]], list[str]]:
    """Extract and map samples from an HDF5 compound (structured) dataset.

    Args:
        path:          Path to the HDF5 file.
        dataset_path:  Dataset path inside the HDF5 file, e.g.
                       'C01:Mixed/posterior_samples'.
        field_mapping: {"m1_source": "<source_field>", ...}

    Returns:
        (canonical_samples, notes)

    Raises:
        ImportError:  h5py not installed.
        ValueError:   Dataset not found, not structured, missing fields,
                      non-numeric or non-finite values, or empty dataset.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            f"[{EXPERIMENT_NAME}] h5py is required for HDF5 input format. "
            "Install with: pip install h5py"
        )

    notes: list[str] = []

    with h5py.File(str(path), "r") as f:
        if dataset_path not in f:
            raise ValueError(
                f"[{EXPERIMENT_NAME}] Dataset '{dataset_path}' not found in "
                f"HDF5 file: {path}"
            )
        ds = f[dataset_path]
        data = ds[()]  # numpy structured array

    if not hasattr(data, "dtype") or data.dtype.names is None:
        raise ValueError(
            f"[{EXPERIMENT_NAME}] Dataset '{dataset_path}' is not a structured "
            f"(compound) array in: {path}"
        )

    available_fields = set(data.dtype.names)

    # Validate all mapped source fields exist
    for canonical, source_field in field_mapping.items():
        if source_field not in available_fields:
            raise ValueError(
                f"[{EXPERIMENT_NAME}] Missing mapped field '{source_field}' "
                f"(-> canonical '{canonical}') in dataset '{dataset_path}': {path}. "
                f"Available fields: {sorted(available_fields)}"
            )

    if len(data) == 0:
        raise ValueError(
            f"[{EXPERIMENT_NAME}] Empty dataset '{dataset_path}' in: {path}"
        )

    # Extract and validate all rows
    canonical_samples: list[dict[str, float]] = []
    for i in range(len(data)):
        csample: dict[str, float] = {}
        for canonical, source_field in field_mapping.items():
            val = data[source_field][i]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"[{EXPERIMENT_NAME}] Non-numeric value {val!r} for field "
                    f"'{source_field}' (-> '{canonical}') in dataset "
                    f"'{dataset_path}' row {i}: {path}"
                )
            if not math.isfinite(fval):
                raise ValueError(
                    f"[{EXPERIMENT_NAME}] Non-finite value {fval} for field "
                    f"'{source_field}' (-> '{canonical}') in dataset "
                    f"'{dataset_path}' row {i}: {path}"
                )
            csample[canonical] = fval
        canonical_samples.append(csample)

    notes.append(
        f"HDF5 dataset '{dataset_path}' extracted {len(canonical_samples)} rows"
    )
    return canonical_samples, notes


# ---------------------------------------------------------------------------
# Canonical JSON builder
# ---------------------------------------------------------------------------


def _build_canonical_json(
    event_id: str, samples: list[dict[str, float]]
) -> dict[str, Any]:
    """Build the canonical posterior JSON payload for one event."""
    return {
        "event_id": event_id,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Source inventory
# ---------------------------------------------------------------------------


def _inventory_source(
    source_dir: Path,
    required_event_ids: list[str],
    input_format: str,
) -> dict[str, Any]:
    """Compare required events vs source files present in source_dir.

    Coverage is determined by _locate_source_file (exact + glob), so an event
    counts as present even if the filename has a suffix after the event_id.
    Returns inventory dict with standard fields plus _located_required (internal).
    """
    required_set = set(required_event_ids)
    ext = ".json" if input_format == "json" else ".h5"

    if source_dir.is_dir():
        present_stems = {
            p.stem for p in source_dir.iterdir()
            if p.suffix == ext and p.is_file()
        }
    else:
        present_stems = set()

    # Coverage: which required events can actually be located (exact or glob)
    located: set[str] = set()
    for eid in required_event_ids:
        path, _ = _locate_source_file(source_dir, eid, input_format)
        if path is not None:
            located.add(eid)

    missing = sorted(required_set - located)
    extra = sorted(present_stems - required_set)
    present_required = sorted(located)

    return {
        "schema_version": SCHEMA_VERSION_INVENTORY,
        "host_run": "",  # filled in by caller
        "source_dir": str(source_dir),
        "input_format": input_format,
        "required_event_ids": sorted(required_event_ids),
        "present_source_events": sorted(present_stems),
        "missing_source_events": missing,
        "extra_source_events": extra,
        "n_required": len(required_event_ids),
        "n_present": len(present_stems),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "_located_required": present_required,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_required_event_ids_txt(path: Path, event_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(event_ids) + ("\n" if event_ids else ""), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    run_id: str,
    source_dir: str | Path,
    input_format: str,
    field_mapping: dict[str, str],
    mode: str,  # "validate_only" | "write_output"
    overwrite_existing: bool = False,
    out_name: str = EXPERIMENT_NAME,
    hdf5_dataset: str | None = None,
) -> dict[str, Any]:
    """Convert real IMR posteriors to canonical Phase4B external input.

    Args:
        run_id:             Host run identifier.
        source_dir:         Directory containing real posterior files.
        input_format:       "json" or "hdf5".
        field_mapping:      {"m1_source": src_field, "m2_source": ...,
                             "chi1": ..., "chi2": ...}.
        mode:               "validate_only" | "write_output".
        overwrite_existing: If True, overwrite existing canonical JSONs in
                            write_output mode (default: skip existing).
        out_name:           Experiment directory name (default: EXPERIMENT_NAME).
        hdf5_dataset:       Dataset path within HDF5 files, required when
                            input_format == "hdf5" (e.g. "C01:Mixed/posterior_samples").

    Returns:
        dict with keys "inventory", "conversion_summary", "stage_summary".

    Raises:
        ValueError:         Invalid mode, input_format, incomplete field_mapping,
                            or missing hdf5_dataset when input_format == "hdf5".
        FileNotFoundError:  Missing RUN_VALID, Phase4 upstream artifacts, or source_dir.
        RuntimeError:       RUN_VALID or Phase4 verdict != PASS, coverage incomplete,
                            or field extraction fails.
    """
    if mode not in ("validate_only", "write_output"):
        raise ValueError(f"[{EXPERIMENT_NAME}] Invalid mode: {mode!r}")
    if input_format not in ("json", "hdf5"):
        raise ValueError(f"[{EXPERIMENT_NAME}] Invalid input_format: {input_format!r}")
    if input_format == "hdf5" and not hdf5_dataset:
        raise ValueError(
            f"[{EXPERIMENT_NAME}] --hdf5-dataset is required when "
            f"--input-format hdf5"
        )

    # Validate field_mapping completeness
    for canon in _CANONICAL_FIELDS:
        if canon not in field_mapping:
            raise ValueError(
                f"[{EXPERIMENT_NAME}] Missing field mapping for '{canon}'"
            )

    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)

    run_dir = out_root / run_id
    source_path = Path(source_dir).resolve()
    phase4_dir = run_dir / "experiment" / PHASE4_UPSTREAM_NAME
    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"

    # --- Gating ---
    _require_phase4_upstream(phase4_dir)

    if not source_path.is_dir():
        raise FileNotFoundError(
            f"[{EXPERIMENT_NAME}] source_dir does not exist or is not a directory: "
            f"{source_path}"
        )

    # --- Extract required event_ids (sole source of truth: Phase4 upstream CSV) ---
    required_event_ids = _read_required_event_ids_from_phase4(phase4_dir)

    # --- Source inventory ---
    inv = _inventory_source(source_path, required_event_ids, input_format)
    inv["host_run"] = run_id

    coverage_complete = inv["n_missing"] == 0
    if not coverage_complete:
        raise RuntimeError(
            f"[{out_name}] Source coverage incomplete: missing events: "
            f"{inv['missing_source_events']}"
        )

    # --- Extract samples for all required events ---
    schema_extractable = True
    numeric_samples_valid = True
    invalid_events: list[str] = []
    per_event_samples: dict[str, list[dict[str, float]]] = {}
    per_event_errors: dict[str, str] = {}
    notes: list[str] = [
        "This experiment converts real IMR posteriors to the canonical external "
        "input consumed by phase4b_hawking_area_law_filter.",
        "Correct chain: Phase3/Phase4A output + external IMR input -> Phase4B.",
    ]
    if input_format == "hdf5":
        notes.append(
            f"HDF5 input: dataset '{hdf5_dataset}'. "
            "event_id derived from filename stem (same rule as JSON path)."
        )

    for eid in required_event_ids:
        src_path, loc_note = _locate_source_file(source_path, eid, input_format)
        if src_path is None:
            # Should not happen after coverage check, but guard anyway
            schema_extractable = False
            numeric_samples_valid = False
            invalid_events.append(eid)
            per_event_errors[eid] = "source file not found after coverage check"
            continue
        try:
            if input_format == "json":
                samples, extract_notes = _extract_json_samples(
                    src_path, eid, field_mapping
                )
            else:  # hdf5
                samples, extract_notes = _extract_hdf5_samples(
                    src_path, hdf5_dataset, field_mapping  # type: ignore[arg-type]
                )
            per_event_samples[eid] = samples
            for en in extract_notes:
                notes.append(en)
        except (ValueError, RuntimeError) as exc:
            schema_extractable = False
            numeric_samples_valid = False
            invalid_events.append(eid)
            per_event_errors[eid] = str(exc)

    if not schema_extractable:
        raise RuntimeError(
            f"[{out_name}] Cannot extract required fields for events: "
            f"{invalid_events}. Errors: {per_event_errors}"
        )

    # --- Write canonical JSONs (write_output mode only) ---
    files_written = 0
    files_skipped_existing = 0

    if mode == "write_output":
        posteriors_dir.mkdir(parents=True, exist_ok=True)
        for eid, samples in per_event_samples.items():
            out_path = posteriors_dir / f"{eid}.json"
            if out_path.exists() and not overwrite_existing:
                files_skipped_existing += 1
                notes.append(
                    f"{eid}: skipped (canonical JSON already exists; "
                    f"use --overwrite-existing to replace)"
                )
                continue
            payload = _build_canonical_json(eid, samples)
            write_json_atomic(out_path, payload)
            files_written += 1

    # --- Build output payloads ---
    inventory_payload: dict[str, Any] = {
        k: v for k, v in inv.items() if not k.startswith("_")
    }

    conversion_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_SUMMARY,
        "host_run": run_id,
        "source_dir": str(source_path),
        "input_format": input_format,
        "hdf5_dataset": hdf5_dataset if input_format == "hdf5" else None,
        "mode": mode,
        "field_mapping": field_mapping,
        "coverage_complete": coverage_complete,
        "schema_extractable": schema_extractable,
        "numeric_samples_valid": numeric_samples_valid,
        "files_written": files_written,
        "files_skipped_existing": files_skipped_existing,
        "invalid_events": invalid_events,
        "notes": notes,
    }

    stage_summary: dict[str, Any] = {
        "experiment_name": out_name,
        "verdict": "PASS",
        "host_run": run_id,
        "created_utc": utc_now_iso(),
        "gating": {
            "host_run_valid": True,
            "phase4_upstream_present": True,
            "phase4_upstream_pass": True,
            "source_dir_present": True,
            "source_event_coverage_complete": coverage_complete,
        },
        "mode": mode,
        "input_format": input_format,
        "hdf5_dataset": hdf5_dataset if input_format == "hdf5" else None,
        "field_mapping": field_mapping,
        "inventory": {
            "n_required": inv["n_required"],
            "n_present": inv["n_present"],
            "n_missing": inv["n_missing"],
            "n_extra": inv["n_extra"],
        },
        "validation": {
            "coverage_complete": coverage_complete,
            "schema_extractable": schema_extractable,
            "numeric_samples_valid": numeric_samples_valid,
        },
        "notes": [
            "This experiment converts real IMR posteriors to the canonical external "
            "input consumed by phase4b_hawking_area_law_filter.",
            "Correct chain: Phase3/Phase4A output + external IMR input -> Phase4B.",
        ] + (
            [f"HDF5 input: dataset '{hdf5_dataset}'. "
             "event_id derived from filename stem (same rule as JSON path)."]
            if input_format == "hdf5" else []
        ),
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

        # 2. source_inventory.json
        out_inventory = tmp_outputs / "source_inventory.json"
        write_json_atomic(out_inventory, inventory_payload)

        # 3. conversion_summary.json
        out_conversion = tmp_outputs / "conversion_summary.json"
        write_json_atomic(out_conversion, conversion_summary)

        outputs = [out_event_ids_txt, out_inventory, out_conversion]
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
        "conversion_summary": conversion_summary,
        "stage_summary": stage_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-4B convert: transform real IMR posteriors to the canonical "
            "external input schema consumed by phase4b_hawking_area_law_filter."
        )
    )
    ap.add_argument("--host-run", required=True, dest="host_run", help="Host run ID")
    ap.add_argument(
        "--source-dir",
        required=True,
        dest="source_dir",
        help="Directory containing real posterior files",
    )
    ap.add_argument(
        "--input-format",
        required=True,
        dest="input_format",
        choices=["json", "hdf5"],
        help="Input format of posterior files",
    )
    # Field mapping
    ap.add_argument(
        "--field-m1", required=True, dest="field_m1",
        help="Source field name mapping to m1_source",
    )
    ap.add_argument(
        "--field-m2", required=True, dest="field_m2",
        help="Source field name mapping to m2_source",
    )
    ap.add_argument(
        "--field-chi1", required=True, dest="field_chi1",
        help="Source field name mapping to chi1",
    )
    ap.add_argument(
        "--field-chi2", required=True, dest="field_chi2",
        help="Source field name mapping to chi2",
    )
    # Optional
    ap.add_argument(
        "--event-id-field",
        dest="event_id_field",
        default=None,
        help="(optional) Source field name for event_id within the JSON (informational)",
    )
    ap.add_argument(
        "--hdf5-dataset",
        dest="hdf5_dataset",
        default=None,
        help=(
            "Dataset path within HDF5 file, required when --input-format hdf5. "
            "Example: C01:Mixed/posterior_samples"
        ),
    )
    # Mode (mutually exclusive)
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        dest="validate_only",
        help="Inspect source only; no writes to external_inputs/gwtc_posteriors/",
    )
    mode_group.add_argument(
        "--write-output",
        action="store_true",
        dest="write_output",
        help="Convert and write canonical JSONs to external_inputs/gwtc_posteriors/",
    )
    ap.add_argument(
        "--overwrite-existing",
        action="store_true",
        dest="overwrite_existing",
        default=False,
        help="(write-output) Overwrite existing canonical JSONs (default: skip existing)",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.input_format == "hdf5" and not args.hdf5_dataset:
        print(
            "error: --hdf5-dataset is required when --input-format hdf5",
            file=sys.stderr,
        )
        return 2

    mode = "validate_only" if args.validate_only else "write_output"
    field_mapping = {
        "m1_source": args.field_m1,
        "m2_source": args.field_m2,
        "chi1": args.field_chi1,
        "chi2": args.field_chi2,
    }
    run_experiment(
        run_id=args.host_run,
        source_dir=args.source_dir,
        input_format=args.input_format,
        field_mapping=field_mapping,
        mode=mode,
        overwrite_existing=args.overwrite_existing,
        hdf5_dataset=args.hdf5_dataset if args.input_format == "hdf5" else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
