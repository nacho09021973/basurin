#!/usr/bin/env python3
"""
stages/ringdown_real_v0_stage.py
--------------------------------
Canonic stage for preparing the real v0 event package for BASURIN.

Produces:
  runs/<run_id>/ringdown_real_v0/outputs/real_v0_events_list.json

If no real data is available (no --data-source-dir provided or empty),
the stage aborts with a categorized MISSING_REAL_DATA contract failure.

Supports --dry-run mode for validation without writing outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1], _here.parents[2]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

import argparse
import json
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "ringdown_real_v0"

REQUIRED_STRAIN_KEYS = {"strain", "h"}
MINIMUM_STRAIN_LENGTH = 64


def abort_contract(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_git_sha() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if res.returncode != 0:
        return None
    value = res.stdout.strip()
    return value or None


def _validate_strain_npz(path: Path) -> Dict[str, Any]:
    """Validate that a strain.npz file is readable and has valid structure."""
    result: Dict[str, Any] = {
        "valid": False,
        "error": None,
        "shape": None,
        "fs": None,
    }
    try:
        data = np.load(path)
        keys = set(data.files)

        strain = None
        for k in ["strain", "h"]:
            if k in keys:
                strain = np.asarray(data[k], dtype=float)
                break

        if strain is None:
            for k in data.files:
                arr = np.asarray(data[k])
                if arr.ndim == 1 and arr.size >= MINIMUM_STRAIN_LENGTH:
                    strain = np.asarray(arr, dtype=float)
                    break

        if strain is None:
            result["error"] = "no_strain_array"
            return result

        if strain.ndim != 1:
            result["error"] = "strain_not_1d"
            return result

        if strain.size < MINIMUM_STRAIN_LENGTH:
            result["error"] = f"strain_too_short:{strain.size}"
            return result

        if not np.all(np.isfinite(strain)):
            result["error"] = "strain_has_nan_or_inf"
            return result

        result["shape"] = list(strain.shape)

        if "fs" in keys:
            result["fs"] = float(np.asarray(data["fs"]).reshape(-1)[0])
        elif "t" in keys:
            t = np.asarray(data["t"], dtype=float)
            if t.size > 1:
                dt = float(np.median(np.diff(t)))
                if dt > 0:
                    result["fs"] = float(1.0 / dt)

        result["valid"] = True

    except Exception as exc:
        result["error"] = f"exception:{type(exc).__name__}:{exc}"

    return result


def _discover_real_events(
    data_source_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Discover real event data from a source directory.

    Expected structure:
      data_source_dir/
        <event_id>/
          strain.npz  (or <event_id>.npz)

    Returns list of event dicts with event_id and source_path.
    """
    events = []

    if not data_source_dir.exists():
        return events

    for item in sorted(data_source_dir.iterdir()):
        if item.is_dir():
            strain_path = item / "strain.npz"
            if strain_path.exists():
                events.append({
                    "event_id": item.name,
                    "source_path": strain_path,
                })
        elif item.is_file() and item.suffix == ".npz":
            events.append({
                "event_id": item.stem,
                "source_path": item,
            })

    return events


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Canonic stage: prepare real v0 event package for ringdown pipeline"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument("--out-root", default="runs", help="runs root (default: runs)")
    ap.add_argument(
        "--data-source-dir",
        type=str,
        default=None,
        help="Directory containing real event data (optional)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, do not write outputs",
    )
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    cases_dir = outputs_dir / "cases"

    data_source_dir = None
    if args.data_source_dir:
        data_source_dir = Path(args.data_source_dir).expanduser().resolve()

    raw_events = []
    if data_source_dir is not None:
        raw_events = _discover_real_events(data_source_dir)

    if not raw_events:
        contract_verdict = {
            "overall_verdict": "FAIL",
            "timestamp": utc_now_iso(),
            "schema_version": "ringdown_real_v0_stage_v1",
            "contracts": [
                {
                    "id": "REAL_V0_DATA_AVAILABLE",
                    "verdict": "FAIL",
                    "violations": ["missing_real_v0_inputs"],
                    "metrics": {
                        "data_source_dir": str(data_source_dir) if data_source_dir else None,
                        "n_events_found": 0,
                    },
                }
            ],
        }

        summary = {
            "stage": STAGE_NAME,
            "run": args.run,
            "created": utc_now_iso(),
            "inputs": {},
            "parameters": {
                "data_source_dir": str(data_source_dir) if data_source_dir else None,
                "dry_run": args.dry_run,
            },
            "outputs": {},
            "results": {
                "overall_verdict": "FAIL",
                "fail_reason": "missing_real_v0_inputs",
                "n_events": 0,
            },
            "version": {"git_sha": _maybe_git_sha()},
        }

        if not args.dry_run:
            contract_path = outputs_dir / "contract_verdict.json"
            with open(contract_path, "w", encoding="utf-8") as f:
                json.dump(contract_verdict, f, indent=2, sort_keys=True)
                f.write("\n")

            summary_written = write_stage_summary(stage_dir, summary)
            write_manifest(
                stage_dir,
                {
                    "contract": contract_path,
                    "stage_summary": summary_written,
                },
                extra={"version": "1"},
            )

        abort_contract(
            "MISSING_REAL_DATA: no real v0 events found. "
            f"data_source_dir={data_source_dir}"
        )

    events_list = []
    validation_results = []
    n_valid = 0
    n_invalid = 0

    for ev in raw_events:
        event_id = ev["event_id"]
        source_path = ev["source_path"]

        validation = _validate_strain_npz(source_path)
        validation_results.append({
            "event_id": event_id,
            "source_path": str(source_path),
            **validation,
        })

        if not validation["valid"]:
            n_invalid += 1
            continue

        n_valid += 1

        case_dir = cases_dir / event_id
        target_strain = case_dir / "strain.npz"

        if not args.dry_run:
            case_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source_path, target_strain)

        rel_path = f"cases/{event_id}/strain.npz"

        entry = {
            "event_id": event_id,
            "strain_npz": rel_path,
        }

        if validation["fs"] is not None:
            entry["fs"] = validation["fs"]

        events_list.append(entry)

    events_list_path = outputs_dir / "real_v0_events_list.json"
    contract_path = outputs_dir / "contract_verdict.json"

    all_valid = n_valid > 0 and n_invalid == 0
    overall_verdict = "PASS" if n_valid > 0 else "FAIL"

    contract_verdict = {
        "overall_verdict": overall_verdict,
        "timestamp": utc_now_iso(),
        "schema_version": "ringdown_real_v0_stage_v1",
        "contracts": [
            {
                "id": "REAL_V0_DATA_AVAILABLE",
                "verdict": "PASS" if n_valid > 0 else "FAIL",
                "violations": [] if n_valid > 0 else ["missing_real_v0_inputs"],
                "metrics": {
                    "data_source_dir": str(data_source_dir) if data_source_dir else None,
                    "n_events_found": len(raw_events),
                    "n_valid": n_valid,
                    "n_invalid": n_invalid,
                },
            },
            {
                "id": "REAL_V0_IO_VALID",
                "verdict": "PASS" if all_valid else "WARN" if n_valid > 0 else "FAIL",
                "violations": [
                    r["event_id"] for r in validation_results if not r["valid"]
                ][:5],
                "metrics": {
                    "n_valid": n_valid,
                    "n_invalid": n_invalid,
                },
            },
        ],
    }

    inputs_hash = {}
    if data_source_dir is not None:
        for ev in raw_events[:10]:
            try:
                inputs_hash[ev["event_id"]] = sha256_file(ev["source_path"])
            except Exception:
                pass

    summary = {
        "stage": STAGE_NAME,
        "run": args.run,
        "created": utc_now_iso(),
        "inputs": {
            "data_source_dir": str(data_source_dir) if data_source_dir else None,
            "source_hashes": inputs_hash,
        },
        "parameters": {
            "data_source_dir": str(data_source_dir) if data_source_dir else None,
            "dry_run": args.dry_run,
        },
        "outputs": {
            "real_v0_events_list": "outputs/real_v0_events_list.json",
            "cases_dir": "outputs/cases",
        },
        "results": {
            "overall_verdict": overall_verdict,
            "n_events": n_valid,
            "n_invalid": n_invalid,
            "validation_results": validation_results,
        },
        "version": {"git_sha": _maybe_git_sha()},
    }

    if not args.dry_run:
        with open(events_list_path, "w", encoding="utf-8") as f:
            json.dump(events_list, f, indent=2, sort_keys=True)
            f.write("\n")

        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract_verdict, f, indent=2, sort_keys=True)
            f.write("\n")

        summary_written = write_stage_summary(stage_dir, summary)

        artifacts = {
            "real_v0_events_list": events_list_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        }

        for entry in events_list:
            event_id = entry["event_id"]
            strain_path = outputs_dir / entry["strain_npz"]
            if strain_path.exists():
                artifacts[f"strain_{event_id}"] = strain_path

        write_manifest(stage_dir, artifacts, extra={"version": "1"})

    if overall_verdict != "PASS":
        abort_contract(
            f"ringdown_real_v0_stage FAIL: n_valid={n_valid}, n_invalid={n_invalid}"
        )

    print(f"OK: ringdown_real_v0_stage PASS ({n_valid} events)")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
