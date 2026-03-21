"""BRUNETE experiment base contract.

Bridges BRUNETE classify runs to the per-event BASURIN canonical subruns
stored under runs/<batch_run_id>/run_batch/event_runs/<event_run_id>/.

Key functions
-------------
validate_classify_run(classify_run_id, runs_root)
    Validate a classify_geometries run and return its geometry_summary payload.

enumerate_event_runs(classify_run_id, mode, runs_root)
    Return {event_id: event_run_dir} for a given mode ("220" or "221").
    Each event_run_dir is a full BASURIN run with the canonical gate layout.

validate_event_run(event_run_dir)
    Validate RUN_VALID of a per-event BASURIN subrun.

All E5 utilities (load_json, sha256_file, ensure_experiment_dir, etc.) are
re-exported here so B5 modules only need a single import.

Governance
----------
- No module writes outside runs/<classify_run_id>/experiment/<name>/.
- No module executes against a classify run whose RUN_VALID verdict is not PASS.
- Per-event subruns that lack RUN_VALID=PASS are skipped with a warning, not
  raised as fatal errors (the batch may have had partial failures).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

# ── Re-export the low-level utilities from BASURIN base_contract ────────────
_here = Path(__file__).resolve()
_repo_root = _here.parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class GovernanceViolation(Exception):
    """Raised when a governance invariant is broken."""


# ── JSON helpers ─────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required artifact missing: {p}")
    with open(p) as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def ensure_experiment_dir(classify_run_dir: Path, experiment_name: str) -> Path:
    exp_dir = classify_run_dir / "experiment" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def write_manifest(output_dir: Path, input_hashes: dict[str, str], extra: dict | None = None) -> Path:
    from datetime import datetime, timezone
    manifest = {
        "input_hashes": input_hashes,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    out = output_dir / "manifest.json"
    _write_json_atomic(out, manifest)
    return out


# ── BRUNETE canonical gate paths ─────────────────────────────────────────────

# Gates on the classify_geometries run dir
BRUNETE_CLASSIFY_GATES = {
    "run_valid":       "classify_geometries/RUN_VALID/verdict.json",
    "stage_summary":   "classify_geometries/stage_summary.json",
    "geometry_summary": "classify_geometries/outputs/geometry_summary.json",
}

# Gates on each per-event BASURIN subrun dir (same layout as BASURIN core)
EVENT_RUN_GATES = {
    "run_valid":      "RUN_VALID/verdict.json",
    "compatible_set": "s4_geometry_filter/outputs/compatible_set.json",
    "stage_summary":  "s4_geometry_filter/stage_summary.json",
    "verdict":        "verdict.json",
    "estimates":      "s3b_multimode_estimates/estimates.json",
}


# ── Run resolution ────────────────────────────────────────────────────────────

def _resolve_runs_root(runs_root: str | Path | None = None) -> Path:
    if runs_root:
        return Path(runs_root)
    env = os.environ.get("BASURIN_RUNS_ROOT")
    return Path(env) if env else Path.cwd() / "runs"


def resolve_classify_run_dir(classify_run_id: str, runs_root: str | Path | None = None) -> Path:
    return _resolve_runs_root(runs_root) / classify_run_id


def validate_classify_run(
    classify_run_id: str,
    runs_root: str | Path | None = None,
) -> tuple[Path, dict]:
    """Validate a classify_geometries run.

    Returns (classify_run_dir, geometry_summary_payload).
    Raises GovernanceViolation if RUN_VALID is not PASS.
    """
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)

    rv_path = run_dir / BRUNETE_CLASSIFY_GATES["run_valid"]
    if not rv_path.exists():
        raise FileNotFoundError(f"classify RUN_VALID verdict missing: {rv_path}")
    rv = load_json(rv_path)
    if rv.get("verdict") != "PASS":
        raise GovernanceViolation(
            f"classify run {classify_run_id!r} RUN_VALID={rv.get('verdict')!r}"
        )

    gs_path = run_dir / BRUNETE_CLASSIFY_GATES["geometry_summary"]
    if not gs_path.exists():
        raise FileNotFoundError(f"geometry_summary.json missing: {gs_path}")
    geometry_summary = load_json(gs_path)

    return run_dir, geometry_summary


def enumerate_event_runs(
    classify_run_id: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
    *,
    skip_invalid: bool = True,
) -> dict[str, Path]:
    """Return {event_id: event_run_dir} for the given mode.

    Navigates from classify_geometries/geometry_summary.json →
    batch_<mode>_run_id → run_batch/event_runs/<event_run_id>/.

    Events whose per-event RUN_VALID is not PASS are skipped with a warning
    when skip_invalid=True (default).  Set skip_invalid=False to raise instead.
    """
    if mode not in ("220", "221"):
        raise ValueError(f"mode must be '220' or '221', got {mode!r}")

    run_dir, geometry_summary = validate_classify_run(classify_run_id, runs_root)
    runs_root_path = _resolve_runs_root(runs_root)

    batch_run_id = geometry_summary.get(f"batch_{mode}_run_id")
    if not batch_run_id:
        raise KeyError(
            f"geometry_summary missing 'batch_{mode}_run_id' in {classify_run_id!r}"
        )

    batch_event_runs_root = (
        runs_root_path / batch_run_id / "run_batch" / "event_runs"
    )

    event_run_id_key = f"event_run_id_{mode}"
    result: dict[str, Path] = {}

    for row in geometry_summary.get("rows", []):
        event_id = row.get("event_id")
        event_run_id = row.get(event_run_id_key)
        if not event_id or not event_run_id:
            continue

        event_run_dir = batch_event_runs_root / event_run_id
        if not event_run_dir.exists():
            warnings.warn(
                f"event_run_dir not found for {event_id!r} (mode {mode}): {event_run_dir}",
                stacklevel=2,
            )
            continue

        rv_path = event_run_dir / EVENT_RUN_GATES["run_valid"]
        if rv_path.exists():
            rv = load_json(rv_path)
            if rv.get("verdict") != "PASS":
                msg = (
                    f"event {event_id!r} mode {mode} RUN_VALID="
                    f"{rv.get('verdict')!r} — skipping"
                )
                if skip_invalid:
                    warnings.warn(msg, stacklevel=2)
                    continue
                else:
                    raise GovernanceViolation(msg)

        result[event_id] = event_run_dir

    return result


def validate_event_run(event_run_dir: Path) -> dict:
    """Validate a per-event BASURIN subrun and return its stage_summary."""
    rv_path = event_run_dir / EVENT_RUN_GATES["run_valid"]
    if not rv_path.exists():
        raise FileNotFoundError(f"event RUN_VALID verdict missing: {rv_path}")
    rv = load_json(rv_path)
    if rv.get("verdict") != "PASS":
        raise GovernanceViolation(f"event RUN_VALID={rv.get('verdict')!r} at {event_run_dir}")

    ss_path = event_run_dir / EVENT_RUN_GATES["stage_summary"]
    if ss_path.exists():
        return load_json(ss_path)
    fallback = event_run_dir / "stage_summary.json"
    if fallback.exists():
        return load_json(fallback)
    raise FileNotFoundError(f"stage_summary.json missing: {ss_path}")
