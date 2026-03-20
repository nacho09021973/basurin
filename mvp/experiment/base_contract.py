#!/usr/bin/env python3
"""Universal entry contract for all E5 experimental alternatives.

Every E5 alternative must call ``assert_run_valid`` before consuming any
artifact from a run.  The canonical gates are declared here so that all
experiments share a single source of truth for input locations.

Governance rule: no experiment may receive a ``run_id`` whose
``stage_summary.json`` does not have ``run_valid: "PASS"``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break


# ── Canonical gate paths (relative to run_dir) ─────────────────────────────

REQUIRED_CANONICAL_GATES = {
    "run_valid": "RUN_VALID/verdict.json",
    "compatible_set": "s4_geometry_filter/outputs/compatible_set.json",
    "stage_summary": "s4_geometry_filter/stage_summary.json",
    "verdict": "verdict.json",
    "estimates": "s3b_multimode_estimates/estimates.json",
}


class GovernanceViolation(Exception):
    """Raised when a governance invariant is broken."""


def load_json(path: str | Path) -> Any:
    """Load JSON from *path* or raise with a clear message."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required artifact missing: {p}")
    with open(p) as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    """Return hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_run_valid(run_valid_path: str | Path) -> dict:
    """Abort if governance says the run is not PASS.

    Canonical path: ``RUN_VALID/verdict.json``.
    Backward-compatible fallback for minimal test fixtures:
    ``run_dir/stage_summary.json`` with ``{"run_valid": "PASS"}``.
    """
    p = Path(run_valid_path)

    if p.exists():
        verdict = load_json(p)
        status = verdict.get("verdict")
        if status != "PASS":
            raise GovernanceViolation(f"RUN_VALID={status} in {p}")
        return verdict

    fallback = p.parent.parent / "stage_summary.json"
    if fallback.exists():
        summary = load_json(fallback)
        status = summary.get("run_valid")
        if status != "PASS":
            raise GovernanceViolation(f"RUN_VALID={status} in {fallback}")
        return {"verdict": status, "source": str(fallback), "compat_fallback": True}

    raise FileNotFoundError(f"Required artifact missing: {p}")


def resolve_run_dir(run_id: str, runs_root: str | Path | None = None) -> Path:
    """Resolve run directory from run_id.  Respects BASURIN_RUNS_ROOT."""
    import os

    if runs_root:
        root = Path(runs_root)
    else:
        env = os.environ.get("BASURIN_RUNS_ROOT")
        root = Path(env) if env else Path.cwd() / "runs"
    return root / run_id


def validate_and_load_run(run_id: str, runs_root: str | Path | None = None) -> tuple[Path, dict]:
    """Validate a run and return ``(run_dir, stage_summary)``.

    Raises GovernanceViolation if run is not PASS.
    Prefers canonical s4_geometry_filter/stage_summary.json and falls back
    to run_dir/stage_summary.json for minimal legacy fixtures.
    """
    run_dir = resolve_run_dir(run_id, runs_root)
    run_valid_path = run_dir / REQUIRED_CANONICAL_GATES["run_valid"]
    assert_run_valid(run_valid_path)

    summary_path = run_dir / REQUIRED_CANONICAL_GATES["stage_summary"]
    if summary_path.exists():
        summary = load_json(summary_path)
    else:
        fallback_summary_path = run_dir / "stage_summary.json"
        if fallback_summary_path.exists():
            summary = load_json(fallback_summary_path)
        else:
            raise FileNotFoundError(f"Required artifact missing: {summary_path}")

    return run_dir, summary


def validate_multiple_runs(
    run_ids: list[str], runs_root: str | Path | None = None
) -> list[tuple[Path, dict]]:
    """Validate N runs, all must be PASS.  Returns list of (run_dir, summary)."""
    results = []
    for rid in run_ids:
        results.append(validate_and_load_run(rid, runs_root))
    return results


def ensure_experiment_dir(run_dir: Path, experiment_name: str) -> Path:
    """Create and return the experiment output directory under a run."""
    exp_dir = run_dir / "experiment" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def write_manifest(output_dir: Path, input_hashes: dict[str, str], extra: dict | None = None) -> Path:
    """Write a manifest.json with SHA-256 hashes of all consumed inputs."""
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


def _write_json_atomic(path: Path, data: Any) -> None:
    """Atomic JSON write: tmp + rename."""
    import tempfile

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
