from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from basurin_io import resolve_out_root


MODE_TO_BATCH_KEY = {
    "220": "batch_220_run_id",
    "221": "batch_221_run_id",
}

MODE_TO_EVENT_RUN_KEY = {
    "220": "event_run_id_220",
    "221": "event_run_id_221",
}


class BruneteGovernanceViolation(Exception):
    """Raised when a BRUNETE classify run or event run breaks governance."""


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required artifact missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_runs_root(runs_root: str | Path | None = None) -> Path:
    return Path(runs_root) if runs_root else resolve_out_root("runs")


def resolve_classify_stage_dir(classify_run_id: str, runs_root: str | Path | None = None) -> Path:
    return resolve_runs_root(runs_root) / classify_run_id / "classify_geometries"


def assert_stage_pass(stage_dir: str | Path) -> dict[str, Any]:
    verdict_path = Path(stage_dir) / "RUN_VALID" / "verdict.json"
    verdict = load_json(verdict_path)
    if verdict.get("verdict") != "PASS":
        raise BruneteGovernanceViolation(
            f"RUN_VALID={verdict.get('verdict')} in {verdict_path}"
        )
    return verdict


def load_geometry_summary(classify_run_id: str, runs_root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    stage_dir = resolve_classify_stage_dir(classify_run_id, runs_root)
    assert_stage_pass(stage_dir)
    summary_path = stage_dir / "outputs" / "geometry_summary.json"
    payload = load_json(summary_path)
    return stage_dir, payload


def get_batch_run_id(classify_run_id: str, mode: str, runs_root: str | Path | None = None) -> str:
    if mode not in MODE_TO_BATCH_KEY:
        raise ValueError(f"Unsupported mode: {mode!r}")
    _stage_dir, summary = load_geometry_summary(classify_run_id, runs_root)
    value = summary.get(MODE_TO_BATCH_KEY[mode])
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"geometry_summary.json missing {MODE_TO_BATCH_KEY[mode]!r} for classify_run_id={classify_run_id}"
        )
    return value


def resolve_event_run_dirs(
    classify_run_id: str,
    mode: str,
    runs_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    if mode not in MODE_TO_EVENT_RUN_KEY:
        raise ValueError(f"Unsupported mode: {mode!r}")

    runs_root_path = resolve_runs_root(runs_root)
    _stage_dir, summary = load_geometry_summary(classify_run_id, runs_root_path)
    batch_run_id = get_batch_run_id(classify_run_id, mode, runs_root_path)
    event_key = MODE_TO_EVENT_RUN_KEY[mode]

    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise ValueError("geometry_summary.json must contain a list at key 'rows'")

    resolved: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_run_id = row.get(event_key)
        if not isinstance(event_run_id, str) or not event_run_id.strip():
            continue
        event_id = row.get("event_id", event_run_id)
        event_stage_dir = runs_root_path / batch_run_id / "run_batch" / "event_runs" / event_run_id
        assert_stage_pass(event_stage_dir)
        resolved.append(
            {
                "event_id": event_id,
                "event_run_id": event_run_id,
                "batch_run_id": batch_run_id,
                "run_dir": event_stage_dir,
            }
        )

    if not resolved:
        raise ValueError(
            f"No valid event_run_id entries found for classify_run_id={classify_run_id!r} mode={mode!r}"
        )
    return resolved


def resolve_event_run_ids(classify_run_id: str, mode: str, runs_root: str | Path | None = None) -> list[str]:
    return [row["event_run_id"] for row in resolve_event_run_dirs(classify_run_id, mode, runs_root)]


def resolve_all_event_run_ids(classify_run_id: str, runs_root: str | Path | None = None) -> list[str]:
    run_ids: list[str] = []
    seen: set[str] = set()
    for mode in ("220", "221"):
        for run_id in resolve_event_run_ids(classify_run_id, mode, runs_root):
            if run_id not in seen:
                seen.add(run_id)
                run_ids.append(run_id)
    return run_ids


def ensure_experiment_dir(classify_run_id: str, experiment_name: str, runs_root: str | Path | None = None) -> Path:
    stage_dir = resolve_classify_stage_dir(classify_run_id, runs_root)
    out_dir = stage_dir / "experiment" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def materialize_event_run_view(
    classify_run_id: str,
    mode: str | None = None,
    runs_root: str | Path | None = None,
) -> tuple[Path, list[str]]:
    """Create a deterministic local view where event runs appear as top-level runs."""
    view_root = resolve_classify_stage_dir(classify_run_id, runs_root) / "experiment" / "_resolved_runs"
    view_root.mkdir(parents=True, exist_ok=True)

    run_ids = resolve_all_event_run_ids(classify_run_id, runs_root) if mode is None else resolve_event_run_ids(classify_run_id, mode, runs_root)
    lookup = {}
    for current_mode in ((mode,) if mode is not None else ("220", "221")):
        for row in resolve_event_run_dirs(classify_run_id, current_mode, runs_root):
            lookup[row["event_run_id"]] = row["run_dir"]

    for run_id in run_ids:
        link = view_root / run_id
        target = lookup[run_id]
        if link.exists() or link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        link.symlink_to(target.resolve(), target_is_directory=True)
    return view_root, run_ids
