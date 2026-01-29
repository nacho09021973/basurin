#!/usr/bin/env python3
"""Select atlas master based on ATLAS_INDEX for a given run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    ensure_stage_dirs,
    get_runs_root,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _resolve_run_valid_path(run_dir: Path) -> Path | None:
    preferred = (run_dir / "RUN_VALID" / "verdict.json").resolve()
    legacy = (run_dir / "RUN_VALID" / "outputs" / "run_valid.json").resolve()
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return None


def _parse_verdict(payload: Any, path: Path) -> str:
    verdict = None
    if isinstance(payload, dict):
        for key in ("overall_verdict", "verdict", "status"):
            if key in payload:
                verdict = payload.get(key)
                break
    if not isinstance(verdict, str):
        print(
            "ERROR: RUN_VALID sin veredicto string en "
            f"{path} (campos esperados: overall_verdict|verdict|status).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return verdict


def _resolve_atlas_index_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        return (Path.cwd() / path).resolve()
    return path.resolve()


def _coerce_dim(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _find_run_entry(index: Any, run_id: str) -> dict[str, Any] | None:
    if not isinstance(index, dict):
        return None
    runs = index.get("runs")
    if isinstance(runs, dict):
        entry = runs.get(run_id)
        if isinstance(entry, dict):
            return entry
    if isinstance(runs, list):
        for entry in runs:
            if not isinstance(entry, dict):
                continue
            if entry.get("run") == run_id or entry.get("run_id") == run_id:
                return entry
    entries = index.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("run") == run_id or entry.get("run_id") == run_id:
                return entry
    return None


def _coerce_atlas_entry(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, str):
        return {"path": entry}
    if isinstance(entry, dict):
        path = entry.get("path") or entry.get("atlas_master")
        if path:
            payload = {"path": path}
            if "sha256" in entry:
                payload["sha256"] = entry.get("sha256")
            return payload
    return None


def _find_atlas_entry(index: dict[str, Any], run_entry: dict[str, Any], dim: int) -> dict[str, Any] | None:
    for key in ("atlas_master", "atlas", "master"):
        if key in run_entry:
            payload = _coerce_atlas_entry(run_entry.get(key))
            if payload:
                return payload
    if "atlas_master_path" in run_entry and isinstance(run_entry["atlas_master_path"], str):
        return {"path": run_entry["atlas_master_path"]}

    for key in ("by_dim", "atlas_by_dim", "masters_by_dim"):
        by_dim = index.get(key)
        if isinstance(by_dim, dict):
            entry = None
            if dim in by_dim:
                entry = by_dim.get(dim)
            if entry is None:
                entry = by_dim.get(str(dim))
            payload = _coerce_atlas_entry(entry)
            if payload:
                return payload

    for key in ("items", "atlases"):
        items = index.get(key)
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_dim = _coerce_dim(item.get("dim") or item.get("dimension") or item.get("d"))
                if item_dim == dim:
                    payload = _coerce_atlas_entry(item)
                    if payload:
                        return payload
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select atlas master based on ATLAS_INDEX for a given run"
    )
    parser.add_argument("--run", required=True, help="run id under runs/<run>")
    parser.add_argument(
        "--atlas-index",
        required=True,
        help="Path to ATLAS_INDEX.json (relative to CWD or absolute)",
    )
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runs_root = get_runs_root()
        out_root = resolve_out_root(args.out_root, runs_root=runs_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = (out_root / args.run).resolve()
    run_valid_path = _resolve_run_valid_path(run_dir)
    if run_valid_path is None:
        preferred = run_dir / "RUN_VALID" / "verdict.json"
        legacy = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
        print(
            "ERROR: RUN_VALID missing; expected "
            f"{preferred} or {legacy}.",
            file=sys.stderr,
        )
        return 2

    try:
        run_valid_payload = json.loads(run_valid_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: RUN_VALID invalid JSON at {run_valid_path}: {exc}", file=sys.stderr)
        return 2

    try:
        verdict = _parse_verdict(run_valid_payload, run_valid_path)
    except SystemExit:
        return 2

    if verdict != "PASS":
        print(
            f"ERROR: RUN_VALID={verdict} at {run_valid_path} (expected PASS).",
            file=sys.stderr,
        )
        return 2

    atlas_index_path = _resolve_atlas_index_path(args.atlas_index)
    if not atlas_index_path.exists():
        print(f"ERROR: no existe ATLAS_INDEX en {atlas_index_path}", file=sys.stderr)
        return 1

    try:
        atlas_index = json.loads(atlas_index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: ATLAS_INDEX invalid JSON at {atlas_index_path}: {exc}", file=sys.stderr)
        return 1

    run_entry = _find_run_entry(atlas_index, args.run)
    if run_entry is None:
        print(
            f"ERROR: ATLAS_INDEX no tiene entrada para run '{args.run}' en {atlas_index_path}",
            file=sys.stderr,
        )
        return 1

    dim = _coerce_dim(run_entry.get("dim") or run_entry.get("dimension") or run_entry.get("d"))
    if dim is None:
        print(
            "ERROR: ATLAS_INDEX sin dim para run "
            f"'{args.run}' en {atlas_index_path}",
            file=sys.stderr,
        )
        return 1

    atlas_entry = _find_atlas_entry(atlas_index, run_entry, dim)
    if atlas_entry is None:
        print(
            f"ERROR: ATLAS_INDEX sin atlas para dim={dim} (run {args.run}) en {atlas_index_path}",
            file=sys.stderr,
        )
        return 1

    selection: dict[str, Any] = {
        "schema_version": "atlas_selection_v1",
        "run": args.run,
        "dim": int(dim),
        "atlas_master": atlas_entry,
        "atlas_index": {
            "path": str(atlas_index_path),
            "sha256": sha256_file(atlas_index_path),
        },
        "run_valid": {
            "path": str(run_valid_path),
            "verdict": verdict,
        },
    }

    atlas_path = None
    if atlas_entry.get("path"):
        atlas_path = Path(str(atlas_entry["path"]))
        if not atlas_path.is_absolute():
            atlas_path = (Path.cwd() / atlas_path).resolve()
        if atlas_path.exists():
            selection["atlas_master"]["sha256"] = sha256_file(atlas_path)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "atlas_select", base_dir=out_root)
    output_path = outputs_dir / "ATLAS_SELECTION.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, sort_keys=True)
        f.write("\n")

    output_hash = sha256_file(output_path)

    summary = {
        "stage": "atlas_select",
        "version": __version__,
        "script": "work/07_atlas_select_stage.py",
        "run": args.run,
        "inputs": {
            "atlas_index": _display_path(atlas_index_path),
            "run_valid": _display_path(run_valid_path),
        },
        "outputs": {
            "atlas_selection": "outputs/ATLAS_SELECTION.json",
        },
        "config": {
            "out_root_requested": args.out_root,
            "out_root_effective": _display_path(out_root),
        },
        "data": {
            "dim": int(dim),
        },
        "hashes": {
            "outputs/ATLAS_SELECTION.json": output_hash,
        },
    }

    summary_path = write_stage_summary(stage_dir, summary)

    write_manifest(
        stage_dir,
        {
            "atlas_selection": output_path,
            "summary": summary_path,
        },
        extra={
            "version": __version__,
            "inputs": {
                "atlas_index": sha256_file(atlas_index_path),
                "run_valid": sha256_file(run_valid_path),
            },
        },
    )

    print(f"ATLAS_SELECTION.json escrito en {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
