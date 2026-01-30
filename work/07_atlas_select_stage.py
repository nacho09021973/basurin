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
    utc_now_iso,
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


def _resolve_master_path(path_str: str, out_root: Path) -> tuple[Path, str]:
    if Path(path_str).is_absolute():
        candidate = Path(path_str).resolve()
        display_path = _display_path(candidate)
    elif path_str.startswith("runs/"):
        candidate = (Path.cwd() / path_str).resolve()
        display_path = path_str
    else:
        candidate = (out_root / path_str).resolve()
        display_path = _display_path(candidate)
    try:
        candidate.relative_to(out_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"atlas_master_path sale de out_root ({out_root}): {path_str}"
        ) from exc
    return candidate, display_path


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
    parser.add_argument(
        "--force-dim",
        type=int,
        help="Forced dimension to select from ATLAS_INDEX (4 or 6).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.force_dim not in (4, 6):
        print(
            "ATLAS_INDEX solo cataloga masters (dim4/dim6). Usa --force-dim 4|6.",
            file=sys.stderr,
        )
        return 2

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

    if not isinstance(atlas_index, dict):
        print(f"ERROR: ATLAS_INDEX inválido en {atlas_index_path}", file=sys.stderr)
        return 1

    items = atlas_index.get("items")
    if not isinstance(items, list):
        print(f"ERROR: ATLAS_INDEX sin items en {atlas_index_path}", file=sys.stderr)
        return 1

    dim = int(args.force_dim)
    target_name = f"dim{dim}"
    atlas_item = None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("name") == target_name:
            atlas_item = item
            break

    if atlas_item is None:
        print(
            f"ERROR: ATLAS_INDEX sin item '{target_name}' en {atlas_index_path}",
            file=sys.stderr,
        )
        return 2

    atlas_path_str = atlas_item.get("path")
    atlas_sha_expected = atlas_item.get("sha256")
    if not atlas_path_str or not isinstance(atlas_path_str, str):
        print(
            f"ERROR: ATLAS_INDEX item '{target_name}' sin path en {atlas_index_path}",
            file=sys.stderr,
        )
        return 2
    if not atlas_sha_expected or not isinstance(atlas_sha_expected, str):
        print(
            f"ERROR: ATLAS_INDEX item '{target_name}' sin sha256 en {atlas_index_path}",
            file=sys.stderr,
        )
        return 2

    try:
        atlas_path, atlas_path_display = _resolve_master_path(atlas_path_str, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not atlas_path.exists():
        print(f"ERROR: atlas master no existe en {atlas_path}", file=sys.stderr)
        return 2

    atlas_sha_computed = sha256_file(atlas_path)
    if atlas_sha_computed != atlas_sha_expected:
        print(
            "ERROR: sha256 mismatch para atlas master "
            f"(index={atlas_sha_expected}, computed={atlas_sha_computed})",
            file=sys.stderr,
        )
        return 2

    selection: dict[str, Any] = {
        "schema_version": "atlas_selection_v1",
        "stage": "atlas_select",
        "version": __version__,
        "run": args.run,
        "created": utc_now_iso(),
        "index_source": {
            "path": _display_path(atlas_index_path),
            "sha256": sha256_file(atlas_index_path),
        },
        "selected": {
            "dim": dim,
            "atlas_master_path": atlas_path_display,
            "atlas_master_sha256": atlas_sha_expected,
            "atlas_master_sha256_computed": atlas_sha_computed,
        },
        "inputs": {
            "RUN_VALID": {
                "path": _display_path(run_valid_path),
                "sha256": sha256_file(run_valid_path),
                "verdict": verdict,
            }
        },
        "decision": {"rule": "forced"},
    }

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
        "inputs": selection["inputs"],
        "index_source": selection["index_source"],
        "outputs": {
            "atlas_selection": "outputs/ATLAS_SELECTION.json",
        },
        "config": {
            "out_root_requested": args.out_root,
            "out_root_effective": _display_path(out_root),
            "force_dim": dim,
        },
        "data": selection["selected"],
        "decision": selection["decision"],
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
