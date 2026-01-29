#!/usr/bin/env python3
"""
Stage BASURIN (core): RUN_VALID

Contrato soberano: un run "existe" solo si este stage emite PASS.

Outputs:
  runs/<run_id>/RUN_VALID/
    - manifest.json
    - stage_summary.json
    - outputs/run_valid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root en sys.path (patrón usado en otros stages)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RUN_VALID: sovereign run validity gate")
    p.add_argument("--run", required=True, help="run_id under runs/<run_id>")
    p.add_argument("--out-root", default="runs", help="Root directory for runs (default: runs)")
    # opcional: lista de paths canónicos mínimos que deben existir para declarar PASS
    p.add_argument(
        "--require",
        action="append",
        default=[],
        help="Relative path under runs/<run_id>/ that must exist (repeatable).",
    )
    return p.parse_args()


def _check_required_paths(run_root: Path, required: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    checks: List[Dict[str, Any]] = []
    ok = True
    for rel in required:
        p = (run_root / rel).resolve()
        exists = p.exists()
        checks.append({"path": str(Path(rel)), "exists": exists})
        if not exists:
            ok = False
    return ok, checks


def main() -> int:
    args = parse_args()

    try:
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except Exception as exc:
        print(f"[BASURIN ABORT] invalid run: {exc}", file=sys.stderr)
        return 1

    stage_dir, outputs_dir = ensure_stage_dirs(out_root, args.run, "RUN_VALID")
    run_root = out_root / args.run

    # Checks mínimos (baseline): el run root debe existir
    base_required = ["."]
    # + cualquier requisito explícito pasado por CLI
    required = base_required + list(args.require)

    ok, checks = _check_required_paths(run_root, required)

    overall = "PASS" if ok else "FAIL"

    out_path = outputs_dir / "run_valid.json"
    payload = {
        "schema_version": "run_valid_v1",
        "run_id": args.run,
        "overall_verdict": overall,
        "timestamp": utc_now_iso(),
        "required_paths": required,
        "checks": checks,
        "stage_version": __version__,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # stage_summary + manifest (hashes)
    inputs = {
        "out_root": str(out_root),
        "required_paths": required,
    }
    results = {"overall_verdict": overall}

    summary = {
        "schema_version": "stage_summary_v1",
        "stage": "RUN_VALID",
        "run_id": args.run,
        "stage_version": __version__,
        "timestamp": utc_now_iso(),
        "inputs": inputs,
        "parameters": {},
        "results": results,
    }
    write_stage_summary(stage_dir, summary)

    files = {"run_valid": str(Path("outputs") / "run_valid.json")}
    hashes = {files["run_valid"]: sha256_file(out_path)}
    write_manifest(stage_dir=stage_dir, files=files, hashes=hashes)

    print(overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
