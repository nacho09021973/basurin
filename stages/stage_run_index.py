#!/usr/bin/env python3
"""BASURIN — Derived canonical stage: RUN_INDEX."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _extract_verdict(summary_path: Path) -> str | None:
    payload = _read_json(summary_path)
    if isinstance(payload.get("verdict"), str):
        return payload["verdict"]
    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get("verdict"), str):
        return results["verdict"]
    return None


def _read_run_valid_verdict(run_dir: Path) -> tuple[str, Path | None]:
    verdict_path = run_dir / "RUN_VALID" / "verdict.json"
    output_path = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"

    if verdict_path.exists():
        verdict_payload = _read_json(verdict_path)
        verdict = verdict_payload.get("verdict")
        if isinstance(verdict, str) and verdict:
            return verdict, verdict_path

    if output_path.exists():
        output_payload = _read_json(output_path)
        verdict = output_payload.get("overall_verdict")
        if not (isinstance(verdict, str) and verdict):
            verdict = output_payload.get("verdict")
        if isinstance(verdict, str) and verdict:
            return verdict, output_path

    return "MISSING", None


def _collect_entries(run_dir: Path) -> list[dict[str, Any]]:
    def _entry(name: str, kind: str, summary_path: Path, manifest_path: Path) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "stage_summary": {
                "path": str(summary_path.relative_to(run_dir)),
                "sha256": sha256_file(summary_path),
            },
            "manifest": {
                "path": str(manifest_path.relative_to(run_dir)),
                "sha256": sha256_file(manifest_path),
            },
        }
        verdict = _extract_verdict(summary_path)
        if verdict is not None:
            entry["verdict"] = verdict
        return entry

    entries: list[dict[str, Any]] = []
    for stage_dir in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if stage_dir.is_dir() and stage_dir.name not in {"RUN_INDEX", "experiment"}:
            summary_path = stage_dir / "stage_summary.json"
            manifest_path = stage_dir / "manifest.json"
            if summary_path.exists() and manifest_path.exists():
                entries.append(_entry(stage_dir.name, "stage", summary_path, manifest_path))

    exp_root = run_dir / "experiment"
    if exp_root.exists():
        for exp_dir in sorted(exp_root.iterdir(), key=lambda p: p.name):
            if exp_dir.is_dir():
                summary_path = exp_dir / "stage_summary.json"
                manifest_path = exp_dir / "manifest.json"
                if summary_path.exists() and manifest_path.exists():
                    entries.append(_entry(exp_dir.name, "experiment", summary_path, manifest_path))

    return sorted(entries, key=lambda item: (item["kind"], item["name"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derived canonical stage RUN_INDEX")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--runs-root", default="runs", help="Runs root path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.run

    verdict, run_valid_path = _read_run_valid_verdict(run_dir)
    if verdict != "PASS":
        print(f"ERROR: RUN_VALID verdict {verdict}", file=sys.stderr)
        return 2

    stage_dir = run_dir / "RUN_INDEX"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    assert run_valid_path is not None
    run_valid_rel = str(run_valid_path.relative_to(run_dir))
    run_valid_sha = sha256_file(run_valid_path)

    index_payload = {
        "schema_version": "1.0.0",
        "stage": "RUN_INDEX",
        "run": args.run,
        "run_valid": {
            "path": run_valid_rel,
            "sha256": run_valid_sha,
            "verdict": verdict,
        },
        "entries": _collect_entries(run_dir),
    }

    index_path = outputs_dir / "index.json"
    _write_json(index_path, index_payload)
    index_sha = sha256_file(index_path)

    created_utc = datetime.now(timezone.utc).isoformat()
    stage_summary_path = stage_dir / "stage_summary.json"
    stage_summary = {
        "stage": "RUN_INDEX",
        "run": args.run,
        "created_utc": created_utc,
        "inputs": {
            "run_valid": {
                "path": run_valid_rel,
                "sha256": run_valid_sha,
                "verdict": verdict,
            }
        },
        "outputs": {
            "index": "outputs/index.json",
        },
        "hashes": {
            "outputs/index.json": index_sha,
        },
        "verdict": "PASS",
    }
    _write_json(stage_summary_path, stage_summary)
    stage_summary_sha = sha256_file(stage_summary_path)

    manifest_path = stage_dir / "manifest.json"
    manifest = {
        "stage": "RUN_INDEX",
        "run": args.run,
        "created_utc": created_utc,
        "files": {
            "index": "outputs/index.json",
            "summary": "stage_summary.json",
            "manifest": "manifest.json",
        },
        "hashes": {
            "outputs/index.json": index_sha,
            "stage_summary.json": stage_summary_sha,
        },
    }
    _write_json(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Manual test:
# RUN="2026-02-06__REPO_INDEX_V0"
# python experiment/run_valid/stage_run_valid.py --run "$RUN"
# python stages/stage_run_index.py --run "$RUN"
# ls runs/$RUN/RUN_INDEX/outputs
