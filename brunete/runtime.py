"""Shared runtime helpers for the BRUNETE public facade.

BRUNETE exposes a small public interface on top of the current BASURIN
checkout. Its public stage contracts live under ``runs/<run_id>/<stage>/...``
and share the same operational metadata:

- ``RUN_VALID/verdict.json``
- ``stage_summary.json``
- ``manifest.json``
- ``outputs/...``
- ``external_inputs/...``
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from basurin_io import (
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

RUN_VALID_SCHEMA_VERSION = "brunete_run_valid_v1"
STAGE_SUMMARY_SCHEMA_VERSION = "brunete_stage_summary_v1"
MANIFEST_SCHEMA_VERSION = "brunete_manifest_v1"
IMPLEMENTATION_STATE = "brunete_public_facade_v1"


@dataclass(frozen=True)
class BruneteContext:
    run_id: str
    stage_name: str
    out_root: Path
    run_dir: Path
    stage_dir: Path
    outputs_dir: Path
    external_inputs_dir: Path
    run_valid_path: Path
    cli_args: dict[str, Any]


def init_stage(run_id: str, stage_name: str, cli_args: dict[str, Any]) -> BruneteContext:
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)

    run_dir = out_root / run_id
    stage_dir = run_dir / stage_name
    outputs_dir = stage_dir / "outputs"
    external_inputs_dir = stage_dir / "external_inputs"
    run_valid_path = stage_dir / "RUN_VALID" / "verdict.json"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    external_inputs_dir.mkdir(parents=True, exist_ok=True)
    return BruneteContext(
        run_id=run_id,
        stage_name=stage_name,
        out_root=out_root,
        run_dir=run_dir,
        stage_dir=stage_dir,
        outputs_dir=outputs_dir,
        external_inputs_dir=external_inputs_dir,
        run_valid_path=run_valid_path,
        cli_args=cli_args,
    )


def write_json_output(ctx: BruneteContext, relative_path: str, payload: Any) -> Path:
    return write_json_atomic(ctx.outputs_dir / relative_path, payload)


def write_text_output(ctx: BruneteContext, relative_path: str, text: str) -> Path:
    path = ctx.outputs_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def copy_input_file(ctx: BruneteContext, source: Path, relative_path: str) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"input not found: {source}")
    destination = ctx.external_inputs_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def finalize_stage(
    ctx: BruneteContext,
    *,
    verdict: str,
    reason: str,
    results: dict[str, Any],
    artifacts: dict[str, Path],
    error: str | None = None,
    notes: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    run_valid_path = write_run_valid(
        ctx,
        verdict=verdict,
        reason=reason,
    )

    summary_artifacts = dict(artifacts)
    summary_artifacts["run_valid_verdict"] = run_valid_path

    summary_payload = {
        "schema_version": STAGE_SUMMARY_SCHEMA_VERSION,
        "created": utc_now_iso(),
        "stage": ctx.stage_name,
        "run_id": ctx.run_id,
        "verdict": verdict,
        "reason": reason,
        "implementation_state": IMPLEMENTATION_STATE,
        "parameters": ctx.cli_args,
        "results": results,
        "artifacts": artifact_rows(ctx.stage_dir, summary_artifacts),
        "notes": notes or [],
    }
    if error is not None:
        summary_payload["error"] = error
    summary_path = write_stage_summary(ctx.stage_dir, summary_payload)

    manifest_artifacts = dict(summary_artifacts)
    manifest_artifacts["stage_summary"] = summary_path
    manifest_path = write_manifest(
        ctx.stage_dir,
        manifest_artifacts,
        extra={
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "implementation_state": IMPLEMENTATION_STATE,
            "stage": ctx.stage_name,
            "run_id": ctx.run_id,
            "verdict": verdict,
            "parameters": ctx.cli_args,
        },
    )
    return run_valid_path, summary_path, manifest_path


def write_run_valid(
    ctx: BruneteContext,
    *,
    verdict: str,
    reason: str,
) -> Path:
    return write_json_atomic(
        ctx.run_valid_path,
        {
            "schema_version": RUN_VALID_SCHEMA_VERSION,
            "created": utc_now_iso(),
            "stage": ctx.stage_name,
            "run_id": ctx.run_id,
            "verdict": verdict,
            "reason": reason,
        },
    )


def artifact_rows(stage_dir: Path, artifacts: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for label, path in sorted(artifacts.items()):
        resolved = Path(path).resolve()
        rows.append(
            {
                "label": label,
                "path": _relative_to_stage(stage_dir, resolved),
                "sha256": sha256_file(resolved) if resolved.exists() and resolved.is_file() else None,
            }
        )
    return rows


def load_event_ids_from_text(path: Path) -> list[str]:
    events: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        event_id = raw.strip()
        if not event_id or event_id.startswith("#") or event_id in seen:
            continue
        seen.add(event_id)
        events.append(event_id)
    return events


def load_event_ids_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events: list[str] = []
    seen: set[str] = set()

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict) and key not in seen:
                seen.add(key)
                events.append(key)
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            candidate = item.get("event_id") or item.get("name") or item.get("id")
            if isinstance(candidate, str):
                event_id = candidate.strip()
                if event_id and event_id not in seen:
                    seen.add(event_id)
                    events.append(event_id)
    else:
        raise ValueError(f"unsupported JSON schema for event extraction: {type(payload).__name__}")

    return events


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _relative_to_stage(stage_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(stage_dir))
    except ValueError:
        return str(path)
