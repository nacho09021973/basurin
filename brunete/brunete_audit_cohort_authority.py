#!/usr/bin/env python3
"""Audit whether a BRUNETE cohort has a single authoritative source in-repo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brunete.runtime import copy_input_file, finalize_stage, init_stage, load_event_ids_from_text, write_json_output

STAGE_NAME = "audit_cohort_authority"
REGISTRY_PATH = _REPO_ROOT / "brunete" / "cohorts" / "authority_registry.json"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BRUNETE: emit a PASS/FAIL verdict on whether a cohort has a unique authoritative source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cohort-key", required=True, help="Registry key to audit, e.g. support_multi or visible_losc_events.")
    ap.add_argument(
        "--registry-path",
        default=str(REGISTRY_PATH),
        help="Authority registry JSON. Primarily for tests; defaults to the repo registry.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry_path = Path(args.registry_path).expanduser().resolve()
    ctx = init_stage(
        args.run_id,
        STAGE_NAME,
        {
            "cohort_key": args.cohort_key,
            "registry_path": str(registry_path),
        },
    )
    artifacts: dict[str, Path] = {}

    try:
        registry = _load_registry(registry_path)
        if args.cohort_key not in registry["cohorts"]:
            raise KeyError(f"cohort_key not found in authority registry: {args.cohort_key}")

        entry = registry["cohorts"][args.cohort_key]
        authority_status = entry["authority_status"]
        authority_kind = entry["authority_kind"]

        results = {
            "cohort_key": args.cohort_key,
            "authority_status": authority_status,
            "authority_kind": authority_kind,
            "description": entry.get("description"),
            "reason": entry.get("reason"),
            "source_path": None,
            "materializers": entry.get("materializers", []),
            "n_events": None,
        }

        if authority_status == "PASS":
            source_path = (_REPO_ROOT / entry["path"]).resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"registry path not found for PASS cohort {args.cohort_key}: {source_path}")
            copied = copy_input_file(ctx, source_path, "authoritative_source.txt")
            artifacts["authoritative_source_txt"] = copied
            events = load_event_ids_from_text(source_path)
            results["source_path"] = str(source_path)
            results["n_events"] = len(events)
            results["event_id_preview"] = events[:10]
            reason = f"unique authoritative source exists for cohort {args.cohort_key}"
            verdict = "PASS"
        else:
            results["event_id_preview"] = []
            reason = entry.get("reason") or f"no unique authoritative source exists for cohort {args.cohort_key}"
            verdict = "FAIL"

        report_path = write_json_output(
            ctx,
            "authority_report.json",
            {
                "schema_version": registry["schema_version"],
                "stage": STAGE_NAME,
                "run_id": args.run_id,
                **results,
            },
        )
        artifacts["authority_report_json"] = report_path

        _, summary_path, manifest_path = finalize_stage(
            ctx,
            verdict=verdict,
            reason=reason,
            results=results,
            artifacts=artifacts,
            error=reason if verdict == "FAIL" else None,
        )
        _log_stage_paths(ctx, summary_path, manifest_path)
        return 0 if verdict == "PASS" else 2

    except Exception as exc:
        report_path = write_json_output(
            ctx,
            "authority_report.json",
            {
                "schema_version": "brunete_cohort_authority_v1",
                "stage": STAGE_NAME,
                "run_id": args.run_id,
                "cohort_key": args.cohort_key,
                "authority_status": "FAIL",
                "authority_kind": "error",
                "description": None,
                "reason": str(exc),
                "source_path": None,
                "materializers": [],
                "n_events": None,
                "event_id_preview": [],
            },
        )
        artifacts["authority_report_json"] = report_path
        _, summary_path, manifest_path = finalize_stage(
            ctx,
            verdict="FAIL",
            reason=str(exc),
            results={
                "cohort_key": args.cohort_key,
                "authority_status": "FAIL",
                "authority_kind": "error",
                "description": None,
                "reason": str(exc),
                "source_path": None,
                "materializers": [],
                "n_events": None,
                "event_id_preview": [],
            },
            artifacts=artifacts,
            error=str(exc),
        )
        _log_stage_paths(ctx, summary_path, manifest_path)
        print(f"ERROR [{STAGE_NAME}]: {exc}", file=sys.stderr)
        return 2


def _load_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"authority registry not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "cohorts" not in payload:
        raise ValueError(f"invalid authority registry schema: {path}")
    return payload


def _log_stage_paths(ctx, stage_summary_path: Path, manifest_path: Path) -> None:
    print(f"OUT_ROOT={ctx.out_root}")
    print(f"STAGE_DIR={ctx.stage_dir}")
    print(f"OUTPUTS_DIR={ctx.outputs_dir}")
    print(f"STAGE_SUMMARY={stage_summary_path}")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
