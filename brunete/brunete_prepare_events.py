#!/usr/bin/env python3
"""Prepare a local BRUNETE event cohort under ``runs/<run_id>/prepare_events``.

This public entrypoint is intentionally small and local-first:

- Input comes from either ``--events-file`` or ``--losc-root``.
- Output is deterministic and stays under ``runs/<run_id>/prepare_events``.
- The stage always writes ``manifest.json``, ``stage_summary.json``, and
  ``RUN_VALID/verdict.json``.
- No online GWOSC bootstrap or t0 inference happens here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    utc_now_iso,
    write_json_atomic,
)
from brunete.runtime import finalize_stage, init_stage

STAGE_NAME = "prepare_events"
OUTPUT_SCHEMA_VERSION = "brunete_prepare_events_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BRUNETE: prepare a local event cohort from an events file or a LOSC cache root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-id", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--events-file",
        default=None,
        help="Plain-text file with one event_id per line.",
    )
    source.add_argument(
        "--losc-root",
        default=None,
        help="Local LOSC cache root with event directories under <root>/<EVENT_ID>/...",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE_NAME,
        {
            "events_file": str(Path(args.events_file).expanduser().resolve()) if args.events_file else None,
            "losc_root": str(Path(args.losc_root).expanduser().resolve()) if args.losc_root else None,
        },
    )

    artifacts: dict[str, Path] = {}

    try:
        if args.events_file is not None:
            source_path = Path(args.events_file).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"events file not found: {source_path}")
            if not source_path.is_file():
                raise ValueError(f"events file is not a regular file: {source_path}")
            events = _normalize_event_ids(source_path.read_text(encoding="utf-8").splitlines())
            source_kind = "events_file"
            source_root = source_path
        else:
            source_path = Path(args.losc_root).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"losc root not found: {source_path}")
            if not source_path.is_dir():
                raise ValueError(f"losc root is not a directory: {source_path}")
            events = _discover_events_from_losc_root(source_path)
            source_kind = "losc_root"
            source_root = source_path

        if not events:
            raise ValueError(
                f"no events discovered from {source_kind}: {source_root}"
            )

        events_txt = ctx.external_inputs_dir / "events.txt"
        events_txt.write_text("".join(f"{event_id}\n" for event_id in events), encoding="utf-8")
        artifacts["events_txt"] = events_txt

        catalog_payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "run_id": args.run_id,
            "created": utc_now_iso(),
            "source_kind": source_kind,
            "source_path": str(source_root),
            "n_events": len(events),
            "event_ids": events,
        }
        catalog_path = write_json_atomic(ctx.outputs_dir / "events_catalog.json", catalog_payload)
        artifacts["events_catalog_json"] = catalog_path

        finalize_stage(
            ctx,
            verdict="PASS",
            reason="prepare_events completed successfully",
            results={
                "source_kind": source_kind,
                "source_path": str(source_root),
                "n_events": len(events),
                "event_id_preview": events[:10],
            },
            artifacts=artifacts,
        )
        return 0

    except Exception as exc:
        finalize_stage(
            ctx,
            verdict="FAIL",
            reason=str(exc),
            results={
                "source_kind": "events_file" if args.events_file is not None else "losc_root",
                "source_path": str(_resolve_optional_input_path(args.events_file or args.losc_root)),
                "n_events": 0,
                "event_id_preview": [],
            },
            artifacts=artifacts,
            error=str(exc),
        )
        print(f"ERROR [{STAGE_NAME}]: {exc}", file=sys.stderr)
        return 2


def _normalize_event_ids(lines: list[str]) -> list[str]:
    event_ids = {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }
    return sorted(event_ids)


def _discover_events_from_losc_root(losc_root: Path) -> list[str]:
    discovered: list[str] = []
    for event_dir in sorted(path for path in losc_root.iterdir() if path.is_dir()):
        has_hdf5 = any(
            child.is_file() and child.suffix.lower() in {".h5", ".hdf5"}
            for child in event_dir.iterdir()
        )
        if has_hdf5:
            discovered.append(event_dir.name.strip())
    return _normalize_event_ids(discovered)


def _resolve_optional_input_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
