#!/usr/bin/env python3
"""List the canonical visible LOSC events under ``data/losc`` for BRUNETE.

This public entrypoint materializes a deterministic snapshot under
``runs/<run_id>/list_events`` so the operator has a canonical place to look
instead of reconstructing the cohort ad hoc from chat history.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import utc_now_iso, write_json_atomic
from brunete.runtime import finalize_stage, init_stage

STAGE_NAME = "list_events"
OUTPUT_SCHEMA_VERSION = "brunete_list_events_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BRUNETE: materialize the canonical sorted list of visible LOSC events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--losc-root",
        default="data/losc",
        help="Local LOSC cache root with event directories under <root>/<EVENT_ID>/...",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    losc_root = Path(args.losc_root).expanduser().resolve()
    ctx = init_stage(
        args.run_id,
        STAGE_NAME,
        {
            "losc_root": str(losc_root),
        },
    )

    artifacts: dict[str, Path] = {}

    try:
        if not losc_root.exists():
            raise FileNotFoundError(f"losc root not found: {losc_root}")
        if not losc_root.is_dir():
            raise ValueError(f"losc root is not a directory: {losc_root}")

        events = _discover_events_from_losc_root(losc_root)
        if not events:
            raise ValueError(f"no visible events discovered under canonical LOSC root: {losc_root}")

        visible_events_txt = ctx.outputs_dir / "visible_events.txt"
        visible_events_txt.write_text("".join(f"{event_id}\n" for event_id in events), encoding="utf-8")
        artifacts["visible_events_txt"] = visible_events_txt

        catalog_payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "run_id": args.run_id,
            "created": utc_now_iso(),
            "source_kind": "losc_root",
            "source_path": str(losc_root),
            "n_events": len(events),
            "event_ids": events,
        }
        catalog_path = write_json_atomic(ctx.outputs_dir / "events_catalog.json", catalog_payload)
        artifacts["events_catalog_json"] = catalog_path

        _, summary_path, manifest_path = finalize_stage(
            ctx,
            verdict="PASS",
            reason="list_events completed successfully",
            results={
                "source_kind": "losc_root",
                "source_path": str(losc_root),
                "n_events": len(events),
                "event_id_preview": events[:10],
            },
            artifacts=artifacts,
        )
        _log_stage_paths(ctx, summary_path, manifest_path)
        sys.stdout.write("".join(f"{event_id}\n" for event_id in events))
        return 0

    except Exception as exc:
        _, summary_path, manifest_path = finalize_stage(
            ctx,
            verdict="FAIL",
            reason=str(exc),
            results={
                "source_kind": "losc_root",
                "source_path": str(losc_root),
                "n_events": 0,
                "event_id_preview": [],
            },
            artifacts=artifacts,
            error=str(exc),
        )
        _log_stage_paths(ctx, summary_path, manifest_path)
        print(f"ERROR [{STAGE_NAME}]: {exc}", file=sys.stderr)
        return 2


def _discover_events_from_losc_root(losc_root: Path) -> list[str]:
    discovered: list[str] = []
    for event_dir in sorted(path for path in losc_root.iterdir() if path.is_dir()):
        has_hdf5 = any(
            child.is_file() and child.suffix.lower() in {".h5", ".hdf5"}
            for child in event_dir.iterdir()
        )
        if has_hdf5:
            discovered.append(event_dir.name.strip())
    return sorted({event_id for event_id in discovered if event_id})


def _log_stage_paths(ctx, stage_summary_path: Path, manifest_path: Path) -> None:
    print(f"OUT_ROOT={ctx.out_root}")
    print(f"STAGE_DIR={ctx.stage_dir}")
    print(f"OUTPUTS_DIR={ctx.outputs_dir}")
    print(f"STAGE_SUMMARY={stage_summary_path}")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
