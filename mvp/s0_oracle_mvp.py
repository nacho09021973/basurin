#!/usr/bin/env python3
"""MVP Stage 0: deterministic viability oracle (offline-first + fail-fast gate)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, finalize


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_local_hdf5(items: list[str]) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    reasons: list[str] = []
    for raw in items:
        det, sep, path_raw = raw.partition("=")
        if not sep:
            reasons.append(f"Invalid --local-hdf5 mapping (expected DET=PATH): {raw}")
            continue
        det_key = det.strip().upper()
        candidate = Path(path_raw.strip())
        if det_key in resolved:
            reasons.append(f"Duplicate --local-hdf5 detector: {det_key}")
            continue
        if det_key not in {"H1", "L1", "V1"}:
            reasons.append(f"Invalid detector in --local-hdf5: {det_key}")
            continue
        if not candidate.exists() or not candidate.is_file():
            reasons.append(f"Missing local_hdf5 file for {det_key}: {candidate}")
            continue
        resolved[det_key] = str(candidate.resolve())
    return resolved, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic oracle precheck for BASURIN runs")
    parser.add_argument("--run", required=True, help="Run identifier")
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--require-offline", action="store_true", default=False)
    parser.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Local detector files used as canonical offline inputs",
    )
    args = parser.parse_args(argv)

    started = _now_iso()
    ctx = init_stage(
        args.run,
        "s0_oracle_mvp",
        params={
            "event_id": args.event_id,
            "require_offline": bool(args.require_offline),
            "local_hdf5": list(args.local_hdf5),
        },
    )

    reasons: list[str] = []
    local_map, parse_reasons = _parse_local_hdf5(args.local_hdf5)
    reasons.extend(parse_reasons)

    if args.require_offline and not local_map:
        reasons.append("Offline policy active (--require-offline) but no valid --local-hdf5 inputs were provided")

    metrics = {
        "require_offline": bool(args.require_offline),
        "local_hdf5_count": len(args.local_hdf5),
        "local_hdf5_valid_count": len(local_map),
        "network_attempted": False,
    }

    verdict = "PASS" if not reasons else "FAIL"
    oracle_payload = {
        "oracle": {
            "name": "oracle_baseline_mvp_v1",
            "verdict": verdict,
            "reasons": reasons,
            "metrics": metrics,
            "timestamps": {"started_utc": started, "ended_utc": _now_iso()},
        },
        "inputs": {"local_hdf5": local_map},
    }
    out_path = ctx.outputs_dir / "oracle_metrics.json"
    out_path.write_text(json.dumps(oracle_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    finalize(
        ctx,
        artifacts={"oracle_metrics": out_path},
        verdict=verdict,
        results={"oracle": oracle_payload["oracle"]},
        extra_summary={"oracle": oracle_payload["oracle"]},
    )

    if verdict != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
