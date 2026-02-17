#!/usr/bin/env python3
"""Stable Stage 4c entrypoint: Kerr consistency summary for multimode pipeline.

This script exists as a stable CLI target for pipeline orchestration
(`mvp/s4c_kerr_consistency.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage

STAGE = "s4c_kerr_consistency"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s4c Kerr consistency wrapper")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas-path", required=False, default=None)
    args = ap.parse_args()

    ctx = init_stage(args.run_id, STAGE, params={"atlas_path": args.atlas_path})

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    compatible_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"

    try:
        check_inputs(
            ctx,
            {
                "s3_estimates": estimates_path,
                "s3b_multimode": multimode_path,
            },
            optional={"s4_compatible_set": compatible_path},
        )

        estimates = _read_json(estimates_path)
        multimode = _read_json(multimode_path)

        compatible = _read_json(compatible_path) if compatible_path.exists() else {}
        n_compatible = compatible.get("n_compatible")
        d2_min = compatible.get("d2_min")

        payload: dict[str, Any] = {
            "schema_version": "mvp_kerr_consistency_v1",
            "run_id": args.run_id,
            "event_id": estimates.get("event_id", "unknown"),
            "kerr_consistent": bool(n_compatible and n_compatible > 0),
            "d2_min": d2_min,
            "chi_best": None,
            "source": {
                "multimode_verdict": multimode.get("results", {}).get("verdict"),
                "compatible_set_present": compatible_path.exists(),
            },
        }

        out_path = ctx.outputs_dir / "kerr_consistency.json"
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"kerr_consistency": out_path},
            verdict="PASS",
            results={
                "kerr_consistent": payload["kerr_consistent"],
                "d2_min": payload["d2_min"],
                "chi_best": payload["chi_best"],
            },
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
