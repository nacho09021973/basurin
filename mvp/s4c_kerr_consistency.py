#!/usr/bin/env python3
"""Stable Stage 4c entrypoint: Kerr consistency summary for multimode pipeline.

This script exists as a stable CLI target for pipeline orchestration
(`mvp/s4c_kerr_consistency.py`).
"""
from __future__ import annotations

import argparse
import json
import os
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
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "s4c_kerr_consistency"
MULTIMODE_OK = "MULTIMODE_OK"
SINGLEMODE_ONLY = "SINGLEMODE_ONLY"
RINGDOWN_NONINFORMATIVE = "RINGDOWN_NONINFORMATIVE"
SKIPPED_MULTIMODE_GATE = "SKIPPED_MULTIMODE_GATE"
ALLOWED_MULTIMODE_VIABILITY_CLASSES = {
    MULTIMODE_OK,
    SINGLEMODE_ONLY,
    RINGDOWN_NONINFORMATIVE,
}
SKIP_VIABILITY_CLASSES = {SINGLEMODE_ONLY, RINGDOWN_NONINFORMATIVE}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _read_multimode_viability(s3b_summary_path: Path) -> dict[str, Any] | None:
    if not s3b_summary_path.exists():
        return None
    summary = _read_json(s3b_summary_path)
    viability = summary.get("multimode_viability")
    if viability is None:
        return None
    if not isinstance(viability, dict):
        raise ValueError(
            "Invalid s3b multimode_viability contract: expected object in "
            f"{s3b_summary_path}"
        )
    viability_class = viability.get("class")
    if not isinstance(viability_class, str) or viability_class not in ALLOWED_MULTIMODE_VIABILITY_CLASSES:
        raise ValueError(
            "Invalid s3b multimode_viability.class in "
            f"{s3b_summary_path}: {viability_class!r}. "
            "Regenerate upstream with: python -m mvp.s3b_multimode_estimates --run-id <RUN_ID>"
        )
    reasons = viability.get("reasons")
    if reasons is not None and not isinstance(reasons, list):
        raise ValueError(
            "Invalid s3b multimode_viability.reasons in "
            f"{s3b_summary_path}: expected list"
        )
    return viability


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s4c Kerr consistency wrapper")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--atlas-path", required=False, default=None)
    args = ap.parse_args()

    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())

    ctx = init_stage(args.run_id, STAGE, params={"atlas_path": args.atlas_path})

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    s3b_summary_path = ctx.run_dir / "s3b_multimode_estimates" / "stage_summary.json"
    compatible_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"

    try:
        check_inputs(
            ctx,
            {
                "s3_estimates": estimates_path,
                "s3b_multimode": multimode_path,
            },
            optional={"s3b_stage_summary": s3b_summary_path, "s4_compatible_set": compatible_path},
        )

        estimates = _read_json(estimates_path)
        multimode = _read_json(multimode_path)
        viability = _read_multimode_viability(s3b_summary_path)
        viability_class = (
            str(viability.get("class")) if isinstance(viability, dict) and viability.get("class") is not None else None
        )
        viability_reasons = (
            [str(r) for r in viability.get("reasons", [])]
            if isinstance(viability, dict) and isinstance(viability.get("reasons"), list)
            else []
        )

        compatible = _read_json(compatible_path) if compatible_path.exists() else {}
        n_compatible = compatible.get("n_compatible")
        d2_min = compatible.get("d2_min")
        skip_by_gate = viability_class in SKIP_VIABILITY_CLASSES
        kerr_consistent = None if skip_by_gate else bool(n_compatible and n_compatible > 0)
        status = SKIPPED_MULTIMODE_GATE if skip_by_gate else "OK"

        payload: dict[str, Any] = {
            "schema_version": "mvp_kerr_consistency_v1",
            "run_id": args.run_id,
            "event_id": estimates.get("event_id", "unknown"),
            "status": status,
            "kerr_consistent": kerr_consistent,
            "d2_min": d2_min,
            "chi_best": None,
            "source": {
                "multimode_verdict": multimode.get("results", {}).get("verdict"),
                "multimode_viability_class": viability_class,
                "multimode_viability_reasons": viability_reasons,
                "compatible_set_present": compatible_path.exists(),
            },
        }
        if viability is not None:
            payload["multimode_viability"] = viability

        out_path = ctx.outputs_dir / "kerr_consistency.json"
        write_json_atomic(out_path, payload)
        extra_summary: dict[str, Any] | None = None
        if viability is not None:
            extra_summary = {"multimode_viability": viability}
        finalize(
            ctx,
            artifacts={"kerr_consistency": out_path},
            verdict="PASS",
            results={
                "status": payload["status"],
                "kerr_consistent": payload["kerr_consistent"],
                "d2_min": payload["d2_min"],
                "chi_best": payload["chi_best"],
                "multimode_viability_class": viability_class,
            },
            extra_summary=extra_summary,
        )
        log_stage_paths(ctx)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
