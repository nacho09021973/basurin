#!/usr/bin/env python3
"""MVP Stage 6c: population descriptive geometry summary from aggregate.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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

STAGE = "s6c_population_geometry"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _weighted_stats(values: list[float], weights: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_weighted": None, "std_weighted": None}
    wsum = sum(weights)
    if wsum <= 0:
        return {"mean_weighted": None, "std_weighted": None}
    mu = sum(v * w for v, w in zip(values, weights)) / wsum
    var = sum(w * (v - mu) ** 2 for v, w in zip(values, weights)) / wsum
    return {"mean_weighted": mu, "std_weighted": math.sqrt(max(0.0, var))}


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if len(values) < 5:
        return {"p05": None, "p50": None, "p95": None}
    vals = sorted(values)

    def pct(q: float) -> float:
        idx = q * (len(vals) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)

    return {"p05": pct(0.05), "p50": pct(0.50), "p95": pct(0.95)}


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s6c population geometry")
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    ctx = init_stage(args.run, STAGE, params={})
    aggregate_path = ctx.run_dir / "s5_aggregate" / "outputs" / "aggregate.json"
    check_inputs(ctx, {"aggregate": aggregate_path})

    try:
        aggregate = _read_json(aggregate_path)
        events = aggregate.get("events", [])
        if not isinstance(events, list):
            abort(ctx, "aggregate.json invalid: events must be a list")

        violations = 0
        reasons = Counter()
        n_inconclusive = 0

        scalar_vals: list[float] = []
        scalar_w: list[float] = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            censor = ev.get("censoring", {}) if isinstance(ev.get("censoring"), dict) else {}
            has_221 = bool(censor.get("has_221"))
            vote = str(censor.get("vote_kerr") or "INCONCLUSIVE")
            wv = float(censor.get("weight_vector4") or 0.0)
            ws = float(censor.get("weight_scalar") if censor.get("weight_scalar") is not None else 1.0)
            reason = censor.get("reason")

            if not has_221 and (vote != "INCONCLUSIVE" or abs(wv - 0.0) > 1e-12):
                violations += 1

            if vote == "INCONCLUSIVE":
                n_inconclusive += 1
            if reason:
                reasons[str(reason)] += 1

            scalar = ev.get("scalar", {}) if isinstance(ev.get("scalar"), dict) else {}
            kt = scalar.get("kerr_tension")
            if isinstance(kt, (int, float)) and math.isfinite(float(kt)) and ws > 0:
                scalar_vals.append(float(kt))
                scalar_w.append(ws)

        if violations > 0:
            abort(
                ctx,
                "Hard censorship rule violation(s): has_221=false requires vote_kerr=INCONCLUSIVE and weight_vector4=0.0",
            )

        stats = _weighted_stats(scalar_vals, scalar_w)
        claim_payload = {
            "schema_version": "population_scalar_claim_v1",
            "run_id": args.run,
            "n_events_total": len(events),
            "n_scalar_included": len(scalar_vals),
            "summary": {
                **stats,
                **_percentiles(scalar_vals),
            },
        }
        diag_payload = {
            "schema_version": "population_diagnostics_v1",
            "run_id": args.run,
            "n_events_total": len(events),
            "n_inconclusive": n_inconclusive,
            "reasons_frequency": dict(sorted(reasons.items())),
            "hard_rule_violations": violations,
        }

        claim_path = ctx.outputs_dir / "population_scalar_claim.json"
        diag_path = ctx.outputs_dir / "population_diagnostics.json"
        write_json_atomic(claim_path, claim_payload)
        write_json_atomic(diag_path, diag_payload)

        finalize(
            ctx,
            artifacts={
                "population_scalar_claim": claim_path,
                "population_diagnostics": diag_path,
            },
            results={
                "n_events_total": len(events),
                "n_scalar_included": len(scalar_vals),
                "n_inconclusive": n_inconclusive,
            },
        )
        print(f"OUT_ROOT={ctx.out_root}")
        print(f"STAGE_DIR={ctx.stage_dir}")
        print(f"OUTPUTS_DIR={ctx.outputs_dir}")
        print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
        print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
