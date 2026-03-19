#!/usr/bin/env python3
"""Evaluate whether a candidate mode-221 upstream strategy improved over a baseline.

This helper is intentionally minimal and transparent:

1. It reads the canonical ``s3b_multimode_estimates/outputs/multimode_estimates.json``
   from two existing runs.
2. It compares the explicit ``mode_221_usable`` gate first.
3. It only uses a small set of human-auditable secondary metrics with configurable
   materiality thresholds.
4. It reports the exact rule path used to reach ``improved``, ``neutral``, or
   ``degraded``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[1], _here.parents[2]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import resolve_out_root, require_run_valid, validate_run_id

TRACKED_METRICS = (
    "valid_fraction",
    "cv_Q",
    "lnQ_span",
    "cv_f",
    "lnf_span",
)
LOWER_IS_BETTER = {"cv_Q", "lnQ_span", "cv_f", "lnf_span"}
HIGHER_IS_BETTER = {"valid_fraction"}


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _extract_mode(payload: dict[str, Any], label: str) -> dict[str, Any]:
    modes = payload.get("modes")
    if not isinstance(modes, list):
        return {}
    for item in modes:
        if isinstance(item, dict) and item.get("label") == label:
            return item
    return {}


def _metric_from_mode(mode_payload: dict[str, Any], key: str) -> float | None:
    fit = mode_payload.get("fit") if isinstance(mode_payload, dict) else None
    stability = fit.get("stability") if isinstance(fit, dict) else None
    if not isinstance(stability, dict):
        return None
    value = stability.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _artifact_path(out_root: Path, run_id: str) -> Path:
    return out_root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"


def _load_run_summary(out_root: Path, run_id: str, *, require_pass: bool) -> dict[str, Any]:
    validate_run_id(run_id, out_root)
    if require_pass:
        require_run_valid(out_root, run_id)
    artifact = _artifact_path(out_root, run_id)
    if not artifact.exists():
        raise FileNotFoundError(
            "missing expected multimode artifact: "
            f"{artifact}\n"
            "expected exact path above under the run directory"
        )
    payload = _read_json_dict(artifact)
    mode_221 = _extract_mode(payload, "221")
    summary = {
        "run_id": run_id,
        "artifact": str(artifact),
        "mode_221_usable": bool(payload.get("mode_221_usable", False)),
        "mode_221_usable_reason": payload.get("mode_221_usable_reason") or "missing_mode_221_usable_reason",
    }
    for metric in TRACKED_METRICS:
        summary[metric] = _metric_from_mode(mode_221, metric)
    return summary


def _metric_delta(metric: str, baseline: float | None, candidate: float | None, threshold: float) -> dict[str, Any]:
    if baseline is None and candidate is None:
        return {"status": "same_missing", "delta": None, "threshold": threshold}
    if baseline is None and candidate is not None:
        return {
            "status": "candidate_recovered_metric",
            "delta": None,
            "threshold": threshold,
            "candidate": candidate,
        }
    if baseline is not None and candidate is None:
        return {
            "status": "candidate_lost_metric",
            "delta": None,
            "threshold": threshold,
            "baseline": baseline,
        }

    assert baseline is not None and candidate is not None
    signed_delta = candidate - baseline
    if metric in HIGHER_IS_BETTER:
        if signed_delta >= threshold:
            status = "better"
        elif signed_delta <= -threshold:
            status = "worse"
        else:
            status = "similar"
    else:
        if signed_delta <= -threshold:
            status = "better"
        elif signed_delta >= threshold:
            status = "worse"
        else:
            status = "similar"
    return {
        "status": status,
        "delta": signed_delta,
        "threshold": threshold,
        "baseline": baseline,
        "candidate": candidate,
    }


def evaluate_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    valid_fraction_delta: float,
    spread_delta: float,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    better_metrics: list[str] = []
    worse_metrics: list[str] = []
    recovered_metrics: list[str] = []
    lost_metrics: list[str] = []

    for metric in TRACKED_METRICS:
        threshold = valid_fraction_delta if metric == "valid_fraction" else spread_delta
        result = _metric_delta(metric, baseline.get(metric), candidate.get(metric), threshold)
        comparisons[metric] = result
        status = result["status"]
        if status == "better":
            better_metrics.append(metric)
        elif status == "worse":
            worse_metrics.append(metric)
        elif status == "candidate_recovered_metric":
            recovered_metrics.append(metric)
        elif status == "candidate_lost_metric":
            lost_metrics.append(metric)

    baseline_usable = bool(baseline.get("mode_221_usable", False))
    candidate_usable = bool(candidate.get("mode_221_usable", False))

    trace: list[str] = [
        "primary rule: compare mode_221_usable before secondary metrics",
        f"baseline usable={baseline_usable} reason={baseline.get('mode_221_usable_reason')}",
        f"candidate usable={candidate_usable} reason={candidate.get('mode_221_usable_reason')}",
    ]

    if not baseline_usable and candidate_usable:
        verdict = "improved"
        trace.append("candidate crossed the main usability gate: false -> true")
    elif baseline_usable and not candidate_usable:
        verdict = "degraded"
        trace.append("candidate lost the main usability gate: true -> false")
    elif baseline_usable and candidate_usable:
        if worse_metrics or lost_metrics:
            if better_metrics or recovered_metrics:
                verdict = "neutral"
                trace.append("both runs are usable but secondary metrics are mixed")
            else:
                verdict = "degraded"
                trace.append("both runs are usable and candidate is materially worse in secondary metrics")
        elif better_metrics or recovered_metrics:
            verdict = "improved"
            trace.append("both runs are usable and candidate is materially better with no material regressions")
        else:
            verdict = "neutral"
            trace.append("both runs are usable and all tracked secondary metrics are within tolerance")
    else:
        if worse_metrics or lost_metrics:
            verdict = "degraded"
            trace.append("both runs remain unusable and candidate also regressed in secondary metrics")
        elif better_metrics or recovered_metrics:
            verdict = "neutral"
            trace.append("candidate improved some secondary metrics but did not solve the main unusable state")
        else:
            verdict = "neutral"
            trace.append("both runs remain unusable with no material secondary change")

    return {
        "schema_version": "evaluate_221_strategy_v1",
        "verdict": verdict,
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": comparisons,
        "summary": {
            "better_metrics": better_metrics,
            "worse_metrics": worse_metrics,
            "recovered_metrics": recovered_metrics,
            "lost_metrics": lost_metrics,
        },
        "trace": trace,
        "rules": {
            "primary": "mode_221_usable dominates the decision",
            "secondary": {
                "valid_fraction": f"higher is better; material if |Δ| >= {valid_fraction_delta}",
                "cv_Q": f"lower is better; material if |Δ| >= {spread_delta}",
                "lnQ_span": f"lower is better; material if |Δ| >= {spread_delta}",
                "cv_f": f"lower is better; material if |Δ| >= {spread_delta}",
                "lnf_span": f"lower is better; material if |Δ| >= {spread_delta}",
            },
            "still_unusable_policy": "if both runs have mode_221_usable=false, the verdict cannot be improved",
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-run", required=True, help="Existing baseline run_id")
    ap.add_argument("--candidate-run", required=True, help="Existing candidate run_id")
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for lookup only")
    ap.add_argument(
        "--allow-nonpass-runs",
        action="store_true",
        help="Skip RUN_VALID=PASS enforcement and only require the s3b artifact to exist.",
    )
    ap.add_argument("--valid-fraction-delta", type=float, default=0.05)
    ap.add_argument("--spread-delta", type=float, default=0.05)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.valid_fraction_delta < 0 or args.spread_delta < 0:
        raise SystemExit("delta thresholds must be >= 0")

    if args.runs_root:
        out_root = Path(args.runs_root).expanduser().resolve()
    else:
        out_root = resolve_out_root("runs")

    baseline = _load_run_summary(out_root, args.baseline_run, require_pass=not args.allow_nonpass_runs)
    candidate = _load_run_summary(out_root, args.candidate_run, require_pass=not args.allow_nonpass_runs)
    result = evaluate_runs(
        baseline,
        candidate,
        valid_fraction_delta=float(args.valid_fraction_delta),
        spread_delta=float(args.spread_delta),
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
