#!/usr/bin/env python3
"""E5-F — Verdict Aggregation (second-order evidence).

Operates EXCLUSIVELY on verdict.json from multiple events.
Does NOT access compatible_set.json, estimates.json, or any raw data.

Produces population-level support rates per family — the core result
for the spectral exclusion paper.

Governance:
  - Only reads verdict.json per run.
  - Writes only under runs/<anchor>/experiment/verdict_aggregation/.
  - Evidence classification is DETERMINISTIC_THRESHOLD, not Bayes factor.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mvp.experiment.base_contract import (
    REQUIRED_CANONICAL_GATES,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_and_load_run,
    write_manifest,
)

SCHEMA_VERSION = "e5f-0.1"
EXPERIMENT_NAME = "verdict_aggregation"

# Evidence strength thresholds (deterministic, not Bayesian)
STRONG_THRESHOLD = 0.8
MODERATE_THRESHOLD = 0.5


def _classify_evidence(rate: float) -> str:
    """Classify support rate into STRONG / MODERATE / WEAK."""
    if rate >= STRONG_THRESHOLD:
        return "STRONG"
    if rate >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "WEAK"


def aggregate_verdicts(
    run_ids: list[str],
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate verdict.json across multiple runs.

    Returns a structured result with support rates per family per mode.
    """
    family_verdicts: dict[str, list[str]] = defaultdict(list)
    mode_verdicts: dict[str, list[str]] = defaultdict(list)
    input_hashes: dict[str, str] = {}
    run_verdicts: list[dict] = []

    for run_id in sorted(run_ids):
        run_dir, _summary = validate_and_load_run(run_id, runs_root)
        verdict_path = run_dir / REQUIRED_CANONICAL_GATES["verdict"]

        if not verdict_path.exists():
            raise FileNotFoundError(f"verdict.json missing for {run_id}: {verdict_path}")

        input_hashes[run_id] = sha256_file(verdict_path)
        verdict = load_json(verdict_path)

        run_verdicts.append({"run_id": run_id, "verdict": verdict})

        # Extract family-level verdicts
        families = verdict.get("family_verdicts", verdict.get("families", {}))
        if isinstance(families, dict):
            for family, fv in families.items():
                v = fv if isinstance(fv, str) else fv.get("verdict", fv.get("status", "UNKNOWN"))
                family_verdicts[family].append(v)

        # Extract mode-level verdicts if present
        modes = verdict.get("mode_verdicts", verdict.get("modes", {}))
        if isinstance(modes, dict):
            for mode, mv in modes.items():
                v = mv if isinstance(mv, str) else mv.get("verdict", mv.get("status", "UNKNOWN"))
                mode_verdicts[mode].append(v)

    n_events = len(run_ids)

    # Compute family support rates
    family_support_rates = {}
    for family, verdicts in sorted(family_verdicts.items()):
        n_support = sum(1 for v in verdicts if v in ("SUPPORT", "SUPPORTED", "COMPATIBLE"))
        n_reject = sum(1 for v in verdicts if v in ("REJECT", "REJECTED", "DISFAVORED"))
        n_inconclusive = sum(1 for v in verdicts if v in ("INCONCLUSIVE", "UNKNOWN"))
        rate = n_support / n_events if n_events > 0 else 0.0
        family_support_rates[family] = {
            "rate": round(rate, 4),
            "n_support": n_support,
            "n_reject": n_reject,
            "n_inconclusive": n_inconclusive,
            "n_total": n_events,
            "evidence_strength": _classify_evidence(rate),
        }

    # Compute mode support rates
    mode_support_rates = {}
    for mode, verdicts in sorted(mode_verdicts.items()):
        n_support = sum(1 for v in verdicts if v in ("SUPPORT", "SUPPORTED", "COMPATIBLE"))
        rate = n_support / n_events if n_events > 0 else 0.0
        mode_support_rates[mode] = {
            "rate": round(rate, 4),
            "n_support": n_support,
            "n_total": n_events,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "n_events_aggregated": n_events,
        "verdict_source_only": True,
        "family_support_rates": family_support_rates,
        "mode_support_rates": mode_support_rates,
        "evidence_classification_policy": "DETERMINISTIC_THRESHOLD — not Bayes factor",
        "thresholds": {
            "strong": STRONG_THRESHOLD,
            "moderate": MODERATE_THRESHOLD,
        },
        "input_hashes": input_hashes,
    }


def run_aggregation(
    run_ids: list[str],
    anchor_run_id: str | None = None,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full aggregation pipeline: validate, aggregate, write results."""
    result = aggregate_verdicts(run_ids, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    anchor = anchor_run_id or sorted(run_ids)[0]
    run_dir, _ = validate_and_load_run(anchor, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    # Write outputs
    _write_json_atomic(out_dir / "population_verdict.json", result)

    _write_json_atomic(
        out_dir / "family_support_rate.json",
        result["family_support_rates"],
    )

    # Evidence strength per family
    evidence = {
        family: data["evidence_strength"]
        for family, data in result["family_support_rates"].items()
    }
    _write_json_atomic(out_dir / "evidence_strength.json", evidence)

    # Manifest
    write_manifest(out_dir, result["input_hashes"], extra={
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
    })

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5-F: Aggregate verdicts across events (second-order evidence)"
    )
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--anchor-run", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_aggregation(
        run_ids=args.run_ids,
        anchor_run_id=args.anchor_run,
        runs_root=args.runs_root,
        dry_run=args.dry_run,
    )

    print(f"Events aggregated: {result['n_events_aggregated']}")
    for family, data in result["family_support_rates"].items():
        print(f"  {family}: {data['rate']*100:.1f}% support ({data['evidence_strength']})")


if __name__ == "__main__":
    main()
