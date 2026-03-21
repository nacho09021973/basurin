from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from mvp.experiment.base_contract import _write_json_atomic, load_json, sha256_file, write_manifest

from brunete.experiment.base_contract import (
    ensure_experiment_dir,
    geometry_summary_input_hashes,
    resolve_joint_support_event_rows,
)

EXPERIMENT_NAME = "b5f_joint_support_rates"
SUPPORT_VERDICTS = {"SUPPORT", "SUPPORTED", "COMPATIBLE"}
REJECT_VERDICTS = {"REJECT", "REJECTED", "DISFAVORED"}
NEUTRAL_VERDICTS = {"INCONCLUSIVE", "UNKNOWN"}


def _normalize_verdict(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().upper()
    if isinstance(value, dict):
        return str(value.get("verdict", value.get("status", "UNKNOWN"))).strip().upper()
    return "UNKNOWN"


def run(classify_run_id: str, runs_root: str | None = None, dry_run: bool = False):
    event_rows = resolve_joint_support_event_rows(classify_run_id, runs_root)
    n_events = len(event_rows)
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n_joint_support": 0,
        "n_supported": 0,
        "n_reject": 0,
        "n_inconclusive": 0,
    })
    input_hashes = geometry_summary_input_hashes(classify_run_id, runs_root)
    events = []

    for row in event_rows:
        verdict_path = row["run_dir"] / "verdict.json"
        if not verdict_path.exists():
            raise FileNotFoundError(f"verdict.json missing for {row['event_run_id']}: {verdict_path}")
        verdict_payload = load_json(verdict_path)
        input_hashes[f"{row['event_run_id']}::verdict"] = sha256_file(verdict_path)
        classify_row = row["classify_row"]
        has_joint_support = bool(classify_row.get("has_joint_support"))
        families = verdict_payload.get("family_verdicts", verdict_payload.get("families", {}))
        per_event_families = {}

        if isinstance(families, dict):
            for family, family_payload in sorted(families.items()):
                verdict = _normalize_verdict(family_payload)
                counts = family_counts[family]
                if has_joint_support and verdict in SUPPORT_VERDICTS:
                    counts["n_joint_support"] += 1
                if verdict in SUPPORT_VERDICTS:
                    counts["n_supported"] += 1
                elif verdict in REJECT_VERDICTS:
                    counts["n_reject"] += 1
                else:
                    counts["n_inconclusive"] += 1
                per_event_families[family] = {
                    "verdict": verdict,
                    "counted_as_joint_support": has_joint_support and verdict in SUPPORT_VERDICTS,
                }

        events.append(
            {
                "event_id": row["event_id"],
                "event_run_id": row["event_run_id"],
                "has_joint_support": has_joint_support,
                "classification": classify_row.get("classification"),
                "support_region_status_221": classify_row.get("support_region_status_221"),
                "support_region_n_final_221": classify_row.get("support_region_n_final_221"),
                "family_verdicts": per_event_families,
            }
        )

    family_joint_support_rates = {}
    for family, counts in sorted(family_counts.items()):
        rate = counts["n_joint_support"] / n_events if n_events else 0.0
        support_rate = counts["n_supported"] / n_events if n_events else 0.0
        family_joint_support_rates[family] = {
            **counts,
            "n_total_events": n_events,
            "joint_support_rate": round(rate, 4),
            "support_rate": round(support_rate, 4),
        }

    result = {
        "schema_version": "b5f-0.2",
        "classify_run_id": classify_run_id,
        "n_events": n_events,
        "n_joint_support_events": sum(1 for row in event_rows if row["classify_row"].get("has_joint_support")),
        "family_joint_support_rates": family_joint_support_rates,
        "events": events,
        "input_hashes": input_hashes,
        "policy": "joint support is consumed from classify_geometries.has_joint_support and never recomputed here",
    }
    if dry_run:
        print(json.dumps(result, indent=2))
        return result
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / "joint_support_rates.json", result)
    _write_json_atomic(out_dir / "joint_support_events.json", events)
    write_manifest(out_dir, result["input_hashes"], extra={"classify_run_id": classify_run_id})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-run-id", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.classify_run_id, args.runs_root, args.dry_run)


if __name__ == "__main__":
    main()
