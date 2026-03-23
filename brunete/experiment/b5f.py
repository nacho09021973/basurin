from __future__ import annotations

import argparse
import json

from mvp.experiment.base_contract import _write_json_atomic, write_manifest
from mvp.experiment.e5f_verdict_aggregation import aggregate_verdicts

from brunete.experiment.base_contract import (
    ensure_experiment_dir,
    load_json,
    materialize_event_run_view,
    resolve_event_run_dirs,
    validate_classify_run,
)

EXPERIMENT_NAME = "b5f_verdict_aggregation"


def run(classify_run_id: str, runs_root: str | None = None, dry_run: bool = False):
    view_root, run_ids = materialize_event_run_view(classify_run_id, None, runs_root)
    result = aggregate_verdicts(run_ids, runs_root=view_root)
    result["classify_run_id"] = classify_run_id
    _add_brunete_compat_fields(result, classify_run_id, runs_root)
    if dry_run:
        print(json.dumps(result, indent=2))
        return result
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / "population_verdict.json", result)
    write_manifest(out_dir, result["input_hashes"], extra={"classify_run_id": classify_run_id})
    return result


def _add_brunete_compat_fields(result: dict, classify_run_id: str, runs_root: str | None) -> None:
    _, geometry_summary = validate_classify_run(classify_run_id, runs_root)
    rows = geometry_summary.get("rows", [])
    resolved = {
        row["event_id"]: row
        for row in resolve_event_run_dirs(classify_run_id, None, runs_root)
    }

    compat_events = []
    family_joint_counts: dict[str, dict[str, int]] = {}

    for row in rows:
        event_id = row.get("event_id")
        has_joint_support = bool(row.get("has_joint_support"))
        compat_row = dict(row)
        compat_row["family_verdicts"] = {}

        resolved_row = resolved.get(event_id)
        if resolved_row is not None:
            verdict_path = resolved_row["event_run_dir"] / "verdict.json"
            if verdict_path.exists():
                verdict_payload = load_json(verdict_path)
                families = verdict_payload.get("family_verdicts", {})
                if isinstance(families, dict):
                    for family, payload in families.items():
                        if isinstance(payload, dict):
                            family_payload = dict(payload)
                        else:
                            family_payload = {"verdict": payload}
                        family_payload["counted_as_joint_support"] = has_joint_support
                        compat_row["family_verdicts"][family] = family_payload

                        stats = family_joint_counts.setdefault(family, {"n_total": 0, "n_joint_support": 0})
                        stats["n_total"] += 1
                        if has_joint_support:
                            stats["n_joint_support"] += 1

        compat_events.append(compat_row)

    family_joint_support_rates = {}
    for family, stats in sorted(family_joint_counts.items()):
        n_total = stats["n_total"]
        n_joint_support = stats["n_joint_support"]
        family_joint_support_rates[family] = {
            "n_total": n_total,
            "n_joint_support": n_joint_support,
            "joint_support_rate": round(n_joint_support / n_total, 4) if n_total else 0.0,
        }

    result["n_events"] = int(result.get("n_events_aggregated", len(compat_events)))
    result["n_joint_support_events"] = sum(1 for row in compat_events if row.get("has_joint_support"))
    result["events"] = compat_events
    result["family_joint_support_rates"] = family_joint_support_rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-run-id", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.classify_run_id, args.runs_root, args.dry_run)


if __name__ == "__main__":
    main()
