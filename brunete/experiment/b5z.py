from __future__ import annotations

import argparse
import json

from mvp.experiment.base_contract import _write_json_atomic, write_manifest
from mvp.experiment.e5z_gpr_emulator import emulate_all_families

from brunete.experiment.base_contract import ensure_experiment_dir, materialize_event_run_view, resolve_event_run_dirs

EXPERIMENT_NAME = "b5z_continuous_emulator"


def run(classify_run_id: str, mode: str, runs_root: str | None = None, dry_run: bool = False):
    resolved = resolve_event_run_dirs(classify_run_id, mode, runs_root)
    per_event = {}
    input_hashes = {}
    view_root, _ = materialize_event_run_view(classify_run_id, mode, runs_root)
    for row in resolved:
        result = emulate_all_families(row["event_run_id"], runs_root=view_root)
        per_event[row["event_id"]] = result
        for fam, fam_result in result.get("per_family_results", {}).items():
            for key, value in fam_result.get("input_hashes", {}).items():
                input_hashes[f"{row['event_run_id']}::{fam}::{key}"] = value
    summary = {"classify_run_id": classify_run_id, "mode": mode, "n_events": len(per_event), "per_event": per_event}
    if dry_run:
        print(json.dumps(summary, indent=2, default=str))
        return summary
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / "predicted_minima_by_event.json", summary)
    write_manifest(out_dir, input_hashes, extra={"classify_run_id": classify_run_id, "mode": mode})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-run-id", required=True)
    ap.add_argument("--mode", choices=["220", "221"], required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.classify_run_id, args.mode, args.runs_root, args.dry_run)


if __name__ == "__main__":
    main()
