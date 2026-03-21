from __future__ import annotations

import argparse
import json

from mvp.experiment.base_contract import _write_json_atomic, write_manifest
from mvp.experiment.e5f_verdict_aggregation import aggregate_verdicts

from brunete.experiment.base_contract import ensure_experiment_dir, materialize_event_run_view

EXPERIMENT_NAME = "b5f_verdict_aggregation"


def run(classify_run_id: str, runs_root: str | None = None, dry_run: bool = False):
    view_root, run_ids = materialize_event_run_view(classify_run_id, None, runs_root)
    result = aggregate_verdicts(run_ids, runs_root=view_root)
    result["classify_run_id"] = classify_run_id
    if dry_run:
        print(json.dumps(result, indent=2))
        return result
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / "population_verdict.json", result)
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
