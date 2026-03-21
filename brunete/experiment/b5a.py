from __future__ import annotations

import argparse
import json

from mvp.experiment.base_contract import _write_json_atomic, write_manifest
from mvp.experiment.e5a_multi_event_aggregation import aggregate_events

from brunete.experiment.base_contract import ensure_experiment_dir, materialize_event_run_view

EXPERIMENT_NAME = "b5a_multi_event_aggregation"


def run(classify_run_id: str, mode: str, runs_root: str | None = None, dry_run: bool = False):
    view_root, run_ids = materialize_event_run_view(classify_run_id, mode, runs_root)
    result = aggregate_events(run_ids, runs_root=view_root)
    result["classify_run_id"] = classify_run_id
    result["mode"] = mode
    if dry_run:
        print(json.dumps(result, indent=2))
        return result
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / "aggregation_result.json", result)
    _write_json_atomic(out_dir / "jaccard_matrix.json", result["jaccard_matrix"])
    _write_json_atomic(out_dir / "persistence_histogram.json", result["persistence_histogram"])
    write_manifest(out_dir, result["input_hashes"], extra={"classify_run_id": classify_run_id, "mode": mode})
    return result


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
