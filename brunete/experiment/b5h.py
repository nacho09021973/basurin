from __future__ import annotations

import argparse
import json

from mvp.experiment.base_contract import _write_json_atomic, write_manifest
from mvp.experiment.e5h_blind_prediction import blind_prediction

from brunete.experiment.base_contract import ensure_experiment_dir, materialize_event_run_view

EXPERIMENT_NAME = "b5h_blind_prediction"


def run(classify_run_id: str, mode: str, strategy: str = "intersection", runs_root: str | None = None, dry_run: bool = False):
    view_root, run_ids = materialize_event_run_view(classify_run_id, mode, runs_root)
    result = blind_prediction(run_ids, prediction_strategy=strategy, runs_root=view_root)
    result["classify_run_id"] = classify_run_id
    result["mode"] = mode
    if dry_run:
        print(json.dumps(result, indent=2))
        return result
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / "prediction_results.json", result)
    write_manifest(out_dir, result["input_hashes"], extra={"classify_run_id": classify_run_id, "mode": mode, "strategy": strategy})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-run-id", required=True)
    ap.add_argument("--mode", choices=["220", "221"], required=True)
    ap.add_argument("--strategy", default="intersection", choices=["intersection", "majority", "frequency_weighted"])
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.classify_run_id, args.mode, args.strategy, args.runs_root, args.dry_run)


if __name__ == "__main__":
    main()
