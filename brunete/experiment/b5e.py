from __future__ import annotations

import argparse
import json

from mvp.experiment.base_contract import _write_json_atomic
from mvp.experiment.e5e_query import execute_query

from brunete.experiment.base_contract import ensure_experiment_dir, materialize_event_run_view

EXPERIMENT_NAME = "b5e_query"


def run(classify_run_id: str, query: str, runs_root: str | None = None, dry_run: bool = False):
    view_root, run_ids = materialize_event_run_view(classify_run_id, None, runs_root)
    result = execute_query(query, run_ids, runs_root=view_root)
    result["classify_run_id"] = classify_run_id
    result["run_ids_consumed"] = sorted(run_ids)
    if dry_run:
        print(json.dumps(result, indent=2))
        return result
    out_dir = ensure_experiment_dir(classify_run_id, EXPERIMENT_NAME, runs_root)
    _write_json_atomic(out_dir / f"query_{result['query_id']}.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-run-id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.classify_run_id, args.query, args.runs_root, args.dry_run)


if __name__ == "__main__":
    main()
