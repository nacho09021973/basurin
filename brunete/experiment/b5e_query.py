#!/usr/bin/env python3
"""B5-E — Reproducible Query Engine (BRUNETE port of E5-E).

A lightweight, deterministic query motor over compatible_set.json artifacts
from the per-event BASURIN subruns of a BRUNETE classify run.

Answers questions like:
  "How many edgb geometries have d2 < 5 across all events?"
  "Which events have more than 3 compatible kerr geometries?"

Governance
----------
- Read-only on compatible_set.json per event subrun.
- Writes only under runs/<classify_run_id>/experiment/query_cache/.
- Idempotent: same query + same inputs → same query_id and same results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from brunete.experiment.base_contract import (
    EVENT_RUN_GATES,
    GovernanceViolation,
    _write_json_atomic,
    ensure_experiment_dir,
    enumerate_event_runs,
    load_json,
    resolve_classify_run_dir,
    sha256_file,
)

SCHEMA_VERSION = "b5e-0.1"
EXPERIMENT_NAME = "query_cache"

_OP_MAP = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: float(a) < float(b),
    "<=": lambda a, b: float(a) <= float(b),
    ">":  lambda a, b: float(a) > float(b),
    ">=": lambda a, b: float(a) >= float(b),
}

_CLAUSE_RE = re.compile(
    r"(\w+)\s*(==|!=|<=|>=|<|>)\s*('[^']*'|\"[^\"]*\"|[\w.+-]+)"
)


def _parse_query(query: str) -> list[tuple[str, str, str]]:
    clauses = _CLAUSE_RE.findall(query)
    if not clauses:
        raise ValueError(f"Cannot parse query: {query!r}")
    return [(field, op, val.strip("'\"")) for field, op, val in clauses]


def _match(geometry: dict, clauses: list[tuple[str, str, str]]) -> bool:
    for field, op, val in clauses:
        gval = geometry.get(field)
        if gval is None:
            return False
        try:
            if not _OP_MAP[op](gval, val):
                return False
        except (ValueError, TypeError):
            return False
    return True


def execute_query(
    classify_run_id: str,
    query: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a query against all compatible_set.json in a classify run.

    Returns per-event match counts and matching geometry_ids.
    Two identical executions produce the same query_id.
    """
    clauses = _parse_query(query)
    query_id = hashlib.sha256(
        f"{classify_run_id}|{mode}|{query}".encode()
    ).hexdigest()[:16]

    event_run_map = enumerate_event_runs(classify_run_id, mode=mode, runs_root=runs_root)
    input_hashes: dict[str, str] = {}
    per_event_matches: list[dict] = []
    total_matched = 0

    for event_id, event_run_dir in sorted(event_run_map.items()):
        cs_path = event_run_dir / EVENT_RUN_GATES["compatible_set"]
        if not cs_path.exists():
            continue
        input_hashes[event_id] = sha256_file(cs_path)
        cs = load_json(cs_path)

        if isinstance(cs, list):
            geometries = cs
        else:
            geometries = cs.get("geometries", cs.get("compatible", []))

        matched = [g for g in geometries if _match(g, clauses)]
        total_matched += len(matched)
        per_event_matches.append({
            "event_id": event_id,
            "n_matched": len(matched),
            "n_total": len(geometries),
            "matched_ids": [
                str(g.get("geometry_id", g.get("id", ""))) for g in matched
            ],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_id,
        "query": query,
        "mode": mode,
        "classify_run_id": classify_run_id,
        "n_events_queried": len(per_event_matches),
        "total_matched": total_matched,
        "per_event_matches": per_event_matches,
        "input_hashes": input_hashes,
    }


def run_b5e(
    classify_run_id: str,
    query: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
) -> Path:
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)
    exp_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    result = execute_query(classify_run_id, query, mode=mode, runs_root=runs_root)

    out_path = exp_dir / f"query_{result['query_id']}.json"
    _write_json_atomic(out_path, result)

    log_path = exp_dir / "query_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "query_id": result["query_id"],
            "query": query,
            "mode": mode,
            "classify_run_id": classify_run_id,
            "total_matched": result["total_matched"],
            "n_events_queried": result["n_events_queried"],
        }) + "\n")

    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-E: reproducible query over compatible geometries in a classify run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-run", required=True)
    ap.add_argument("--query", required=True,
                    help="Query string, e.g. \"family == 'edgb' AND mahalanobis_d2 < 5.0\"")
    ap.add_argument("--mode", choices=["220", "221"], default="220")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = execute_query(args.classify_run, args.query,
                           mode=args.mode, runs_root=args.runs_root)

    if args.dry_run:
        print(json.dumps({k: v for k, v in result.items()
                          if k != "per_event_matches"}, indent=2))
        return 0

    out_path = run_b5e(args.classify_run, args.query,
                       mode=args.mode, runs_root=args.runs_root)
    print(f"B5-E written: {out_path}")
    print(f"  query_id      : {result['query_id']}")
    print(f"  total_matched : {result['total_matched']}")
    print(f"  events queried: {result['n_events_queried']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
