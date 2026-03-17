#!/usr/bin/env python3
"""E5-E — Reproducible Query Engine (no heavy stage re-execution).

A lightweight query motor over frozen artifact snapshots.  Answers questions
like "how many edgb geometries have d² < 5 across all events?" in a fully
deterministic, idempotent way without re-running any pipeline stage.

Governance:
  - Read-only access to canonical artifacts.
  - All outputs go under runs/<anchor>/experiment/query_cache/.
  - Idempotent: two identical runs produce byte-identical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from experiment.base_contract import (
    GovernanceViolation,
    REQUIRED_CANONICAL_GATES,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_and_load_run,
)

SCHEMA_VERSION = "e5e-0.1"
EXPERIMENT_NAME = "query_cache"


# ── Mini query parser ───────────────────────────────────────────────────────

_OP_MAP = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: float(a) < float(b),
    "<=": lambda a, b: float(a) <= float(b),
    ">": lambda a, b: float(a) > float(b),
    ">=": lambda a, b: float(a) >= float(b),
}

_CLAUSE_RE = re.compile(
    r"(\w+)\s*(==|!=|<=|>=|<|>)\s*('[^']*'|\"[^\"]*\"|[\w.+-]+)"
)


def _parse_query(query: str) -> list[tuple[str, str, str]]:
    """Parse a simple AND-connected query into (field, op, value) triples."""
    clauses = _CLAUSE_RE.findall(query)
    if not clauses:
        raise ValueError(f"Cannot parse query: {query!r}")
    parsed = []
    for field, op, val in clauses:
        val = val.strip("'\"")
        parsed.append((field, op, val))
    return parsed


def _match(geometry: dict, clauses: list[tuple[str, str, str]]) -> bool:
    """Return True if geometry matches ALL clauses."""
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


# ── Core query execution ────────────────────────────────────────────────────

def execute_query(
    query: str,
    run_ids: list[str],
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute query over multiple runs.  Returns deterministic result dict."""
    clauses = _parse_query(query)
    query_hash = hashlib.sha256(query.encode()).hexdigest()

    input_snapshots: dict[str, str] = {}
    matching_geometries: list[dict] = []
    total_scanned = 0

    t0 = time.monotonic()

    for run_id in sorted(run_ids):
        run_dir, _summary = validate_and_load_run(run_id, runs_root)
        cs_path = run_dir / REQUIRED_CANONICAL_GATES["compatible_set"]
        if not cs_path.exists():
            raise FileNotFoundError(f"compatible_set.json missing for {run_id}: {cs_path}")

        input_snapshots[run_id] = sha256_file(cs_path)
        cs = load_json(cs_path)

        geometries = cs if isinstance(cs, list) else cs.get("geometries", cs.get("compatible", []))
        for g in geometries:
            total_scanned += 1
            if _match(g, clauses):
                entry = dict(g)
                entry["_source_run_id"] = run_id
                matching_geometries.append(entry)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_hash,
        "query": query,
        "result_count": len(matching_geometries),
        "total_scanned": total_scanned,
        "matching_geometries": matching_geometries,
        "input_snapshots_hashed": input_snapshots,
        "reproducible": True,
        "execution_ms": elapsed_ms,
    }


def run_query(
    query: str,
    run_ids: list[str],
    anchor_run_id: str | None = None,
    output_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full query pipeline: validate, execute, write results."""
    result = execute_query(query, run_ids, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    # Determine output location
    anchor = anchor_run_id or sorted(run_ids)[0]
    if output_dir:
        out_dir = Path(output_dir)
    else:
        run_dir, _ = validate_and_load_run(anchor, runs_root)
        out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write query result
    query_file = out_dir / f"query_{result['query_id'][:16]}.json"
    _write_json_atomic(query_file, result)

    # Append to query log
    log_file = out_dir / "query_log.jsonl"
    log_entry = {
        "query_id": result["query_id"],
        "query": result["query"],
        "result_count": result["result_count"],
        "execution_ms": result["execution_ms"],
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, sort_keys=True) + "\n")

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5-E: Reproducible query engine over frozen artifacts"
    )
    parser.add_argument("--query", required=True, help="Query string, e.g. \"family == 'edgb' AND mahalanobis_d2 < 5.0\"")
    parser.add_argument("--run-ids", nargs="+", required=True, help="Run IDs to query")
    parser.add_argument("--anchor-run", default=None, help="Anchor run for output location")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--runs-root", default=None, help="Runs root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print result to stdout, don't write files")
    args = parser.parse_args()

    result = run_query(
        query=args.query,
        run_ids=args.run_ids,
        anchor_run_id=args.anchor_run,
        output_dir=args.output,
        runs_root=args.runs_root,
        dry_run=args.dry_run,
    )

    print(f"Query: {result['query']}")
    print(f"Results: {result['result_count']} / {result['total_scanned']} scanned")
    print(f"Time: {result['execution_ms']}ms")


if __name__ == "__main__":
    main()
