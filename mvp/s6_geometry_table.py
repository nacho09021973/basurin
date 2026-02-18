#!/usr/bin/env python3
"""Derived extractor: geometry/stability table from s3b multimode outputs.

Stdlib-only utility intended for experiment trees and canonical runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import sha256_file, write_json_atomic

SEED_RE = re.compile(r"(?:^|/)t0_sweep_full_seed(\d+)(?:/|$)")
T0MS_RE = re.compile(r"__t0ms(\d+)")

TSV_HEADER = [
    "seed",
    "t0_ms",
    "s3b_seed_param",
    "lnQ_span",
    "cv_Q",
    "valid_fraction",
    "verdict",
    "flags",
    "path",
]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_out_root() -> Path:
    env = os.environ.get("BASURIN_RUNS_ROOT")
    if env:
        return Path(env).resolve()
    return (Path.cwd() / "runs").resolve()


def _parse_seed_from_path(path_text: str) -> str:
    m = SEED_RE.search(path_text.replace("\\", "/"))
    return m.group(1) if m else "na"


def _parse_t0ms_from_path(path_text: str) -> str:
    m = T0MS_RE.search(path_text.replace("\\", "/"))
    return m.group(1) if m else "na"


def _safe_mode_value(payload: dict[str, Any], mode_label: str, field: str) -> str:
    modes = payload.get("modes")
    if not isinstance(modes, list):
        return "na"
    for row in modes:
        if not isinstance(row, dict):
            continue
        if str(row.get("label")) != mode_label:
            continue
        fit = row.get("fit")
        if not isinstance(fit, dict):
            return "na"
        stability = fit.get("stability")
        if not isinstance(stability, dict):
            return "na"
        val = stability.get(field)
        if isinstance(val, (int, float)):
            return f"{float(val):.12g}"
        return "na"
    return "na"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid JSON object: {path}")
    return data


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _default_out_path(scan_root: Path, out_root: Path, run_id: str) -> Path:
    canonical_run_root = out_root / run_id
    experiment_root = canonical_run_root / "experiment"
    if scan_root.resolve() == canonical_run_root.resolve():
        return scan_root / "derived_geometry_table" / "outputs" / "geometry_table.tsv"
    if scan_root.resolve() == experiment_root.resolve():
        return scan_root / "derived" / "geometry_table.tsv"
    return scan_root / "derived" / "geometry_table.tsv"


def _sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    seed = row["seed"]
    t0 = row["t0_ms"]
    seed_ok = 0 if seed.isdigit() else 1
    t0_ok = 0 if t0.isdigit() else 1
    seed_num = int(seed) if seed_ok == 0 else 0
    t0_num = int(t0) if t0_ok == 0 else 0
    return (seed_ok, seed_num, t0_ok, t0_num, row["path"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build deterministic geometry table from s3b outputs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scan-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jsonl-out", default=None)
    ap.add_argument("--mode-label", default="221")
    ap.add_argument("--no-run-valid-check", action="store_true")
    args = ap.parse_args()

    out_root = _resolve_out_root()
    run_root = out_root / args.run_id

    if not args.no_run_valid_check:
        verdict_path = run_root / "RUN_VALID" / "verdict.json"
        if not verdict_path.exists():
            raise FileNotFoundError(f"RUN_VALID verdict not found: {verdict_path}")
        verdict = _read_json(verdict_path).get("verdict")
        if verdict != "PASS":
            raise RuntimeError(f"RUN_VALID verdict is not PASS: {verdict!r}")

    scan_root = Path(args.scan_root).resolve() if args.scan_root else run_root.resolve()
    if not scan_root.exists():
        raise FileNotFoundError(f"scan root does not exist: {scan_root}")

    if not _within(scan_root, run_root):
        raise RuntimeError(f"scan root must stay under run root: {scan_root} not in {run_root}")

    out_path = Path(args.out).resolve() if args.out else _default_out_path(scan_root, out_root, args.run_id).resolve()
    if not _within(out_path, scan_root):
        raise RuntimeError(f"--out must stay under scan root: {out_path} not in {scan_root}")

    jsonl_path = Path(args.jsonl_out).resolve() if args.jsonl_out else None
    if jsonl_path is not None and not _within(jsonl_path, scan_root):
        raise RuntimeError(f"--jsonl-out must stay under scan root: {jsonl_path} not in {scan_root}")

    mm_paths = sorted(scan_root.rglob("multimode_estimates.json"))

    rows: list[dict[str, str]] = []
    input_records: list[dict[str, str]] = []
    input_seen: set[str] = set()
    rows_with_span = 0

    for mm_path in mm_paths:
        rel_path = mm_path.resolve().relative_to(scan_root).as_posix()
        payload = _read_json(mm_path)

        seed = _parse_seed_from_path(rel_path)
        t0_ms = _parse_t0ms_from_path(rel_path)

        s3b_seed_param = "na"
        s3b_summary = mm_path.parent.parent / "stage_summary.json"
        if s3b_summary.exists():
            summary_payload = _read_json(s3b_summary)
            params = summary_payload.get("parameters")
            if isinstance(params, dict) and isinstance(params.get("seed"), (int, float, str)):
                s3b_seed_param = str(params.get("seed"))
            rel_s3b = s3b_summary.resolve().relative_to(scan_root).as_posix()
            key = f"s3b_stage_summary:{rel_s3b}"
            if key not in input_seen:
                input_seen.add(key)
                input_records.append({
                    "label": "s3b_stage_summary",
                    "path": rel_s3b,
                    "sha256": sha256_file(s3b_summary),
                })

        lnq_span = _safe_mode_value(payload, args.mode_label, "lnQ_span")
        cv_q = _safe_mode_value(payload, args.mode_label, "cv_Q")
        valid_fraction = _safe_mode_value(payload, args.mode_label, "valid_fraction")

        if lnq_span != "na":
            rows_with_span += 1

        results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
        verdict = str(results.get("verdict", "na")) if results else "na"

        flags_value = "na"
        qflags = results.get("quality_flags") if isinstance(results, dict) else None
        if isinstance(qflags, list):
            flags = sorted(str(x) for x in qflags)
            flags_value = ",".join(flags) if flags else ""

        row = {
            "seed": seed,
            "t0_ms": t0_ms,
            "s3b_seed_param": s3b_seed_param,
            "lnQ_span": lnq_span,
            "cv_Q": cv_q,
            "valid_fraction": valid_fraction,
            "verdict": verdict,
            "flags": flags_value,
            "path": rel_path,
        }
        rows.append(row)

        key_mm = f"multimode_estimates:{rel_path}"
        if key_mm not in input_seen:
            input_seen.add(key_mm)
            input_records.append({
                "label": "multimode_estimates",
                "path": rel_path,
                "sha256": sha256_file(mm_path),
            })

    rows.sort(key=_sort_key)
    input_records.sort(key=lambda x: (x["label"], x["path"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(TSV_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(row[key] for key in TSV_HEADER) + "\n")

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    summary_path = out_path.parent / "stage_summary.json"
    summary = {
        "created": _iso_now(),
        "parameters": {
            "run_id": args.run_id,
            "scan_root": str(scan_root),
            "out": str(out_path),
            "jsonl_out": str(jsonl_path) if jsonl_path else None,
            "mode_label": args.mode_label,
            "no_run_valid_check": bool(args.no_run_valid_check),
        },
        "inputs": input_records,
        "counts": {
            "total_files": len(mm_paths),
            "rows_written": len(rows),
            "rows_with_221_lnQ_span": rows_with_span if args.mode_label == "221" else 0,
            "rows_with_mode_lnQ_span": rows_with_span,
        },
    }
    write_json_atomic(summary_path, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
