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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import sha256_file, write_json_atomic

SEED_RE = re.compile(r"(?:^|/)t0_sweep_full_seed(\d+)(?:/|$)")
T0MS_RE = re.compile(r"__t0ms(\d{1,6})")

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


def _resolve_runs_root() -> Path:
    env = os.environ.get("BASURIN_RUNS_ROOT")
    if env:
        return Path(env).resolve()
    return (Path.cwd() / "runs").resolve()


def _iter_stage_summaries(scan_root: Path) -> list[Path]:
    pattern = "**/s3b_multimode_estimates/stage_summary.json"
    found = [p.absolute() for p in scan_root.glob(pattern) if p.is_file() and not has_symlink_ancestor(p, scan_root)]
    return sorted(found, key=lambda p: p.as_posix())


def has_symlink_ancestor(path: Path, root: Path) -> bool:
    """Return True when ``path`` has any symlink ancestor between ``root`` and the file.

    The check walks from ``path.parent`` upwards and rejects deterministically if
    ``path`` is not under ``root``.
    """
    path_abs = path.absolute()
    root_abs = root.absolute()
    try:
        path_abs.relative_to(root_abs)
    except ValueError:
        return True

    current = path_abs.parent
    while True:
        if current.is_symlink():
            return True
        if current == root_abs:
            return False
        if current == current.parent:
            return True
        current = current.parent


def parse_context(path_abs: Path, scan_root_abs: Path) -> tuple[str, str]:
    stage_summary_path = path_abs.parent.parent / "stage_summary.json"
    try:
        summary_payload = _read_json(stage_summary_path)
    except FileNotFoundError:
        summary_payload = None
    except Exception as exc:
        print(f"WARN invalid stage summary at {stage_summary_path.resolve()}: {exc}", file=sys.stderr)
        summary_payload = None

    seed_from_params = "na"
    t0_from_params = "na"
    if isinstance(summary_payload, dict):
        params = summary_payload.get("params")
        if isinstance(params, dict):
            seed_raw = params.get("seed")
            t0_raw = params.get("t0_ms")
            if isinstance(seed_raw, int):
                seed_from_params = str(seed_raw)
            if isinstance(t0_raw, int):
                t0_from_params = str(t0_raw)

    path_text = path_abs.as_posix()
    t0_match = T0MS_RE.search(path_text)
    t0_ms = str(int(t0_match.group(1))) if t0_match else "na"
    if t0_from_params != "na":
        t0_ms = t0_from_params

    seed_match = SEED_RE.search(path_text)
    if seed_match:
        seed = seed_match.group(1)
        if seed_from_params != "na":
            seed = seed_from_params
        return seed, t0_ms

    if seed_from_params != "na":
        return seed_from_params, t0_ms

    seed_scan_root = re.match(r"^t0_sweep_full_seed(\d+)$", scan_root_abs.name)
    if seed_scan_root:
        return seed_scan_root.group(1), t0_ms

    return "na", t0_ms


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
    _ = scan_root
    return (out_root / run_id / "experiment" / "derived" / "geometry_table.tsv").resolve()


def _sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    seed = row["seed"]
    t0 = row["t0_ms"]
    seed_is_na = 1 if seed == "na" else 0
    t0_is_na = 1 if t0 == "na" else 0
    seed_num = int(seed) if seed.isdigit() else 0
    t0_num = int(t0) if t0.isdigit() else 0
    return (seed_is_na, seed_num, t0_is_na, t0_num, row["path"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build deterministic geometry table from s3b outputs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--base-runs-root", default=None)
    ap.add_argument("--scan-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-path", default=None)
    ap.add_argument("--jsonl-out", default=None)
    ap.add_argument("--mode-label", default="221")
    ap.add_argument("--no-run-valid-check", action="store_true")
    args = ap.parse_args()

    runs_root_arg = args.runs_root if args.runs_root else args.base_runs_root
    runs_root = Path(runs_root_arg).resolve() if runs_root_arg else _resolve_runs_root()
    run_root = (runs_root / args.run_id).resolve()

    if not args.no_run_valid_check:
        verdict_path = run_root / "RUN_VALID" / "verdict.json"
        if not verdict_path.exists():
            print(f"RUN_VALID verdict not found: {verdict_path.resolve()}", file=sys.stderr)
            raise SystemExit(2)
        try:
            verdict = _read_json(verdict_path).get("verdict")
        except Exception as exc:
            print(f"failed to read RUN_VALID verdict at {verdict_path.resolve()}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        if verdict != "PASS":
            print(f"RUN_VALID verdict is not PASS: {verdict!r} at {verdict_path.resolve()}", file=sys.stderr)
            raise SystemExit(2)

    scan_root = Path(args.scan_root).resolve() if args.scan_root else run_root.resolve()
    if not scan_root.exists():
        raise FileNotFoundError(f"scan root does not exist: {scan_root.resolve()}")

    if not _within(scan_root, run_root):
        raise RuntimeError(f"scan root must stay under run root: {scan_root} not in {run_root}")

    out_arg = args.out_path if args.out_path else args.out
    out_path = Path(out_arg).resolve() if out_arg else _default_out_path(scan_root, runs_root, args.run_id).resolve()
    if not _within(out_path, run_root):
        raise RuntimeError(f"--out-path/--out must stay under run root: {out_path.resolve()} not in {run_root.resolve()}")

    jsonl_path = Path(args.jsonl_out).resolve() if args.jsonl_out else None
    if jsonl_path is not None and not _within(jsonl_path, run_root):
        raise RuntimeError(f"--jsonl-out must stay under run root: {jsonl_path.resolve()} not in {run_root.resolve()}")

    stage_pattern = "**/s3b_multimode_estimates/stage_summary.json"
    s3b_summaries = _iter_stage_summaries(scan_root)

    rows: list[dict[str, str]] = []
    input_records: list[dict[str, str]] = []
    input_seen: set[str] = set()
    rows_with_span = 0
    skipped_invalid = 0
    skipped_missing_payload = 0

    for s3b_summary in s3b_summaries:
        s3b_root = s3b_summary.parent
        mm_path = s3b_root / "outputs" / "multimode_estimates.json"
        rel_path = mm_path.resolve().relative_to(scan_root).as_posix()
        seed, t0_ms = parse_context(mm_path.resolve(), scan_root.resolve())
        s3b_seed_param = "na"
        try:
            summary_payload = _read_json(s3b_summary)
        except Exception as exc:
            print(f"WARN invalid stage summary at {s3b_summary.resolve()}: {exc}", file=sys.stderr)
        else:
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

        if not mm_path.exists():
            skipped_missing_payload += 1
            rows.append({
                "seed": seed,
                "t0_ms": t0_ms,
                "s3b_seed_param": s3b_seed_param,
                "lnQ_span": "na",
                "cv_Q": "na",
                "valid_fraction": "na",
                "verdict": "na",
                "flags": "missing_multimode_estimates_json",
                "path": rel_path,
            })
            continue

        try:
            payload = _read_json(mm_path)
        except Exception as exc:
            skipped_invalid += 1
            print(f"WARN skip invalid multimode json at {mm_path.resolve()}: {exc}", file=sys.stderr)
            continue

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

        rows.append({
            "seed": seed,
            "t0_ms": t0_ms,
            "s3b_seed_param": s3b_seed_param,
            "lnQ_span": lnq_span,
            "cv_Q": cv_q,
            "valid_fraction": valid_fraction,
            "verdict": verdict,
            "flags": flags_value,
            "path": rel_path,
        })

        key_mm = f"multimode_estimates:{rel_path}"
        if key_mm not in input_seen:
            input_seen.add(key_mm)
            input_records.append({
                "label": "multimode_estimates",
                "path": rel_path,
                "sha256": sha256_file(mm_path),
            })

    if len(rows) == 0:
        print(
            "No geometry rows discovered: "
            f"scan_root_abs={scan_root.resolve()} pattern={stage_pattern}. "
            "Hint: the layout likely does not contain s3b_multimode_estimates/stage_summary.json",
            file=sys.stderr,
        )
        raise SystemExit(2)

    rows.sort(key=_sort_key)
    input_records.sort(key=lambda x: (x["label"], x["path"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=out_path.parent, encoding="utf-8", newline="\n") as tf:
        tf.write("\t".join(TSV_HEADER) + "\n")
        for row in rows:
            tf.write("\t".join(row[key] for key in TSV_HEADER) + "\n")
        tmp_name = tf.name
    os.replace(tmp_name, out_path)

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
            "total_files": len(s3b_summaries),
            "rows_written": len(rows),
            "files_skipped_invalid": skipped_invalid,
            "files_skipped_missing_payload": skipped_missing_payload,
            "rows_with_221_lnQ_span": rows_with_span if args.mode_label == "221" else 0,
            "rows_with_mode_lnQ_span": rows_with_span,
        },
    }
    write_json_atomic(summary_path, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
