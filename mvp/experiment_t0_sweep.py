#!/usr/bin/env python3
"""Compatibility wrapper for t0 sweep.

Legacy entrypoint kept for CLI stability. Canonical implementation is
``mvp/experiment_t0_sweep_full.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

EXPERIMENT_STAGE = "experiment/t0_sweep"
RESULTS_NAME = "t0_sweep_results.json"
DEV_TOOL_BANNER = (
    "DEPRECATED WRAPPER: mvp/experiment_t0_sweep.py delega a "
    "mvp/experiment_t0_sweep_full.py"
)


def _build_full_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mvp.experiment_t0_sweep_full",
        "--run-id",
        args.run_id,
        "--phase",
        "run",
        "--seed",
        str(int(args.seed)),
        "--detector",
        args.detector,
        "--n-bootstrap",
        str(int(args.n_bootstrap)),
        "--stage-timeout-s",
        "300",
    ]
    if args.t0_grid_ms:
        cmd += ["--t0-grid-ms", args.t0_grid_ms]
    else:
        if args.t0_start_ms is not None:
            cmd += ["--t0-start-ms", str(int(args.t0_start_ms))]
        if args.t0_stop_ms is not None:
            cmd += ["--t0-stop-ms", str(int(args.t0_stop_ms))]
        if args.t0_step_ms is not None:
            cmd += ["--t0-step-ms", str(int(args.t0_step_ms))]
    if args.atlas_path:
        cmd += ["--atlas-path", args.atlas_path]
    return cmd


def _build_legacy_payload(full_payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    summary = full_payload.get("summary", {}) if isinstance(full_payload.get("summary"), dict) else {}
    points = full_payload.get("points", []) if isinstance(full_payload.get("points"), list) else []
    n_ok = int(summary.get("n_ok", 0))
    n_ins = int(summary.get("n_insufficient", 0))
    n_failed = int(summary.get("n_failed", 0))
    verdict_note = "EXECUTED" if (n_ok + n_ins + n_failed) > 0 else "SKIPPED_UNSUPPORTED"
    return {
        "schema_version": "experiment_t0_sweep_v1",
        "run_id": run_id,
        "source": full_payload.get("source", {}),
        "grid": full_payload.get("grid", {}),
        "mode": "single",
        "summary": {
            "n_points": int(summary.get("n_points", len(points))),
            "n_ok": n_ok,
            "n_insufficient": n_ins,
            "n_failed": n_failed,
            "best_point": summary.get("best_point", {}),
            "verdict": verdict_note,
        },
        "points": points,
    }


def run_t0_sweep(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    cmd = _build_full_cmd(args)
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if cp.returncode != 0:
        msg = (cp.stderr or cp.stdout).strip()
        raise RuntimeError(f"t0_sweep_full failed exit={cp.returncode}: {msg}")

    full_results = (
        out_root
        / args.run_id
        / "experiment"
        / f"t0_sweep_full_seed{int(args.seed)}"
        / "outputs"
        / "t0_sweep_full_results.json"
    )
    if not full_results.exists():
        raise FileNotFoundError(
            "Input faltante para wrapper deprecado. "
            f"Ruta esperada exacta: {full_results}. "
            f"Comando para regenerar upstream: {' '.join(cmd)}."
        )

    full_payload = json.loads(full_results.read_text(encoding="utf-8"))
    legacy_payload = _build_legacy_payload(full_payload, args.run_id)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run_id, EXPERIMENT_STAGE, base_dir=out_root)
    out_path = outputs_dir / RESULTS_NAME
    write_json_atomic(out_path, legacy_payload)

    stage_summary = {
        "stage": EXPERIMENT_STAGE,
        "run_id": args.run_id,
        "verdict": "PASS",
        "created": utc_now_iso(),
        "results": {
            "n_points": legacy_payload["summary"]["n_points"],
            "n_ok": legacy_payload["summary"]["n_ok"],
            "n_insufficient": legacy_payload["summary"]["n_insufficient"],
            "n_failed": legacy_payload["summary"]["n_failed"],
            "experiment_verdict": legacy_payload["summary"]["verdict"],
            "delegated_to": "mvp.experiment_t0_sweep_full",
            "results_sha256": sha256_file(out_path),
        },
    }
    stage_summary_path = write_stage_summary(stage_dir, stage_summary)
    manifest_path = write_manifest(stage_dir, {"t0_sweep_results": out_path, "stage_summary": stage_summary_path})

    print(f"OUT_ROOT={out_root}")
    print(f"STAGE_DIR={stage_dir}")
    print(f"OUTPUTS_DIR={outputs_dir}")
    print(f"STAGE_SUMMARY={stage_summary_path}")
    print(f"MANIFEST={manifest_path}")

    return legacy_payload, stage_summary, out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Experiment (deprecated wrapper): deterministic t0 sweep over existing s2 outputs. "
            "Delegates to mvp/experiment_t0_sweep_full.py"
        )
    )
    ap.add_argument("--run-id", "--run", dest="run_id", required=True)
    ap.add_argument("--t0-grid-ms", default=None)
    ap.add_argument("--t0-start-ms", type=int, default=None)
    ap.add_argument("--t0-stop-ms", type=int, default=None)
    ap.add_argument("--t0-step-ms", type=int, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mode", choices=["single", "multimode"], default="single")
    ap.add_argument("--detector", choices=["H1", "L1", "auto"], default="auto")
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument("--quiet", action="store_true", help="Suppress deprecation banner")
    args = ap.parse_args()

    if args.mode != "single":
        print("[experiment_t0_sweep] WARNING: wrapper supports only single-mode output; forcing canonical run phase", file=sys.stderr)

    try:
        if not args.quiet:
            print(f"[experiment_t0_sweep] {DEV_TOOL_BANNER}", file=sys.stderr)
        run_t0_sweep(args)
        return 0
    except Exception as exc:
        print(f"[experiment_t0_sweep] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
