#!/usr/bin/env python3
"""Ex3 wrapper: run experiment_t0_sweep_full over a list of golden events."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
)

EXPERIMENT_STAGE = "experiment/ex3_t0_golden"
RESULTS_NAME = "t0_sweep_golden_results.json"


def _parse_events(raw: str) -> list[str]:
    events = [e.strip() for e in str(raw).split(",") if e.strip()]
    if not events:
        raise ValueError("--golden-events vacío; usa formato GW150914,GW151226,...")
    return events


def _resolve_event_run_map(out_root: Path, parent_run_id: str) -> tuple[dict[str, str], Path]:
    agg_path = out_root / parent_run_id / "s5_aggregate" / "outputs" / "aggregate.json"
    if not agg_path.exists():
        candidates = sorted(str(p) for p in out_root.glob("*/s5_aggregate/outputs/aggregate.json"))
        cands = "\n".join(f"  - {c}" for c in candidates[:20]) or "  - (none)"
        raise FileNotFoundError(
            "Input faltante para ex3_t0_golden.\n"
            f"Ruta canónica esperada: runs/{parent_run_id}/s5_aggregate/outputs/aggregate.json\n"
            f"Ruta esperada exacta: {agg_path}\n"
            "Comando exacto para regenerar upstream: "
            f"python mvp/pipeline.py multi --events <E1,E2,...> --atlas-path <ATLAS_PATH> --agg-run-id {parent_run_id}\n"
            f"Candidatos detectados:\n{cands}"
        )

    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise TypeError(f"aggregate.json inválido: 'events' debe ser list ({agg_path})")

    mapping: dict[str, str] = {}
    for row in events:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if event_id and run_id:
            mapping[event_id] = run_id
    return mapping, agg_path


def _load_source_run_valid_sha(out_root: Path, event_id: str, run_id: str) -> str:
    verdict_path = out_root / run_id / "RUN_VALID" / "verdict.json"
    if not verdict_path.exists():
        raise FileNotFoundError(
            "ERROR: [experiment_ex3_golden_sweep] Source run "
            f"'{run_id}' for event '{event_id}'\n"
            "  does not have RUN_VALID == PASS.\n"
            f"  Expected: {verdict_path}\n"
            "  Regenerate: "
            f"python mvp/pipeline.py single --event-id {event_id} --atlas-path <atlas>"
        )

    verdict_payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    if str(verdict_payload.get("verdict", "")).strip() != "PASS":
        raise RuntimeError(
            "ERROR: [experiment_ex3_golden_sweep] Source run "
            f"'{run_id}' for event '{event_id}'\n"
            "  does not have RUN_VALID == PASS.\n"
            f"  Expected: {verdict_path}\n"
            "  Regenerate: "
            f"python mvp/pipeline.py single --event-id {event_id} --atlas-path <atlas>"
        )

    return sha256_file(verdict_path)


def _build_cmd(
    *,
    script_path: Path,
    event_run_id: str,
    out_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--run",
        event_run_id,
        "--runs-root",
        str(out_root),
        "--t0-start-ms",
        str(args.t0_start_ms),
        "--t0-stop-ms",
        str(args.t0_stop_ms),
        "--t0-step-ms",
        str(args.t0_step_ms),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--seed",
        str(args.seed),
        "--detector",
        str(args.detector),
        "--stage-timeout-s",
        str(args.stage_timeout_s),
        "--max-retries-per-pair",
        str(args.max_retries_per_pair),
        "--resume-batch-size",
        str(args.resume_batch_size),
    ]
    if args.t0_grid_ms:
        cmd.extend(["--t0-grid-ms", args.t0_grid_ms])
    if args.resume_missing:
        cmd.append("--resume-missing")
    if args.atlas_path:
        cmd.extend(["--atlas-path", args.atlas_path])
    return cmd


def _read_event_result(out_root: Path, event_run_id: str, seed: int) -> dict[str, Any]:
    result_path = (
        out_root
        / event_run_id
        / "experiment"
        / f"t0_sweep_full_seed{int(seed)}"
        / "outputs"
        / "t0_sweep_full_results.json"
    )
    if not result_path.exists():
        raise FileNotFoundError(
            "Output faltante de experiment_t0_sweep_full.\n"
            f"Ruta esperada exacta: {result_path}\n"
            "Comando exacto para regenerar upstream: "
            f"python mvp/experiment_t0_sweep_full.py --run {event_run_id} --seed {int(seed)}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "path": str(result_path),
        "sha256": sha256_file(result_path),
        "payload": payload,
    }


def run_experiment(args: argparse.Namespace) -> int:
    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    event_ids = _parse_events(args.golden_events)
    event_run_map, aggregate_path = _resolve_event_run_map(out_root, args.run_id)

    missing = [ev for ev in event_ids if ev not in event_run_map]
    if missing:
        known = sorted(event_run_map.keys())
        known_msg = ", ".join(known) if known else "(none)"
        raise KeyError(
            "Eventos solicitados no presentes en aggregate.json. "
            f"missing={missing}; disponibles={known_msg}"
        )

    source_run_valid_sha: dict[str, str] = {}
    for event_id in event_ids:
        if not re.match(r"^[A-Za-z0-9_-]+$", event_id):
            raise ValueError(
                "ERROR: [experiment_ex3_golden_sweep] Invalid event_id for filename usage: "
                f"'{event_id}'. Allowed pattern: ^[A-Za-z0-9_-]+$"
            )
        run_id = event_run_map[event_id]
        source_run_valid_sha[event_id] = _load_source_run_valid_sha(out_root, event_id, run_id)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run_id, EXPERIMENT_STAGE, base_dir=out_root)
    per_event_dir = outputs_dir / "per_event"
    per_event_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve().parent / "experiment_t0_sweep_full.py"

    results: list[dict[str, Any]] = []
    for event_id in event_ids:
        event_run_id = event_run_map[event_id]

        cmd = _build_cmd(script_path=script_path, event_run_id=event_run_id, out_root=out_root, args=args)
        env = dict(os.environ)
        env["BASURIN_RUNS_ROOT"] = str(out_root)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "experiment_t0_sweep_full falló (fail-fast). "
                f"event_id={event_id} event_run={event_run_id} returncode={proc.returncode}\n"
                f"CMD={' '.join(cmd)}\nSTDOUT={proc.stdout[-1000:]}\nSTDERR={proc.stderr[-1000:]}"
            )

        event_res = _read_event_result(out_root, event_run_id, int(args.seed))
        per_event_payload = {
            "schema_version": "ex3_per_event_v1",
            "parent_run_id": args.run_id,
            "event_id": event_id,
            "source_run_id": event_run_id,
            "source_run_valid_sha256": source_run_valid_sha[event_id],
            "event_run_id": event_run_id,
            "seed": int(args.seed),
            "source": {
                "t0_sweep_full_results": event_res["path"],
                "t0_sweep_full_results_sha256": event_res["sha256"],
            },
            "science_diagnostics": {
                "n_points": len(event_res["payload"].get("results", [])) if isinstance(event_res["payload"], dict) else None,
                "keys": sorted(event_res["payload"].keys()) if isinstance(event_res["payload"], dict) else [],
            },
            "result": event_res["payload"],
        }
        per_event_path = per_event_dir / f"{event_id}_t0_sweep.json"
        write_json_atomic(per_event_path, per_event_payload)

        results.append(
            {
                "event_id": event_id,
                "event_run_id": event_run_id,
                "output": str(per_event_path.relative_to(stage_dir)),
                "output_sha256": sha256_file(per_event_path),
                "science_diagnostics": per_event_payload["science_diagnostics"],
            }
        )

    agg_path = outputs_dir / RESULTS_NAME
    agg_payload = {
        "schema_version": "ex3_golden_sweep_v1",
        "stage": EXPERIMENT_STAGE,
        "parent_run_id": args.run_id,
        "parent_aggregate_sha256": sha256_file(aggregate_path),
        "golden_events": event_ids,
        "created": utc_now_iso(),
        "seed": int(args.seed),
        "results": results,
    }
    write_json_atomic(agg_path, agg_payload)

    summary_path = stage_dir / "stage_summary.json"
    summary_payload = {
        "schema_version": "ex3_t0_golden_summary_v1",
        "stage": EXPERIMENT_STAGE,
        "run_id": args.run_id,
        "is_experiment": True,
        "results": {
            "n_events": len(results),
            "event_ids": event_ids,
        },
        "outputs": [
            {
                "path": str(agg_path.relative_to(stage_dir)),
                "sha256": sha256_file(agg_path),
            }
        ]
        + [
            {
                "path": item["output"],
                "sha256": item["output_sha256"],
            }
            for item in results
        ],
    }
    write_json_atomic(summary_path, summary_payload)

    manifest_path = stage_dir / "manifest.json"
    manifest_payload = {
        "schema_version": "mvp_manifest_v1",
        "stage": EXPERIMENT_STAGE,
        "run_id": args.run_id,
        "artifacts": {
            "golden_results": str(agg_path.relative_to(stage_dir)),
            "stage_summary": "stage_summary.json",
        },
        "hashes": {
            "golden_results": sha256_file(agg_path),
            "stage_summary": sha256_file(summary_path),
        },
        "per_event": {
            item["event_id"]: {
                "path": item["output"],
                "sha256": item["output_sha256"],
            }
            for item in results
        },
    }
    write_json_atomic(manifest_path, manifest_payload)

    print(f"OUT_ROOT={out_root}")
    print(f"STAGE_DIR={stage_dir}")
    print(f"OUTPUTS_DIR={outputs_dir}")
    print(f"STAGE_SUMMARY={summary_path}")
    print(f"MANIFEST={manifest_path}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ex3 wrapper: golden events t0 sweep")
    ap.add_argument("--run", "--run-id", dest="run_id", required=True)
    ap.add_argument("--golden-events", required=True, help="CSV list, e.g. GW150914,GW151226")
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument("--t0-grid-ms", default=None)
    ap.add_argument("--t0-start-ms", type=int, default=0)
    ap.add_argument("--t0-stop-ms", type=int, default=30)
    ap.add_argument("--t0-step-ms", type=int, default=5)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--detector", choices=["auto", "H1", "L1"], default="auto")
    ap.add_argument("--stage-timeout-s", type=int, default=300)
    ap.add_argument("--resume-missing", action="store_true")
    ap.add_argument("--max-retries-per-pair", type=int, default=2)
    ap.add_argument("--resume-batch-size", type=int, default=50)
    return ap


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
