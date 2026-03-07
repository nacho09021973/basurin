#!/usr/bin/env python3
"""Non-canonical population experiment for beyond-Kerr scores."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import resolve_out_root, sha256_file, validate_run_id, write_json_atomic
from mvp import contracts

EXPERIMENT_STAGE = "experiment/population_kerr"


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _truncate_text(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _resolve_runs_root(runs_root_arg: str | None) -> Path:
    if runs_root_arg:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(runs_root_arg).expanduser().resolve())
    return resolve_out_root("runs")


def _resolve_experiment_dir(out_root: Path, experiment_run_id: str) -> Path:
    validate_run_id(experiment_run_id, out_root)
    host_run_dir = out_root / experiment_run_id
    host_run_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = host_run_dir / "experiment" / f"population_kerr_{_utc_stamp()}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _inventory_runs(
    out_root: Path,
    *,
    experiment_run_id: str,
    batch_prefix: str | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((p for p in out_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        run_id = run_dir.name
        if batch_prefix and not run_id.startswith(batch_prefix):
            continue

        try:
            validate_run_id(run_id, out_root)
            run_id_valid = True
        except Exception:
            run_id_valid = False

        run_valid_path = run_dir / "RUN_VALID" / "verdict.json"
        has_run_valid_pass = False
        if run_valid_path.exists():
            try:
                has_run_valid_pass = _read_json_object(run_valid_path).get("verdict") == "PASS"
            except Exception:
                has_run_valid_pass = False

        s4d_path = run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
        s7_path = run_dir / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json"
        has_s4d = s4d_path.exists()
        has_s7 = s7_path.exists()
        eligible = bool(run_id_valid and has_run_valid_pass and has_s4d)

        if not run_id_valid:
            exclusion_reason = "INVALID_RUN_ID"
        elif not has_run_valid_pass:
            exclusion_reason = "RUN_VALID_NOT_PASS"
        elif not has_s4d:
            exclusion_reason = "MISSING_S4D_KERR_EXTRACTION"
        else:
            exclusion_reason = None

        rows.append(
            {
                "run_id": run_id,
                "eligible": eligible,
                "has_run_valid_pass": bool(has_run_valid_pass),
                "has_s4d_kerr_extraction": bool(has_s4d),
                "has_s7_beyond_kerr_score": bool(has_s7),
                "exclusion_reason": exclusion_reason,
            }
        )

    return {
        "schema_name": "population_kerr_inventory",
        "schema_version": "v1",
        "created_utc": _utc_now_z(),
        "experiment_run_id": experiment_run_id,
        "runs_root": str(out_root),
        "batch_prefix": batch_prefix,
        "n_runs_inspected": len(rows),
        "n_runs_eligible": sum(1 for r in rows if r["eligible"]),
        "runs": rows,
    }


def _run_missing_s7(
    out_root: Path,
    *,
    experiment_run_id: str,
    inventory_payload: dict[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(out_root)

    runs = inventory_payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    for row in runs:
        if not isinstance(row, dict) or not row.get("eligible"):
            continue
        run_id = str(row.get("run_id", ""))
        score_path = out_root / run_id / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json"
        if score_path.exists():
            entries.append(
                {
                    "run_id": run_id,
                    "status": "skipped_existing",
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "command": [],
                }
            )
            continue

        cmd = [sys.executable, "-m", "mvp.s7_beyond_kerr_deviation_score", "--run-id", run_id]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        status = "executed_ok" if proc.returncode == 0 else "failed"
        entries.append(
            {
                "run_id": run_id,
                "status": status,
                "returncode": int(proc.returncode),
                "stdout": _truncate_text(proc.stdout or ""),
                "stderr": _truncate_text(proc.stderr or ""),
                "command": cmd,
            }
        )

    return {
        "schema_name": "population_kerr_run_log",
        "schema_version": "v1",
        "created_utc": _utc_now_z(),
        "experiment_run_id": experiment_run_id,
        "n_runs_eligible": sum(1 for r in runs if isinstance(r, dict) and r.get("eligible")),
        "n_runs_logged": len(entries),
        "n_runs_failed": sum(1 for e in entries if e["status"] == "failed"),
        "entries": entries,
    }


def _aggregate_population(
    out_root: Path,
    *,
    experiment_run_id: str,
    inventory_payload: dict[str, Any],
) -> dict[str, Any]:
    runs = inventory_payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    eligible_run_ids = [str(r.get("run_id")) for r in runs if isinstance(r, dict) and r.get("eligible")]
    rows: list[dict[str, Any]] = []
    for run_id in eligible_run_ids:
        score_path = out_root / run_id / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json"
        if not score_path.exists():
            continue
        try:
            payload = _read_json_object(score_path)
        except Exception:
            continue

        verdict = str(payload.get("verdict", ""))
        if verdict.startswith("SKIPPED_"):
            continue
        epsilon_f = _as_finite_float(payload.get("epsilon_f"))
        epsilon_tau = _as_finite_float(payload.get("epsilon_tau"))
        if epsilon_f is None or epsilon_tau is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "verdict": verdict,
                "epsilon_f": epsilon_f,
                "epsilon_tau": epsilon_tau,
            }
        )

    epsilon_f_values = [r["epsilon_f"] for r in rows]
    epsilon_tau_values = [r["epsilon_tau"] for r in rows]
    n_total = len(eligible_run_ids)
    n_agg = len(rows)
    n_consistent = sum(1 for r in rows if r["verdict"] == "GR_CONSISTENT")
    n_tension = sum(1 for r in rows if r["verdict"] == "GR_TENSION")
    n_inconsistent = sum(1 for r in rows if r["verdict"] == "GR_INCONSISTENT")

    if n_agg == 0:
        mean_f = None
        mean_tau = None
        stderr_f = None
        stderr_tau = None
        z_f = None
        z_tau = None
        population_verdict = "NO_DATA"
    else:
        mean_f = sum(epsilon_f_values) / n_agg
        mean_tau = sum(epsilon_tau_values) / n_agg
        var_f = sum((x - mean_f) ** 2 for x in epsilon_f_values) / max(n_agg - 1, 1)
        var_tau = sum((x - mean_tau) ** 2 for x in epsilon_tau_values) / max(n_agg - 1, 1)
        stderr_f = math.sqrt(var_f / n_agg)
        stderr_tau = math.sqrt(var_tau / n_agg)
        z_f = mean_f / max(stderr_f, 1e-12)
        z_tau = mean_tau / max(stderr_tau, 1e-12)
        if abs(z_f) < 2.0 and abs(z_tau) < 2.0:
            population_verdict = "GR_CONSISTENT_POPULATION"
        elif abs(z_f) < 3.0 and abs(z_tau) < 3.0:
            population_verdict = "GR_TENSION_POPULATION"
        else:
            population_verdict = "GR_INCONSISTENT_POPULATION"

    outlier_runs = [r["run_id"] for r in rows if r["verdict"] == "GR_INCONSISTENT"]
    return {
        "schema_name": "population_kerr_summary",
        "schema_version": "v1",
        "created_utc": _utc_now_z(),
        "experiment_run_id": experiment_run_id,
        "n_events_total": n_total,
        "n_events_aggregated": n_agg,
        "n_events_gr_consistent": n_consistent,
        "n_events_gr_tension": n_tension,
        "n_events_gr_inconsistent": n_inconsistent,
        "epsilon_f_values": epsilon_f_values,
        "epsilon_tau_values": epsilon_tau_values,
        "population_epsilon_f_mean": mean_f,
        "population_epsilon_f_stderr": stderr_f,
        "population_Z_f": z_f,
        "population_epsilon_tau_mean": mean_tau,
        "population_epsilon_tau_stderr": stderr_tau,
        "population_Z_tau": z_tau,
        "population_verdict": population_verdict,
        "outlier_runs": outlier_runs,
    }


def _write_manifest(
    experiment_dir: Path,
    *,
    experiment_run_id: str,
    inventory_path: Path,
    run_log_path: Path,
    summary_path: Path,
) -> Path:
    payload = {
        "schema_name": "population_kerr_manifest",
        "schema_version": "v1",
        "created_utc": _utc_now_z(),
        "experiment_run_id": experiment_run_id,
        "artifacts": {
            "inventory": "inventory.json",
            "run_log": "run_log.json",
            "population_kerr_summary": "population_kerr_summary.json",
        },
        "hashes": {
            "inventory": sha256_file(inventory_path),
            "run_log": sha256_file(run_log_path),
            "population_kerr_summary": sha256_file(summary_path),
        },
    }
    return write_json_atomic(experiment_dir / "manifest.json", payload)


def _log_experiment_paths(out_root: Path, experiment_run_id: str, experiment_dir: Path) -> None:
    ctx = contracts.StageContext(
        run_id=experiment_run_id,
        stage_name=EXPERIMENT_STAGE,
        contract=contracts.StageContract(
            name=EXPERIMENT_STAGE,
            required_inputs=[],
            produced_outputs=[],
            upstream_stages=[],
            check_run_valid=False,
        ),
        out_root=out_root,
        run_dir=out_root / experiment_run_id,
        stage_dir=experiment_dir,
        outputs_dir=experiment_dir,
    )
    contracts.log_stage_paths(ctx)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Population beyond-Kerr experiment over existing runs")
    parser.add_argument("--experiment-run-id", required=True)
    parser.add_argument("--phase", choices=["inventory", "run", "aggregate", "all"], default="all")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--batch-prefix", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    out_root = _resolve_runs_root(args.runs_root)
    validate_run_id(args.experiment_run_id, out_root)
    experiment_dir = _resolve_experiment_dir(out_root, args.experiment_run_id)

    inventory_payload = _inventory_runs(
        out_root,
        experiment_run_id=args.experiment_run_id,
        batch_prefix=args.batch_prefix,
    )
    inventory_path = write_json_atomic(experiment_dir / "inventory.json", inventory_payload)

    if args.phase in {"run", "all"}:
        run_log_payload = _run_missing_s7(
            out_root,
            experiment_run_id=args.experiment_run_id,
            inventory_payload=inventory_payload,
        )
    else:
        run_log_payload = {
            "schema_name": "population_kerr_run_log",
            "schema_version": "v1",
            "created_utc": _utc_now_z(),
            "experiment_run_id": args.experiment_run_id,
            "n_runs_eligible": sum(1 for r in inventory_payload["runs"] if r["eligible"]),
            "n_runs_logged": 0,
            "n_runs_failed": 0,
            "entries": [],
            "note": f"phase={args.phase}: run phase not executed",
        }
    run_log_path = write_json_atomic(experiment_dir / "run_log.json", run_log_payload)

    summary_payload = _aggregate_population(
        out_root,
        experiment_run_id=args.experiment_run_id,
        inventory_payload=inventory_payload,
    )
    summary_path = write_json_atomic(experiment_dir / "population_kerr_summary.json", summary_payload)

    _write_manifest(
        experiment_dir,
        experiment_run_id=args.experiment_run_id,
        inventory_path=inventory_path,
        run_log_path=run_log_path,
        summary_path=summary_path,
    )
    _log_experiment_paths(out_root, args.experiment_run_id, experiment_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
