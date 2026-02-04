#!/usr/bin/env python3
"""
Run the canonical real-data ringdown pipeline with governance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1], _here.parents[2]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

from basurin_io import (
    get_run_dir,
    get_runs_root,
    require_run_valid,
    sha256_file,
    validate_run_id,
)


@dataclass(frozen=True)
class StagePlan:
    stage_name: str
    command: list[str]


def _parse_band_hz(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("band-hz debe tener formato 'low,high'")
    try:
        low = float(parts[0])
        high = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band-hz debe ser numérico") from exc
    return [low, high]


def _compute_suffix(dt_start_s: float, duration_s: float) -> tuple[str, int, int]:
    dt_ms = int(round(dt_start_s * 1000))
    dur_ms = int(round(duration_s * 1000))
    suffix = f"dt{dt_ms:04d}ms__dur{dur_ms:04d}ms"
    return suffix, dt_ms, dur_ms


def _extract_verdict(payload: dict[str, Any]) -> str | None:
    for key in ("verdict", "overall_verdict", "status", "decision", "result"):
        if key in payload:
            value = payload[key]
            if isinstance(value, dict) and "verdict" in value:
                return value["verdict"]
            if isinstance(value, str):
                return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON inválido en {path}: {exc}") from exc


def _stage_summary_verdict(stage_dir: Path) -> str | None:
    summary_path = stage_dir / "stage_summary.json"
    if not summary_path.exists():
        return None
    payload = _read_json(summary_path)
    return _extract_verdict(payload)


def _run_command(
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
) -> None:
    result = subprocess.run(command, cwd=repo_root, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"falló comando: {' '.join(command)} (exit={result.returncode})")


def _build_stage_plans(
    run_id: str,
    dt_start_s: float,
    duration_s: float,
    window_stage_base: str,
) -> dict[str, StagePlan]:
    suffix, _, _ = _compute_suffix(dt_start_s, duration_s)
    window_stage_name = f"{window_stage_base}__{suffix}"
    observables_stage_name = f"ringdown_real_observables_v0__{suffix}"
    features_stage_name = f"ringdown_real_features_v0__{suffix}"
    inference_stage_name = f"ringdown_real_inference_v0__{suffix}"

    plans = {
        "window": StagePlan(
            stage_name=window_stage_name,
            command=[
                sys.executable,
                "stages/ringdown_real_ringdown_window_v1_stage.py",
                "--run",
                run_id,
                "--dt-start-s",
                str(dt_start_s),
                "--duration-s",
                str(duration_s),
                "--stage-name",
                window_stage_name,
            ],
        ),
        "observables": StagePlan(
            stage_name=observables_stage_name,
            command=[
                sys.executable,
                "stages/ringdown_real_observables_v0_stage.py",
                "--run",
                run_id,
                "--window-stage",
                window_stage_name,
                "--stage-name",
                observables_stage_name,
            ],
        ),
        "features": StagePlan(
            stage_name=features_stage_name,
            command=[
                sys.executable,
                "stages/ringdown_real_features_v0_stage.py",
                "--run",
                run_id,
                "--window-stage",
                window_stage_name,
                "--stage-name",
                features_stage_name,
            ],
        ),
        "inference": StagePlan(
            stage_name=inference_stage_name,
            command=[
                sys.executable,
                "stages/ringdown_real_inference_v0_stage.py",
                "--run",
                run_id,
                "--window-stage",
                window_stage_name,
                "--stage-name",
                inference_stage_name,
            ],
        ),
    }
    return plans


def _maybe_run_ringdown_real_v0(
    run_id: str,
    run_dir: Path,
    repo_root: Path,
    env: dict[str, str],
) -> None:
    stage_dir = run_dir / "ringdown_real_v0"
    outputs_dir = stage_dir / "outputs"
    summary_path = stage_dir / "stage_summary.json"
    outputs_present = outputs_dir.exists() and any(outputs_dir.iterdir())
    if summary_path.exists() and outputs_present:
        return
    command = [sys.executable, "stages/ringdown_real_v0_stage.py", "--run", run_id]
    _run_command(command, repo_root, env)


def _ensure_stage_reexecution_allowed(run_dir: Path, stage_name: str, force: bool) -> None:
    stage_dir = run_dir / stage_name
    if stage_dir.exists() and not force:
        raise RuntimeError(
            f"stage dir {stage_dir} ya existe; usa --force para re-ejecutar"
        )


def _collect_artifact(path: Path, run_dir: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"artifact faltante: {path}")
    rel = path.relative_to(run_dir)
    return {"path": str(rel), "sha256": sha256_file(path)}


def _write_summary(
    run_dir: Path,
    run_id: str,
    event_id: str | None,
    t0_gps: float | None,
    params: dict[str, Any],
    stage_names: dict[str, str],
    verdicts: dict[str, Any],
    artifacts: dict[str, dict[str, dict[str, str]]],
) -> Path:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "params": params,
        "stage_names": stage_names,
        "artifacts": artifacts,
        "verdicts": verdicts,
    }
    if event_id is not None:
        payload["event_id"] = event_id
    if t0_gps is not None:
        payload["t0_gps"] = t0_gps

    summary_path = run_dir / "REAL_PIPELINE_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical real ringdown pipeline")
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument("--event-id", help="event id (summary metadata only)")
    ap.add_argument("--t0-gps", type=float, help="t0_gps (summary metadata only)")
    ap.add_argument("--fs-hz", type=int, default=4096, help="sampling rate metadata")
    ap.add_argument("--dt-start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.25)
    ap.add_argument("--band-hz", default="150,400", type=_parse_band_hz)
    ap.add_argument("--nproc", type=int, default=4, help="metadata only")
    ap.add_argument(
        "--window-stage-base",
        default="ringdown_real_ringdown_window_v1",
        help="base stage name for ringdown window",
    )
    ap.add_argument("--dry-run", action="store_true", help="print plan only")
    ap.add_argument("--force", action="store_true", help="re-run stages if dir exists")

    args = ap.parse_args()

    runs_root = get_runs_root()
    validate_run_id(args.run, runs_root)
    run_dir = get_run_dir(args.run, base_dir=runs_root).resolve()

    try:
        run_valid_payload = require_run_valid(runs_root, args.run)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    plans = _build_stage_plans(
        args.run,
        args.dt_start_s,
        args.duration_s,
        args.window_stage_base,
    )

    if args.dry_run:
        print("[DRY-RUN] Stage names:")
        for key in ("window", "observables", "features", "inference"):
            print(f"- {key}: {plans[key].stage_name}")
        print("[DRY-RUN] Commands:")
        for key in ("window", "observables", "features", "inference"):
            print(f"- {key}: {' '.join(plans[key].command)}")
        return 0

    try:
        _maybe_run_ringdown_real_v0(args.run, run_dir, repo_root, env)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for key in ("window", "observables", "features", "inference"):
        plan = plans[key]
        try:
            _ensure_stage_reexecution_allowed(run_dir, plan.stage_name, args.force)
            _run_command(plan.command, repo_root, env)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    suffix, dt_ms, dur_ms = _compute_suffix(args.dt_start_s, args.duration_s)
    stage_names = {
        "window": plans["window"].stage_name,
        "observables": plans["observables"].stage_name,
        "features": plans["features"].stage_name,
        "inference": plans["inference"].stage_name,
        "suffix": suffix,
        "dt_ms": dt_ms,
        "duration_ms": dur_ms,
    }

    params = {
        "fs_hz": args.fs_hz,
        "band_hz": args.band_hz,
        "dt_start_s": float(args.dt_start_s),
        "duration_s": float(args.duration_s),
        "nproc": args.nproc,
    }

    verdicts: dict[str, Any] = {
        "RUN_VALID": _extract_verdict(run_valid_payload),
        "window": _stage_summary_verdict(run_dir / plans["window"].stage_name),
        "observables": _stage_summary_verdict(run_dir / plans["observables"].stage_name),
        "features": _stage_summary_verdict(run_dir / plans["features"].stage_name),
        "inference": _stage_summary_verdict(run_dir / plans["inference"].stage_name),
    }

    inference_report = (
        run_dir
        / plans["inference"].stage_name
        / "outputs"
        / "inference_report.json"
    )
    if inference_report.exists():
        payload = _read_json(inference_report)
        decision = payload.get("decision")
        if isinstance(decision, dict):
            verdicts["decision.verdict"] = decision.get("verdict")
        else:
            verdicts["decision.verdict"] = payload.get("verdict")

    artifacts = {
        "window": {
            "H1_rd.npz": _collect_artifact(
                run_dir / plans["window"].stage_name / "outputs" / "H1_rd.npz",
                run_dir,
            ),
            "L1_rd.npz": _collect_artifact(
                run_dir / plans["window"].stage_name / "outputs" / "L1_rd.npz",
                run_dir,
            ),
            "segments_rd.json": _collect_artifact(
                run_dir
                / plans["window"].stage_name
                / "outputs"
                / "segments_rd.json",
                run_dir,
            ),
        },
        "observables": {
            "observables.jsonl": _collect_artifact(
                run_dir
                / plans["observables"].stage_name
                / "outputs"
                / "observables.jsonl",
                run_dir,
            ),
        },
        "features": {
            "features.jsonl": _collect_artifact(
                run_dir
                / plans["features"].stage_name
                / "outputs"
                / "features.jsonl",
                run_dir,
            ),
        },
        "inference": {
            "inference_report.json": _collect_artifact(
                run_dir
                / plans["inference"].stage_name
                / "outputs"
                / "inference_report.json",
                run_dir,
            ),
            "contract_verdict.json": _collect_artifact(
                run_dir
                / plans["inference"].stage_name
                / "outputs"
                / "contract_verdict.json",
                run_dir,
            ),
        },
    }

    try:
        _write_summary(
            run_dir,
            args.run,
            args.event_id,
            args.t0_gps,
            params,
            stage_names,
            verdicts,
            artifacts,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
