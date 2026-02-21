#!/usr/bin/env python3
"""Full deterministic t0 sweep using isolated subruns (s3 -> s3b -> s4c)."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import statistics
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from mvp.oracles.oracle_v1_plateau import WindowMetrics, load_window_metrics_from_subrun, run_oracle_v1
from mvp.oracles.t0_input_schema import WindowSummaryV1Error, map_sweep_point_to_window_summary_v1

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

EXPERIMENT_STAGE = "experiment/t0_sweep_full"
RESULTS_NAME = "t0_sweep_full_results.json"
S3_NO_VALID_ESTIMATE_MSG = "No detector produced a valid estimate"
SEED_RE = re.compile(r"t0_sweep_full_seed(\d+)")
T0MS_RE = re.compile(r"__t0ms(\d+)")


def run_cmd(cmd: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, check=False, capture_output=True, text=True, timeout=timeout)


def _parse_grid(args: argparse.Namespace) -> list[int]:
    if args.t0_grid_ms:
        vals = [int(round(float(x.strip()))) for x in args.t0_grid_ms.split(",") if x.strip()]
        if not vals:
            raise ValueError("--t0-grid-ms provided but empty")
        return vals

    start = int(args.t0_start_ms)
    stop = int(args.t0_stop_ms)
    step = int(args.t0_step_ms)
    if step <= 0:
        raise ValueError("--t0-step-ms must be > 0")
    if stop < start:
        raise ValueError("--t0-stop-ms must be >= --t0-start-ms")
    return list(range(start, stop + 1, step))


def _pick_detector(outputs_dir: Path, detector: str) -> tuple[str, Path]:
    available: dict[str, Path] = {}
    for det in ("H1", "L1"):
        p = outputs_dir / f"{det}_rd.npz"
        if p.exists():
            available[det] = p

    if not available:
        raise FileNotFoundError(f"No H1/L1 *_rd.npz files in {outputs_dir}")

    if detector != "auto":
        if detector not in available:
            raise FileNotFoundError(f"Requested detector {detector} not found in {outputs_dir}")
        return detector, available[detector]

    if "H1" in available:
        return "H1", available["H1"]
    return "L1", available["L1"]


def _load_npz(npz_path: Path, window_meta: dict[str, Any]) -> tuple[Any, float]:
    import numpy as np

    data = np.load(npz_path)
    if "strain" not in data:
        raise RuntimeError(f"corrupt s2 NPZ: missing strain in {npz_path}")
    strain = np.asarray(data["strain"], dtype=float)

    fs = None
    for key in ("sample_rate_hz", "fs"):
        if key in data:
            fs = float(np.asarray(data[key]).flat[0])
            break
    if fs is None:
        for key in ("sample_rate_hz", "fs"):
            if key in window_meta:
                fs = float(window_meta[key])
                break

    if strain.ndim != 1 or strain.size < 16 or not np.all(np.isfinite(strain)):
        raise RuntimeError("corrupt s2 NPZ: invalid strain")
    if fs is None or not math.isfinite(fs) or fs <= 0:
        raise RuntimeError("corrupt s2 NPZ: missing sample_rate_hz")
    return strain, fs


def _subrun_id(base_run_id: str, t0_ms: int) -> str:
    return f"{base_run_id}__t0ms{int(t0_ms):04d}"


def _init_subrun_run_valid(subrun_dir: Path) -> None:
    rv_path = subrun_dir / "RUN_VALID" / "verdict.json"
    rv_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_path, {"verdict": "PASS"})


def _init_parent_run_valid(runs_root: Path, run_id: str) -> None:
    rv_path = runs_root / run_id / "RUN_VALID" / "verdict.json"
    if rv_path.exists():
        return
    rv_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_path, {"verdict": "PASS"})


def _write_subrun_shadow_s2(
    subrun_dir: Path,
    detector: str,
    trimmed: Any,
    fs: float,
    t0_ms: int,
    offset_samples: int,
    original_npz: Path,
    original_sha: str,
    *,
    base_window_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    import numpy as np

    _init_subrun_run_valid(subrun_dir)

    s2_stage_dir = subrun_dir / "s2_ringdown_window"
    s2_out = s2_stage_dir / "outputs"
    s2_out.mkdir(parents=True, exist_ok=True)
    subrun_npz = s2_out / f"{detector}_rd.npz"
    np.savez(subrun_npz, strain=trimmed.astype(np.float64), sample_rate_hz=np.array([fs], dtype=np.float64))
    subrun_sha = sha256_file(subrun_npz)
    base = dict(base_window_meta or {})
    t_start = base.get("t_start_gps")
    if t_start is not None:
        try:
            t_start = float(t_start) + (float(t0_ms) / 1000.0)
        except (TypeError, ValueError):
            t_start = None
    duration_s = float(trimmed.size) / float(fs)
    window_meta = {
        "event_id": base.get("event_id", "unknown"),
        "t0_source": base.get("t0_source"),
        "sample_rate_hz": float(fs),
        "detectors": [detector],
        "n_samples": int(trimmed.size),
        "duration_s": duration_s,
        "t0_offset_ms": int(t0_ms),
        "offset_samples": int(offset_samples),
        "derived_from": str(original_npz),
    }
    if t_start is not None:
        window_meta["t_start_gps"] = t_start
        window_meta["t_end_gps"] = t_start + duration_s
    meta_path = s2_out / "window_meta.json"
    write_json_atomic(meta_path, window_meta)
    window_meta_sha = sha256_file(meta_path)

    stage_summary = {
        "stage": "s2_ringdown_window",
        "run": subrun_dir.name,
        "verdict": "PASS",
        "outputs": [
            {"path": f"s2_ringdown_window/outputs/{detector}_rd.npz", "sha256": subrun_sha},
            {"path": "s2_ringdown_window/outputs/window_meta.json", "sha256": window_meta_sha},
        ],
        "results": {
            "detector": detector,
            "fs_hz": float(fs),
            "t0_offset_ms": int(t0_ms),
            "offset_samples": int(offset_samples),
            "npz_original_sha256": original_sha,
            "npz_trimmed_sha256": subrun_sha,
            "window_meta_sha256": window_meta_sha,
            "derived_from": str(original_npz),
        },
    }
    write_stage_summary(s2_stage_dir, stage_summary)

    manifest = {
        "schema_version": "mvp_manifest_v1",
        "artifacts": {f"{detector}_rd": f"outputs/{detector}_rd.npz", "window_meta": "outputs/window_meta.json"},
        "hashes": {f"{detector}_rd": subrun_sha, "window_meta": window_meta_sha},
        "provenance": {
            "source_stage": "s2_ringdown_window",
            "npz_original_sha256": original_sha,
            "npz_trimmed_sha256": subrun_sha,
            "t0_offset_ms": int(t0_ms),
            "offset_samples": int(offset_samples),
        },
    }
    write_json_atomic(s2_stage_dir / "manifest.json", manifest)
    return {
        "npz_trimmed_sha256": subrun_sha,
        "offset_samples": int(offset_samples),
        "window_meta_sha256": window_meta_sha,
        "window_meta_path": str(meta_path),
    }


def _require_subrun_window_meta(subrun_dir: Path) -> Path:
    path = subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing required subrun provenance: {path}")
    return path


def _expected_subrun_window_meta_path(subrun_runs_root: Path, subrun_id: str) -> Path:
    return subrun_runs_root / subrun_id / "s2_ringdown_window" / "outputs" / "window_meta.json"


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_fact(path: Path, *, include_sha: bool = True) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": bool(exists),
    }
    if include_sha:
        payload["sha256"] = sha256_file(path) if exists else None
    return payload


def _write_preflight_report_or_abort(
    *,
    run_id: str,
    out_root: Path,
    base_root: Path,
    scan_root_abs: Path,
    base_run_dir: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    run_valid = base_run_dir / "RUN_VALID" / "verdict.json"
    s1_strain = base_run_dir / "s1_fetch_strain" / "outputs" / "strain.npz"
    s3_estimates = base_run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"

    rv_verdict = "MISSING"
    if run_valid.exists():
        try:
            rv_payload = json.loads(run_valid.read_text(encoding="utf-8"))
            rv_verdict = "PASS" if str(rv_payload.get("verdict")) == "PASS" else "FAIL"
        except Exception:
            rv_verdict = "FAIL"

    checks = {
        "RUN_VALID": {
            "path": str(run_valid),
            "exists": run_valid.exists(),
            "verdict": rv_verdict,
        },
        "s1_strain_npz": _artifact_fact(s1_strain),
        "s3_estimates_json": _artifact_fact(s3_estimates),
    }

    can_run = True
    reason = "ok"
    missing_path = None
    if not checks["RUN_VALID"]["exists"]:
        can_run = False
        reason = "missing RUN_VALID verdict"
        missing_path = checks["RUN_VALID"]["path"]
    elif checks["RUN_VALID"]["verdict"] != "PASS":
        can_run = False
        reason = f"RUN_VALID verdict is {checks['RUN_VALID']['verdict']}"
        missing_path = checks["RUN_VALID"]["path"]
    elif not checks["s1_strain_npz"]["exists"]:
        can_run = False
        reason = "missing base s1 strain NPZ"
        missing_path = checks["s1_strain_npz"]["path"]

    report = {
        "run_id": run_id,
        "runs_root_abs": str(out_root),
        "repo_root_abs": str(base_root),
        "scan_root_abs": str(scan_root_abs),
        "base_artifacts": checks,
        "tooling": {
            "python": sys.executable,
            "cwd": str(Path.cwd()),
            "env_BASURIN_RUNS_ROOT": os.environ.get("BASURIN_RUNS_ROOT"),
        },
        "decision": {
            "can_run": bool(can_run),
            "reason": reason,
        },
    }
    write_json_atomic(preflight_path, report)
    if not can_run:
        msg = f"[experiment_t0_sweep_full] preflight failed: reason={reason} path={missing_path}"
        if reason == "missing base s1 strain NPZ":
            repo_root = Path(__file__).resolve().parents[1]
            s1_cmd = (
                f"python mvp/s1_fetch_strain.py --run {run_id} --event-id <EVENT_ID> --detectors H1,L1 "
                f"--hdf5-root {repo_root / 'data' / 'losc'}"
            )
            msg += (
                "\nPrimero ejecuta s1_fetch_strain; los HDF5 externos deben estar en "
                f"{repo_root / 'data' / 'losc' / '<EVENT_ID>'}/ ..."
                f"\nEjemplo: {s1_cmd}"
                f"\nComprobación: find {repo_root / 'data' / 'losc' / '<EVENT_ID>'} \( -iname '*.hdf5' -o -iname '*.h5' \) -type f"
            )
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    return report


def _new_subrun_trace(subrun_id: str, seed: int, t0_ms: int, stages: list[list[str]]) -> dict[str, Any]:
    return {
        "subrun_id": subrun_id,
        "seed": int(seed),
        "t0_ms": int(t0_ms),
        "planned_stages": [Path(cmd[1]).stem for cmd in stages if len(cmd) > 1],
        "stages": [],
        "aborted": False,
        "abort_stage": None,
        "exception": None,
        "contract": {"missing": []},
    }


def _output_presence_map(subrun_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    expected = {
        "s2_ringdown_window": {"window_meta": subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"},
        "s3_ringdown_estimates": {"estimates": subrun_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"},
        "s3b_multimode_estimates": {
            "multimode_estimates": subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
        },
        "s4c_kerr_consistency": {"kerr_consistency": subrun_dir / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json"},
    }
    stage_expected = expected.get(stage, {})
    return {key: _artifact_fact(path) for key, path in stage_expected.items()}


def _write_subrun_trace(path: Path, trace: dict[str, Any]) -> None:
    write_json_atomic(path, trace)


def _extract_s3b(payload: dict[str, Any] | None) -> dict[str, Any]:
    empty = {
        "verdict": "INSUFFICIENT_DATA",
        "has_221": False,
        "ln_f_220": None,
        "ln_Q_220": None,
        "ln_f_221": None,
        "ln_Q_221": None,
    }
    if not payload:
        return empty

    results = payload.get("results", {})
    modes = payload.get("modes", [])
    by_label = {m.get("label"): m for m in modes if isinstance(m, dict)}
    m220 = by_label.get("220", {})
    m221 = by_label.get("221", {})
    return {
        "verdict": results.get("verdict", "INSUFFICIENT_DATA"),
        "has_221": m221.get("ln_f") is not None and m221.get("ln_Q") is not None,
        "ln_f_220": m220.get("ln_f"),
        "ln_Q_220": m220.get("ln_Q"),
        "ln_f_221": m221.get("ln_f"),
        "ln_Q_221": m221.get("ln_Q"),
    }


def _extract_s4c(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "verdict": "FAIL",
            "kerr_consistent_95": None,
            "chi_best": None,
            "d2_min": None,
            "delta_logfreq": None,
            "delta_logQ": None,
        }

    source = payload.get("source", {})
    mm_verdict = source.get("multimode_verdict")
    verdict = "OK" if mm_verdict == "OK" else ("INSUFFICIENT_DATA" if mm_verdict == "INSUFFICIENT_DATA" else "FAIL")
    return {
        "verdict": verdict,
        "kerr_consistent_95": payload.get("kerr_consistent"),
        "chi_best": payload.get("chi_best"),
        "d2_min": payload.get("d2_min"),
        "delta_logfreq": payload.get("delta_logfreq"),
        "delta_logQ": payload.get("delta_logQ"),
    }


def build_subrun_stage_cmds(
    python: str,
    s2_script: str,
    s3_script: str,
    s3b_script: str,
    s4c_script: str,
    subrun_runs_root: str,
    subrun_id: str,
    event_id: str,
    dt_start_s: float,
    duration_s: float,
    strain_npz: str,
    n_bootstrap: int,
    s3b_seed: int,
    atlas_path: str | None,
) -> list[list[str]]:
    """
    Pure helper: build the per-subrun stage command lists.

    NOTE: This is intentionally pure and deterministic: it only assembles argv.
    Seed propagation to s3b is explicit via --seed <s3b_seed>.
    """
    s3_estimates = f"runs/{subrun_id}/s3_ringdown_estimates/outputs/estimates.json"
    cmds = [
        [
            python,
            s2_script,
            "--run",
            subrun_id,
            "--run-id",
            subrun_id,
            "--runs-root",
            subrun_runs_root,
            "--event-id",
            event_id,
            "--dt-start-s",
            str(float(dt_start_s)),
            "--duration-s",
            str(float(duration_s)),
            "--strain-npz",
            strain_npz,
        ],
        [python, s3_script, "--run", subrun_id, "--run-id", subrun_id, "--runs-root", subrun_runs_root],
        [
            python,
            s3b_script,
            "--run-id",
            subrun_id,
            "--runs-root",
            subrun_runs_root,
            "--s3-estimates",
            s3_estimates,
            "--n-bootstrap",
            str(int(n_bootstrap)),
            "--seed",
            str(int(s3b_seed)),
        ],
    ]

    atlas_stage = [python, s4c_script, "--run-id", subrun_id, "--runs-root", subrun_runs_root]
    cmds.append(atlas_stage)

    needs_atlas = any(len(cmd) > 1 and Path(cmd[1]).stem == "s4c_kerr_consistency" for cmd in cmds)
    if needs_atlas:
        if not atlas_path:
            print("stage=s4c_kerr_consistency requiere --atlas-path", file=sys.stderr)
            raise SystemExit(2)
        atlas_stage.extend(["--atlas-path", atlas_path])

    return cmds


def build_subrun_execution_plan(
    *,
    subrun_dir: Path,
    python: str,
    s2_script: str,
    s3_script: str,
    s3b_script: str,
    s4c_script: str,
    subrun_runs_root: Path,
    subrun_id: str,
    event_id: str,
    dt_start_s: float,
    duration_s: float,
    strain_npz: str,
    n_bootstrap: int,
    s3b_seed: int,
    atlas_path: str,
) -> dict[str, Any]:
    """Pure plan for one subrun including local provenance path expectations."""
    window_meta = _expected_subrun_window_meta_path(subrun_runs_root, subrun_id)
    return {
        "subrun_id": subrun_id,
        "expected_inputs": {
            "s2_window_meta": str(window_meta),
        },
        "commands": build_subrun_stage_cmds(
            python=python,
            s2_script=s2_script,
            s3_script=s3_script,
            s3b_script=s3b_script,
            s4c_script=s4c_script,
            subrun_runs_root=str(subrun_runs_root),
            subrun_id=subrun_id,
            event_id=event_id,
            dt_start_s=dt_start_s,
            duration_s=duration_s,
            strain_npz=strain_npz,
            n_bootstrap=n_bootstrap,
            s3b_seed=s3b_seed,
            atlas_path=atlas_path,
        ),
    }


def execute_subrun_stages_or_abort(
    *,
    stages: list[list[str]],
    subrun_dir: Path,
    env: dict[str, str],
    stage_timeout_s: int,
    run_cmd_fn: Callable[[list[str], dict[str, str], int], Any],
    point: dict[str, Any],
    subrun_runs_root: Path | None = None,
    trace: dict[str, Any] | None = None,
    trace_path: Path | None = None,
) -> tuple[bool, bool]:
    """Execute per-subrun commands with mandatory post-s2 window_meta precheck.

    Returns ``(failed, skip_to_insufficient)``.
    """
    failed = False
    skip_to_insufficient = False
    for cmd in stages:
        stage_stem = Path(cmd[1]).stem if len(cmd) > 1 else ""
        stage_trace = {
            "stage": stage_stem,
            "started": True,
            "finished": False,
            "exit_code": None,
            "outputs_present": {},
        }
        if trace is not None:
            trace["stages"].append(stage_trace)
            if trace_path is not None:
                _write_subrun_trace(trace_path, trace)
        try:
            cp = run_cmd_fn(cmd, env, stage_timeout_s)
        except Exception as exc:
            point["messages"].append(f"subprocess error for {' '.join(cmd)}: {type(exc).__name__}: {exc}")
            stage_trace["finished"] = True
            stage_trace["exit_code"] = -1
            stage_trace["outputs_present"] = _output_presence_map(subrun_dir, stage_stem)
            if trace is not None:
                trace["aborted"] = True
                trace["abort_stage"] = stage_stem
                trace["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if trace_path is not None:
                    _write_subrun_trace(trace_path, trace)
            failed = True
            break

        stage_trace["finished"] = True
        stage_trace["exit_code"] = int(getattr(cp, "returncode", 1))
        stage_trace["outputs_present"] = _output_presence_map(subrun_dir, stage_stem)
        if trace is not None and trace_path is not None:
            _write_subrun_trace(trace_path, trace)
        if int(getattr(cp, "returncode", 1)) != 0:
            stderr = (getattr(cp, "stderr", "") or "").strip()
            is_s3 = stage_stem == "s3_ringdown_estimates"
            if is_s3 and S3_NO_VALID_ESTIMATE_MSG in stderr:
                point["status"] = "INSUFFICIENT_DATA"
                point["quality_flags"].append("s3_no_valid_estimate")
                if stderr:
                    point["messages"].append(stderr)
                skip_to_insufficient = True
                break
            point["messages"].append(f"stage failed rc={cp.returncode}: {' '.join(cmd)}")
            if stderr:
                point["messages"].append(stderr)
            if trace is not None:
                trace["aborted"] = True
                trace["abort_stage"] = stage_stem
                trace["exception"] = {
                    "type": "CalledProcessError",
                    "message": stderr or f"stage returned {cp.returncode}",
                }
                if trace_path is not None:
                    _write_subrun_trace(trace_path, trace)
            failed = True
            break

        if stage_stem == "s2_ringdown_window":
            effective_runs_root = subrun_runs_root if subrun_runs_root is not None else subrun_dir.parent
            expected_meta_path = _expected_subrun_window_meta_path(effective_runs_root, subrun_dir.name)
            if trace is not None:
                trace["s2_window_meta_check"] = {
                    "expected_window_meta_path": str(expected_meta_path),
                    "found": False,
                }
                if trace_path is not None:
                    _write_subrun_trace(trace_path, trace)
            try:
                meta_path = _require_subrun_window_meta(subrun_dir)
                point["quality_flags"].append(f"subrun_window_meta={meta_path}")
                if trace is not None:
                    trace["s2_window_meta_check"] = {
                        "expected_window_meta_path": str(expected_meta_path),
                        "found": True,
                    }
                    if trace_path is not None:
                        _write_subrun_trace(trace_path, trace)
            except FileNotFoundError:
                subrun_id = str((trace or {}).get("subrun_id", subrun_dir.name))
                seed = (trace or {}).get("seed")
                t0_ms = (trace or {}).get("t0_ms")
                msg = (
                    "[experiment_t0_sweep_full] ERROR: missing window_meta after s2 "
                    f"subrun_id={subrun_id} seed={seed} t0_ms={t0_ms} "
                    f"expected_path={expected_meta_path}"
                )
                point["messages"].append(msg)
                print(msg, file=sys.stderr)
                raise SystemExit(2)
    return failed, skip_to_insufficient


def compute_experiment_paths(run_id: str) -> tuple[Path, Path, Path]:
    """Resolve output root and directories for ``experiment/t0_sweep_full``.

    Returns ``(out_root, stage_dir, subruns_root)``.
    """
    out_root = resolve_out_root("runs")
    stage_dir = out_root / run_id / "experiment" / "t0_sweep_full"
    subruns_root = stage_dir / "runs"
    return out_root, stage_dir, subruns_root


def enforce_isolated_runsroot(out_root: Path, run_id: str) -> None:
    """Abort if BASURIN_RUNS_ROOT aliases the legacy global experiment tree.

    We intentionally reject symlinked/aliased seed-specific roots because those
    can collapse multiple seeds into the same physical inode tree.
    """
    env_root = os.environ.get("BASURIN_RUNS_ROOT")
    if not env_root:
        return

    requested_root = Path(env_root).expanduser()
    if not requested_root.is_absolute():
        requested_root = (Path.cwd() / requested_root)
    requested_root = requested_root.absolute()
    resolved_root = out_root.resolve()

    if resolved_root != requested_root:
        raise RuntimeError(
            "BASURIN_RUNS_ROOT must not traverse symlinks: "
            f"requested={requested_root} resolved={resolved_root}"
        )

    legacy_stage = (Path.cwd() / "runs" / run_id / "experiment" / "t0_sweep_full").resolve()
    seed_stage = (out_root / run_id / "experiment" / "t0_sweep_full").resolve()
    default_runs_root = (Path.cwd() / "runs").resolve()
    if resolved_root != default_runs_root and seed_stage == legacy_stage:
        raise RuntimeError(
            "BASURIN_RUNS_ROOT aliases legacy runs/<run_id>/experiment/t0_sweep_full; "
            "use a physical directory per seed instead of symlink/alias paths"
        )


def resolve_experiment_paths(run_id: str, *, out_root: Path | None = None) -> tuple[Path, Path, Path]:
    """Backwards-compatible wrapper returning ``(stage_dir, outputs_dir, subruns_root)``."""
    if out_root is None:
        _, stage_dir, subruns_root = compute_experiment_paths(run_id)
    else:
        stage_dir = out_root / run_id / "experiment" / "t0_sweep_full"
        subruns_root = stage_dir / "runs"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    subruns_root.mkdir(parents=True, exist_ok=True)
    return stage_dir, outputs_dir, subruns_root


def ensure_seed_runsroot_layout(runsroot: Path, run_id: str) -> Path:
    """Ensure ``runsroot/<run_id>`` exists as a real directory (never symlink)."""
    run_root = runsroot / run_id
    if run_root.exists() and run_root.is_symlink():
        target = os.readlink(run_root)
        raise RuntimeError(f"runsroot/{run_id} must be a real directory; found symlink to {target}")
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def run_t0_sweep_full(
    args: argparse.Namespace,
    *,
    run_cmd_fn: Callable[[list[str], dict[str, str], int], Any] = run_cmd,
) -> tuple[dict[str, Any], Path]:
    out_root = resolve_out_root("runs").resolve()
    base_root = Path(args.base_runs_root).expanduser().resolve()
    base_run_dir = out_root / args.run_id
    scan_root_abs = (
        Path(args.scan_root).expanduser().resolve()
        if getattr(args, "scan_root", None)
        else (base_run_dir / "experiment").resolve()
    )
    stage_dir = out_root / args.run_id / "experiment" / f"t0_sweep_full_seed{int(args.seed)}"
    subruns_root = stage_dir / "runs"
    trace_path = out_root / args.run_id / "experiment" / "derived" / "run_trace.json"
    preflight_path = out_root / args.run_id / "experiment" / "derived" / "preflight_report.json"
    trace_payload: dict[str, Any] = {
        "phase": "run",
        "run_id": args.run_id,
        "runs_root": str(out_root),
        "repo_root": str(base_root),
        "scan_root_abs": str(scan_root_abs),
        "seed": int(args.seed),
        "t0_grid_ms": str(args.t0_grid_ms) if args.t0_grid_ms is not None else None,
        "creating_seed_dir": str(stage_dir.resolve()),
        "created_seed_dir": False,
        "exception": None,
    }

    print(
        "[experiment_t0_sweep_full] "
        f"phase=run run_id={args.run_id} runs_root_abs={out_root} "
        f"scan_root_abs={scan_root_abs} "
        f"seed={int(args.seed)} t0_grid_ms={trace_payload['t0_grid_ms']}",
        file=sys.stderr,
    )

    try:
        enforce_isolated_runsroot(out_root, args.run_id)
        ensure_seed_runsroot_layout(out_root, args.run_id)
        validate_run_id(args.run_id, out_root)
        validate_run_id(args.run_id, base_root)
        _init_parent_run_valid(out_root, args.run_id)
        require_run_valid(out_root, args.run_id)
        _write_preflight_report_or_abort(
            run_id=args.run_id,
            out_root=out_root,
            base_root=base_root,
            scan_root_abs=scan_root_abs,
            base_run_dir=base_run_dir,
            preflight_path=preflight_path,
        )

        print(f"[experiment_t0_sweep_full] creating_seed_dir={stage_dir.resolve()}", file=sys.stderr)
        stage_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir = stage_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        subruns_root.mkdir(parents=True, exist_ok=True)
        trace_payload["created_seed_dir"] = stage_dir.exists()
        print(f"[experiment_t0_sweep_full] seed_dir_exists={str(stage_dir.exists()).lower()}", file=sys.stderr)
        write_json_atomic(trace_path, trace_payload)

        s2_dir = base_run_dir / "s2_ringdown_window"
        s2_manifest = s2_dir / "manifest.json"
        if not s2_manifest.exists():
            raise FileNotFoundError(f"Missing s2 manifest required for phase=run: {s2_manifest}")

        s2_outputs = s2_dir / "outputs"
        detector, source_npz = _pick_detector(s2_outputs, args.detector)

        window_meta_path = s2_outputs / "window_meta.json"
        window_meta = json.loads(window_meta_path.read_text(encoding="utf-8")) if window_meta_path.exists() else {}
        strain, fs = _load_npz(source_npz, window_meta)
        source_sha = sha256_file(source_npz)
        grid = _parse_grid(args)

        python = sys.executable
        s2_script = str((_here.parent / "s2_ringdown_window.py").resolve())
        s3_script = str((_here.parent / "s3_ringdown_estimates.py").resolve())
        s3b_script = str((_here.parent / "s3b_multimode_estimates.py").resolve())
        s4c_script = str((_here.parent / "s4c_kerr_consistency.py").resolve())

        points: list[dict[str, Any]] = []
        n_ok = n_ins = n_failed = 0

        for t0_ms in grid:
            subrun_id = _subrun_id(args.run_id, int(t0_ms))
            point = {
                "t0_ms": int(t0_ms),
                "subrun_id": subrun_id,
                "status": "FAILED_POINT",
                "s3b": {
                    "verdict": "INSUFFICIENT_DATA",
                    "has_221": False,
                    "ln_f_220": None,
                    "ln_Q_220": None,
                    "ln_f_221": None,
                    "ln_Q_221": None,
                },
                "s4c": {
                    "verdict": "FAIL",
                    "kerr_consistent_95": None,
                    "chi_best": None,
                    "d2_min": None,
                    "delta_logfreq": None,
                    "delta_logQ": None,
                },
                "quality_flags": [],
                "messages": [],
            }

            if t0_ms < 0:
                point["status"] = "SKIPPED_POINT_EARLY"
                point["messages"].append("t0 offset < 0 unsupported")
                points.append(point)
                continue

            offset_samples = int(round((float(t0_ms) / 1000.0) * fs))
            if offset_samples >= strain.size:
                point["status"] = "INSUFFICIENT_DATA"
                point["messages"].append("offset exceeds window length")
                n_ins += 1
                points.append(point)
                continue

            subrun_dir = subruns_root / subrun_id
            subrun_trace_path = subrun_dir / "derived" / "subrun_trace.json"
            if subrun_dir.exists():
                shutil.rmtree(subrun_dir)
            _init_subrun_run_valid(subrun_dir)

            base_dt_start_s = float(window_meta.get("dt_start_s", 0.0))
            base_duration_s = float(window_meta.get("duration_s", float(strain.size) / float(fs)))
            s1_strain_npz = base_run_dir / "s1_fetch_strain" / "outputs" / "strain.npz"
            if not s1_strain_npz.exists():
                raise FileNotFoundError(
                    "Missing base s1 strain NPZ required for subrun s2: "
                    f"{s1_strain_npz}"
                )

            env = os.environ.copy()
            env["BASURIN_RUNS_ROOT"] = str(subruns_root)

            stages = build_subrun_stage_cmds(
                python=python,
                s2_script=s2_script,
                s3_script=s3_script,
                s3b_script=s3b_script,
                s4c_script=s4c_script,
                subrun_runs_root=str(subruns_root),
                subrun_id=subrun_id,
                event_id=str(window_meta.get("event_id", "GW150914")),
                dt_start_s=base_dt_start_s + (float(t0_ms) / 1000.0),
                duration_s=base_duration_s - (float(t0_ms) / 1000.0),
                strain_npz=str(s1_strain_npz),
                n_bootstrap=int(args.n_bootstrap),
                s3b_seed=int(args.seed),
                atlas_path=str(args.atlas_path),
            )
            subrun_trace = _new_subrun_trace(subrun_id, int(args.seed), int(t0_ms), stages)
            failed, skip_to_insufficient = execute_subrun_stages_or_abort(
                stages=stages,
                subrun_dir=subrun_dir,
                env=env,
                stage_timeout_s=args.stage_timeout_s,
                run_cmd_fn=run_cmd_fn,
                point=point,
                subrun_runs_root=subruns_root,
                trace=subrun_trace,
                trace_path=subrun_trace_path,
            )

            contract_ok = _check_subrun_contract(subrun_dir, subrun_trace)
            if not contract_ok:
                failed = True
                point["status"] = "FAILED_POINT"
                point["messages"].append("subrun contract failed: missing local provenance outputs")
                for miss in subrun_trace["contract"]["missing"]:
                    point["messages"].append(f"missing {miss['key']}: {miss['path']}")
            _write_subrun_trace(subrun_trace_path, subrun_trace)

            s3b_payload = _read_json_if_exists(
                subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
            )
            s4c_payload = _read_json_if_exists(subrun_dir / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json")
            point["s3b"] = _extract_s3b(s3b_payload)
            point["s4c"] = _extract_s4c(s4c_payload)
            if skip_to_insufficient:
                point["s4c"] = {
                    "verdict": "INSUFFICIENT_DATA",
                    "kerr_consistent_95": None,
                    "chi_best": None,
                    "d2_min": None,
                    "delta_logfreq": None,
                    "delta_logQ": None,
                }

            if skip_to_insufficient:
                n_ins += 1
            elif failed:
                point["status"] = "FAILED_POINT"
                n_failed += 1
            elif point["s3b"]["verdict"] == "INSUFFICIENT_DATA":
                point["status"] = "INSUFFICIENT_DATA"
                n_ins += 1
            elif point["s3b"]["verdict"] == "OK" and point["s4c"]["verdict"] == "OK":
                point["status"] = "OK"
                n_ok += 1
            else:
                point["status"] = "FAILED_POINT"
                n_failed += 1

            if s3b_payload:
                point["quality_flags"].extend(s3b_payload.get("results", {}).get("quality_flags", []))
                point["messages"].extend(s3b_payload.get("results", {}).get("messages", []))

            # Optional backward-compatible postprocess payload for the t0 oracle input contract.
            try:
                point["window_summary_v1"] = map_sweep_point_to_window_summary_v1(
                    {"subruns_root": str(subruns_root)},
                    point,
                )
            except WindowSummaryV1Error as exc:
                point["window_summary_v1"] = None
                point["messages"].append(str(exc))

            points.append(point)

        best_t0 = None
        best_value = None
        for p in points:
            if p.get("s4c", {}).get("verdict") != "OK":
                continue
            d2 = p.get("s4c", {}).get("d2_min")
            if d2 is None:
                continue
            try:
                val = float(d2)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val):
                continue
            if best_value is None or val < best_value:
                best_value = val
                best_t0 = int(p["t0_ms"])

        results = {
            "schema_version": "experiment_t0_sweep_full_v1",
            "run_id": args.run_id,
            "atlas_path": str(args.atlas_path),
            "subruns_root": str(subruns_root),
            "source": {
                "stage": "s2_ringdown_window",
                "detector": detector,
                "fs_hz": float(fs),
                "npz_original_sha256": source_sha,
                "t0_base_s": window_meta.get("t0_start_s"),
            },
            "grid": {
                "t0_offsets_ms": [int(x) for x in grid],
                "interpreted_as": "offsets_from_s2_window_start_nonnegative_only",
            },
            "summary": {
                "n_points": len(points),
                "n_ok": n_ok,
                "n_insufficient": n_ins,
                "n_failed": n_failed,
                "best_point": {
                    "t0_ms": best_t0,
                    "metric": "min_d2",
                    "value": best_value,
                },
            },
            "points": points,
        }

        out_path = outputs_dir / RESULTS_NAME
        write_json_atomic(out_path, results)

        stage_summary = {
            "stage": EXPERIMENT_STAGE,
            "run_id": args.run_id,
            "verdict": "PASS",
            "results": {
                "n_points": len(points),
                "n_ok": n_ok,
                "n_insufficient": n_ins,
                "n_failed": n_failed,
                "results_sha256": sha256_file(out_path),
            },
        }
        write_stage_summary(stage_dir, stage_summary)
        write_manifest(
            stage_dir,
            {"t0_sweep_full_results": out_path},
            extra={"subruns_root": str(subruns_root), "source_npz_sha256": source_sha},
        )
        trace_payload["created_seed_dir"] = stage_dir.exists()
        trace_payload["exception"] = None
        write_json_atomic(trace_path, trace_payload)
        return results, out_path
    except Exception as exc:
        trace_payload["created_seed_dir"] = stage_dir.exists()
        trace_payload["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_json_atomic(trace_path, trace_payload)
        raise


def _parse_int_csv(raw: str | None) -> list[int]:
    if raw is None:
        return []
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted(set(vals))


def _resolve_runs_root_arg(runs_root: str | None) -> Path:
    if runs_root:
        return Path(runs_root).expanduser().resolve()
    env_root = os.environ.get("BASURIN_RUNS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return resolve_out_root("runs").resolve()


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        tmp.write(rendered)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _pair_key(seed: int, t0_ms: int) -> str:
    return f"seed={int(seed)},t0_ms={int(t0_ms)}"


def _inventory_path(runs_root_abs: Path, base_run: str) -> Path:
    return runs_root_abs / base_run / "experiment" / "derived" / "sweep_inventory.json"


def _oracle_inventory_path(runs_root_abs: Path, base_run: str) -> Path:
    return runs_root_abs / base_run / "experiment" / "derived" / "t0_sweep_inventory.json"


def _oracle_report_path(runs_root_abs: Path, base_run: str) -> Path:
    return runs_root_abs / base_run / "experiment" / "derived" / "oracle_report.json"


def _best_window_path(runs_root_abs: Path, base_run: str) -> Path:
    return runs_root_abs / base_run / "experiment" / "derived" / "best_window.json"


def _diagnose_path(runs_root_abs: Path, base_run: str) -> Path:
    return runs_root_abs / base_run / "experiment" / "derived" / "diagnose_report.json"


def _backfill_window_meta_report_path(runs_root_abs: Path, base_run: str) -> Path:
    return runs_root_abs / base_run / "experiment" / "derived" / "backfill_window_meta_report.json"


def _discover_subrun_dirs(scan_root_abs: Path) -> list[Path]:
    return sorted([p for p in scan_root_abs.glob("**/*__t0ms*") if p.is_dir()], key=lambda p: p.as_posix())


def _parse_t0_offset_ms_from_subrun(subrun_dir: Path) -> int:
    match = T0MS_RE.search(subrun_dir.name)
    if not match:
        return 0
    return int(match.group(1))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_results_artifact(seed_dir: Path) -> dict[str, Any] | None:
    outputs = seed_dir / "outputs" / RESULTS_NAME
    if not outputs.exists():
        return None
    payload = json.loads(outputs.read_text(encoding="utf-8"))
    stage_summary = _read_json_if_exists(seed_dir / "stage_summary.json") or {}
    manifest = _read_json_if_exists(seed_dir / "manifest.json") or {}
    return {
        "seed_dir": seed_dir,
        "results_path": outputs,
        "results": payload,
        "stage_summary": stage_summary,
        "manifest": manifest,
    }


def _build_windows_from_sweep_results(
    artifacts: list[dict[str, Any]],
    *,
    sigma_floor_f: float,
    sigma_floor_tau: float,
    scale_floor_f: float,
    scale_floor_tau: float,
    gate_221: bool,
) -> tuple[list[tuple[Path, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for art in artifacts:
        seed_match = SEED_RE.search(art["seed_dir"].name)
        seed = int(seed_match.group(1)) if seed_match else None
        points = art["results"].get("points", [])
        for point in points:
            t0 = int(point.get("t0_ms", 0))
            s3b = point.get("s3b", {})
            ln_f_220 = _safe_float(s3b.get("ln_f_220"))
            ln_q_220 = _safe_float(s3b.get("ln_Q_220"))
            ln_f_221 = _safe_float(s3b.get("ln_f_221"))
            ln_q_221 = _safe_float(s3b.get("ln_Q_221"))
            status = str(point.get("status", ""))
            flags_raw = s3b.get("quality_flags", point.get("quality_flags", []))
            flags = [str(flag) for flag in flags_raw if isinstance(flag, (str, int, float))]
            if ln_f_220 is None or ln_q_220 is None:
                continue
            grouped.setdefault(t0, []).append(
                {
                    "seed": seed,
                    "status": status,
                    "ln_f_220": ln_f_220,
                    "ln_q_220": ln_q_220,
                    "ln_f_221": ln_f_221,
                    "ln_q_221": ln_q_221,
                    "has_221": bool(s3b.get("has_221")),
                    "flags_221": [flag for flag in flags if "221" in flag],
                }
            )

    windows: list[tuple[Path, Any]] = []
    for t0 in sorted(grouped):
        rows = grouped[t0]
        f_values = [math.exp(r["ln_f_220"]) for r in rows]
        q_values = [math.exp(r["ln_q_220"]) for r in rows]
        tau_values = [q_values[i] / (math.pi * f_values[i]) for i in range(len(rows))]
        f_median = statistics.median(f_values)
        tau_median = statistics.median(tau_values)
        f_sigma = max(statistics.pstdev(f_values) if len(f_values) > 1 else 0.0, sigma_floor_f)
        tau_sigma = max(statistics.pstdev(tau_values) if len(tau_values) > 1 else 0.0, sigma_floor_tau)
        cv_f = f_sigma / max(abs(f_median), scale_floor_f)
        cv_tau = tau_sigma / max(abs(tau_median), scale_floor_tau)
        has_any_221 = any(r["has_221"] for r in rows)
        flags_221 = sorted({flag for r in rows for flag in r.get("flags_221", [])})
        warnings: list[str] = []
        if not has_any_221:
            warnings.append("WARN_221_MISSING")
        if gate_221 and not has_any_221:
            continue
        windows.append(
            (
                artifacts[0]["seed_dir"] / f"aggregate__t0ms{t0:04d}",
                WindowMetrics(
                    t0=float(t0),
                    T=1.0,
                    f_median=f_median,
                    f_sigma=f_sigma,
                    tau_median=tau_median,
                    tau_sigma=tau_sigma,
                    cond_number=1.0,
                    delta_bic=20.0,
                    p_ljungbox=1.0,
                    n_samples=256,
                    snr=30.0,
                    flags_221=flags_221,
                ),
            )
        )
        audit_rows.append(
            {
                "t0_ms": t0,
                "n_seeds": len(rows),
                "f_median": f_median,
                "tau_median": tau_median,
                "f_sigma": f_sigma,
                "tau_sigma": tau_sigma,
                "cv_f": cv_f,
                "cv_tau": cv_tau,
                "warnings": warnings,
            }
        )

    return windows, audit_rows


def _build_backfilled_window_meta(subrun_dir: Path) -> dict[str, Any]:
    return {
        "event_id": "unknown",
        "t0_source": "backfill_window_meta",
        "sample_rate_hz": None,
        "detectors": [],
        "n_samples": None,
        "duration_s": None,
        "t0_offset_ms": _parse_t0_offset_ms_from_subrun(subrun_dir),
        "offset_samples": 0,
        "derived_from": "backfill_window_meta",
    }


def run_backfill_window_meta_phase(args: argparse.Namespace) -> dict[str, Any]:
    runs_root_abs = _resolve_runs_root_arg(getattr(args, "runs_root", None))
    base_run = args.run_id
    base_run_dir = runs_root_abs / base_run
    scan_root_abs = Path(args.scan_root).expanduser().resolve() if getattr(args, "scan_root", None) else (base_run_dir / "experiment").resolve()

    touched: list[dict[str, str]] = []
    scanned_count = 0
    backfilled_count = 0
    skipped_count = 0

    for subrun_dir in _discover_subrun_dirs(scan_root_abs):
        scanned_count += 1
        run_valid_path = subrun_dir / "RUN_VALID" / "verdict.json"
        s3_estimates = subrun_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
        s3b_payload = subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
        window_meta_path = subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"

        if window_meta_path.exists():
            skipped_count += 1
            continue
        if not (run_valid_path.exists() and s3_estimates.exists() and s3b_payload.exists()):
            skipped_count += 1
            continue

        verdict = _read_json_if_exists(run_valid_path) or {}
        if str(verdict.get("verdict")) != "PASS":
            skipped_count += 1
            continue

        window_meta_path.parent.mkdir(parents=True, exist_ok=True)
        window_meta = _build_backfilled_window_meta(subrun_dir)
        write_json_atomic(window_meta_path, window_meta)
        window_meta_sha = sha256_file(window_meta_path)

        s2_stage_dir = subrun_dir / "s2_ringdown_window"
        summary_path = s2_stage_dir / "stage_summary.json"
        if not summary_path.exists():
            write_stage_summary(
                s2_stage_dir,
                {
                    "stage": "s2_ringdown_window",
                    "run": subrun_dir.name,
                    "verdict": "PASS",
                    "outputs": [{"path": "s2_ringdown_window/outputs/window_meta.json", "sha256": window_meta_sha}],
                    "results": {
                        "window_meta_sha256": window_meta_sha,
                        "source": "backfill_window_meta",
                    },
                },
            )

        manifest_path = s2_stage_dir / "manifest.json"
        if not manifest_path.exists():
            write_manifest(
                s2_stage_dir,
                {
                    "window_meta": window_meta_path,
                },
                extra={
                    "provenance": {"source": "backfill_window_meta"},
                },
            )

        backfilled_count += 1
        if len(touched) < 100:
            touched.append({"subrun": str(subrun_dir), "window_meta_sha256": window_meta_sha})

    payload = {
        "schema_version": "t0_sweep_backfill_window_meta_v1",
        "run_id": base_run,
        "scan_root": str(scan_root_abs),
        "scanned_count": scanned_count,
        "backfilled_count": backfilled_count,
        "skipped_count": skipped_count,
        "touched_subruns": touched,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json_dump(_backfill_window_meta_report_path(runs_root_abs, base_run), payload)
    return payload


def _contract_expectations(subrun_dir: Path) -> dict[str, Path]:
    return {
        "window_meta": subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json",
        "estimates": subrun_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        "multimode_estimates": subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
    }


def _check_subrun_contract(subrun_dir: Path, trace: dict[str, Any]) -> bool:
    missing: list[dict[str, str]] = []
    for key, path in _contract_expectations(subrun_dir).items():
        if not path.exists():
            missing.append({"key": key, "path": str(path)})
    trace["contract"] = {
        "missing": missing,
        "ok": len(missing) == 0,
    }
    return len(missing) == 0


def run_diagnose_phase(args: argparse.Namespace) -> dict[str, Any]:
    runs_root_abs = _resolve_runs_root_arg(getattr(args, "runs_root", None))
    base_run = args.run_id
    base_run_dir = runs_root_abs / base_run
    scan_root_abs = Path(args.scan_root).expanduser().resolve() if getattr(args, "scan_root", None) else (base_run_dir / "experiment").resolve()
    generated_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        counts_missing: dict[str, int] = {
            "window_meta": 0,
            "s3_estimates": 0,
            "s3b_payload": 0,
            "RUN_VALID": 0,
        }
        exception_counts: dict[str, int] = {}
        worst: list[dict[str, Any]] = []
        subrun_dirs = _discover_subrun_dirs(scan_root_abs)

        for subrun_dir in subrun_dirs:
            missing_keys: list[str] = []
            checks = {
                "window_meta": subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json",
                "s3_estimates": subrun_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
                "s3b_payload": subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
                "RUN_VALID": subrun_dir / "RUN_VALID" / "verdict.json",
            }
            for key, path in checks.items():
                if not path.exists():
                    counts_missing[key] += 1
                    missing_keys.append(key)

            derived_trace = subrun_dir / "derived" / "subrun_trace.json"
            trace = _read_json_if_exists(derived_trace)
            error_msg = None
            if trace:
                exc = trace.get("exception") or {}
                msg = str(exc.get("message", "")).strip()
                if msg:
                    exception_counts[msg] = exception_counts.get(msg, 0) + 1
                    error_msg = msg

            if (missing_keys or error_msg) and len(worst) < 20:
                worst.append(
                    {
                        "path": str(subrun_dir),
                        "missing": sorted(missing_keys),
                        "error": error_msg,
                    }
                )

        top_errors = sorted(
            [{"message": msg, "count": count} for msg, count in exception_counts.items()],
            key=lambda x: (-int(x["count"]), x["message"]),
        )[:10]
        worst_subruns = sorted(worst, key=lambda item: str(item.get("path", "")))
        payload = {
            "schema_version": "t0_sweep_diagnose_v1",
            "run_id": base_run,
            "scan_root": str(scan_root_abs),
            "generated_at_utc": generated_at_utc,
            "n_subruns_scanned": len(subrun_dirs),
            "counts_missing": counts_missing,
            "top_errors": top_errors,
            "worst_subruns": worst_subruns,
        }
    except Exception as exc:
        payload = {
            "schema_version": "t0_sweep_diagnose_v1",
            "run_id": base_run,
            "scan_root": str(scan_root_abs),
            "generated_at_utc": generated_at_utc,
            "n_subruns_scanned": 0,
            "counts_missing": {},
            "top_errors": [],
            "worst_subruns": [],
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    _atomic_json_dump(_diagnose_path(runs_root_abs, base_run), payload)
    return payload


def _flag_present(argv: list[str], *flags: str) -> bool:
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def _validate_phase_contracts(args: argparse.Namespace, argv: list[str]) -> None:
    if args.phase not in {"inventory", "finalize"}:
        return

    has_explicit_seeds = _flag_present(argv, "--inventory-seeds")
    has_explicit_grid = _flag_present(argv, "--t0-grid-ms") or (
        _flag_present(argv, "--t0-start-ms")
        and _flag_present(argv, "--t0-stop-ms")
        and _flag_present(argv, "--t0-step-ms")
    )
    if not (has_explicit_seeds and has_explicit_grid):
        print(
            "inventory/finalize requiere definir expected_pairs: "
            "pasa --inventory-seeds y --t0-grid-ms (o start/stop/step) para evitar defaults",
            file=sys.stderr,
        )
        raise SystemExit(2)


def run_inventory_phase(args: argparse.Namespace) -> dict[str, Any]:
    runs_root_abs = _resolve_runs_root_arg(getattr(args, "runs_root", None))
    base_run = args.run_id
    base_run_dir = runs_root_abs / base_run
    scan_root_abs = Path(args.scan_root).expanduser().resolve() if getattr(args, "scan_root", None) else (base_run_dir / "experiment").resolve()

    expected_seeds = _parse_int_csv(getattr(args, "inventory_seeds", None)) or [int(args.seed)]
    expected_t0 = _parse_grid(args)
    expected_pairs = [{"seed": int(seed), "t0_ms": int(t0)} for seed in expected_seeds for t0 in expected_t0]
    expected_pairs.sort(key=lambda p: (p["seed"], p["t0_ms"]))

    observed_set: set[tuple[int, int]] = set()
    pattern = "**/s3b_multimode_estimates/outputs/multimode_estimates.json"
    if scan_root_abs.exists():
        for path in sorted(scan_root_abs.glob(pattern), key=lambda p: p.as_posix()):
            path_text = path.as_posix()
            seed_match = SEED_RE.search(path_text)
            t0_match = T0MS_RE.search(path_text)
            if not seed_match or not t0_match:
                continue
            observed_set.add((int(seed_match.group(1)), int(t0_match.group(1))))

    expected_set = {(p["seed"], p["t0_ms"]) for p in expected_pairs}
    observed_pairs = [{"seed": seed, "t0_ms": t0} for (seed, t0) in sorted(observed_set)]
    missing_pairs = [{"seed": seed, "t0_ms": t0} for (seed, t0) in sorted(expected_set - observed_set)]

    counts_by_seed = {str(seed): 0 for seed in expected_seeds}
    counts_by_t0_ms = {str(t0): 0 for t0 in expected_t0}
    for seed, t0 in observed_set:
        counts_by_seed[str(seed)] = counts_by_seed.get(str(seed), 0) + 1
        counts_by_t0_ms[str(t0)] = counts_by_t0_ms.get(str(t0), 0) + 1

    out_path = _inventory_path(runs_root_abs, base_run)
    previous_payload = _read_json_if_exists(out_path) or {}
    retry_counts = previous_payload.get("retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
    normalized_retry_counts: dict[str, int] = {}
    for key, value in retry_counts.items():
        try:
            normalized_retry_counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue

    max_retries_per_pair = int(getattr(args, "max_retries_per_pair", 2))
    blocked_pairs = [
        pair for pair in missing_pairs if normalized_retry_counts.get(_pair_key(pair["seed"], pair["t0_ms"]), 0) >= max_retries_per_pair
    ]
    eligible_missing_pairs = [
        pair for pair in missing_pairs if normalized_retry_counts.get(_pair_key(pair["seed"], pair["t0_ms"]), 0) < max_retries_per_pair
    ]

    payload = {
        "base_run": base_run,
        "base_run_dir": str(base_run_dir),
        "scan_root_abs": str(scan_root_abs),
        "runs_root_abs": str(runs_root_abs),
        "expected_pairs": expected_pairs,
        "observed_pairs": observed_pairs,
        "missing_pairs": missing_pairs,
        "blocked_pairs": blocked_pairs,
        "counts_by_seed": counts_by_seed,
        "counts_by_t0_ms": counts_by_t0_ms,
        "observed_payload_count": len(observed_pairs),
        "expected_payload_count": len(expected_pairs),
        "acceptance": {
            "max_missing_abs": int(getattr(args, "max_missing_abs", 0)),
            "max_missing_frac": float(getattr(args, "max_missing_frac", 0.0)),
        },
        "max_retries_per_pair": max_retries_per_pair,
        "retry_counts": normalized_retry_counts,
        "last_attempted_pairs": previous_payload.get("last_attempted_pairs", []),
    }

    missing_abs = len(missing_pairs)
    expected_payload_count = payload["expected_payload_count"]
    missing_frac = (missing_abs / expected_payload_count) if expected_payload_count > 0 else 0.0
    payload["missing_abs"] = missing_abs
    payload["missing_frac"] = missing_frac

    phase = getattr(args, "phase", "run")
    fail = (missing_abs > payload["acceptance"]["max_missing_abs"]) or (
        missing_frac > payload["acceptance"]["max_missing_frac"]
    )
    if phase == "finalize":
        has_blocked = len(blocked_pairs) > 0
        payload["status"] = "FAIL" if fail else "PASS"
        if fail:
            if has_blocked:
                reason = "blocked_pairs exhausted retries"
            else:
                reason = "missing_pairs exceed acceptance thresholds"
        else:
            reason = "missing within acceptance thresholds"
        payload["decision"] = {
            "fail": bool(fail),
            "reason": reason,
            "missing_abs": missing_abs,
            "missing_frac": missing_frac,
            "blocked_pairs": len(blocked_pairs),
            "eligible_missing_pairs": len(eligible_missing_pairs),
        }
    else:
        payload["status"] = "IN_PROGRESS"

    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()

    _atomic_json_dump(out_path, payload)

    if missing_pairs:
        first = ", ".join(f"(seed={p['seed']},t0_ms={p['t0_ms']})" for p in missing_pairs[:20])
        summary = (
            "[experiment_t0_sweep_full] inventory missing pairs: "
            f"total={len(missing_pairs)} first={first} "
            f"missing_abs={missing_abs} missing_frac={missing_frac:.6f} "
            f"threshold_abs={payload['acceptance']['max_missing_abs']} "
            f"threshold_frac={payload['acceptance']['max_missing_frac']:.6f} phase={phase}"
        )
        print(summary, file=sys.stderr)

    if phase == "finalize" and fail:
        print("[experiment_t0_sweep_full] finalize status=FAIL", file=sys.stderr)
        raise SystemExit(2)

    return payload


def _set_experiment_run_valid(runs_root_abs: Path, run_id: str, verdict: str, reason: str) -> None:
    rv_path = runs_root_abs / run_id / "RUN_VALID" / "verdict.json"
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rv_payload = {
        "verdict": verdict,
        "created": timestamp_utc,
        "timestamp_utc": timestamp_utc,
        "reason": reason,
        "stage": "experiment/t0_sweep_full/finalize",
    }
    write_json_atomic(rv_path, rv_payload)


def _finalize_inventory_decision(payload: dict[str, Any], args: argparse.Namespace) -> bool:
    missing_abs = int(payload.get("missing_abs", 0))
    missing_frac = float(payload.get("missing_frac", 0.0))
    max_abs = int(getattr(args, "max_missing_abs", 0))
    max_frac = float(getattr(args, "max_missing_frac", 0.0))
    blocked_pairs = payload.get("blocked_pairs", [])
    fail = (missing_abs > max_abs) or (missing_frac > max_frac)
    payload["status"] = "FAIL" if fail else "PASS"
    if fail:
        reason = "blocked_pairs exhausted retries" if blocked_pairs else "missing_pairs exceed acceptance thresholds"
    else:
        reason = "missing within acceptance thresholds"
    payload["decision"] = {
        "fail": bool(fail),
        "reason": reason,
        "missing_abs": missing_abs,
        "missing_frac": missing_frac,
        "blocked_pairs": len(blocked_pairs),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return fail


def run_finalize_phase(args: argparse.Namespace) -> dict[str, Any]:
    runs_root_abs = _resolve_runs_root_arg(getattr(args, "runs_root", None))
    base_run = args.run_id
    base_run_dir = runs_root_abs / base_run
    scan_root_abs = Path(args.scan_root).expanduser().resolve() if getattr(args, "scan_root", None) else (base_run_dir / "experiment").resolve()

    inventory_args = copy.copy(args)
    inventory_args.phase = "inventory"
    inventory_payload = run_inventory_phase(inventory_args)
    inventory_fail = _finalize_inventory_decision(inventory_payload, args)
    _atomic_json_dump(_inventory_path(runs_root_abs, base_run), inventory_payload)

    try:
        require_run_valid(runs_root_abs, base_run)
    except FileNotFoundError as exc:
        raise SystemExit(2) from exc

    rows: list[dict[str, Any]] = []
    windows = []

    seed_artifacts: list[dict[str, Any]] = []
    seeds = _parse_int_csv(getattr(args, "inventory_seeds", None))
    for seed in seeds:
        seed_dir = scan_root_abs / f"t0_sweep_full_seed{seed}"
        loaded = _load_results_artifact(seed_dir)
        if loaded is not None:
            seed_artifacts.append(loaded)

    if seed_artifacts:
        windows, rows = _build_windows_from_sweep_results(
            seed_artifacts,
            sigma_floor_f=float(getattr(args, "sigma_floor_f", 1e-12)),
            sigma_floor_tau=float(getattr(args, "sigma_floor_tau", 1e-12)),
            scale_floor_f=float(getattr(args, "scale_floor_f", 1e-12)),
            scale_floor_tau=float(getattr(args, "scale_floor_tau", 1e-12)),
            gate_221=bool(getattr(args, "gate_221", False)),
        )

    for subrun_dir in ([] if seed_artifacts else _discover_subrun_dirs(scan_root_abs)):
        subrun_id = subrun_dir.name
        row: dict[str, Any] = {
            "subrun_id": subrun_id,
            "subrun_root": str(subrun_dir),
            "t0_ms": _parse_t0_offset_ms_from_subrun(subrun_dir),
            "paths": {},
            "hashes": {},
        }
        for stage in ("s2_ringdown_window", "s3_ringdown_estimates", "s3b_multimode_estimates", "s4c_kerr_consistency"):
            st_path = subrun_dir / stage / "stage_summary.json"
            mf_path = subrun_dir / stage / "manifest.json"
            row["paths"][f"{stage}.stage_summary"] = str(st_path)
            row["paths"][f"{stage}.manifest"] = str(mf_path)
            row["hashes"][f"{stage}.stage_summary"] = sha256_file(st_path) if st_path.exists() else None
            row["hashes"][f"{stage}.manifest"] = sha256_file(mf_path) if mf_path.exists() else None

        try:
            metrics = load_window_metrics_from_subrun(subrun_dir)
            windows.append((subrun_dir, metrics))
            row["window"] = {
                "t0": metrics.t0,
                "T": metrics.T,
                "f_median": metrics.f_median,
                "tau_median": metrics.tau_median,
            }
            row["metrics_loaded"] = True
        except Exception as exc:
            row["metrics_loaded"] = False
            row["metrics_error"] = str(exc)
        rows.append(row)

    windows_only = [item[1] for item in windows]
    oracle_report = run_oracle_v1(windows_only, chi2_coh_max=None)

    inv_payload = {
        "schema_version": "t0_sweep_inventory_v1",
        "run_id": base_run,
        "scan_root_abs": str(scan_root_abs),
        "n_subruns_scanned": len(rows),
        "n_windows_loaded": len(windows_only),
        "windows": rows,
    }
    _atomic_json_dump(_oracle_inventory_path(runs_root_abs, base_run), inv_payload)
    _atomic_json_dump(_oracle_report_path(runs_root_abs, base_run), oracle_report)

    best_path = _best_window_path(runs_root_abs, base_run)
    if best_path.exists():
        best_path.unlink()

    oracle_pass = oracle_report.get("final_verdict") == "PASS"
    oracle_fail_reason = oracle_report.get("fail_global_reason")
    final_fail = bool(inventory_fail or not oracle_pass)

    if oracle_pass:
        chosen_t0 = oracle_report.get("chosen_t0")
        chosen_T = oracle_report.get("chosen_T")
        selected = None
        for subrun_dir, metrics in windows:
            if float(metrics.t0) == float(chosen_t0) and float(metrics.T) == float(chosen_T):
                selected = (subrun_dir, metrics)
                break
        if selected is not None and not final_fail:
            subrun_dir, _metrics = selected
            s2_stage_summary = subrun_dir / "s2_ringdown_window" / "stage_summary.json"
            s2_manifest = subrun_dir / "s2_ringdown_window" / "manifest.json"
            best_payload = {
                "chosen_t0": chosen_t0,
                "chosen_T": chosen_T,
                "subrun_id": subrun_dir.name,
                "paths": {
                    "subrun_root": str(subrun_dir),
                    "stage_summary_path": str(s2_stage_summary),
                    "manifest_path": str(s2_manifest),
                },
                "hashes": {
                    "sha256(stage_summary)": sha256_file(s2_stage_summary) if s2_stage_summary.exists() else None,
                    "sha256(manifest)": sha256_file(s2_manifest) if s2_manifest.exists() else None,
                },
            }
            _atomic_json_dump(best_path, best_payload)

    derived_dir = runs_root_abs / base_run / "experiment" / "derived"
    experiment_dir = runs_root_abs / base_run / "experiment"
    manifest_path = experiment_dir / "manifest.json"
    summary_path = experiment_dir / "stage_summary.json"

    artifacts = {
        "oracle_report": _oracle_report_path(runs_root_abs, base_run),
        "t0_sweep_inventory": _oracle_inventory_path(runs_root_abs, base_run),
        "sweep_inventory": _inventory_path(runs_root_abs, base_run),
    }
    if best_path.exists():
        artifacts["best_window"] = best_path

    manifest_payload = {
        "schema_version": "mvp_manifest_v1",
        "stage": "experiment/t0_sweep_full/finalize",
        "derived_root": str(derived_dir),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "hashes": {k: sha256_file(v) for k, v in artifacts.items()},
    }
    _atomic_json_dump(manifest_path, manifest_payload)

    summary_payload = {
        "stage": "experiment/t0_sweep_full/finalize",
        "run_id": base_run,
        "verdict": "FAIL" if final_fail else "PASS",
        "results": {
            "final_verdict": oracle_report.get("final_verdict"),
            "fail_global_reason": oracle_fail_reason,
            "chosen_t0": oracle_report.get("chosen_t0"),
            "inventory_status": inventory_payload.get("status"),
            "oracle_final_verdict": oracle_report.get("final_verdict"),
            "oracle_fail_global_reason": oracle_fail_reason,
            "reasons": [
                reason
                for reason in [
                    "INVENTORY_FAIL" if inventory_fail else None,
                    oracle_fail_reason,
                ]
                if reason is not None
            ],
            "best_window_convention": "best_window.json is generated only when RUN_VALID=PASS",
            "n_subruns_scanned": len(rows),
            "n_windows_loaded": len(windows_only),
        },
        "outputs": [{"path": str(v), "sha256": sha256_file(v)} for v in artifacts.values()],
    }
    _atomic_json_dump(summary_path, summary_payload)

    if final_fail:
        _set_experiment_run_valid(
            runs_root_abs,
            base_run,
            "FAIL",
            (
                "t0_sweep_full finalize oracle/inventory gating "
                f"inventory_fail={inventory_fail} oracle_fail_reason={oracle_fail_reason}"
            ),
        )
        raise SystemExit(2)

    return {
        "inventory": inventory_payload,
        "oracle_report": oracle_report,
        "oracle_inventory": inv_payload,
    }


def _update_retry_state(
    args: argparse.Namespace,
    attempted_pairs: list[dict[str, int]],
) -> None:
    runs_root_abs = _resolve_runs_root_arg(getattr(args, "runs_root", None))
    out_path = _inventory_path(runs_root_abs, args.run_id)
    payload = _read_json_if_exists(out_path) or {}
    retry_counts = payload.get("retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}

    for pair in attempted_pairs:
        key = _pair_key(pair["seed"], pair["t0_ms"])
        retry_counts[key] = int(retry_counts.get(key, 0)) + 1

    payload["retry_counts"] = retry_counts
    payload["last_attempted_pairs"] = [
        {"seed": int(pair["seed"]), "t0_ms": int(pair["t0_ms"])} for pair in attempted_pairs
    ]

    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json_dump(out_path, payload)


def main() -> int:
    raw_argv = sys.argv[1:]
    ap = argparse.ArgumentParser(description="Experiment: full deterministic t0 sweep with subruns (requires s1 outputs; external HDF5 live in data/losc/<EVENT_ID>/)")
    ap.add_argument("--run-id", "--run", dest="run_id", required=True)
    ap.add_argument("--base-runs-root", type=Path, default=Path.cwd() / "runs")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--scan-root", default=None)
    ap.add_argument("--inventory-seeds", default=None)
    ap.add_argument("--phase", choices=["run", "inventory", "finalize", "diagnose", "backfill_window_meta"], default="run")
    ap.add_argument("--sigma-floor-f", type=float, default=1e-12)
    ap.add_argument("--sigma-floor-tau", type=float, default=1e-12)
    ap.add_argument("--scale-floor-f", type=float, default=1e-12)
    ap.add_argument("--scale-floor-tau", type=float, default=1e-12)
    ap.add_argument("--gate-221", action="store_true")
    ap.add_argument("--max-missing-abs", type=int, default=0)
    ap.add_argument("--max-missing-frac", type=float, default=0.0)
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
    args = ap.parse_args()

    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())

    try:
        _validate_phase_contracts(args, raw_argv)
        if args.phase == "inventory":
            run_inventory_phase(args)
            return 0

        if args.phase == "finalize":
            run_finalize_phase(args)
            return 0

        if args.phase == "diagnose":
            run_diagnose_phase(args)
            return 0

        if args.phase == "backfill_window_meta":
            run_backfill_window_meta_phase(args)
            return 0

        if args.phase == "run":
            if args.runs_root and not _flag_present(raw_argv, "--base-runs-root"):
                args.base_runs_root = Path(args.runs_root)
            run_t0_sweep_full(args)

            if args.resume_missing:
                inventory_payload = run_inventory_phase(args)
                missing_pairs = list(inventory_payload.get("missing_pairs", []))
                retry_counts = dict(inventory_payload.get("retry_counts", {}))
                max_retries = int(args.max_retries_per_pair)
                selected_pairs: list[dict[str, int]] = []
                for pair in missing_pairs:
                    seed = int(pair["seed"])
                    t0_ms = int(pair["t0_ms"])
                    key = _pair_key(seed, t0_ms)
                    if retry_counts.get(key, 0) >= max_retries:
                        continue
                    if seed != int(args.seed):
                        continue
                    selected_pairs.append({"seed": seed, "t0_ms": t0_ms})
                    if len(selected_pairs) >= int(args.resume_batch_size):
                        break

                if selected_pairs:
                    _update_retry_state(args, selected_pairs)
                for pair in selected_pairs:
                    single_args = copy.copy(args)
                    single_args.seed = int(pair["seed"])
                    single_args.t0_grid_ms = str(int(pair["t0_ms"]))
                    run_t0_sweep_full(single_args)

            if not args.inventory_seeds and _flag_present(raw_argv, "--seed"):
                args.inventory_seeds = str(int(args.seed))
            run_inventory_phase(args)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[experiment_t0_sweep_full] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
