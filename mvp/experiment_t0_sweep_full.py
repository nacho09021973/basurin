#!/usr/bin/env python3
"""Full deterministic t0 sweep using isolated subruns (s3 -> s3b -> s4c)."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    ensure_stage_dirs,
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


def _load_npz(npz_path: Path, window_meta: dict[str, Any]) -> tuple[np.ndarray, float]:
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


def _write_subrun_shadow_s2(
    subrun_dir: Path,
    detector: str,
    trimmed: np.ndarray,
    fs: float,
    t0_ms: int,
    offset_samples: int,
    original_npz: Path,
    original_sha: str,
) -> dict[str, Any]:
    rv_path = subrun_dir / "RUN_VALID" / "verdict.json"
    rv_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_path, {"verdict": "PASS"})

    s2_stage_dir = subrun_dir / "s2_ringdown_window"
    s2_out = s2_stage_dir / "outputs"
    s2_out.mkdir(parents=True, exist_ok=True)
    subrun_npz = s2_out / f"{detector}_rd.npz"
    np.savez(subrun_npz, strain=trimmed.astype(np.float64), sample_rate_hz=np.array([fs], dtype=np.float64))
    subrun_sha = sha256_file(subrun_npz)

    stage_summary = {
        "stage": "s2_ringdown_window",
        "run_id": subrun_dir.name,
        "verdict": "PASS",
        "results": {
            "detector": detector,
            "fs_hz": float(fs),
            "t0_offset_ms": int(t0_ms),
            "offset_samples": int(offset_samples),
            "npz_original_sha256": original_sha,
            "npz_trimmed_sha256": subrun_sha,
            "derived_from": str(original_npz),
        },
    }
    write_stage_summary(s2_stage_dir, stage_summary)

    manifest = {
        "schema_version": "mvp_manifest_v1",
        "artifacts": {f"{detector}_rd": f"outputs/{detector}_rd.npz"},
        "hashes": {f"{detector}_rd": subrun_sha},
        "provenance": {
            "source_stage": "s2_ringdown_window",
            "npz_original_sha256": original_sha,
            "npz_trimmed_sha256": subrun_sha,
            "t0_offset_ms": int(t0_ms),
            "offset_samples": int(offset_samples),
        },
    }
    write_json_atomic(s2_stage_dir / "manifest.json", manifest)
    return {"npz_trimmed_sha256": subrun_sha, "offset_samples": int(offset_samples)}


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def run_t0_sweep_full(
    args: argparse.Namespace,
    *,
    run_cmd_fn: Callable[[list[str], dict[str, str], int], Any] = run_cmd,
) -> tuple[dict[str, Any], Path]:
    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    base_run_dir = out_root / args.run_id
    s2_dir = base_run_dir / "s2_ringdown_window"
    s2_manifest = s2_dir / "manifest.json"
    if not s2_manifest.exists():
        raise FileNotFoundError(f"Missing s2 manifest: {s2_manifest}")

    s2_outputs = s2_dir / "outputs"
    detector, source_npz = _pick_detector(s2_outputs, args.detector)

    window_meta_path = s2_outputs / "window_meta.json"
    window_meta = json.loads(window_meta_path.read_text(encoding="utf-8")) if window_meta_path.exists() else {}
    strain, fs = _load_npz(source_npz, window_meta)
    source_sha = sha256_file(source_npz)
    grid = _parse_grid(args)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run_id, EXPERIMENT_STAGE, base_dir=out_root)
    subruns_root = stage_dir / "runs"
    subruns_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
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

        trimmed = strain[offset_samples:]
        subrun_dir = subruns_root / subrun_id
        if subrun_dir.exists():
            shutil.rmtree(subrun_dir)
        provenance = _write_subrun_shadow_s2(
            subrun_dir,
            detector,
            trimmed,
            fs,
            int(t0_ms),
            offset_samples,
            source_npz,
            source_sha,
        )
        point["quality_flags"].append(
            f"provenance:offset_samples={provenance['offset_samples']},trimmed_sha={provenance['npz_trimmed_sha256']}"
        )

        env = os.environ.copy()
        env["BASURIN_RUNS_ROOT"] = str(subruns_root)

        stages = [
            [python, s3_script, "--run-id", subrun_id],
            [python, s3b_script, "--run-id", subrun_id, "--n-bootstrap", str(args.n_bootstrap), "--seed", str(args.seed)],
            [python, s4c_script, "--run-id", subrun_id, "--atlas-path", str(args.atlas_path)],
        ]

        failed = False
        for cmd in stages:
            try:
                cp = run_cmd_fn(cmd, env, args.stage_timeout_s)
            except Exception as exc:
                point["messages"].append(f"subprocess error for {' '.join(cmd)}: {type(exc).__name__}: {exc}")
                failed = True
                break
            if int(getattr(cp, "returncode", 1)) != 0:
                stderr = (getattr(cp, "stderr", "") or "").strip()
                point["messages"].append(f"stage failed rc={cp.returncode}: {' '.join(cmd)}")
                if stderr:
                    point["messages"].append(stderr)
                failed = True
                break

        s3b_payload = _read_json_if_exists(subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json")
        s4c_payload = _read_json_if_exists(subrun_dir / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json")
        point["s3b"] = _extract_s3b(s3b_payload)
        point["s4c"] = _extract_s4c(s4c_payload)

        if failed:
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
    return results, out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment: full deterministic t0 sweep with subruns")
    ap.add_argument("--run-id", "--run", dest="run_id", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--t0-grid-ms", default=None)
    ap.add_argument("--t0-start-ms", type=int, default=0)
    ap.add_argument("--t0-stop-ms", type=int, default=30)
    ap.add_argument("--t0-step-ms", type=int, default=5)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--detector", choices=["auto", "H1", "L1"], default="auto")
    ap.add_argument("--stage-timeout-s", type=int, default=300)
    args = ap.parse_args()

    try:
        run_t0_sweep_full(args)
        return 0
    except Exception as exc:
        print(f"[experiment_t0_sweep_full] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
