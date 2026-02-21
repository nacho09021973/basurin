from __future__ import annotations

import json
import os
from pathlib import Path

from mvp import experiment_oracle_t0_ringdown as oracle_exp


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mk_subrun(subruns_root: Path, subrun_id: str, *, t0_ms: int, ln_f: float = 4.7, ln_q: float = 3.2) -> None:
    subrun_root = subruns_root / subrun_id
    _write_json(
        subrun_root / "s2_ringdown_window" / "outputs" / "window_meta.json",
        {"duration_s": 1.0, "t0_offset_ms": t0_ms, "n_samples": 256},
    )
    _write_json(
        subrun_root / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "modes": [
                {
                    "label": "220",
                    "ln_f": ln_f,
                    "ln_Q": ln_q,
                    "Sigma": [[0.001, 0.0], [0.0, 0.002]],
                    "fit": {"stability": {"sigma_cond": 1.5}},
                }
            ]
        },
    )
    _write_json(
        subrun_root / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {"combined": {"snr_peak": 20.0}, "combined_uncertainty": {"cov_logf_logQ": 0.0, "r": 0.0}},
    )


def _mk_seed_results(seed_dir: Path, subruns_root: Path) -> None:
    points = []
    for t0 in (0, 2, 4):
        subrun_id = f"BASE_RUN__t0ms{t0:04d}"
        _mk_subrun(subruns_root, subrun_id, t0_ms=t0)
        points.append(
            {
                "t0_ms": t0,
                "status": "OK",
                "subrun_id": subrun_id,
                "s3b": {"ln_f_220": 4.7, "ln_Q_220": 3.2},
            }
        )
    _write_json(
        seed_dir / "outputs" / "t0_sweep_full_results.json",
        {
            "schema_version": "experiment_t0_sweep_full_v1",
            "run_id": "BASE_RUN",
            "subruns_root": str(subruns_root),
            "points": points,
            "summary": {"n_points": 3},
        },
    )


def test_integration_lite_generates_artifacts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "BASE_RUN"
    run_root = runs_root / run_id
    _write_json(run_root / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    seed_dir = run_root / "experiment" / "t0_sweep_full_seed101"
    subruns_root = run_root / "experiment" / "subruns"
    _mk_seed_results(seed_dir, subruns_root)

    oracle_exp.run(["--run-id", run_id, "--runs-root", str(runs_root), "--seed-dir", str(seed_dir)])

    stage_dir = run_root / "experiment" / "oracle_t0_ringdown"
    assert (stage_dir / "outputs" / "oracle_report.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "manifest.json").exists()


def test_pathing_supports_batch_root_env(tmp_path: Path) -> None:
    batch_runs_root = tmp_path / "runs" / "batch_foo" / "runs"
    run_id = "BASE_RUN"
    run_root = batch_runs_root / run_id
    _write_json(run_root / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    seed_dir = run_root / "experiment" / "t0_sweep_full_seed101"
    subruns_root = run_root / "experiment" / "subruns"
    _mk_seed_results(seed_dir, subruns_root)

    prev = os.environ.get("BASURIN_RUNS_ROOT")
    os.environ["BASURIN_RUNS_ROOT"] = str(batch_runs_root)
    try:
        oracle_exp.run(["--run-id", run_id])
    finally:
        if prev is None:
            os.environ.pop("BASURIN_RUNS_ROOT", None)
        else:
            os.environ["BASURIN_RUNS_ROOT"] = prev

    assert (run_root / "experiment" / "oracle_t0_ringdown" / "outputs" / "oracle_report.json").exists()
