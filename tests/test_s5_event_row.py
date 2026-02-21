from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _init_base_layout(tmp_path: Path, run_id: str, seed: int, t0_ms: int) -> tuple[Path, Path]:
    runs_root = tmp_path / "runs"
    run_root = runs_root / run_id
    _write_json(run_root / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    subrun_id = f"{run_id}__t0ms{int(t0_ms):04d}"
    subrun_root = run_root / "experiment" / f"t0_sweep_full_seed{seed}" / "runs" / subrun_id
    return runs_root, subrun_root


def _cmd(repo_root: Path, runs_root: Path, run_id: str, seed: int, t0_ms: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mvp.s5_event_row",
        "--runs-root",
        str(runs_root),
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--t0-ms",
        str(t0_ms),
    ]


def test_s5_event_row_requires_inputs_and_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "run_req"
    seed = 101
    t0_ms = 5
    runs_root, subrun_root = _init_base_layout(tmp_path, run_id, seed, t0_ms)

    proc_gate = subprocess.run(
        _cmd(repo_root, runs_root, run_id, seed, t0_ms),
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc_gate.returncode != 0
    assert "Missing required inputs" in proc_gate.stderr

    _write_json(subrun_root / "s2_ringdown_window" / "outputs" / "window_meta.json", {"event_id": "GW150914"})
    _write_json(subrun_root / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json", {"results": {}, "modes": []})
    proc_missing = subprocess.run(
        _cmd(repo_root, runs_root, run_id, seed, t0_ms),
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc_missing.returncode != 0
    assert "kerr_consistency" in proc_missing.stderr

    _write_json((runs_root / run_id / "RUN_VALID" / "verdict.json"), {"verdict": "FAIL"})
    proc_gate_fail = subprocess.run(
        _cmd(repo_root, runs_root, run_id, seed, t0_ms),
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc_gate_fail.returncode != 0
    assert "RUN_VALID verdict is not PASS" in proc_gate_fail.stderr


def test_s5_event_row_writes_only_under_runs_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "run_paths"
    seed = 101
    t0_ms = 0
    runs_root, subrun_root = _init_base_layout(tmp_path, run_id, seed, t0_ms)

    _write_json(subrun_root / "s2_ringdown_window" / "outputs" / "window_meta.json", {"event_id": "GW150914", "dt_start_s": 0.0})
    _write_json(
        subrun_root / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "results": {"verdict": "OK", "quality_flags": ["a"]},
            "modes": [
                {"label": "220", "ln_f": 1.0, "ln_Q": 2.0, "fit": {"stability": {"valid_fraction": 0.9, "lnf_span": 0.1, "lnQ_span": 0.2, "n_failed": 1}}},
                {"label": "221", "ln_f": 1.1, "ln_Q": 2.1, "fit": {"stability": {"valid_fraction": 0.8, "lnf_span": 0.2, "lnQ_span": 0.3, "n_failed": 2}}},
            ],
        },
    )
    _write_json(subrun_root / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json", {"chi_best": 0.7, "d2_min": 1.5, "deltas": {"delta_logfreq": 0.02, "delta_logQ": 0.03}, "verdict": "PASS"})

    proc = subprocess.run(_cmd(repo_root, runs_root, run_id, seed, t0_ms), cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / run_id / "s5_event_row"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "outputs" / "event_row.json").exists()

    unexpected = repo_root / "runs" / run_id / "s5_event_row"
    assert not unexpected.exists()


def test_s5_event_row_schema_stable(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "run_schema"
    seed = 202
    t0_ms = 10
    runs_root, subrun_root = _init_base_layout(tmp_path, run_id, seed, t0_ms)

    _write_json(subrun_root / "s2_ringdown_window" / "outputs" / "window_meta.json", {"event_id": "GW150914", "duration_s": 0.05})
    _write_json(
        subrun_root / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "results": {"verdict": "OK", "quality_flags": []},
            "modes": [
                {"label": "220", "ln_f": 1.0, "ln_Q": 2.0, "fit": {"stability": {"valid_fraction": 1.0, "lnf_span": 0.1, "lnQ_span": 0.2, "n_failed": 0}}},
                {"label": "221", "ln_f": 1.2, "ln_Q": 2.2, "fit": {"stability": {"valid_fraction": 0.9, "lnf_span": 0.2, "lnQ_span": 0.3, "n_failed": 1}}},
            ],
        },
    )
    _write_json(subrun_root / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json", {"chi_best": 0.5, "d2_min": 0.9, "delta_logfreq": 0.01, "delta_logQ": 0.02, "verdict": "PASS"})

    proc = subprocess.run(_cmd(repo_root, runs_root, run_id, seed, t0_ms), cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((runs_root / run_id / "s5_event_row" / "outputs" / "event_row.json").read_text(encoding="utf-8"))
    assert {"schema_version", "run_id", "subrun_id", "event_window", "s3b", "s4c", "geometry", "artifacts"}.issubset(payload.keys())
    assert {"modes", "verdict", "quality_flags"}.issubset(payload["s3b"].keys())
    assert {"chi_best", "d2_min", "deltas", "verdict"}.issubset(payload["s4c"].keys())
