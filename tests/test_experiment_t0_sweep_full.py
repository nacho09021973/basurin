from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mvp import experiment_t0_sweep_full as exp


def _write_min_run(runs_root: Path, run_id: str) -> None:
    run_dir = runs_root / run_id
    rv_dir = run_dir / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    s2_dir = run_dir / "s2_ringdown_window"
    s2_out = s2_dir / "outputs"
    s2_out.mkdir(parents=True, exist_ok=True)

    fs = 4096.0
    t = np.arange(0, 0.25, 1.0 / fs)
    strain = np.exp(-t / 0.05) * np.sin(2.0 * np.pi * 220.0 * t)
    np.savez(s2_out / "H1_rd.npz", strain=strain.astype(np.float64), sample_rate_hz=np.array([fs]))

    (s2_out / "window_meta.json").write_text(
        json.dumps({"sample_rate_hz": fs, "t0_start_s": 0.0}),
        encoding="utf-8",
    )
    (s2_dir / "manifest.json").write_text(json.dumps({"artifacts": {"H1_rd": "outputs/H1_rd.npz"}}), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_runner_factory(subruns_root: Path):
    def _run(cmd: list[str], env: dict[str, str], timeout: int):
        assert timeout == 30
        run_id = cmd[cmd.index("--run-id") + 1]
        stage = Path(cmd[1]).stem
        run_dir = subruns_root / run_id

        if stage == "s3b_multimode_estimates":
            out = run_dir / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            payload = {
                "results": {"verdict": "OK", "quality_flags": [], "messages": []},
                "modes": [
                    {"label": "220", "ln_f": 5.4, "ln_Q": 2.1},
                    {"label": "221", "ln_f": 5.1, "ln_Q": 1.8},
                ],
            }
            (out / "multimode_estimates.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        if stage == "s4c_kerr_consistency":
            out = run_dir / "s4c_kerr_consistency" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            t0_ms = int(run_id.split("t0ms")[1])
            payload = {
                "kerr_consistent": True,
                "chi_best": 0.68,
                "d2_min": float(t0_ms),
                "delta_logfreq": 0.01,
                "delta_logQ": 0.02,
                "source": {"multimode_verdict": "OK"},
            }
            (out / "kerr_consistency.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return _run


def test_experiment_t0_sweep_full_deterministic(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_t0_sweep_full"
    _write_min_run(runs_root, run_id)

    args = SimpleNamespace(
        run_id=run_id,
        atlas_path="docs/ringdown/atlas/atlas_berti_v2.json",
        t0_grid_ms="0,5,10",
        t0_start_ms=0,
        t0_stop_ms=10,
        t0_step_ms=5,
        n_bootstrap=50,
        seed=123,
        detector="auto",
        stage_timeout_s=30,
    )

    with patch("mvp.experiment_t0_sweep_full.resolve_out_root", return_value=runs_root):
        exp.run_t0_sweep_full(args, run_cmd_fn=_fake_runner_factory(runs_root / run_id / "experiment" / "t0_sweep_full" / "runs"))

    out_json = runs_root / run_id / "experiment" / "t0_sweep_full" / "outputs" / "t0_sweep_full_results.json"
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "experiment_t0_sweep_full_v1"
    assert [p["t0_ms"] for p in payload["points"]] == [0, 5, 10]

    sha_first = _sha(out_json)
    with patch("mvp.experiment_t0_sweep_full.resolve_out_root", return_value=runs_root):
        exp.run_t0_sweep_full(args, run_cmd_fn=_fake_runner_factory(runs_root / run_id / "experiment" / "t0_sweep_full" / "runs"))
    assert _sha(out_json) == sha_first
