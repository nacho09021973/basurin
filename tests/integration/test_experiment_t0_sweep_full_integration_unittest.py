from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest("integration: requires numpy") from exc

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

    s1_out = run_dir / "s1_fetch_strain" / "outputs"
    s1_out.mkdir(parents=True, exist_ok=True)
    np.savez(s1_out / "strain.npz", H1=strain.astype(np.float64), sample_rate_hz=np.array([fs]))

    np.savez(s2_out / "H1_rd.npz", strain=strain.astype(np.float64), sample_rate_hz=np.array([fs]))

    (s2_out / "window_meta.json").write_text(
        json.dumps({"sample_rate_hz": fs, "t0_start_s": 0.0}),
        encoding="utf-8",
    )
    (s2_dir / "manifest.json").write_text(json.dumps({"artifacts": {"H1_rd": "outputs/H1_rd.npz"}}), encoding="utf-8")


class TestExperimentT0SweepFullIntegration(unittest.TestCase):
    def _sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fake_runner_factory(self, subruns_root: Path, observed_cmds: list[list[str]]):
        def _run(cmd: list[str], env: dict[str, str], timeout: int):
            self.assertEqual(timeout, 30)
            observed_cmds.append(cmd)
            run_id = cmd[cmd.index("--run-id") + 1] if "--run-id" in cmd else cmd[cmd.index("--run") + 1]
            stage = Path(cmd[1]).stem
            run_dir = subruns_root / run_id

            if stage == "s3b_multimode_estimates":
                out = run_dir / "s3b_multimode_estimates" / "outputs"
                out.mkdir(parents=True, exist_ok=True)
                payload = {
                    "results": {"verdict": "OK", "quality_flags": [], "messages": []},
                    "modes": [{"label": "220", "ln_f": 5.4, "ln_Q": 2.1}, {"label": "221", "ln_f": 5.1, "ln_Q": 1.8}],
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

    def test_deterministic_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "test_t0_sweep_full"
            _write_min_run(runs_root, run_id)
            args = SimpleNamespace(run_id=run_id, runs_root=runs_root, base_runs_root=runs_root, atlas_path="docs/ringdown/atlas/atlas_berti_v2.json", t0_grid_ms="0,5,10", t0_start_ms=0, t0_stop_ms=10, t0_step_ms=5, n_bootstrap=50, seed=123, detector="auto", stage_timeout_s=30)

            observed_cmds: list[list[str]] = []
            exp.run_t0_sweep_full(args, run_cmd_fn=self._fake_runner_factory(runs_root / run_id / "experiment" / "t0_sweep_full_seed123" / "runs", observed_cmds))

            out_json = runs_root / run_id / "experiment" / "t0_sweep_full_seed123" / "outputs" / "t0_sweep_full_results.json"
            digest1 = self._sha(out_json)
            observed_cmds.clear()
            exp.run_t0_sweep_full(args, run_cmd_fn=self._fake_runner_factory(runs_root / run_id / "experiment" / "t0_sweep_full_seed123" / "runs", observed_cmds))
            self.assertEqual(self._sha(out_json), digest1)

    def test_build_subrun_stage_cmds_passes_seed_to_s3b(self) -> None:
        cmds = exp.build_subrun_stage_cmds(
            python="python",
            s2_script="mvp/s2_ringdown_window.py",
            s3_script="mvp/s3_ringdown_estimates.py",
            s3b_script="mvp/s3b_multimode_estimates.py",
            s4c_script="mvp/s4c_kerr_consistency.py",
            subrun_runs_root="runs/rid",
            subrun_id="rid__t0ms0008",
            event_id="GW150914",
            dt_start_s=0.003,
            duration_s=0.06,
            strain_npz="runs/BASE_RUN/s1_fetch_strain/outputs/strain.npz",
            n_bootstrap=200,
            s3b_seed=101,
            atlas_path="docs/ringdown/atlas/atlas_berti_v2_s4.json",
        )
        s3b_cmd = next(cmd for cmd in cmds if Path(cmd[1]).stem == "s3b_multimode_estimates")
        self.assertIn("--seed", s3b_cmd)
        self.assertEqual(s3b_cmd[s3b_cmd.index("--seed") + 1], "101")


if __name__ == "__main__":
    unittest.main()
