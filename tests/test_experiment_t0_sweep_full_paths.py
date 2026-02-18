from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeStrain:
    size = 32

    def __getitem__(self, key):
        return self


class TestExperimentT0SweepFullPaths(unittest.TestCase):
    def test_compute_experiment_paths_follow_seed_runsroot(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            seed101_root = base / "runsroot_seed101"
            seed505_root = base / "runsroot_seed505"

            os.environ["BASURIN_RUNS_ROOT"] = str(seed101_root)
            out_root_101, stage_dir_101, subruns_root_101 = exp.compute_experiment_paths("BASE_RUN")

            self.assertEqual(out_root_101, seed101_root.resolve())
            self.assertEqual(stage_dir_101, seed101_root.resolve() / "BASE_RUN" / "experiment" / "t0_sweep_full")
            self.assertEqual(subruns_root_101, stage_dir_101 / "runs")

            os.environ["BASURIN_RUNS_ROOT"] = str(seed505_root)
            out_root_505, stage_dir_505, subruns_root_505 = exp.compute_experiment_paths("BASE_RUN")

            self.assertEqual(out_root_505, seed505_root.resolve())
            self.assertEqual(stage_dir_505, seed505_root.resolve() / "BASE_RUN" / "experiment" / "t0_sweep_full")
            self.assertEqual(subruns_root_505, stage_dir_505 / "runs")
            self.assertNotEqual(stage_dir_101, stage_dir_505)

            os.environ.pop("BASURIN_RUNS_ROOT", None)

    def test_enforce_isolated_runsroot_rejects_symlink_alias(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            real_root = base / "real_runsroot"
            real_root.mkdir(parents=True, exist_ok=True)
            alias_root = base / "alias_runsroot"
            alias_root.symlink_to(real_root, target_is_directory=True)

            os.environ["BASURIN_RUNS_ROOT"] = str(alias_root)
            out_root, _, _ = exp.compute_experiment_paths("BASE_RUN")

            with self.assertRaises(RuntimeError):
                exp.enforce_isolated_runsroot(out_root, "BASE_RUN")

            os.environ.pop("BASURIN_RUNS_ROOT", None)


    def test_run_t0_sweep_full_precheck_reads_s2_manifest_from_base_runs_root(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            base_runs_root = base / "A"
            out_runs_root = base / "B"
            run_id = "BASE_RUN"

            base_s2 = base_runs_root / run_id / "s2_ringdown_window"
            base_s2_out = base_s2 / "outputs"
            base_s2_out.mkdir(parents=True, exist_ok=True)
            (base_s2 / "manifest.json").write_text("{}", encoding="utf-8")
            fake_source_npz = base_s2_out / "H1_rd.npz"
            fake_source_npz.write_bytes(b"npz-placeholder")

            args = SimpleNamespace(
                run_id=run_id,
                base_runs_root=base_runs_root,
                detector="auto",
                t0_grid_ms="-1",
                t0_start_ms=0,
                t0_stop_ms=0,
                t0_step_ms=1,
                n_bootstrap=10,
                seed=101,
                atlas_path="atlas.json",
                stage_timeout_s=1,
            )

            os.environ["BASURIN_RUNS_ROOT"] = str(out_runs_root)

            with (
                mock.patch.object(exp, "require_run_valid"),
                mock.patch.object(exp, "_pick_detector", return_value=("H1", fake_source_npz)),
                mock.patch.object(exp, "_load_npz", return_value=(FakeStrain(), 1024.0)),
                mock.patch.object(exp, "sha256_file", return_value="sha"),
            ):
                result, _ = exp.run_t0_sweep_full(args, run_cmd_fn=lambda *_: SimpleNamespace(returncode=0, stderr=""))

            self.assertEqual(result["summary"]["n_points"], 1)
            os.environ.pop("BASURIN_RUNS_ROOT", None)

    def test_run_t0_sweep_full_passes_subruns_root_to_subprocess_env(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs_root"
            run_id = "BASE_RUN"

            args = SimpleNamespace(
                run_id=run_id,
                base_runs_root=runs_root,
                detector="auto",
                t0_grid_ms="0",
                t0_start_ms=0,
                t0_stop_ms=0,
                t0_step_ms=1,
                n_bootstrap=10,
                seed=505,
                atlas_path="atlas.json",
                stage_timeout_s=1,
            )

            os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
            expected_out_root, _, expected_subruns_root = exp.compute_experiment_paths(run_id)

            fake_source_npz = expected_out_root / run_id / "s2_ringdown_window" / "outputs" / "H1_rd.npz"
            fake_source_npz.parent.mkdir(parents=True, exist_ok=True)
            fake_source_npz.write_bytes(b"npz-placeholder")
            (expected_out_root / run_id / "s2_ringdown_window" / "manifest.json").write_text("{}", encoding="utf-8")

            seen_envs: list[dict[str, str]] = []

            def _fake_run_cmd(cmd: list[str], env: dict[str, str], timeout: int) -> SimpleNamespace:
                seen_envs.append(dict(env))
                return SimpleNamespace(returncode=0, stderr="")

            with (
                mock.patch.object(exp, "require_run_valid"),
                mock.patch.object(exp, "_pick_detector", return_value=("H1", fake_source_npz)),
                mock.patch.object(exp, "_load_npz", return_value=(FakeStrain(), 1024.0)),
                mock.patch.object(exp, "_write_subrun_shadow_s2", return_value={"offset_samples": 0, "npz_trimmed_sha256": "abc"}),
                mock.patch.object(exp, "sha256_file", return_value="sha"),
                mock.patch.object(exp, "_read_json_if_exists", return_value=None),
                mock.patch.object(exp, "write_json_atomic"),
                mock.patch.object(exp, "write_stage_summary"),
                mock.patch.object(exp, "write_manifest"),
            ):
                exp.run_t0_sweep_full(args, run_cmd_fn=_fake_run_cmd)

            self.assertTrue(seen_envs)
            self.assertTrue(all(e.get("BASURIN_RUNS_ROOT") == str(expected_subruns_root) for e in seen_envs))
            os.environ.pop("BASURIN_RUNS_ROOT", None)


if __name__ == "__main__":
    unittest.main()
