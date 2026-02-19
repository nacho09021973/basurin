from __future__ import annotations

import importlib
import json
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
    def test_subrun_plan_includes_s2_before_s3b_and_writes_in_subrun(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = Path(tmpdir) / "runs" / "BASE_RUN__t0ms0008"
            subrun_dir.mkdir(parents=True, exist_ok=True)

            plan = exp.build_subrun_execution_plan(
                subrun_dir=subrun_dir,
                python="python",
                s2_script="mvp/s2_ringdown_window.py",
                s3_script="mvp/s3_ringdown_estimates.py",
                s3b_script="mvp/s3b_multimode_estimates.py",
                s4c_script="mvp/s4c_kerr_consistency.py",
                subrun_runs_root=Path(tmpdir) / "runs",
                subrun_id="BASE_RUN__t0ms0008",
                event_id="GW150914",
                dt_start_s=0.011,
                duration_s=0.052,
                strain_npz="/tmp/base/s1_fetch_strain/outputs/strain.npz",
                n_bootstrap=200,
                s3b_seed=101,
                atlas_path="atlas.json",
            )

            stage_names = [Path(cmd[1]).stem for cmd in plan["commands"]]
            self.assertLess(stage_names.index("s2_ringdown_window"), stage_names.index("s3b_multimode_estimates"))
            self.assertEqual(
                plan["expected_inputs"]["s2_window_meta"],
                str(subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"),
            )
            s2_cmd = next(cmd for cmd in plan["commands"] if Path(cmd[1]).stem == "s2_ringdown_window")
            self.assertEqual(s2_cmd[s2_cmd.index("--run-id") + 1], "BASE_RUN__t0ms0008")
            self.assertEqual(s2_cmd[s2_cmd.index("--runs-root") + 1], str(Path(tmpdir) / "runs"))

    def test_missing_window_meta_aborts_before_s3b(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = Path(tmpdir) / "runs" / "BASE_RUN__t0ms0008"
            subrun_dir.mkdir(parents=True, exist_ok=True)
            point = {"messages": [], "quality_flags": [], "status": "FAILED_POINT"}
            seen: list[str] = []

            def _fake_run(cmd: list[str], env: dict[str, str], timeout: int) -> SimpleNamespace:
                seen.append(Path(cmd[1]).stem)
                return SimpleNamespace(returncode=0, stderr="")

            stages = [
                ["python", "mvp/s2_ringdown_window.py", "--run", "BASE_RUN__t0ms0008"],
                ["python", "mvp/s3_ringdown_estimates.py", "--run", "BASE_RUN__t0ms0008"],
                ["python", "mvp/s3b_multimode_estimates.py", "--run-id", "BASE_RUN__t0ms0008"],
            ]

            with self.assertRaises(SystemExit) as ctx:
                exp.execute_subrun_stages_or_abort(
                    stages=stages,
                    subrun_dir=subrun_dir,
                    env={},
                    stage_timeout_s=1,
                    run_cmd_fn=_fake_run,
                    point=point,
                )

            self.assertEqual(ctx.exception.code, 2)
            self.assertEqual(seen, ["s2_ringdown_window"])

    def test_window_meta_precheck_passes_when_expected_file_exists(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = Path(tmpdir) / "runs" / "BASE_RUN__t0ms0008"
            meta_path = subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text("{}", encoding="utf-8")
            point = {"messages": [], "quality_flags": [], "status": "FAILED_POINT"}
            trace = exp._new_subrun_trace("BASE_RUN__t0ms0008", seed=606, t0_ms=8, stages=[])

            def _fake_run(cmd: list[str], env: dict[str, str], timeout: int) -> SimpleNamespace:
                return SimpleNamespace(returncode=0, stderr="")

            stages = [["python", "mvp/s2_ringdown_window.py", "--run-id", "BASE_RUN__t0ms0008"]]
            failed, skip = exp.execute_subrun_stages_or_abort(
                stages=stages,
                subrun_dir=subrun_dir,
                env={},
                stage_timeout_s=1,
                run_cmd_fn=_fake_run,
                point=point,
                trace=trace,
                trace_path=subrun_dir / "derived" / "subrun_trace.json",
            )

            self.assertFalse(failed)
            self.assertFalse(skip)
            self.assertEqual(trace["s2_window_meta_check"]["expected_window_meta_path"], str(meta_path))
            self.assertTrue(trace["s2_window_meta_check"]["found"])

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
            (base_runs_root / run_id / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (base_runs_root / run_id / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
            (base_runs_root / run_id / "s1_fetch_strain" / "outputs").mkdir(parents=True, exist_ok=True)
            (base_runs_root / run_id / "s1_fetch_strain" / "outputs" / "strain.npz").write_bytes(b"npz-placeholder")
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
            expected_out_root, _, _ = exp.compute_experiment_paths(run_id)
            expected_subruns_root = expected_out_root / run_id / "experiment" / f"t0_sweep_full_seed{int(args.seed)}" / "runs"

            fake_source_npz = expected_out_root / run_id / "s2_ringdown_window" / "outputs" / "H1_rd.npz"
            fake_source_npz.parent.mkdir(parents=True, exist_ok=True)
            fake_source_npz.write_bytes(b"npz-placeholder")
            (expected_out_root / run_id / "s2_ringdown_window" / "manifest.json").write_text("{}", encoding="utf-8")
            (expected_out_root / run_id / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (expected_out_root / run_id / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
            (expected_out_root / run_id / "s1_fetch_strain" / "outputs").mkdir(parents=True, exist_ok=True)
            (expected_out_root / run_id / "s1_fetch_strain" / "outputs" / "strain.npz").write_bytes(b"npz-placeholder")

            seen_envs: list[dict[str, str]] = []

            def _fake_run_cmd(cmd: list[str], env: dict[str, str], timeout: int) -> SimpleNamespace:
                seen_envs.append(dict(env))
                if Path(cmd[1]).stem == "s2_ringdown_window":
                    run_id_arg = cmd[cmd.index("--run") + 1]
                    meta = expected_subruns_root / run_id_arg / "s2_ringdown_window" / "outputs" / "window_meta.json"
                    meta.parent.mkdir(parents=True, exist_ok=True)
                    meta.write_text("{}", encoding="utf-8")
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

    def test_phase_run_creates_seed_dir_and_trace_file(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runs_root = base / "runs"
            run_id = "BASE"

            run_dir = runs_root / run_id
            (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
            s2_dir = run_dir / "s2_ringdown_window"
            (s2_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (s2_dir / "manifest.json").write_text("{}", encoding="utf-8")
            source_npz = s2_dir / "outputs" / "H1_rd.npz"
            source_npz.write_bytes(b"npz-placeholder")
            (run_dir / "s1_fetch_strain" / "outputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "s1_fetch_strain" / "outputs" / "strain.npz").write_bytes(b"npz-placeholder")

            def _fake_run(cmd: list[str], env: dict[str, str], timeout: int) -> SimpleNamespace:
                if Path(cmd[1]).stem == "s2_ringdown_window":
                    subrun_id = cmd[cmd.index("--run") + 1]
                    meta = (
                        runs_root
                        / run_id
                        / "experiment"
                        / "t0_sweep_full_seed606"
                        / "runs"
                        / subrun_id
                        / "s2_ringdown_window"
                        / "outputs"
                        / "window_meta.json"
                    )
                    meta.parent.mkdir(parents=True, exist_ok=True)
                    meta.write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr="")

            argv = [
                "experiment_t0_sweep_full.py",
                "--phase",
                "run",
                "--run-id",
                run_id,
                "--runs-root",
                str(runs_root),
                "--base-runs-root",
                str(runs_root),
                "--atlas-path",
                "dummy_atlas.h5",
                "--seed",
                "606",
                "--t0-grid-ms",
                "8",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(exp, "run_cmd", side_effect=_fake_run),
                mock.patch.object(exp, "sha256_file", return_value="sha"),
                mock.patch.object(exp, "_pick_detector", return_value=("H1", source_npz)),
                mock.patch.object(exp, "_load_npz", return_value=(FakeStrain(), 1024.0)),
                mock.patch.object(exp, "_read_json_if_exists", return_value=None),
            ):
                rc = exp.main()

            self.assertEqual(rc, 0)
            seed_dir = runs_root / run_id / "experiment" / "t0_sweep_full_seed606"
            self.assertTrue(seed_dir.exists())
            trace_path = runs_root / run_id / "experiment" / "derived" / "run_trace.json"
            self.assertTrue(trace_path.exists())
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertTrue(trace["created_seed_dir"])

    def test_phase_run_missing_base_strain_aborts_with_trace(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runs_root = base / "runs"
            run_id = "BASE"

            run_dir = runs_root / run_id
            (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
            s2_dir = run_dir / "s2_ringdown_window"
            (s2_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (s2_dir / "manifest.json").write_text("{}", encoding="utf-8")
            source_npz = s2_dir / "outputs" / "H1_rd.npz"
            source_npz.write_bytes(b"npz-placeholder")

            argv = [
                "experiment_t0_sweep_full.py",
                "--phase",
                "run",
                "--run-id",
                run_id,
                "--runs-root",
                str(runs_root),
                "--base-runs-root",
                str(runs_root),
                "--atlas-path",
                "dummy_atlas.h5",
                "--seed",
                "606",
                "--t0-grid-ms",
                "8",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(exp, "_pick_detector", return_value=("H1", source_npz)),
                mock.patch.object(exp, "_load_npz", return_value=(FakeStrain(), 1024.0)),
                mock.patch.object(exp, "sha256_file", return_value="sha"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    raise SystemExit(exp.main())

            self.assertEqual(ctx.exception.code, 2)
            preflight_path = runs_root / run_id / "experiment" / "derived" / "preflight_report.json"
            self.assertTrue(preflight_path.exists())
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            self.assertFalse(preflight["decision"]["can_run"])
            self.assertIn("s1_fetch_strain/outputs/strain.npz", preflight["base_artifacts"]["s1_strain_npz"]["path"])


if __name__ == "__main__":
    unittest.main()
