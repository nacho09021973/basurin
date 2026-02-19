from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from mvp import experiment_t0_sweep_full as exp


class ExperimentT0SweepFullInventoryTests(unittest.TestCase):
    def _mk_payload(self, scan_root: Path, seed: int, t0_ms: int) -> None:
        subrun = scan_root / f"t0_sweep_full_seed{seed}" / f"segment__t0ms{t0_ms:04d}"
        out = subrun / "s3b_multimode_estimates" / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

    def _args(self, tmp: Path, seeds: str, grid: str) -> SimpleNamespace:
        return SimpleNamespace(
            run_id="BASE_RUN",
            runs_root=str(tmp / "runs"),
            scan_root=None,
            inventory_seeds=seeds,
            t0_grid_ms=grid,
            t0_start_ms=0,
            t0_stop_ms=0,
            t0_step_ms=1,
            seed=101,
            phase="inventory",
            max_missing_abs=0,
            max_missing_frac=0.0,
        )

    def test_inventory_complete_exit_0(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            for seed in (101, 202):
                for t0_ms in (0, 2):
                    self._mk_payload(scan_root, seed, t0_ms)

            args = self._args(tmp, "101,202", "0,2")
            payload = exp.run_inventory_phase(args)

            self.assertEqual(payload["expected_payload_count"], 4)
            self.assertEqual(payload["observed_payload_count"], 4)
            self.assertEqual(payload["missing_pairs"], [])

            out = tmp / "runs" / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json"
            self.assertTrue(out.exists())
            on_disk = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["missing_pairs"], [])

    def test_phase_run_incomplete_exit_0_and_status_in_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            out = scan_root / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            args = self._args(tmp, "101,202", "0,2")
            args.phase = "run"
            payload = exp.run_inventory_phase(args)

            self.assertEqual(payload["status"], "IN_PROGRESS")
            self.assertGreater(payload["missing_abs"], 0)

    def test_phase_inventory_incomplete_exit_0(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            out = scan_root / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            args = self._args(tmp, "101,202", "0,2")
            args.phase = "inventory"
            payload = exp.run_inventory_phase(args)

            self.assertEqual(payload["status"], "IN_PROGRESS")
            self.assertEqual(payload["expected_payload_count"], 4)
            self.assertEqual(payload["observed_payload_count"], 1)

    def test_phase_finalize_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            out = scan_root / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            args_fail = self._args(tmp, "101,202", "0,2")
            args_fail.phase = "finalize"
            args_fail.max_missing_abs = 0
            args_fail.max_missing_frac = 0.0
            with self.assertRaises(SystemExit) as ctx:
                exp.run_inventory_phase(args_fail)
            self.assertEqual(ctx.exception.code, 2)

            out = tmp / "runs" / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json"
            fail_disk = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(fail_disk["status"], "FAIL")

            missing_count = len(fail_disk["missing_pairs"])

            args_pass = self._args(tmp, "101,202", "0,2")
            args_pass.phase = "finalize"
            args_pass.max_missing_abs = missing_count
            args_pass.max_missing_frac = 1.0
            payload_pass = exp.run_inventory_phase(args_pass)
            self.assertEqual(payload_pass["status"], "PASS")

            on_disk = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["status"], "PASS")


class ExperimentT0SweepFullMainContractTests(unittest.TestCase):
    def _base_argv(self) -> list[str]:
        return [
            "prog",
            "--run-id",
            "BASE_RUN",
            "--runs-root",
            "/tmp/runs",
            "--scan-root",
            "/tmp/runs/BASE_RUN/experiment",
            "--inventory-seeds",
            "101,202",
            "--t0-grid-ms",
            "0,2",
        ]

    def test_inventory_does_not_require_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            out = scan_root / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            argv = [
                "prog", "--phase", "inventory", "--run-id", "BASE_RUN",
                "--runs-root", str(tmp / "runs"), "--scan-root", str(scan_root),
                "--inventory-seeds", "101", "--t0-grid-ms", "0"
            ]
            with mock.patch("sys.argv", argv):
                rc = exp.main()
            self.assertEqual(rc, 0)
            self.assertTrue((tmp / "runs" / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json").exists())

    def test_finalize_does_not_require_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            out = scan_root / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            argv_fail = [
                "prog", "--phase", "finalize", "--run-id", "BASE_RUN",
                "--runs-root", str(tmp / "runs"), "--scan-root", str(scan_root),
                "--inventory-seeds", "101,202", "--t0-grid-ms", "0,2",
                "--max-missing-abs", "0", "--max-missing-frac", "0.0"
            ]
            with mock.patch("sys.argv", argv_fail):
                with self.assertRaises(SystemExit) as ctx:
                    exp.main()
            self.assertEqual(ctx.exception.code, 2)

            argv_pass = [
                "prog", "--phase", "finalize", "--run-id", "BASE_RUN",
                "--runs-root", str(tmp / "runs"), "--scan-root", str(scan_root),
                "--inventory-seeds", "101,202", "--t0-grid-ms", "0,2",
                "--max-missing-abs", "3", "--max-missing-frac", "1.0"
            ]
            with mock.patch("sys.argv", argv_pass):
                rc = exp.main()
            self.assertEqual(rc, 0)

    def test_inventory_requires_explicit_seeds_and_grid(self) -> None:
        argv = ["prog", "--phase", "inventory", "--run-id", "BASE_RUN"]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                exp.main()
        self.assertEqual(ctx.exception.code, 2)

    def test_acceptance_written_from_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            argv = [
                "prog", "--phase", "inventory", "--run-id", "BASE_RUN",
                "--runs-root", str(tmp / "runs"), "--scan-root", str(scan_root),
                "--inventory-seeds", "101,202", "--t0-grid-ms", "0,2",
                "--max-missing-abs", "14", "--max-missing-frac", "0.4"
            ]
            with mock.patch("sys.argv", argv):
                rc = exp.main()
            self.assertEqual(rc, 0)
            payload = json.loads((tmp / "runs" / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["acceptance"], {"max_missing_abs": 14, "max_missing_frac": 0.4})

    def test_resume_missing_attempts_only_missing_pairs_for_seed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            runs_root = tmp / "runs"
            scan_root = runs_root / "BASE_RUN" / "experiment"

            out = scan_root / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates" / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            invoked_t0: list[str] = []

            def fake_run_t0(args):
                invoked_t0.append(str(args.t0_grid_ms))
                out = (
                    scan_root
                    / f"t0_sweep_full_seed{int(args.seed)}"
                    / f"segment__t0ms{int(args.t0_grid_ms):04d}"
                    / "s3b_multimode_estimates"
                    / "outputs"
                )
                out.mkdir(parents=True, exist_ok=True)
                (out / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            argv = [
                "prog", "--phase", "run", "--run-id", "BASE_RUN",
                "--runs-root", str(runs_root), "--scan-root", str(scan_root),
                "--inventory-seeds", "101", "--t0-grid-ms", "0,2,4",
                "--seed", "101", "--atlas-path", "atlas.json",
                "--resume-missing", "--resume-batch-size", "2",
            ]

            with mock.patch("sys.argv", argv):
                with mock.patch("mvp.experiment_t0_sweep_full.run_t0_sweep_full", side_effect=fake_run_t0):
                    rc = exp.main()

            self.assertEqual(rc, 0)
            self.assertEqual(invoked_t0, ["2", "4"])

            payload = json.loads((runs_root / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["missing_pairs"], [])
            self.assertEqual(payload["last_attempted_pairs"], [{"seed": 101, "t0_ms": 2}, {"seed": 101, "t0_ms": 4}])
            self.assertEqual(payload["retry_counts"]["seed=101,t0_ms=2"], 1)
            self.assertEqual(payload["retry_counts"]["seed=101,t0_ms=4"], 1)

    def test_finalize_reports_blocked_pairs_when_retries_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            runs_root = tmp / "runs"
            scan_root = runs_root / "BASE_RUN" / "experiment"

            argv_run = [
                "prog", "--phase", "run", "--run-id", "BASE_RUN",
                "--runs-root", str(runs_root), "--scan-root", str(scan_root),
                "--inventory-seeds", "101", "--t0-grid-ms", "0,2",
                "--seed", "101", "--atlas-path", "atlas.json",
                "--resume-missing", "--max-retries-per-pair", "0",
            ]

            with mock.patch("sys.argv", argv_run):
                with mock.patch("mvp.experiment_t0_sweep_full.run_t0_sweep_full") as run_mock:
                    rc = exp.main()
            self.assertEqual(rc, 0)
            run_mock.assert_not_called()

            argv_finalize = [
                "prog", "--phase", "finalize", "--run-id", "BASE_RUN",
                "--runs-root", str(runs_root), "--scan-root", str(scan_root),
                "--inventory-seeds", "101", "--t0-grid-ms", "0,2",
                "--max-missing-abs", "0", "--max-missing-frac", "0.0",
                "--max-retries-per-pair", "0",
            ]

            with mock.patch("sys.argv", argv_finalize):
                with self.assertRaises(SystemExit) as ctx:
                    exp.main()
            self.assertEqual(ctx.exception.code, 2)

            payload = json.loads((runs_root / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["blocked_pairs"]), 2)
            self.assertIn("blocked_pairs", payload["decision"]["reason"])


class ExperimentT0SweepFullPlanAndLayoutTests(unittest.TestCase):
    def test_build_subrun_stage_cmds_includes_s2_before_s3b(self) -> None:
        cmds = exp.build_subrun_stage_cmds(
            python="python",
            s2_script="mvp/s2_ringdown_window.py",
            s3_script="mvp/s3_ringdown_estimates.py",
            s3b_script="mvp/s3b_multimode_estimates.py",
            s4c_script="mvp/s4c_kerr_consistency.py",
            subrun_id="rid__t0ms0008",
            event_id="GW150914",
            dt_start_s=0.003,
            duration_s=0.06,
            strain_npz="runs/BASE_RUN/s1_fetch_strain/outputs/strain.npz",
            n_bootstrap=200,
            s3b_seed=101,
            atlas_path="docs/ringdown/atlas/atlas_berti_v2_s4.json",
        )
        stage_names = [Path(cmd[1]).stem for cmd in cmds]
        self.assertIn("s2_ringdown_window", stage_names)
        self.assertLess(stage_names.index("s2_ringdown_window"), stage_names.index("s3b_multimode_estimates"))

    def test_require_subrun_window_meta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            subrun = Path(td) / "rid__t0ms0000"
            with self.assertRaises(FileNotFoundError):
                exp._require_subrun_window_meta(subrun)

            meta = subrun / "s2_ringdown_window" / "outputs" / "window_meta.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text("{}", encoding="utf-8")
            self.assertEqual(exp._require_subrun_window_meta(subrun), meta)

    def test_symlink_runsroot_base_run_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runsroot = Path(td) / "runsroot"
            runsroot.mkdir(parents=True, exist_ok=True)
            (runsroot / "BASE_RUN").symlink_to("../../..", target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, r"runsroot/BASE_RUN must be a real directory; found symlink to"):
                exp.ensure_seed_runsroot_layout(runsroot, "BASE_RUN")

    def test_clean_runsroot_base_run_is_real_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runsroot = Path(td) / "runsroot"
            runsroot.mkdir(parents=True, exist_ok=True)
            created = exp.ensure_seed_runsroot_layout(runsroot, "BASE_RUN")
            self.assertEqual(created, runsroot / "BASE_RUN")
            self.assertTrue(created.is_dir())
            self.assertFalse(created.is_symlink())


if __name__ == "__main__":
    unittest.main()
