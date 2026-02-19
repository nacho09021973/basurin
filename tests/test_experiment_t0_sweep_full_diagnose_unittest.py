from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mvp import experiment_t0_sweep_full as exp


class TestExperimentT0SweepFullDiagnose(unittest.TestCase):
    def test_preflight_report_written_and_blocks_missing_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "BASE"
            run_dir = root / "runs" / run_id
            (run_dir / "s2_ringdown_window" / "outputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "s2_ringdown_window" / "manifest.json").write_text("{}", encoding="utf-8")
            (run_dir / "s2_ringdown_window" / "outputs" / "H1_rd.npz").write_bytes(b"npz")
            args = SimpleNamespace(
                run_id=run_id,
                base_runs_root=root / "runs",
                runs_root=str(root / "runs"),
                scan_root=None,
                detector="auto",
                t0_grid_ms="8",
                t0_start_ms=0,
                t0_stop_ms=0,
                t0_step_ms=1,
                n_bootstrap=10,
                seed=606,
                atlas_path="atlas.h5",
                stage_timeout_s=1,
            )

            with (
                mock.patch.object(exp, "require_run_valid"),
                mock.patch.object(exp, "_pick_detector", return_value=("H1", run_dir / "s2_ringdown_window" / "outputs" / "H1_rd.npz")),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    exp.run_t0_sweep_full(args)

            self.assertEqual(ctx.exception.code, 2)
            preflight = root / "runs" / run_id / "experiment" / "derived" / "preflight_report.json"
            self.assertTrue(preflight.exists())
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertFalse(payload["decision"]["can_run"])
            self.assertEqual(payload["base_artifacts"]["RUN_VALID"]["verdict"], "MISSING")

            (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
            with (
                mock.patch.object(exp, "require_run_valid"),
                mock.patch.object(exp, "_pick_detector", return_value=("H1", run_dir / "s2_ringdown_window" / "outputs" / "H1_rd.npz")),
            ):
                with self.assertRaises(SystemExit) as ctx2:
                    exp.run_t0_sweep_full(args)
            self.assertEqual(ctx2.exception.code, 2)
            payload2 = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertFalse(payload2["decision"]["can_run"])
            self.assertFalse(payload2["base_artifacts"]["s1_strain_npz"]["exists"])

    def test_phase_diagnose_writes_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "BASE"
            scan_root = root / "runs" / run_id / "experiment" / "t0_sweep_full_seed606" / "runs"

            complete = scan_root / f"{run_id}__t0ms0000"
            missing_window = scan_root / f"{run_id}__t0ms0008"
            missing_mm = scan_root / f"{run_id}__t0ms0010"

            for subrun in (complete, missing_window, missing_mm):
                (subrun / "s2_ringdown_window" / "outputs").mkdir(parents=True, exist_ok=True)
                (subrun / "s3_ringdown_estimates" / "outputs").mkdir(parents=True, exist_ok=True)
                (subrun / "s3b_multimode_estimates" / "outputs").mkdir(parents=True, exist_ok=True)

            (complete / "s2_ringdown_window" / "outputs" / "window_meta.json").write_text("{}", encoding="utf-8")
            (complete / "s3_ringdown_estimates" / "outputs" / "estimates.json").write_text("{}", encoding="utf-8")
            (complete / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            (missing_window / "s3_ringdown_estimates" / "outputs" / "estimates.json").write_text("{}", encoding="utf-8")
            (missing_window / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json").write_text("{}", encoding="utf-8")

            (missing_mm / "s2_ringdown_window" / "outputs" / "window_meta.json").write_text("{}", encoding="utf-8")
            (missing_mm / "s3_ringdown_estimates" / "outputs" / "estimates.json").write_text("{}", encoding="utf-8")

            args = SimpleNamespace(run_id=run_id, runs_root=str(root / "runs"), scan_root=str(scan_root))
            payload = exp.run_diagnose_phase(args)

            self.assertEqual(payload["stage_missing_counts"]["window_meta"], 1)
            self.assertEqual(payload["stage_missing_counts"]["multimode_estimates"], 1)
            self.assertEqual(payload["stage_missing_counts"]["estimates"], 0)

            report = root / "runs" / run_id / "experiment" / "derived" / "diagnose_report.json"
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
