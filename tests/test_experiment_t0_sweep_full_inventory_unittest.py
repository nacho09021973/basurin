from __future__ import annotations

import json
import tempfile
import unittest
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
            self._mk_payload(scan_root, 101, 0)

            args = self._args(tmp, "101,202", "0,2")
            args.phase = "run"
            payload = exp.run_inventory_phase(args)

            self.assertEqual(payload["status"], "IN_PROGRESS")
            self.assertGreater(payload["missing_abs"], 0)

    def test_phase_inventory_incomplete_exit_0(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            self._mk_payload(scan_root, 101, 0)

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
            self._mk_payload(scan_root, 101, 0)

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


if __name__ == "__main__":
    unittest.main()
