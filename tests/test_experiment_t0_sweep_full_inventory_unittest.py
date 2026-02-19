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

    def test_inventory_incomplete_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scan_root = tmp / "runs" / "BASE_RUN" / "experiment"
            self._mk_payload(scan_root, 101, 0)

            args = self._args(tmp, "101,202", "0,2")
            with self.assertRaises(SystemExit) as ctx:
                exp.run_inventory_phase(args)
            self.assertEqual(ctx.exception.code, 2)

            out = tmp / "runs" / "BASE_RUN" / "experiment" / "derived" / "sweep_inventory.json"
            on_disk = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["expected_payload_count"], 4)
            self.assertEqual(on_disk["observed_payload_count"], 1)
            self.assertIn({"seed": 101, "t0_ms": 2}, on_disk["missing_pairs"])
            self.assertIn({"seed": 202, "t0_ms": 0}, on_disk["missing_pairs"])
            self.assertIn({"seed": 202, "t0_ms": 2}, on_disk["missing_pairs"])


if __name__ == "__main__":
    unittest.main()
