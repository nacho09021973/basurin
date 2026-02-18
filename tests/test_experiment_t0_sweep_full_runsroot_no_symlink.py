from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path


class TestExperimentT0SweepFullRunsrootNoSymlink(unittest.TestCase):
    def test_symlink_runsroot_base_run_raises(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            runsroot = Path(tmpdir) / "runsroot"
            runsroot.mkdir(parents=True, exist_ok=True)
            run_id = "BASE_RUN"
            legacy_target = "../../.."
            (runsroot / run_id).symlink_to(legacy_target, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, r"runsroot/BASE_RUN must be a real directory; found symlink to"):
                exp.ensure_seed_runsroot_layout(runsroot, run_id)

    def test_clean_runsroot_base_run_is_real_directory(self) -> None:
        exp = importlib.import_module("mvp.experiment_t0_sweep_full")

        with tempfile.TemporaryDirectory() as tmpdir:
            runsroot = Path(tmpdir) / "runsroot"
            runsroot.mkdir(parents=True, exist_ok=True)
            run_id = "BASE_RUN"

            created = exp.ensure_seed_runsroot_layout(runsroot, run_id)

            self.assertEqual(created, runsroot / run_id)
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())
            self.assertFalse(created.is_symlink())


if __name__ == "__main__":
    unittest.main()
