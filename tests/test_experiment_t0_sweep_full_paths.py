from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


if __name__ == "__main__":
    unittest.main()
