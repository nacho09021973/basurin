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
    def test_compute_experiment_paths_follow_active_runsroot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runsroot = Path(tmpdir) / "RUNSROOT"
            os.environ["BASURIN_RUNS_ROOT"] = str(runsroot)
            try:
                exp = importlib.import_module("mvp.experiment_t0_sweep_full")
                out_root, stage_dir, subruns_root = exp.compute_experiment_paths("BASE_RUN")

                resolved_root = exp.resolve_out_root("runs")
                self.assertEqual(out_root, resolved_root)
                self.assertEqual(stage_dir, out_root / "BASE_RUN" / "experiment" / "t0_sweep_full")
                self.assertEqual(subruns_root, stage_dir / "runs")
            finally:
                os.environ.pop("BASURIN_RUNS_ROOT", None)


if __name__ == "__main__":
    unittest.main()
