import unittest
from pathlib import Path

from mvp import experiment_t0_sweep_full as exp


class TestT0SweepFullPlan(unittest.TestCase):
    def test_build_subrun_stage_cmds_includes_s2_before_s3b(self) -> None:
        cmds = exp.build_subrun_stage_cmds(
            python="python",
            s3_script="mvp/s3_ringdown_estimates.py",
            s3b_script="mvp/s3b_multimode_estimates.py",
            s4c_script="mvp/s4c_kerr_consistency.py",
            subrun_id="rid__t0ms0008",
            n_bootstrap=200,
            s3b_seed=101,
            atlas_path="docs/ringdown/atlas/atlas_berti_v2_s4.json",
        )
        stage_names = [Path(cmd[1]).stem for cmd in cmds]
        self.assertIn("shadow_s2_materialized", stage_names)
        self.assertLess(stage_names.index("shadow_s2_materialized"), stage_names.index("s3b_multimode_estimates"))

    def test_build_subrun_execution_plan_has_subrun_local_window_meta(self) -> None:
        subrun_dir = Path("/tmp/example/runs/rid__t0ms0008")
        plan = exp.build_subrun_execution_plan(
            subrun_dir=subrun_dir,
            python="python",
            s3_script="mvp/s3_ringdown_estimates.py",
            s3b_script="mvp/s3b_multimode_estimates.py",
            s4c_script="mvp/s4c_kerr_consistency.py",
            subrun_id="rid__t0ms0008",
            n_bootstrap=10,
            s3b_seed=7,
            atlas_path="atlas.json",
        )

        expected = subrun_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
        self.assertEqual(plan["expected_inputs"]["s2_window_meta"], str(expected))
        self.assertIn("rid__t0ms0008", plan["expected_inputs"]["s2_window_meta"])

    def test_require_subrun_window_meta(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subrun = Path(td) / "rid__t0ms0000"
            with self.assertRaises(FileNotFoundError):
                exp._require_subrun_window_meta(subrun)

            meta = subrun / "s2_ringdown_window" / "outputs" / "window_meta.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text("{}", encoding="utf-8")
            found = exp._require_subrun_window_meta(subrun)
            self.assertEqual(found, meta)


if __name__ == "__main__":
    unittest.main()
