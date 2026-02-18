from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGeometryTableScript(unittest.TestCase):
    def test_builds_sorted_tsv_from_experiment_tree(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "mvp" / "s6_geometry_table.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id

            (run_root / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (run_root / "RUN_VALID" / "verdict.json").write_text(
                json.dumps({"verdict": "PASS"}, sort_keys=True),
                encoding="utf-8",
            )

            p_seed202 = run_root / "experiment" / "t0_sweep_full_seed202" / "segment__t0ms0050" / "s3b_multimode_estimates"
            p_seed101 = run_root / "experiment" / "t0_sweep_full_seed101" / "segment__t0ms0000" / "s3b_multimode_estimates"
            (p_seed202 / "outputs").mkdir(parents=True, exist_ok=True)
            (p_seed101 / "outputs").mkdir(parents=True, exist_ok=True)

            payload_202 = {
                "results": {"verdict": "OK", "quality_flags": ["b", "a"]},
                "modes": [
                    {
                        "label": "221",
                        "fit": {"stability": {"lnQ_span": 2.2, "cv_Q": 0.4, "valid_fraction": 0.7}},
                    }
                ],
            }
            payload_101 = {
                "results": {"verdict": "OK", "quality_flags": ["x"]},
                "modes": [
                    {
                        "label": "221",
                        "fit": {"stability": {"lnQ_span": 1.1, "cv_Q": 0.2, "valid_fraction": 0.9}},
                    }
                ],
            }

            (p_seed202 / "outputs" / "multimode_estimates.json").write_text(
                json.dumps(payload_202, sort_keys=True), encoding="utf-8"
            )
            (p_seed101 / "outputs" / "multimode_estimates.json").write_text(
                json.dumps(payload_101, sort_keys=True), encoding="utf-8"
            )
            (p_seed202 / "stage_summary.json").write_text(
                json.dumps({"parameters": {"seed": 202}}, sort_keys=True), encoding="utf-8"
            )
            (p_seed101 / "stage_summary.json").write_text(
                json.dumps({"parameters": {"seed": 101}}, sort_keys=True), encoding="utf-8"
            )

            cmd = [
                sys.executable,
                str(script),
                "--run-id",
                run_id,
                "--scan-root",
                f"runs/{run_id}/experiment",
            ]
            subprocess.run(cmd, cwd=base, check=True)

            tsv_path = run_root / "experiment" / "derived" / "geometry_table.tsv"
            self.assertTrue(tsv_path.exists())

            lines = tsv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(
                lines[0],
                "seed\tt0_ms\ts3b_seed_param\tlnQ_span\tcv_Q\tvalid_fraction\tverdict\tflags\tpath",
            )

            # Stable sort by numeric seed first (101 before 202).
            self.assertTrue(lines[1].startswith("101\t0000\t101\t1.1"))
            self.assertIn("a,b", lines[2])
            self.assertTrue(lines[2].startswith("202\t0050\t202\t2.2"))


if __name__ == "__main__":
    unittest.main()
