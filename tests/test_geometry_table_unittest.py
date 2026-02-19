from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGeometryTableScript(unittest.TestCase):
    def _write_seed101_tree(self, run_root: Path) -> tuple[str, Path]:
        run_id = run_root.name
        (run_root / "RUN_VALID").mkdir(parents=True, exist_ok=True)
        (run_root / "RUN_VALID" / "verdict.json").write_text(
            json.dumps({"verdict": "PASS"}, sort_keys=True),
            encoding="utf-8",
        )

        seed_root = run_root / "experiment" / "t0_sweep_full_seed101"
        p_t0_0006 = seed_root / "segment__t0ms0006" / "s3b_multimode_estimates"
        p_t0_0008 = seed_root / "segment__t0ms0008" / "s3b_multimode_estimates"
        (p_t0_0006 / "outputs").mkdir(parents=True, exist_ok=True)
        (p_t0_0008 / "outputs").mkdir(parents=True, exist_ok=True)

        payload = {
            "results": {"verdict": "OK", "quality_flags": []},
            "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 1.0}}}],
        }
        for p in (p_t0_0006, p_t0_0008):
            (p / "outputs" / "multimode_estimates.json").write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            (p / "stage_summary.json").write_text(
                json.dumps({"parameters": {"seed": 101}}, sort_keys=True),
                encoding="utf-8",
            )

        return run_id, seed_root

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

    def test_ignores_symlinked_directories_during_scan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "mvp" / "s6_geometry_table.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            scan_root = run_root / "experiment"
            outside_root = base / "outside_seed"

            (run_root / "RUN_VALID").mkdir(parents=True, exist_ok=True)
            (run_root / "RUN_VALID" / "verdict.json").write_text(
                json.dumps({"verdict": "PASS"}, sort_keys=True),
                encoding="utf-8",
            )

            ok_path = scan_root / "ok" / "segment__t0ms0001" / "s3b_multimode_estimates" / "outputs"
            (ok_path).mkdir(parents=True, exist_ok=True)
            (ok_path / "multimode_estimates.json").write_text(
                json.dumps(
                    {
                        "results": {"verdict": "OK", "quality_flags": []},
                        "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 1.23}}}],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            outside_path = outside_root / "bad" / "segment__t0ms9999" / "s3b_multimode_estimates" / "outputs"
            outside_path.mkdir(parents=True, exist_ok=True)
            (outside_path / "multimode_estimates.json").write_text(
                json.dumps(
                    {
                        "results": {"verdict": "OK", "quality_flags": ["outside"]},
                        "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 999}}}],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            (scan_root / "link").symlink_to(outside_root, target_is_directory=True)

            cmd = [
                sys.executable,
                str(script),
                "--run-id",
                run_id,
                "--scan-root",
                str(scan_root),
            ]
            proc = subprocess.run(cmd, cwd=base, check=True, capture_output=True, text=True)

            tsv_path = run_root / "experiment" / "derived" / "geometry_table.tsv"
            lines = tsv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("SKIP_SYMLINK_DIR", proc.stderr)
            self.assertIn("\tok/segment__t0ms0001/s3b_multimode_estimates/outputs/multimode_estimates.json", lines[1])
            self.assertNotIn("999", "\n".join(lines))
            self.assertNotIn("outside", "\n".join(lines))

    def test_seed_scan_root_infers_seed_from_scan_root_basename(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "mvp" / "s6_geometry_table.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_root = base / "runs" / "BASE_RUN"
            run_id, seed_root = self._write_seed101_tree(run_root)

            out_path = run_root / "experiment" / "t0_sweep_full_seed101" / "derived" / "geometry_table.tsv"
            cmd = [
                sys.executable,
                str(script),
                "--run-id",
                run_id,
                "--scan-root",
                str(seed_root),
                "--out-path",
                str(out_path),
            ]
            subprocess.run(cmd, cwd=base, check=True)

            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[1].split("\t")[0], "101")
            self.assertEqual(lines[2].split("\t")[0], "101")

    def test_seed_counts_match_between_seed_scan_root_and_experiment_scan_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "mvp" / "s6_geometry_table.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_root = base / "runs" / "BASE_RUN"
            run_id, seed_root = self._write_seed101_tree(run_root)
            experiment_root = run_root / "experiment"

            seed_out = run_root / "experiment" / "t0_sweep_full_seed101" / "derived" / "geometry_table.tsv"
            exp_out = run_root / "experiment" / "derived" / "geometry_table.tsv"

            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-id",
                    run_id,
                    "--scan-root",
                    str(seed_root),
                    "--out-path",
                    str(seed_out),
                ],
                cwd=base,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-id",
                    run_id,
                    "--scan-root",
                    str(experiment_root),
                    "--out-path",
                    str(exp_out),
                ],
                cwd=base,
                check=True,
            )

            def _seed_counts(tsv_path: Path) -> dict[str, int]:
                counts: dict[str, int] = {}
                lines = tsv_path.read_text(encoding="utf-8").splitlines()[1:]
                for line in lines:
                    seed = line.split("\t", 1)[0]
                    counts[seed] = counts.get(seed, 0) + 1
                return counts

            self.assertEqual(_seed_counts(seed_out), {"101": 2})
            self.assertEqual(_seed_counts(exp_out), {"101": 2})


if __name__ == "__main__":
    unittest.main()
