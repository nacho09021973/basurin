from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGeometryTableScript(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "mvp" / "s6_geometry_table.py"

    def _write_verdict_pass(self, run_root: Path) -> None:
        (run_root / "RUN_VALID").mkdir(parents=True, exist_ok=True)
        (run_root / "RUN_VALID" / "verdict.json").write_text(
            json.dumps({"verdict": "PASS"}, sort_keys=True),
            encoding="utf-8",
        )

    def _write_multimode(self, root: Path, payload: dict, stage_seed: int | None = None) -> None:
        outputs = root / "s3b_multimode_estimates" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "multimode_estimates.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        self._write_stage_summary(root, stage_seed)

    def _write_stage_summary(self, root: Path, stage_seed: int | None = None) -> None:
        stage_dir = root / "s3b_multimode_estimates"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage_seed is not None:
            (stage_dir / "stage_summary.json").write_text(
                json.dumps({"parameters": {"seed": stage_seed}}, sort_keys=True),
                encoding="utf-8",
            )

    def _run(self, base: Path, run_id: str, scan_root: Path, out_path: Path | None = None) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            str(self.script),
            "--run-id",
            run_id,
            "--runs-root",
            str(base / "runs"),
            "--scan-root",
            str(scan_root),
        ]
        if out_path is not None:
            cmd.extend(["--out-path", str(out_path)])
        return subprocess.run(cmd, cwd=base, check=True, capture_output=True, text=True)

    def _read_rows(self, tsv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
        lines = tsv_path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        rows = []
        for line in lines[1:]:
            vals = line.split("\t")
            rows.append(dict(zip(header, vals)))
        return header, rows

    def test_scan_root_global_and_seed_root_yield_same_rows_for_that_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            seed_root = run_root / "experiment" / "t0_sweep_full_seed101"
            nested_root = seed_root / "runsroot" / run_id / "experiment" / "t0_sweep_full"
            p6 = nested_root / f"runs/{run_id}__t0ms0006"
            p8 = nested_root / f"runs/{run_id}__t0ms0008"

            payload6 = {
                "results": {"verdict": "OK", "quality_flags": ["z", "a"]},
                "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 2.6, "cv_Q": 0.26, "valid_fraction": 0.76}}}],
            }
            payload8 = {
                "results": {"verdict": "OK", "quality_flags": ["b"]},
                "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 2.8, "cv_Q": 0.28, "valid_fraction": 0.78}}}],
            }
            self._write_multimode(p6, payload6, stage_seed=101)
            self._write_multimode(p8, payload8, stage_seed=101)

            out_exp = run_root / "experiment" / "derived" / "geometry_table.tsv"
            out_seed = seed_root / "derived" / "geometry_table.tsv"
            self._run(base, run_id, run_root / "experiment", out_exp)
            self._run(base, run_id, seed_root, out_seed)

            _, exp_rows = self._read_rows(out_exp)
            _, seed_rows = self._read_rows(out_seed)
            exp_seed101 = [r for r in exp_rows if r["seed"] == "101"]

            def norm(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
                fields = ["seed", "t0_ms", "s3b_seed_param", "lnQ_span", "cv_Q", "valid_fraction", "verdict", "flags"]
                return sorted(tuple(r[k] for k in fields) for r in rows)

            self.assertEqual(norm(exp_seed101), norm(seed_rows))

    def test_t0ms_variable_digits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            exp = run_root / "experiment" / "t0_sweep_full_seed101"
            self._write_multimode(exp / "segment__t0ms8", {"results": {}, "modes": []}, stage_seed=101)
            self._write_multimode(exp / "segment__t0ms0008", {"results": {}, "modes": []}, stage_seed=101)

            out = exp / "derived" / "geometry_table.tsv"
            self._run(base, run_id, exp, out)
            _, rows = self._read_rows(out)

            self.assertEqual([r["t0_ms"] for r in rows], ["8", "8"])

    def test_missing_mode_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            exp = run_root / "experiment" / "t0_sweep_full_seed101"
            self._write_multimode(exp / "segment__t0ms0001", {"results": {"verdict": "WARN", "quality_flags": ["f"]}}, stage_seed=101)

            out = exp / "derived" / "geometry_table.tsv"
            self._run(base, run_id, exp, out)
            header, rows = self._read_rows(out)

            self.assertEqual(
                header,
                ["seed", "t0_ms", "s3b_seed_param", "lnQ_span", "cv_Q", "valid_fraction", "verdict", "flags", "path"],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["lnQ_span"], "na")
            self.assertEqual(rows[0]["cv_Q"], "na")
            self.assertEqual(rows[0]["valid_fraction"], "na")
            self.assertEqual(rows[0]["verdict"], "WARN")
            self.assertEqual(rows[0]["flags"], "f")
            self.assertTrue(rows[0]["path"].endswith("multimode_estimates.json"))

    def test_deterministic_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            exp = run_root / "experiment"
            self._write_multimode(
                exp / "t0_sweep_full_seed202" / "segment__t0ms0010",
                {"results": {}, "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 1.0}}}]},
                stage_seed=202,
            )
            self._write_multimode(
                exp / "t0_sweep_full_seed101" / "segment__t0ms0002",
                {"results": {}, "modes": [{"label": "221", "fit": {"stability": {"lnQ_span": 2.0}}}]},
                stage_seed=101,
            )

            out = run_root / "experiment" / "derived" / "geometry_table.tsv"
            self._run(base, run_id, exp, out)
            digest1 = hashlib.sha256(out.read_bytes()).hexdigest()

            self._run(base, run_id, exp, out)
            digest2 = hashlib.sha256(out.read_bytes()).hexdigest()

            self.assertEqual(digest1, digest2)

    def test_stage_summary_without_payload_writes_row_with_missing_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            exp = run_root / "experiment" / "t0_sweep_full_seed101" / "segment__t0ms0001"
            self._write_stage_summary(exp, stage_seed=101)

            out = run_root / "experiment" / "derived" / "geometry_table.tsv"
            self._run(base, run_id, run_root / "experiment", out)
            _, rows = self._read_rows(out)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], "101")
            self.assertEqual(rows[0]["t0_ms"], "1")
            self.assertEqual(rows[0]["s3b_seed_param"], "101")
            self.assertEqual(rows[0]["lnQ_span"], "na")
            self.assertEqual(rows[0]["cv_Q"], "na")
            self.assertEqual(rows[0]["valid_fraction"], "na")
            self.assertEqual(rows[0]["verdict"], "na")
            self.assertEqual(rows[0]["flags"], "missing_multimode_estimates_json")

            summary = json.loads((out.parent / "stage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["files_skipped_missing_payload"], 1)

    def test_scan_root_without_stage_summary_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            scan_root = run_root / "experiment"
            scan_root.mkdir(parents=True, exist_ok=True)
            out = run_root / "experiment" / "derived" / "geometry_table.tsv"

            cmd = [
                sys.executable,
                str(self.script),
                "--run-id",
                run_id,
                "--runs-root",
                str(base / "runs"),
                "--scan-root",
                str(scan_root),
                "--out-path",
                str(out),
            ]
            proc = subprocess.run(cmd, cwd=base, capture_output=True, text=True)

            self.assertEqual(proc.returncode, 2)
            self.assertIn("scan_root_abs=", proc.stderr)
            self.assertIn("s3b_multimode_estimates/stage_summary.json", proc.stderr)

    def test_parse_context_prefers_stage_summary_params_over_path_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            exp = run_root / "experiment" / "generic_seedless" / "segment_without_t0"
            outputs = exp / "s3b_multimode_estimates" / "outputs"
            outputs.mkdir(parents=True, exist_ok=True)
            (outputs / "multimode_estimates.json").write_text(
                json.dumps({"results": {}, "modes": []}, sort_keys=True),
                encoding="utf-8",
            )
            (exp / "s3b_multimode_estimates" / "stage_summary.json").write_text(
                json.dumps({"params": {"seed": 777, "t0_ms": 42}}, sort_keys=True),
                encoding="utf-8",
            )

            out = run_root / "experiment" / "derived" / "geometry_table.tsv"
            self._run(base, run_id, run_root / "experiment", out)
            _, rows = self._read_rows(out)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], "777")
            self.assertEqual(rows[0]["t0_ms"], "42")

    def test_symlink_intermediate_directory_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "BASE_RUN"
            run_root = base / "runs" / run_id
            self._write_verdict_pass(run_root)

            scan_root = run_root / "experiment" / "t0_sweep_full_seed101"
            self._write_multimode(
                scan_root / "segment__t0ms0001",
                {"results": {"verdict": "OK"}, "modes": []},
                stage_seed=101,
            )

            payload_dir = base / "payload_store" / "segment__t0ms0002"
            self._write_multimode(
                payload_dir,
                {"results": {"verdict": "OK"}, "modes": []},
                stage_seed=101,
            )
            (scan_root / "via_symlink").symlink_to(payload_dir, target_is_directory=True)

            out = run_root / "experiment" / "derived" / "geometry_table.tsv"
            self._run(base, run_id, scan_root, out)
            _, rows = self._read_rows(out)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["t0_ms"], "1")


if __name__ == "__main__":
    unittest.main()
