from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import write_json_atomic
from mvp.contracts import init_stage


@unittest.skipUnless(hasattr(os, "symlink"), "symlink not supported on this platform")
class SymlinkGuardTests(unittest.TestCase):
    def test_write_json_atomic_blocks_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            evil = tmpdir / "evil"
            os.symlink(tmpdir, evil)

            target = evil / "sub" / "file.json"
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                write_json_atomic(target, {"ok": False})

    def test_init_stage_blocks_symlink_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            runs_root = tmpdir / "runs"
            runs_root.mkdir(parents=True)

            run_id = "test_symlink_run"
            real_run = tmpdir / "real_run"
            real_run.mkdir(parents=True)
            os.symlink(real_run, runs_root / run_id)

            old = os.environ.get("BASURIN_RUNS_ROOT")
            os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
            try:
                with self.assertRaises(SystemExit) as cm:
                    init_stage(run_id, "s2_ringdown_window")
                self.assertEqual(cm.exception.code, 2)
            finally:
                if old is None:
                    os.environ.pop("BASURIN_RUNS_ROOT", None)
                else:
                    os.environ["BASURIN_RUNS_ROOT"] = old

    def test_positive_without_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            runs_root = tmpdir / "runs"
            run_id = "ok_run"

            run_valid = runs_root / run_id / "RUN_VALID"
            run_valid.mkdir(parents=True)
            (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

            out = runs_root / run_id / "s1_fetch_strain" / "outputs" / "ok.json"
            write_json_atomic(out, {"ok": True})
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["ok"], True)

            old = os.environ.get("BASURIN_RUNS_ROOT")
            os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
            try:
                ctx = init_stage(run_id, "s2_ringdown_window")
            finally:
                if old is None:
                    os.environ.pop("BASURIN_RUNS_ROOT", None)
                else:
                    os.environ["BASURIN_RUNS_ROOT"] = old

            self.assertTrue(ctx.stage_dir.exists())
            self.assertTrue(ctx.outputs_dir.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
