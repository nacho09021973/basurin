from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.contracts import check_inputs, finalize, init_stage
from mvp.s2_ringdown_window import _ensure_window_meta_contract


class TestS2RingdownWindowContract(unittest.TestCase):
    def test_window_meta_written_and_registered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "contract_s2_ok"
            rv = runs_root / run_id / "RUN_VALID"
            rv.mkdir(parents=True, exist_ok=True)
            (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

            strain = runs_root / run_id / "s1_fetch_strain" / "outputs" / "strain.npz"
            strain.parent.mkdir(parents=True, exist_ok=True)
            strain.write_bytes(b"fake-npz")

            old = os.environ.get("BASURIN_RUNS_ROOT")
            os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
            try:
                ctx = init_stage(run_id, "s2_ringdown_window", params={"event_id": "GW150914"})
                check_inputs(ctx, {"strain_npz": strain})

                window_meta_path = ctx.outputs_dir / "window_meta.json"
                write_json_atomic(window_meta_path, {"event_id": "GW150914", "n_samples": 16})
                artifacts = {"window_meta": window_meta_path}
                _ensure_window_meta_contract(ctx, artifacts, strain)
                finalize(ctx, artifacts, verdict="PASS", results={"n_samples": 16})
            finally:
                if old is None:
                    os.environ.pop("BASURIN_RUNS_ROOT", None)
                else:
                    os.environ["BASURIN_RUNS_ROOT"] = old

            self.assertTrue(window_meta_path.exists())

            manifest = json.loads((ctx.stage_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"]["window_meta"], "outputs/window_meta.json")

            summary = json.loads((ctx.stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["verdict"], "PASS")
            self.assertTrue(summary["inputs"])
            output_entry = next(item for item in summary["outputs"] if item["path"].endswith("window_meta.json"))
            self.assertEqual(len(output_entry["sha256"]), 64)

    def test_guardrail_pass_without_outputs_becomes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "guardrail_fail"

            old = os.environ.get("BASURIN_RUNS_ROOT")
            os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
            try:
                ctx = init_stage(run_id, "s1_fetch_strain", params={})
                finalize(ctx, artifacts={}, verdict="PASS")
            finally:
                if old is None:
                    os.environ.pop("BASURIN_RUNS_ROOT", None)
                else:
                    os.environ["BASURIN_RUNS_ROOT"] = old

            summary = json.loads((ctx.stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["verdict"], "FAIL")
            self.assertEqual(summary["error"], "PASS_WITHOUT_OUTPUTS")


if __name__ == "__main__":
    unittest.main()
