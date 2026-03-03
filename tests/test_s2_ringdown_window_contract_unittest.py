from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    np = None

from basurin_io import write_json_atomic
from mvp.contracts import check_inputs, finalize, init_stage
from mvp.s2_ringdown_window import _ensure_window_meta_contract, main as s2_main


@unittest.skipIf(np is None, "numpy is required for s2 ringdown window contract tests")
class TestS2RingdownWindowContract(unittest.TestCase):
    def _run_s2(
        self,
        runs_root: Path,
        run_id: str,
        *,
        clip_window: bool,
        event_id: str = "GW150914",
        offline: bool = False,
        catalog_payload: dict[str, object] | None = None,
    ) -> int:
        old = os.environ.get("BASURIN_RUNS_ROOT")
        old_argv = sys.argv[:]
        window_catalog = runs_root / "window_catalog.json"
        window_catalog.write_text(
            json.dumps(catalog_payload if catalog_payload is not None else {"GW150914": {"t0_gps": 1_000_100.0}}),
            encoding="utf-8",
        )
        argv = [
            "s2_ringdown_window.py",
            "--run", run_id,
            "--event-id", event_id,
            "--dt-start-s", "0.003",
            "--duration-s", "0.06",
            "--window-catalog", str(window_catalog),
        ]
        if clip_window:
            argv.append("--clip-window")
        if offline:
            argv.append("--offline")
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        sys.argv = argv
        try:
            return s2_main()
        finally:
            sys.argv = old_argv
            if old is None:
                os.environ.pop("BASURIN_RUNS_ROOT", None)
            else:
                os.environ["BASURIN_RUNS_ROOT"] = old

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

    def test_out_of_range_without_clip_fails_with_window_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "s2_oob_fail"

            rv = runs_root / run_id / "RUN_VALID"
            rv.mkdir(parents=True, exist_ok=True)
            (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

            s1_outputs = runs_root / run_id / "s1_fetch_strain" / "outputs"
            s1_outputs.mkdir(parents=True, exist_ok=True)
            np.savez(
                s1_outputs / "strain.npz",
                H1=np.zeros(131072, dtype=np.float64),
                gps_start=np.float64(1_000_000.0),
                sample_rate_hz=np.float64(4096.0),
            )

            rc = self._run_s2(runs_root, run_id, clip_window=False)
            self.assertEqual(rc, 2)

            summary = json.loads(
                (runs_root / run_id / "s2_ringdown_window" / "stage_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["verdict"], "FAIL")
            self.assertIn("Window out of range for H1", summary["error"])

            status = summary["results"]["window_status"]["H1"]
            self.assertFalse(status["ok"])
            self.assertEqual(status["reason"], "out_of_range")
            self.assertEqual(status["n"], 131072)

    def test_out_of_range_with_clip_passes_and_marks_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "s2_oob_clip"

            rv = runs_root / run_id / "RUN_VALID"
            rv.mkdir(parents=True, exist_ok=True)
            (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

            s1_outputs = runs_root / run_id / "s1_fetch_strain" / "outputs"
            s1_outputs.mkdir(parents=True, exist_ok=True)
            np.savez(
                s1_outputs / "strain.npz",
                H1=np.ones(131072, dtype=np.float64),
                gps_start=np.float64(1_000_000.0),
                sample_rate_hz=np.float64(4096.0),
            )

            rc = self._run_s2(runs_root, run_id, clip_window=True)
            self.assertEqual(rc, 0)

            summary = json.loads(
                (runs_root / run_id / "s2_ringdown_window" / "stage_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["verdict"], "PASS")

            status = summary["results"]["window_status"]["H1"]
            self.assertTrue(status["ok"])
            self.assertEqual(status["reason"], "clipped")
            self.assertTrue(status["clipped"])
            self.assertGreater(status["clip_right_samples"], 0)


    def test_offline_missing_t0_sources_fails_with_stable_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "s2_offline_missing_t0"
            rv = runs_root / run_id / "RUN_VALID"
            rv.mkdir(parents=True, exist_ok=True)
            (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

            s1_outputs = runs_root / run_id / "s1_fetch_strain" / "outputs"
            s1_outputs.mkdir(parents=True, exist_ok=True)
            np.savez(
                s1_outputs / "strain.npz",
                H1=np.ones(4096, dtype=np.float64),
                gps_start=np.float64(1_000_000.0),
                sample_rate_hz=np.float64(4096.0),
            )

            rc = self._run_s2(
                runs_root,
                run_id,
                clip_window=False,
                event_id="GW_NOT_IN_CATALOG",
                offline=True,
                catalog_payload={"GW150914": {"t0_gps": 1_000_100.0}},
            )
            self.assertEqual(rc, 2)

            summary = json.loads((runs_root / run_id / "s2_ringdown_window" / "stage_summary.json").read_text("utf-8"))
            self.assertEqual(summary["verdict"], "FAIL")
            self.assertIn("missing_t0_gps_offline", summary["error"])

    def test_online_fetch_writes_external_input_cache_and_records_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "s2_online_t0_cache"
            rv = runs_root / run_id / "RUN_VALID"
            rv.mkdir(parents=True, exist_ok=True)
            (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

            s1_outputs = runs_root / run_id / "s1_fetch_strain" / "outputs"
            s1_outputs.mkdir(parents=True, exist_ok=True)
            np.savez(
                s1_outputs / "strain.npz",
                H1=np.ones(4096, dtype=np.float64),
                gps_start=np.float64(1_000_000.0),
                sample_rate_hz=np.float64(4096.0),
            )

            with mock.patch("mvp.s2_ringdown_window._fetch_gwosc_event_gps", return_value=1_000_000.0):
                rc = self._run_s2(
                    runs_root,
                    run_id,
                    clip_window=False,
                    event_id="GW_REMOTE_ONLY",
                    offline=False,
                    catalog_payload={"GW150914": {"t0_gps": 1_000_100.0}},
                )
            self.assertEqual(rc, 0)

            cache_file = runs_root / run_id / "external_inputs" / "gwosc" / "event_time" / "GW_REMOTE_ONLY.json"
            self.assertTrue(cache_file.exists())

            summary = json.loads((runs_root / run_id / "s2_ringdown_window" / "stage_summary.json").read_text("utf-8"))
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(summary["results"]["gwosc_cache_path"], str(cache_file))
            cached_input = next(item for item in summary["inputs"] if item["label"] == "gwosc_event_time")
            self.assertEqual(cached_input["path"], "external_inputs/gwosc/event_time/GW_REMOTE_ONLY.json")
            self.assertEqual(len(cached_input["sha256"]), 64)

            manifest = json.loads((runs_root / run_id / "s2_ringdown_window" / "manifest.json").read_text("utf-8"))
            manifest_inputs = manifest.get("inputs", [])
            self.assertTrue(any(i.get("path") == "external_inputs/gwosc/event_time/GW_REMOTE_ONLY.json" for i in manifest_inputs))


if __name__ == "__main__":
    unittest.main()
