from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mvp import pipeline


class TestMultimodeWiring(unittest.TestCase):
    def test_s3b_declares_s3_estimates_input_in_command(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
            calls.append({"label": label, "args": list(args), "run_id": run_id})
            timeline["stages"].append(
                {
                    "stage": label,
                    "script": script,
                    "command": [script] + list(args),
                    "started_utc": "now",
                    "ended_utc": "now",
                    "duration_s": 0.0,
                    "returncode": 0,
                    "timed_out": False,
                }
            )
            pipeline._write_timeline(out_root, run_id, timeline)
            return 0

        with mock.patch.object(pipeline, "_run_stage", side_effect=fake_run_stage):
            with mock.patch.object(pipeline, "_parse_multimode_results", return_value={}):
                with mock.patch.dict("os.environ", {"BASURIN_RUNS_ROOT": str(Path("/tmp") / "basurin_wiring_test")}, clear=False):
                    rc, _ = pipeline.run_multimode_event(
                        event_id="GW150914",
                        atlas_path="mvp/test_atlas_fixture.json",
                        run_id="wire_a",
                        synthetic=True,
                        band_low=120.0,
                        band_high=280.0,
                    )
                    self.assertEqual(rc, 0)

                    rc, _ = pipeline.run_multimode_event(
                        event_id="GW150914",
                        atlas_path="mvp/test_atlas_fixture.json",
                        run_id="wire_b",
                        synthetic=True,
                        band_low=180.0,
                        band_high=360.0,
                    )
                    self.assertEqual(rc, 0)

        s3_calls = [c for c in calls if c["label"] == "s3_ringdown_estimates"]
        self.assertEqual(len(s3_calls), 2)
        self.assertIn("120.0", s3_calls[0]["args"])
        self.assertIn("280.0", s3_calls[0]["args"])
        self.assertIn("180.0", s3_calls[1]["args"])
        self.assertIn("360.0", s3_calls[1]["args"])

        s3b_calls = [c for c in calls if c["label"] == "s3b_multimode_estimates"]
        self.assertEqual(len(s3b_calls), 2)

        for call in s3b_calls:
            args = call["args"]
            run_id = call["run_id"]
            idx = args.index("--s3-estimates")
            self.assertEqual(args[idx + 1], f"{run_id}/s3_ringdown_estimates/outputs/estimates.json")


class TestMultimodePipelineBehavior(unittest.TestCase):
    def test_single_with_t0_sweep_missing_script_is_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"

            def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
                timeline["stages"].append(
                    {
                        "stage": label,
                        "script": script,
                        "command": [script] + list(args),
                        "started_utc": "now",
                        "ended_utc": "now",
                        "duration_s": 0.0,
                        "returncode": 0,
                        "timed_out": False,
                    }
                )
                pipeline._write_timeline(out_root, run_id, timeline)
                return 0

            with mock.patch.dict("os.environ", {"BASURIN_RUNS_ROOT": str(runs_root)}, clear=False):
                with mock.patch.object(pipeline, "_run_stage", side_effect=fake_run_stage):
                    with mock.patch.object(pipeline, "MVP_DIR", Path(td) / "missing_mvp"):
                        rc, run_id = pipeline.run_single_event(
                            event_id="GW150914",
                            atlas_path="mvp/test_atlas_fixture.json",
                            synthetic=True,
                            duration_s=4.0,
                            with_t0_sweep=True,
                        )

            self.assertEqual(rc, 0)
            timeline = json.loads((runs_root / run_id / "pipeline_timeline.json").read_text(encoding="utf-8"))
            exp_entries = [s for s in timeline["stages"] if s["stage"] == "experiment_t0_sweep"]
            self.assertEqual(len(exp_entries), 1)
            self.assertTrue(exp_entries[0]["best_effort"])
            self.assertEqual(exp_entries[0]["status"], "SKIPPED")
            self.assertIsNone(exp_entries[0]["returncode"])

    def test_multimode_writes_results_and_stage_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"

            def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
                stage_dir = out_root / run_id / label / "outputs"
                stage_dir.mkdir(parents=True, exist_ok=True)

                if label == "s3b_multimode_estimates":
                    (stage_dir / "multimode_estimates.json").write_text(
                        json.dumps({"results": {"verdict": "INSUFFICIENT_DATA"}}),
                        encoding="utf-8",
                    )
                if label == "s4c_kerr_consistency":
                    (stage_dir / "kerr_consistency.json").write_text(
                        json.dumps({"consistent_kerr_95": True, "chi_best_fit": 0.69, "d2_min": 1.2}),
                        encoding="utf-8",
                    )

                timeline["stages"].append(
                    {
                        "stage": label,
                        "script": script,
                        "command": [script] + list(args),
                        "started_utc": "now",
                        "ended_utc": "now",
                        "duration_s": 0.0,
                        "returncode": 0,
                        "timed_out": False,
                    }
                )
                pipeline._write_timeline(out_root, run_id, timeline)
                return 0

            with mock.patch.dict("os.environ", {"BASURIN_RUNS_ROOT": str(runs_root)}, clear=False):
                with mock.patch.object(pipeline, "_run_stage", side_effect=fake_run_stage):
                    rc, run_id = pipeline.run_multimode_event(
                        event_id="GW150914",
                        atlas_path="mvp/test_atlas_fixture.json",
                        synthetic=True,
                        duration_s=4.0,
                    )

            self.assertEqual(rc, 0)
            timeline = json.loads((runs_root / run_id / "pipeline_timeline.json").read_text(encoding="utf-8"))
            self.assertEqual([s["stage"] for s in timeline["stages"]], [
                "s0_oracle_mvp", "s1_fetch_strain", "s2_ringdown_window", "s3_ringdown_estimates",
                "s3b_multimode_estimates", "s4_geometry_filter", "s4c_kerr_consistency",
            ])
            self.assertEqual(timeline["multimode_results"]["extraction_quality"], "INSUFFICIENT_DATA")

    def test_multi_forwards_with_t0_sweep_to_each_event(self) -> None:
        calls = []

        def fake_single_event(**kwargs):
            calls.append(kwargs)
            return 0, f"run_{kwargs['event_id']}"

        def fake_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
            timeline["stages"].append(
                {
                    "stage": label,
                    "script": script,
                    "command": [script] + list(args),
                    "started_utc": "now",
                    "ended_utc": "now",
                    "duration_s": 0.0,
                    "returncode": 0,
                    "timed_out": False,
                }
            )
            pipeline._write_timeline(out_root, run_id, timeline)
            return 0

        with mock.patch.object(pipeline, "run_single_event", side_effect=fake_single_event):
            with mock.patch.object(pipeline, "_run_stage", side_effect=fake_stage):
                rc, _ = pipeline.run_multi_event(
                    events=["GW1", "GW2"],
                    atlas_path="mvp/test_atlas_fixture.json",
                    synthetic=True,
                    with_t0_sweep=True,
                )

        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(call["with_t0_sweep"])


if __name__ == "__main__":
    unittest.main()
