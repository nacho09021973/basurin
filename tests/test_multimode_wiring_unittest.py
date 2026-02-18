from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
