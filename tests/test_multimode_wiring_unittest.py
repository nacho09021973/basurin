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
            stage_dir = out_root / run_id / label / "outputs"
            stage_dir.mkdir(parents=True, exist_ok=True)
            if label == "s8_family_router":
                (stage_dir / "family_router.json").write_text(
                    json.dumps({"primary_family": "GR_KERR_BH", "families_to_run": ["GR_KERR_BH"]}),
                    encoding="utf-8",
                )
            if label == "s4e_kerr_ratio_filter":
                (stage_dir / "ratio_filter_result.json").write_text(
                    json.dumps({"kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "LOW"}, "filtering": {"n_ratio_compatible": 1}}),
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
                        s3b_method="spectral_two_pass",
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

        for idx_call, call in enumerate(s3b_calls):
            args = call["args"]
            run_id = call["run_id"]
            idx = args.index("--s3-estimates")
            self.assertEqual(args[idx + 1], f"{run_id}/s3_ringdown_estimates/outputs/estimates.json")
            m_idx = args.index("--method")
            expected_method = "spectral_two_pass" if idx_call == 1 else "hilbert_peakband"
            self.assertEqual(args[m_idx + 1], expected_method)


class TestMultimodePipelineBehavior(unittest.TestCase):
    def test_parse_multimode_results_tracks_viability_and_s4d_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "parse_multimode_status"
            run_dir = runs_root / run_id

            s4c_out = run_dir / "s4c_kerr_consistency" / "outputs"
            s4c_out.mkdir(parents=True, exist_ok=True)
            (s4c_out / "kerr_consistency.json").write_text(
                json.dumps(
                    {
                        "status": "SKIPPED_MULTIMODE_GATE",
                        "kerr_consistent": None,
                        "d2_min": 12.34,
                        "source": {
                            "multimode_viability_class": "RINGDOWN_NONINFORMATIVE",
                            "multimode_viability_reasons": [
                                "rel_iqr_f220=0.833 > 0.5: fundamental frequency poorly constrained"
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            s3b_out = run_dir / "s3b_multimode_estimates" / "outputs"
            s3b_out.mkdir(parents=True, exist_ok=True)
            (s3b_out / "multimode_estimates.json").write_text(
                json.dumps({"results": {"verdict": "INSUFFICIENT_DATA"}}),
                encoding="utf-8",
            )
            (run_dir / "s3b_multimode_estimates" / "stage_summary.json").write_text(
                json.dumps(
                    {
                        "multimode_viability": {
                            "class": "RINGDOWN_NONINFORMATIVE",
                            "reasons": [
                                "rel_iqr_f220=0.833 > 0.5: fundamental frequency poorly constrained"
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            s4d_out = run_dir / "s4d_kerr_from_multimode" / "outputs"
            s4d_out.mkdir(parents=True, exist_ok=True)
            (s4d_out / "kerr_from_multimode.json").write_text(
                json.dumps(
                    {
                        "status": "SKIPPED_MULTIMODE_GATE",
                        "multimode_viability": {
                            "class": "RINGDOWN_NONINFORMATIVE",
                            "reasons": [
                                "rel_iqr_f220=0.833 > 0.5: fundamental frequency poorly constrained"
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = pipeline._parse_multimode_results(runs_root, run_id)

            self.assertEqual(parsed["s4c_status"], "SKIPPED_MULTIMODE_GATE")
            self.assertEqual(parsed["kerr_from_multimode_status"], "SKIPPED_MULTIMODE_GATE")
            self.assertEqual(parsed["multimode_viability_class"], "RINGDOWN_NONINFORMATIVE")
            self.assertEqual(
                parsed["multimode_viability_reasons"],
                ["rel_iqr_f220=0.833 > 0.5: fundamental frequency poorly constrained"],
            )

    def test_parse_multimode_results_propagates_singlemode_only_out_of_domain_family_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            run_id = "gw170817_like_singlemode_chain"
            run_dir = runs_root / run_id

            s3b_out = run_dir / "s3b_multimode_estimates" / "outputs"
            s3b_out.mkdir(parents=True, exist_ok=True)
            (s3b_out / "multimode_estimates.json").write_text(
                json.dumps({"results": {"verdict": "OK"}}),
                encoding="utf-8",
            )
            (run_dir / "s3b_multimode_estimates" / "stage_summary.json").write_text(
                json.dumps(
                    {
                        "multimode_viability": {
                            "class": "SINGLEMODE_ONLY",
                            "reasons": [
                                "mode_221_ok=false: overtone posterior not usable for multimode inference"
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            s4d_out = run_dir / "s4d_kerr_from_multimode" / "outputs"
            s4d_out.mkdir(parents=True, exist_ok=True)
            (s4d_out / "kerr_from_multimode.json").write_text(
                json.dumps({"status": "SKIPPED_MULTIMODE_GATE"}),
                encoding="utf-8",
            )

            router_out = run_dir / "s8_family_router" / "outputs"
            router_out.mkdir(parents=True, exist_ok=True)
            (router_out / "family_router.json").write_text(
                json.dumps(
                    {
                        "primary_family": "BNS_REMNANT",
                        "families_to_run": [
                            "BNS_REMNANT",
                            "LOW_MASS_BH_POSTMERGER",
                            "GR_KERR_BH",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            s8b_out = run_dir / "s8b_family_bns" / "outputs"
            s8b_out.mkdir(parents=True, exist_ok=True)
            (s8b_out / "bns_family.json").write_text(
                json.dumps(
                    {
                        "status": "EVALUATED",
                        "assessment": "INCONCLUSIVE",
                        "reason": (
                            "analysis band has no physically useful overlap with the BNS "
                            "post-merger atlas envelope for this event domain"
                        ),
                        "model_status": "PHENOMENOLOGICAL_V1",
                    }
                ),
                encoding="utf-8",
            )

            s8c_out = run_dir / "s8c_family_low_mass_bh_postmerger" / "outputs"
            s8c_out.mkdir(parents=True, exist_ok=True)
            (s8c_out / "low_mass_bh_family.json").write_text(
                json.dumps(
                    {
                        "status": "EVALUATED",
                        "assessment": "INCONCLUSIVE",
                        "reason": (
                            "analysis band has no physically useful overlap with the low-mass "
                            "Kerr modal envelope for this event domain"
                        ),
                        "model_status": "LOW_MASS_KERR_PRIOR_V1",
                    }
                ),
                encoding="utf-8",
            )

            parsed = pipeline._parse_multimode_results(runs_root, run_id)

            self.assertEqual(parsed["multimode_viability_class"], "SINGLEMODE_ONLY")
            self.assertTrue(
                any("mode_221_ok=false" in reason for reason in parsed["multimode_viability_reasons"])
            )
            self.assertEqual(parsed["kerr_from_multimode_status"], "SKIPPED_MULTIMODE_GATE")
            self.assertEqual(parsed["primary_family"], "BNS_REMNANT")
            self.assertEqual(
                parsed["families_to_run"],
                ["BNS_REMNANT", "LOW_MASS_BH_POSTMERGER", "GR_KERR_BH"],
            )

            bns = parsed["family_assessments"]["BNS_REMNANT"]
            low_mass = parsed["family_assessments"]["LOW_MASS_BH_POSTMERGER"]

            self.assertEqual(bns["assessment"], "INCONCLUSIVE")
            self.assertEqual(low_mass["assessment"], "INCONCLUSIVE")
            self.assertIn("no physically useful overlap", bns["reason"] or "")
            self.assertIn("no physically useful overlap", low_mass["reason"] or "")

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
                    (out_root / run_id / "s3b_multimode_estimates" / "stage_summary.json").write_text(
                        json.dumps({"multimode_viability": {"class": "MULTIMODE_OK", "reasons": []}}),
                        encoding="utf-8",
                    )
                if label == "s4d_kerr_from_multimode":
                    (stage_dir / "kerr_from_multimode.json").write_text(
                        json.dumps({"status": "PASS"}),
                        encoding="utf-8",
                    )
                    (stage_dir / "kerr_extraction.json").write_text(
                        json.dumps({"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67}),
                        encoding="utf-8",
                    )
                if label == "s7_beyond_kerr_deviation_score":
                    (stage_dir / "beyond_kerr_score.json").write_text(
                        json.dumps({"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 1.0}),
                        encoding="utf-8",
                    )
                if label == "s8_family_router":
                    (stage_dir / "family_router.json").write_text(
                        json.dumps({"primary_family": "GR_KERR_BH", "families_to_run": ["GR_KERR_BH"]}),
                        encoding="utf-8",
                    )
                if label == "s4e_kerr_ratio_filter":
                    (stage_dir / "ratio_filter_result.json").write_text(
                        json.dumps({"kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "MODERATE"}, "filtering": {"n_ratio_compatible": 4}}),
                        encoding="utf-8",
                    )
                if label == "s8a_family_gr_kerr":
                    (stage_dir / "gr_kerr_family.json").write_text(
                        json.dumps({"status": "EVALUATED", "assessment": "SUPPORTED", "reason": "test"}),
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
            stages = [s["stage"] for s in timeline["stages"]]
            expected_prefix = [
                "s0_oracle_mvp", "s1_fetch_strain", "s2_ringdown_window", "s3_ringdown_estimates",
                "s3b_multimode_estimates", "s4_geometry_filter", "s4c_kerr_consistency",
                "s4d_kerr_from_multimode", "s7_beyond_kerr_deviation_score",
                "s8_family_router", "s4e_kerr_ratio_filter", "s8a_family_gr_kerr",
            ]
            self.assertEqual(stages[:len(expected_prefix)], expected_prefix)
            self.assertEqual(timeline["multimode_results"]["kerr_consistent"], True)
            self.assertEqual(timeline["multimode_results"]["chi_best"], 0.69)
            self.assertEqual(timeline["multimode_results"]["d2_min"], 1.2)
            self.assertEqual(timeline["multimode_results"]["extraction_quality"], "INSUFFICIENT_DATA")
            self.assertEqual(timeline["multimode_results"]["primary_family"], "GR_KERR_BH")
            self.assertEqual(timeline["multimode_results"]["ratio_rf_consistent"], True)
            self.assertEqual(
                timeline["multimode_results"]["family_assessments"]["GR_KERR_BH"]["assessment"],
                "SUPPORTED",
            )

    def test_multimode_with_t0_sweep_reinjects_selected_offset_to_s2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            s2_calls: list[list[str]] = []
            s3_calls: list[list[str]] = []

            def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
                stage_dir = out_root / run_id / label / "outputs"
                stage_dir.mkdir(parents=True, exist_ok=True)

                if label == "s2_ringdown_window":
                    s2_calls.append(list(args))
                if label == "s3_ringdown_estimates":
                    s3_calls.append(list(args))
                if label == "s3b_multimode_estimates":
                    (stage_dir / "multimode_estimates.json").write_text(
                        json.dumps({"results": {"verdict": "INSUFFICIENT_DATA"}}),
                        encoding="utf-8",
                    )
                if label == "s4c_kerr_consistency":
                    (stage_dir / "kerr_consistency.json").write_text(
                        json.dumps({"kerr_consistent": False, "d2_min": 12.34}),
                        encoding="utf-8",
                    )
                    (out_root / run_id / "s3b_multimode_estimates" / "stage_summary.json").write_text(
                        json.dumps({"multimode_viability": {"class": "MULTIMODE_OK", "reasons": []}}),
                        encoding="utf-8",
                    )
                if label == "s4d_kerr_from_multimode":
                    (stage_dir / "kerr_from_multimode.json").write_text(
                        json.dumps({"status": "PASS"}),
                        encoding="utf-8",
                    )
                    (stage_dir / "kerr_extraction.json").write_text(
                        json.dumps({"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67}),
                        encoding="utf-8",
                    )
                if label == "s7_beyond_kerr_deviation_score":
                    (stage_dir / "beyond_kerr_score.json").write_text(
                        json.dumps({"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 1.0}),
                        encoding="utf-8",
                    )
                if label == "s8_family_router":
                    (stage_dir / "family_router.json").write_text(
                        json.dumps({"primary_family": "GR_KERR_BH", "families_to_run": ["GR_KERR_BH"]}),
                        encoding="utf-8",
                    )
                if label == "s4e_kerr_ratio_filter":
                    (stage_dir / "ratio_filter_result.json").write_text(
                        json.dumps({"kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "LOW"}, "filtering": {"n_ratio_compatible": 2}}),
                        encoding="utf-8",
                    )
                if label == "s8a_family_gr_kerr":
                    (stage_dir / "gr_kerr_family.json").write_text(
                        json.dumps({"status": "EVALUATED", "assessment": "SUPPORTED", "reason": "test"}),
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
                    with mock.patch.object(pipeline, "_run_optional_experiment_t0_sweep", return_value=20.0) as sweep_mock:
                        rc, _ = pipeline.run_multimode_event(
                            event_id="GW150914",
                            atlas_path="mvp/test_atlas_fixture.json",
                            synthetic=True,
                            duration_s=4.0,
                            with_t0_sweep=True,
                        )

            self.assertEqual(rc, 0)
            sweep_mock.assert_called_once()
            self.assertEqual(len(s2_calls), 2)
            first_dt_idx = s2_calls[0].index("--dt-start-s")
            second_dt_idx = s2_calls[1].index("--dt-start-s")
            self.assertEqual(s2_calls[0][first_dt_idx + 1], "0.003")
            self.assertEqual(s2_calls[1][second_dt_idx + 1], "0.023")
            self.assertEqual(len(s3_calls), 1)
            self.assertNotIn("--t0-scan-ms", s3_calls[0])

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
