from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mvp import pipeline


def _write_fake_s3_estimates(out_root: Path, run_id: str) -> None:
    stage_dir = out_root / run_id / "s3_ringdown_estimates" / "outputs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "estimates.json").write_text(
        json.dumps(
            {
                "combined": {
                    "f_hz": 250.0,
                    "tau_s": 0.004,
                    "Q": 3.141592653589793,
                    "sigma_f_hz": 5.0,
                    "sigma_tau_s": 0.0005,
                    "sigma_Q": 0.2,
                },
                "combined_uncertainty": {
                    "sigma_f_hz": 5.0,
                    "sigma_tau_s": 0.0005,
                    "sigma_Q": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_mode_payload(label: str, *, ln_f: float, ln_q: float, valid_fraction: float = 0.95) -> dict[str, object]:
    return {
        "label": label,
        "ln_f": ln_f,
        "ln_Q": ln_q,
        "Sigma": [[0.01, 0.001], [0.001, 0.02]],
        "fit": {"stability": {"valid_fraction": valid_fraction}},
    }


def _write_fake_s3b_outputs(
    out_root: Path,
    run_id: str,
    *,
    verdict: str = "INSUFFICIENT_DATA",
    viability_class: str = "MULTIMODE_OK",
    viability_reasons: list[str] | None = None,
) -> None:
    stage_dir = out_root / run_id / "s3b_multimode_estimates" / "outputs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "multimode_estimates.json").write_text(
        json.dumps(
            {
                "results": {"verdict": verdict},
                "modes": [
                    _fake_mode_payload("220", ln_f=5.52, ln_q=1.15),
                    _fake_mode_payload("221", ln_f=5.67, ln_q=0.95),
                ],
            }
        ),
        encoding="utf-8",
    )
    (out_root / run_id / "s3b_multimode_estimates" / "stage_summary.json").write_text(
        json.dumps(
            {
                "multimode_viability": {
                    "class": viability_class,
                    "reasons": viability_reasons or [],
                }
            }
        ),
        encoding="utf-8",
    )


class TestMultimodeWiring(unittest.TestCase):
    def test_s3b_declares_s3_estimates_input_in_command(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
            calls.append({"label": label, "args": list(args), "run_id": run_id})
            stage_dir = out_root / run_id / label / "outputs"
            stage_dir.mkdir(parents=True, exist_ok=True)
            if label == "s3_ringdown_estimates":
                _write_fake_s3_estimates(out_root, run_id)
            if label == "s3b_multimode_estimates":
                _write_fake_s3b_outputs(out_root, run_id)
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
                        estimator="spectral",
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
                        estimator="spectral",
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

    def test_dual_estimator_runs_dual_method_pipeline(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
            calls.append({"script": script, "label": label, "args": list(args), "run_id": run_id})
            stage_dir = out_root / run_id / label / "outputs"
            stage_dir.mkdir(parents=True, exist_ok=True)
            if label == "s3_ringdown_estimates":
                _write_fake_s3_estimates(out_root, run_id)
            if label == "s3_spectral_estimates":
                spec_dir = out_root / run_id / "s3_spectral_estimates" / "outputs"
                spec_dir.mkdir(parents=True, exist_ok=True)
                (spec_dir / "spectral_estimates.json").write_text(
                    json.dumps({
                        "combined": {
                            "f_hz": 260.0,
                            "tau_s": 0.005,
                            "Q": 4.084070449666731,
                            "sigma_f_hz": 4.0,
                            "sigma_tau_s": 0.0004,
                            "sigma_Q": 0.3,
                        },
                        "combined_uncertainty": {
                            "sigma_f_hz": 4.0,
                            "sigma_tau_s": 0.0004,
                            "sigma_Q": 0.3,
                        },
                    }),
                    encoding="utf-8",
                )
            if label == "experiment_dual_method":
                exp_dir = out_root / run_id / "experiment" / "DUAL_METHOD_V1"
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "dual_method_comparison.json").write_text(
                    json.dumps({"recommendation": "spectral"}),
                    encoding="utf-8",
                )
            if label == "s3b_multimode_estimates":
                _write_fake_s3b_outputs(out_root, run_id)
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
            timeline["stages"].append({
                "stage": label,
                "script": script,
                "command": [script] + list(args),
                "started_utc": "now",
                "ended_utc": "now",
                "duration_s": 0.0,
                "returncode": 0,
                "timed_out": False,
            })
            pipeline._write_timeline(out_root, run_id, timeline)
            return 0

        with mock.patch.object(pipeline, "_run_stage", side_effect=fake_run_stage):
            with mock.patch.object(pipeline, "_parse_multimode_results", return_value={}):
                with mock.patch.dict("os.environ", {"BASURIN_RUNS_ROOT": str(Path("/tmp") / "basurin_wiring_test_dual")}, clear=False):
                    rc, _ = pipeline.run_multimode_event(
                        event_id="GW150914",
                        atlas_path="mvp/test_atlas_fixture.json",
                        run_id="wire_dual",
                        synthetic=True,
                        estimator="dual",
                    )
                    self.assertEqual(rc, 0)

        labels = [c["label"] for c in calls]
        self.assertIn("s3_ringdown_estimates", labels)
        self.assertIn("s3_spectral_estimates", labels)
        self.assertIn("experiment_dual_method", labels)

        s3_call = next(c for c in calls if c["label"] == "s3_ringdown_estimates")
        self.assertIn("--method", s3_call["args"])
        self.assertIn("hilbert_envelope", s3_call["args"])

        s3b_call = next(c for c in calls if c["label"] == "s3b_multimode_estimates")
        idx = s3b_call["args"].index("--s3-estimates")
        self.assertEqual(
            s3b_call["args"][idx + 1],
            "wire_dual/s3_spectral_estimates/outputs/spectral_estimates.json",
        )


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

            s4k_out = run_dir / "s4k_event_support_region" / "outputs"
            s4k_out.mkdir(parents=True, exist_ok=True)
            (s4k_out / "event_support_region.json").write_text(
                json.dumps(
                    {
                        "analysis_path": "MULTIMODE_INTERSECTION",
                        "support_region_status": "SUPPORT_REGION_AVAILABLE",
                        "n_final_geometries": 507,
                        "downstream_status": {
                            "class": "GEOMETRY_PRESENT_BUT_NONINFORMATIVE",
                            "reasons": ["multimode_viability=RINGDOWN_NONINFORMATIVE"],
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
            self.assertEqual(parsed["downstream_status_class"], "GEOMETRY_PRESENT_BUT_NONINFORMATIVE")
            self.assertEqual(
                parsed["downstream_status_reasons"],
                ["multimode_viability=RINGDOWN_NONINFORMATIVE"],
            )
            self.assertEqual(parsed["support_region_status"], "SUPPORT_REGION_AVAILABLE")
            self.assertEqual(parsed["support_region_n_final"], 507)
            self.assertEqual(parsed["support_region_analysis_path"], "MULTIMODE_INTERSECTION")

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
                json.dumps(
                    {
                        "status": "SKIPPED_MULTIMODE_GATE",
                        "multimode_fallback": {
                            "classification": "MULTIMODE_UNAVAILABLE_221",
                            "fallback_path": "220_ATLAS",
                            "program_classification": "SINGLE_MODE_CONSTRAINED_PROGRAM",
                            "reason": "mode_221_ok=false: overtone posterior not usable for multimode inference",
                        },
                    }
                ),
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
                        "program_classification": "SINGLE_MODE_CONSTRAINED_PROGRAM",
                        "fallback_classification": "MULTIMODE_UNAVAILABLE_221",
                        "fallback_path": "220_ATLAS",
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
            self.assertEqual(parsed["program_classification"], "SINGLE_MODE_CONSTRAINED_PROGRAM")
            self.assertEqual(parsed["fallback_classification"], "MULTIMODE_UNAVAILABLE_221")
            self.assertEqual(parsed["fallback_path"], "220_ATLAS")
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

                if label == "s3_ringdown_estimates":
                    _write_fake_s3_estimates(out_root, run_id)
                if label == "s3b_multimode_estimates":
                    _write_fake_s3b_outputs(out_root, run_id)
                if label == "s4g_mode220_geometry_filter":
                    obs = json.loads(
                        (out_root / run_id / "s4g_mode220_geometry_filter" / "inputs" / "mode220_obs.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    (stage_dir / "mode220_filter.json").write_text(
                        json.dumps({"accepted_geometry_ids": ["geo_A", "geo_B"], **obs}),
                        encoding="utf-8",
                    )
                if label == "s4h_mode221_geometry_filter":
                    obs_path = out_root / run_id / "s4h_mode221_geometry_filter" / "inputs" / "mode221_obs.json"
                    if obs_path.exists():
                        obs = json.loads(obs_path.read_text(encoding="utf-8"))
                        (stage_dir / "mode221_filter.json").write_text(
                            json.dumps({"geometry_ids": ["geo_A"], "verdict": "PASS", **obs}),
                            encoding="utf-8",
                        )
                    else:
                        (stage_dir / "mode221_filter.json").write_text(
                            json.dumps({"geometry_ids": [], "verdict": "SKIPPED_221_UNAVAILABLE"}),
                            encoding="utf-8",
                        )
                if label == "s4i_common_geometry_intersection":
                    (stage_dir / "common_intersection.json").write_text(
                        json.dumps(
                            {
                                "common_geometry_ids": ["geo_A"],
                                "n_common": 1,
                                "mode221_skipped": False,
                                "verdict": "PASS",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4f_area_observation":
                    (stage_dir / "area_obs.json").write_text(
                        json.dumps(
                            {
                                "area_data": {"geo_A": {"area_final": 2.0, "area_initial": 1.0}},
                                "observation_status": "AREA_DATA_AVAILABLE",
                                "policy": "mass_only_lower_bound_v1",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4j_hawking_area_filter":
                    (stage_dir / "hawking_area_filter.json").write_text(
                        json.dumps(
                            {
                                "golden_geometry_ids": ["geo_A"],
                                "n_golden": 1,
                                "area_data": {"geo_A": {"area_final": 2.0, "area_initial": 1.0}},
                                "area_obs_present": True,
                                "area_constraint_applied": True,
                                "verdict": "PASS",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4c_kerr_consistency":
                    (stage_dir / "kerr_consistency.json").write_text(
                        json.dumps({"consistent_kerr_95": True, "chi_best_fit": 0.69, "d2_min": 1.2}),
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
                if label == "s4k_event_support_region":
                    (stage_dir / "event_support_region.json").write_text(
                        json.dumps(
                            {
                                "analysis_path": "MULTIMODE_INTERSECTION",
                                "support_region_status": "SUPPORT_REGION_AVAILABLE",
                                "n_final_geometries": 1,
                                "downstream_status": {
                                    "class": "MULTIMODE_USABLE",
                                    "reasons": ["support_region_status=SUPPORT_REGION_AVAILABLE"],
                                },
                            }
                        ),
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
                        estimator="spectral",
                    )

            self.assertEqual(rc, 0)
            timeline = json.loads((runs_root / run_id / "pipeline_timeline.json").read_text(encoding="utf-8"))
            stages = [s["stage"] for s in timeline["stages"]]
            expected_prefix = [
                "s0_oracle_mvp", "s1_fetch_strain", "s2_ringdown_window", "s3_ringdown_estimates",
                "s3b_multimode_estimates", "s4g_mode220_geometry_filter", "s4h_mode221_geometry_filter",
                "s4i_common_geometry_intersection", "s4f_area_observation", "s4j_hawking_area_filter",
                "s4_geometry_filter", "s4c_kerr_consistency", "s4d_kerr_from_multimode",
                "s4k_event_support_region", "s7_beyond_kerr_deviation_score",
                "s8_family_router", "s4e_kerr_ratio_filter", "s8a_family_gr_kerr",
            ]
            self.assertEqual(stages[:len(expected_prefix)], expected_prefix)
            self.assertEqual(timeline["multimode_results"]["kerr_consistent"], True)
            self.assertEqual(timeline["multimode_results"]["chi_best"], 0.69)
            self.assertEqual(timeline["multimode_results"]["d2_min"], 1.2)
            self.assertEqual(timeline["multimode_results"]["extraction_quality"], "INSUFFICIENT_DATA")
            self.assertEqual(timeline["multimode_results"]["primary_family"], "GR_KERR_BH")
            self.assertEqual(timeline["multimode_results"]["ratio_rf_consistent"], True)
            self.assertEqual(timeline["multimode_results"]["downstream_status_class"], "MULTIMODE_USABLE")
            self.assertEqual(
                timeline["multimode_results"]["downstream_status_reasons"],
                ["support_region_status=SUPPORT_REGION_AVAILABLE"],
            )
            self.assertEqual(
                timeline["multimode_results"]["support_region_status"],
                "SUPPORT_REGION_AVAILABLE",
            )
            self.assertEqual(timeline["multimode_results"]["support_region_n_final"], 1)
            self.assertEqual(
                timeline["multimode_results"]["support_region_analysis_path"],
                "MULTIMODE_INTERSECTION",
            )
            self.assertEqual(
                timeline["multimode_results"]["family_assessments"]["GR_KERR_BH"]["assessment"],
                "SUPPORTED",
            )

    def test_multimode_degrades_to_mode220_plus_hawking_when_221_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            mode221_input_seen = {"exists": None}

            def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
                stage_dir = out_root / run_id / label / "outputs"
                stage_dir.mkdir(parents=True, exist_ok=True)

                if label == "s3_ringdown_estimates":
                    _write_fake_s3_estimates(out_root, run_id)
                if label == "s3b_multimode_estimates":
                    _write_fake_s3b_outputs(
                        out_root,
                        run_id,
                        verdict="OK",
                        viability_class="SINGLEMODE_ONLY",
                        viability_reasons=[
                            "mode_221_ok=false: overtone posterior not usable for multimode inference"
                        ],
                    )
                if label == "s4g_mode220_geometry_filter":
                    obs = json.loads(
                        (out_root / run_id / "s4g_mode220_geometry_filter" / "inputs" / "mode220_obs.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    (stage_dir / "mode220_filter.json").write_text(
                        json.dumps({"accepted_geometry_ids": ["geo_A", "geo_B"], **obs}),
                        encoding="utf-8",
                    )
                if label == "s4h_mode221_geometry_filter":
                    obs_path = out_root / run_id / "s4h_mode221_geometry_filter" / "inputs" / "mode221_obs.json"
                    mode221_input_seen["exists"] = obs_path.exists()
                    (stage_dir / "mode221_filter.json").write_text(
                        json.dumps({"geometry_ids": [], "verdict": "SKIPPED_221_UNAVAILABLE"}),
                        encoding="utf-8",
                    )
                if label == "s4i_common_geometry_intersection":
                    (stage_dir / "common_intersection.json").write_text(
                        json.dumps(
                            {
                                "common_geometry_ids": ["geo_A", "geo_B"],
                                "n_common": 2,
                                "mode221_skipped": True,
                                "verdict": "PASS",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4f_area_observation":
                    (stage_dir / "area_obs.json").write_text(
                        json.dumps(
                            {
                                "area_data": {},
                                "observation_status": "MISSING_SOURCE_MASSES",
                                "policy": "mass_only_lower_bound_v1",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4j_hawking_area_filter":
                    (stage_dir / "hawking_area_filter.json").write_text(
                        json.dumps(
                            {
                                "golden_geometry_ids": ["geo_A", "geo_B"],
                                "n_golden": 2,
                                "area_data": {},
                                "area_obs_present": True,
                                "area_constraint_applied": False,
                                "verdict": "PASS",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4c_kerr_consistency":
                    (stage_dir / "kerr_consistency.json").write_text(
                        json.dumps({"status": "SKIPPED_MULTIMODE_GATE", "kerr_consistent": None, "d2_min": 42.0}),
                        encoding="utf-8",
                    )
                if label == "s4d_kerr_from_multimode":
                    (stage_dir / "kerr_from_multimode.json").write_text(
                        json.dumps(
                            {
                                "status": "SKIPPED_MULTIMODE_GATE",
                                "multimode_fallback": {
                                    "classification": "MULTIMODE_UNAVAILABLE_221",
                                    "fallback_path": "220_HAWKING",
                                    "program_classification": "SINGLE_MODE_CONSTRAINED_PROGRAM",
                                    "reason": "mode_221_ok=false: overtone posterior not usable for multimode inference",
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4k_event_support_region":
                    (stage_dir / "event_support_region.json").write_text(
                        json.dumps(
                            {
                                "analysis_path": "MODE220_NO_AREA_CONSTRAINT",
                                "support_region_status": "SUPPORT_REGION_AVAILABLE",
                                "n_final_geometries": 2,
                                "downstream_status": {
                                    "class": "GEOMETRY_PRESENT_BUT_NONINFORMATIVE",
                                    "reasons": ["multimode_viability=SINGLEMODE_ONLY"],
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s7_beyond_kerr_deviation_score":
                    (stage_dir / "beyond_kerr_score.json").write_text(
                        json.dumps({"verdict": "INCONCLUSIVE", "chi2_kerr_2dof": None}),
                        encoding="utf-8",
                    )
                if label == "s8_family_router":
                    self.assertTrue(
                        (out_root / run_id / "s4j_hawking_area_filter" / "outputs" / "hawking_area_filter.json").exists()
                    )
                    (stage_dir / "family_router.json").write_text(
                        json.dumps(
                            {
                                "primary_family": "GR_KERR_BH",
                                "families_to_run": ["GR_KERR_BH", "LOW_MASS_BH_POSTMERGER", "BNS_REMNANT"],
                                "program_classification": "SINGLE_MODE_CONSTRAINED_PROGRAM",
                                "fallback_classification": "MULTIMODE_UNAVAILABLE_221",
                                "fallback_path": "220_HAWKING",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4e_kerr_ratio_filter":
                    (stage_dir / "ratio_filter_result.json").write_text(
                        json.dumps(
                            {
                                "kerr_consistency": {"Rf_consistent": True},
                                "diagnostics": {"informativity_class": "UNINFORMATIVE"},
                                "filtering": {"n_ratio_compatible": 0},
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s8a_family_gr_kerr":
                    (stage_dir / "gr_kerr_family.json").write_text(
                        json.dumps({"status": "EVALUATED", "assessment": "INCONCLUSIVE", "reason": "test"}),
                        encoding="utf-8",
                    )
                if label == "s8b_family_bns":
                    (stage_dir / "bns_family.json").write_text(
                        json.dumps({"status": "EVALUATED", "assessment": "INCONCLUSIVE", "reason": "test"}),
                        encoding="utf-8",
                    )
                if label == "s8c_family_low_mass_bh_postmerger":
                    (stage_dir / "low_mass_bh_family.json").write_text(
                        json.dumps({"status": "EVALUATED", "assessment": "INCONCLUSIVE", "reason": "test"}),
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
                        event_id="GW190924_021846",
                        atlas_path="mvp/test_atlas_fixture.json",
                        synthetic=True,
                        duration_s=4.0,
                    )

            self.assertEqual(rc, 0)
            self.assertFalse(mode221_input_seen["exists"])
            timeline = json.loads((runs_root / run_id / "pipeline_timeline.json").read_text(encoding="utf-8"))
            self.assertEqual(timeline["multimode_results"]["fallback_path"], "220_HAWKING")
            self.assertEqual(
                timeline["multimode_results"]["support_region_analysis_path"],
                "MODE220_NO_AREA_CONSTRAINT",
            )
            self.assertEqual(
                timeline["multimode_results"]["downstream_status_class"],
                "GEOMETRY_PRESENT_BUT_NONINFORMATIVE",
            )
            self.assertEqual(timeline["multimode_results"]["support_region_n_final"], 2)

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
                    _write_fake_s3_estimates(out_root, run_id)
                if label == "s3b_multimode_estimates":
                    _write_fake_s3b_outputs(out_root, run_id)
                if label == "s4g_mode220_geometry_filter":
                    obs = json.loads(
                        (out_root / run_id / "s4g_mode220_geometry_filter" / "inputs" / "mode220_obs.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    (stage_dir / "mode220_filter.json").write_text(
                        json.dumps({"accepted_geometry_ids": ["geo_A", "geo_B"], **obs}),
                        encoding="utf-8",
                    )
                if label == "s4h_mode221_geometry_filter":
                    obs = json.loads(
                        (out_root / run_id / "s4h_mode221_geometry_filter" / "inputs" / "mode221_obs.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    (stage_dir / "mode221_filter.json").write_text(
                        json.dumps({"geometry_ids": ["geo_A"], "verdict": "PASS", **obs}),
                        encoding="utf-8",
                    )
                if label == "s4i_common_geometry_intersection":
                    (stage_dir / "common_intersection.json").write_text(
                        json.dumps(
                            {
                                "common_geometry_ids": ["geo_A"],
                                "n_common": 1,
                                "mode221_skipped": False,
                                "verdict": "PASS",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4f_area_observation":
                    (stage_dir / "area_obs.json").write_text(
                        json.dumps(
                            {
                                "area_data": {"geo_A": {"area_final": 2.0, "area_initial": 1.0}},
                                "observation_status": "AREA_DATA_AVAILABLE",
                                "policy": "mass_only_lower_bound_v1",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4j_hawking_area_filter":
                    (stage_dir / "hawking_area_filter.json").write_text(
                        json.dumps(
                            {
                                "golden_geometry_ids": ["geo_A"],
                                "n_golden": 1,
                                "area_data": {"geo_A": {"area_final": 2.0, "area_initial": 1.0}},
                                "area_obs_present": True,
                                "area_constraint_applied": True,
                                "verdict": "PASS",
                            }
                        ),
                        encoding="utf-8",
                    )
                if label == "s4c_kerr_consistency":
                    (stage_dir / "kerr_consistency.json").write_text(
                        json.dumps({"kerr_consistent": False, "d2_min": 12.34}),
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
                if label == "s4k_event_support_region":
                    (stage_dir / "event_support_region.json").write_text(
                        json.dumps(
                            {
                                "analysis_path": "MULTIMODE_INTERSECTION",
                                "support_region_status": "SUPPORT_REGION_AVAILABLE",
                                "n_final_geometries": 1,
                                "downstream_status": {
                                    "class": "MULTIMODE_USABLE",
                                    "reasons": ["support_region_status=SUPPORT_REGION_AVAILABLE"],
                                },
                            }
                        ),
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


    def test_psd_path_forwarded_to_s3_spectral_and_s3b(self) -> None:
        """--psd-path must be forwarded to s3_spectral_estimates and s3b_multimode_estimates."""
        calls: list[dict[str, object]] = []

        def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
            calls.append({"label": label, "args": list(args)})
            stage_dir = out_root / run_id / label / "outputs"
            stage_dir.mkdir(parents=True, exist_ok=True)
            if label == "s3_ringdown_estimates":
                _write_fake_s3_estimates(out_root, run_id)
            if label == "s3b_multimode_estimates":
                _write_fake_s3b_outputs(out_root, run_id)
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
            timeline["stages"].append({
                "stage": label, "script": script, "command": [script] + list(args),
                "started_utc": "now", "ended_utc": "now",
                "duration_s": 0.0, "returncode": 0, "timed_out": False,
            })
            pipeline._write_timeline(out_root, run_id, timeline)
            return 0

        with mock.patch.object(pipeline, "_run_stage", side_effect=fake_run_stage):
            with mock.patch.object(pipeline, "_parse_multimode_results", return_value={}):
                with mock.patch.dict("os.environ", {"BASURIN_RUNS_ROOT": str(Path("/tmp") / "basurin_psd_wiring")}, clear=False):
                    rc, _ = pipeline.run_multimode_event(
                        event_id="GW150914",
                        atlas_path="mvp/test_atlas_fixture.json",
                        run_id="wire_psd",
                        synthetic=True,
                        estimator="dual",
                        psd_path="/tmp/fake_psd.json",
                    )
                    self.assertEqual(rc, 0)

        # Collect args by label
        by_label = {c["label"]: c["args"] for c in calls}

        s3_spectral_args = by_label.get("s3_spectral_estimates", [])
        self.assertIn("--psd-path", s3_spectral_args,
                      f"--psd-path not found in s3_spectral_estimates args: {s3_spectral_args}")
        psd_idx = s3_spectral_args.index("--psd-path")
        self.assertEqual(s3_spectral_args[psd_idx + 1], "/tmp/fake_psd.json")

        s3b_args = by_label.get("s3b_multimode_estimates", [])
        self.assertIn("--psd-path", s3b_args,
                      f"--psd-path not found in s3b_multimode_estimates args: {s3b_args}")
        psd_idx_b = s3b_args.index("--psd-path")
        self.assertEqual(s3b_args[psd_idx_b + 1], "/tmp/fake_psd.json")

    def test_single_forwards_psd_path_to_s3_spectral(self) -> None:
        """run_single_event must forward psd_path to s3_spectral_estimates in dual mode."""
        calls: list[dict[str, object]] = []

        def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
            calls.append({"label": label, "args": list(args)})
            stage_dir = out_root / run_id / label / "outputs"
            stage_dir.mkdir(parents=True, exist_ok=True)
            if label == "s3_ringdown_estimates":
                _write_fake_s3_estimates(out_root, run_id)
            timeline["stages"].append({
                "stage": label, "script": script, "command": [script] + list(args),
                "started_utc": "now", "ended_utc": "now",
                "duration_s": 0.0, "returncode": 0, "timed_out": False,
            })
            pipeline._write_timeline(out_root, run_id, timeline)
            return 0

        with mock.patch.object(pipeline, "_run_stage", side_effect=fake_run_stage):
            with mock.patch.dict("os.environ", {"BASURIN_RUNS_ROOT": str(Path("/tmp") / "basurin_single_psd")}, clear=False):
                rc, _ = pipeline.run_single_event(
                    event_id="GW150914",
                    atlas_path="mvp/test_atlas_fixture.json",
                    run_id="wire_single_psd",
                    synthetic=True,
                    estimator="dual",
                    psd_path="/tmp/fake_psd.json",
                )
                # rc may be non-zero (later stages may not be stubbed), only check s3 wiring
                by_label = {c["label"]: c["args"] for c in calls}
                s3_spectral_args = by_label.get("s3_spectral_estimates", [])
                self.assertIn("--psd-path", s3_spectral_args,
                              f"--psd-path not in s3_spectral_estimates args: {s3_spectral_args}")


if __name__ == "__main__":
    unittest.main()
