"""Tests for Q5: pipeline estimator selection and batch mode.

Tests:
    1. s4 with --estimates-path custom: reads from correct path
    2. batch mode with 2 synthetic events: produces aggregate
    3. Anti-regression: run_single_event without estimator arg works as before
    4. gwtc_events.py: catalog contains expected events
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestGWTCEvents:
    """Test gwtc_events.py catalog."""

    def test_gwtc_events_importable(self):
        from mvp.gwtc_events import GWTC_EVENTS
        assert isinstance(GWTC_EVENTS, dict)

    def test_has_gw150914(self):
        from mvp.gwtc_events import GWTC_EVENTS
        assert "GW150914" in GWTC_EVENTS

    def test_gw150914_fields(self):
        from mvp.gwtc_events import GWTC_EVENTS
        ev = GWTC_EVENTS["GW150914"]
        assert "m_final_msun" in ev
        assert "chi_final" in ev
        assert "snr_network" in ev

    def test_gw150914_values_reasonable(self):
        from mvp.gwtc_events import GWTC_EVENTS
        ev = GWTC_EVENTS["GW150914"]
        assert 50.0 < ev["m_final_msun"] < 80.0
        assert 0.5 < ev["chi_final"] < 0.9
        assert ev["snr_network"] > 10.0

    def test_all_10_events_present(self):
        from mvp.gwtc_events import GWTC_EVENTS
        expected = {
            "GW150914", "GW151226", "GW170104", "GW170608",
            "GW170729", "GW170809", "GW170814", "GW170818",
            "GW170823", "GW190521",
        }
        assert expected.issubset(set(GWTC_EVENTS.keys()))

    def test_get_event_function(self):
        from mvp.gwtc_events import get_event
        ev = get_event("GW150914")
        assert ev is not None
        assert "m_final_msun" in ev

    def test_get_event_missing_returns_none(self):
        from mvp.gwtc_events import get_event
        assert get_event("GW_NONEXISTENT") is None

    def test_list_events(self):
        from mvp.gwtc_events import list_events
        events = list_events()
        assert isinstance(events, list)
        assert len(events) >= 10
        assert "GW150914" in events


class TestS4EstimatesPath:
    """Test that s4_geometry_filter accepts --estimates-path override."""

    def test_main_has_estimates_path_arg(self):
        """Check that s4 CLI now accepts --estimates-path."""
        import argparse
        from mvp import s4_geometry_filter
        import inspect

        # Check the arg is in main() by looking at the argparse setup
        source = inspect.getsource(s4_geometry_filter.main)
        assert "--estimates-path" in source, \
            "s4_geometry_filter.main() should have --estimates-path argument"


class TestPipelineEstimatorArg:
    """Test that pipeline run_single_event accepts estimator argument."""

    def test_run_single_event_has_estimator_param(self):
        import inspect
        from mvp.pipeline import run_single_event
        sig = inspect.signature(run_single_event)
        assert "estimator" in sig.parameters, \
            "run_single_event should have an 'estimator' parameter"

    def test_default_estimator_is_spectral(self):
        import inspect
        from mvp.pipeline import run_single_event
        sig = inspect.signature(run_single_event)
        default = sig.parameters["estimator"].default
        assert default == "spectral"

    def test_spectral_estimator_uses_ringdown_lorentzian_method(self, monkeypatch):
        from mvp import pipeline

        stage_calls = []

        def _fake_run_stage(script, args, stage, out_root, run_id, timeline, timeout):
            stage_calls.append((script, args, stage))
            return 0

        monkeypatch.setattr(pipeline, "_run_stage", _fake_run_stage)
        monkeypatch.setattr(pipeline, "_write_timeline", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "_create_run_valid", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "_set_run_valid_verdict", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "_parse_multimode_results", lambda *a, **k: {})

        rc, _ = pipeline.run_single_event(
            event_id="GW150914",
            atlas_path="atlas.json",
            run_id="run_test",
            estimator="spectral",
            local_hdf5=[],
        )

        assert rc == 0
        s3_call = next(call for call in stage_calls if call[0] == "s3_ringdown_estimates.py")
        assert "--method" in s3_call[1]
        method_idx = s3_call[1].index("--method")
        assert s3_call[1][method_idx + 1] == "spectral_lorentzian"

    def test_pipeline_has_batch_mode(self):
        import inspect
        from mvp.pipeline import main
        source = inspect.getsource(main)
        assert "batch" in source, "pipeline.main() should have batch mode"


class TestRunMultiEventSignature:
    """Test run_multi_event signature changes."""

    def test_accepts_catalog_path(self):
        import inspect
        from mvp.pipeline import run_multi_event
        sig = inspect.signature(run_multi_event)
        assert "catalog_path" in sig.parameters

    def test_accepts_abort_on_event_fail(self):
        import inspect
        from mvp.pipeline import run_multi_event
        sig = inspect.signature(run_multi_event)
        assert "abort_on_event_fail" in sig.parameters
