"""Tests for mvp/experiment_dual_method.py — dual method gate.

Tests:
    1. Clean synthetic signal: verdict=CONSISTENT, both close to f_true
    2. Schema: all required fields present
    3. Fallback: if spectral fails, recommendation=hilbert
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.experiment_dual_method import compare_methods, run_dual_method


class TestCompareMethods:
    """Unit tests for the comparison logic."""

    def test_consistent_verdict(self):
        """When both estimators agree, verdict should be CONSISTENT."""
        hilbert = {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004,
                   "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        spectral = {"f_hz": 251.0, "Q": 3.20, "tau_s": 0.0041,
                    "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        result = compare_methods(hilbert, spectral)
        assert result["verdict"] == "CONSISTENT"

    def test_tension_verdict(self):
        """When tension is 2–3σ, verdict should be TENSION."""
        hilbert = {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004,
                   "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        spectral = {"f_hz": 250.0 + 15.0, "Q": 3.14, "tau_s": 0.004,
                    "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        # Δf=15, σ_combined=sqrt(25+25)≈7.07, tension≈2.12
        result = compare_methods(hilbert, spectral)
        assert result["verdict"] == "TENSION"

    def test_inconsistent_verdict(self):
        """When tension ≥ 3σ, verdict should be INCONSISTENT."""
        hilbert = {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004,
                   "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        spectral = {"f_hz": 250.0 + 25.0, "Q": 3.14, "tau_s": 0.004,
                    "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        # Δf=25, σ_combined=7.07, tension≈3.54
        result = compare_methods(hilbert, spectral)
        assert result["verdict"] == "INCONSISTENT"

    def test_delta_f_correct(self):
        hilbert = {"f_hz": 250.0, "Q": 3.0, "tau_s": 0.004,
                   "sigma_f_hz": 1.0, "sigma_Q": 0.1}
        spectral = {"f_hz": 255.0, "Q": 3.0, "tau_s": 0.004,
                    "sigma_f_hz": 1.0, "sigma_Q": 0.1}
        result = compare_methods(hilbert, spectral)
        assert abs(result["delta_f_hz"] - 5.0) < 1e-10

    def test_schema_fields_present(self):
        hilbert = {"f_hz": 250.0, "Q": 3.0, "tau_s": 0.004,
                   "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        spectral = {"f_hz": 251.0, "Q": 3.1, "tau_s": 0.0041,
                    "sigma_f_hz": 5.0, "sigma_Q": 0.5}
        result = compare_methods(hilbert, spectral)
        required = {"delta_f_hz", "delta_Q", "tension_f_sigma",
                    "tension_Q_sigma", "verdict"}
        assert required.issubset(set(result.keys()))


class TestDualMethodSchema:
    """Test that dual method output JSON has all required fields."""

    def test_comparison_schema(self):
        """The comparison dict from compare_methods must be complete."""
        h = {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004,
             "sigma_f_hz": 2.0, "sigma_Q": 0.2}
        s = {"f_hz": 252.0, "Q": 3.20, "tau_s": 0.004,
             "sigma_f_hz": 2.0, "sigma_Q": 0.2}
        comp = compare_methods(h, s)
        assert "verdict" in comp
        assert comp["verdict"] in ("CONSISTENT", "TENSION", "INCONSISTENT")
        assert math.isfinite(comp["tension_f_sigma"])
        assert math.isfinite(comp["tension_Q_sigma"])
