from __future__ import annotations

import math

import pytest

from mvp.kerr_qnm_fits import kerr_ratio_curve
from mvp.s4e_kerr_ratio_filter import (
    classify_informativity,
    compute_observed_ratios,
    extract_geometry_spin,
    run_ratio_filter,
)

def _mode_payload(label: str, f_hz: float, q_factor: float, frac: float = 0.01) -> dict:
    tau_s = q_factor / (math.pi * f_hz)
    return {
        "label": label,
        "estimates": {
            "f_hz": {
                "p10": f_hz * (1.0 - frac),
                "p50": f_hz,
                "p90": f_hz * (1.0 + frac),
            },
            "tau_s": {
                "p10": tau_s * (1.0 - frac),
                "p50": tau_s,
                "p90": tau_s * (1.0 + frac),
            },
            "Q": {
                "p10": q_factor * (1.0 - frac),
                "p50": q_factor,
                "p90": q_factor * (1.0 + frac),
            },
        },
    }

def _multimode_payload(f220_hz: float, q220: float, f221_hz: float, q221: float, frac: float = 0.01) -> dict:
    return {
        "modes": [
            _mode_payload("220", f220_hz, q220, frac=frac),
            _mode_payload("221", f221_hz, q221, frac=frac),
        ]
    }

def _compatible_payload(rows: list[dict]) -> dict:
    return {"compatible_geometries": rows}

def _rf_at_spin(spin: float) -> float:
    return float(kerr_ratio_curve(chi_grid=[spin])["Rf_grid"][0])

def test_kerr_ratio_curve_basic() -> None:
    result = kerr_ratio_curve(n_points=10)
    assert len(result["chi_grid"]) == 10
    assert len(result["Rf_grid"]) == 10
    assert len(result["RQ_grid"]) == 10
    assert result["Rf_range"]["min"] < result["Rf_range"]["max"]
    assert result["RQ_range"]["min"] < result["RQ_range"]["max"]
    assert all(0.5 < value < 1.5 for value in result["Rf_grid"])
    assert all(0.0 < value < 1.0 for value in result["RQ_grid"])

def test_kerr_ratio_curve_schwarzschild() -> None:
    result = kerr_ratio_curve(chi_grid=[0.0])
    assert abs(result["Rf_grid"][0] - 0.9267) < 0.01
    assert abs(result["RQ_grid"][0] - 0.3038) < 0.01

def test_kerr_ratio_curve_non_monotonic_with_high_spin_turnover() -> None:
    result = kerr_ratio_curve(n_points=2000)
    rf_grid = result["Rf_grid"]
    chi_grid = result["chi_grid"]

    peak_idx = max(range(len(rf_grid)), key=lambda i: rf_grid[i])
    peak_chi = chi_grid[peak_idx]

    assert 0.94 < peak_chi < 0.98
    assert rf_grid[0] < rf_grid[peak_idx]
    assert rf_grid[-1] < rf_grid[peak_idx]

def test_kerr_ratio_curve_range_is_stable_with_coarse_grid() -> None:
    coarse = kerr_ratio_curve(n_points=10)
    dense = kerr_ratio_curve(n_points=4000)

    assert abs(coarse["Rf_range"]["max"] - dense["Rf_range"]["max"]) < 1e-6
    assert abs(coarse["Rf_range"]["min"] - dense["Rf_range"]["min"]) < 1e-6

def test_extract_geometry_spin_from_metadata_and_geometry_id() -> None:
    assert extract_geometry_spin({"geometry_id": "Kerr_M90_a0.8631"}) == pytest.approx(0.8631)
    assert extract_geometry_spin({"geometry_id": "x", "metadata": {"chi": 0.67}}) == pytest.approx(0.67)
    assert extract_geometry_spin({"geometry_id": "x"}) is None

def test_ratio_filter_keeps_compatible_geometry() -> None:
    spin = 0.67
    rf_obs = _rf_at_spin(spin)
    multimode = _multimode_payload(251.0, 4.3, 251.0 * rf_obs, 1.5, frac=0.01)
    compatible = _compatible_payload([{"geometry_id": "kerr_a0.67", "metadata": {"spin": spin}, "d2": 1.2}])

    result = run_ratio_filter(
        run_id="run_test",
        multimode_estimates=multimode,
        compatible_payload=compatible,
        ranked_all_full=None,
        estimates_sha256="a",
        compatible_set_sha256="b",
        sigma_rf=2.0,
        sigma_rq=2.0,
        chi_grid_points=100,
        apply_rq=False,
    )

    rows = result["filtering"]["compatible_geometries"]
    assert len(rows) == 1
    assert rows[0]["status"] == "RATIO_COMPATIBLE"
    assert rows[0]["spin"] == pytest.approx(spin)

def test_ratio_filter_excludes_incompatible_geometry() -> None:
    multimode = _multimode_payload(250.0, 4.2, 245.0, 1.45, frac=0.002)
    compatible = _compatible_payload([{"geometry_id": "kerr_a0.01", "metadata": {"spin": 0.01}}])

    result = run_ratio_filter(
        run_id="run_test",
        multimode_estimates=multimode,
        compatible_payload=compatible,
        ranked_all_full=None,
        estimates_sha256="a",
        compatible_set_sha256="b",
        sigma_rf=2.0,
        sigma_rq=2.0,
        chi_grid_points=100,
        apply_rq=False,
    )

    excluded = result["filtering"]["excluded_geometries"]
    assert len(excluded) == 1
    assert excluded[0]["status"] == "RATIO_EXCLUDED"
    assert excluded[0]["tension_Rf"] is not None
    assert excluded[0]["tension_Rf"] > 0.0

def test_ratio_filter_keeps_no_spin_geometry_as_not_applicable() -> None:
    spin = 0.75
    rf_obs = _rf_at_spin(spin)
    multimode = _multimode_payload(240.0, 4.0, 240.0 * rf_obs, 1.5, frac=0.01)
    compatible = _compatible_payload([{"geometry_id": "bardeen_variant", "metadata": {"theory": "Bardeen"}}])

    result = run_ratio_filter(
        run_id="run_test",
        multimode_estimates=multimode,
        compatible_payload=compatible,
        ranked_all_full=None,
        estimates_sha256="a",
        compatible_set_sha256="b",
        sigma_rf=2.0,
        sigma_rq=2.0,
        chi_grid_points=100,
        apply_rq=False,
    )

    row = result["filtering"]["compatible_geometries"][0]
    assert row["status"] == "RATIO_NOT_APPLICABLE"
    assert row["spin"] is None

def test_compute_observed_ratios_requires_mode_221() -> None:
    multimode = {"modes": [_mode_payload("220", 250.0, 4.0)]}
    with pytest.raises(ValueError, match=r"\(2,2,1\)"):
        compute_observed_ratios(multimode, sigma_rf=2.0, sigma_rq=2.0)

def test_reduction_fraction_and_spin_constraints() -> None:
    spin_target = 0.75
    rf_obs = _rf_at_spin(spin_target)
    multimode = _multimode_payload(250.0, 4.2, 250.0 * rf_obs, 1.5, frac=0.004)
    rows = [
        {"geometry_id": "kerr_a0.00", "metadata": {"spin": 0.00}},
        {"geometry_id": "kerr_a0.02", "metadata": {"spin": 0.02}},
        {"geometry_id": "kerr_a0.05", "metadata": {"spin": 0.05}},
        {"geometry_id": "kerr_a0.65", "metadata": {"spin": 0.65}},
        {"geometry_id": "kerr_a0.70", "metadata": {"spin": 0.70}},
        {"geometry_id": "kerr_a0.75", "metadata": {"spin": 0.75}},
        {"geometry_id": "kerr_a0.80", "metadata": {"spin": 0.80}},
        {"geometry_id": "kerr_a0.85", "metadata": {"spin": 0.85}},
        {"geometry_id": "nostat_1", "metadata": {"theory": "Alt"}},
        {"geometry_id": "nostat_2", "metadata": {"theory": "Alt"}},
    ]
    result = run_ratio_filter(
        run_id="run_test",
        multimode_estimates=multimode,
        compatible_payload=_compatible_payload(rows),
        ranked_all_full=None,
        estimates_sha256="a",
        compatible_set_sha256="b",
        sigma_rf=2.0,
        sigma_rq=2.0,
        chi_grid_points=100,
        apply_rq=False,
    )

    assert result["filtering"]["n_ratio_excluded"] == 3
    assert result["filtering"]["n_ratio_compatible"] == 5
    assert result["filtering"]["n_ratio_not_applicable"] == 2
    assert result["filtering"]["reduction_fraction"] == pytest.approx(0.3)
    assert result["spin_constraints"]["chi_range_before_ratio"] == [0.0, 0.85]
    assert result["spin_constraints"]["chi_range_after_ratio"] == [0.65, 0.85]
    assert result["spin_constraints"]["spin_range_reduction_fraction"] is not None
    assert result["spin_constraints"]["spin_range_reduction_fraction"] > 0.0

def test_informativity_classification_thresholds() -> None:
    assert classify_informativity(0.833) == "HIGH"
    assert classify_informativity(0.167) == "LOW"
    assert classify_informativity(0.0) == "UNINFORMATIVE"

def test_ratio_filter_result_schema_fields_present() -> None:
    spin = 0.70
    rf_obs = _rf_at_spin(spin)
    multimode = _multimode_payload(240.0, 4.0, 240.0 * rf_obs, 1.45, frac=0.01)
    result = run_ratio_filter(
        run_id="run_test",
        multimode_estimates=multimode,
        compatible_payload=_compatible_payload([{"geometry_id": "kerr_a0.70", "metadata": {"spin": spin}}]),
        ranked_all_full=None,
        estimates_sha256="sha_est",
        compatible_set_sha256="sha_cs",
        sigma_rf=2.0,
        sigma_rq=2.0,
        chi_grid_points=100,
        apply_rq=True,
    )

    assert {
        "schema_version",
        "run_id",
        "created",
        "inputs",
        "parameters",
        "observed_ratios",
        "kerr_reference",
        "kerr_consistency",
        "filtering",
        "spin_constraints",
        "diagnostics",
        "verdict",
    } <= set(result.keys())
