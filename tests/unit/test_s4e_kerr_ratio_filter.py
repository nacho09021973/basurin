from __future__ import annotations

import math

import pytest

from mvp.contracts import CONTRACTS
from mvp.kerr_qnm_fits import kerr_ratio_curve
from mvp.s4e_kerr_ratio_filter import (
    classify_informativity,
    compute_observed_ratios,
    run_ratio_filter,
)


def _mode_payload(label: str, f_hz: float, q_factor: float, frac: float = 0.01) -> dict[str, object]:
    tau_s = q_factor / (math.pi * f_hz)
    return {
        "label": label,
        "estimates": {
            "f_hz": {"p10": f_hz * (1.0 - frac), "p50": f_hz, "p90": f_hz * (1.0 + frac)},
            "tau_s": {"p10": tau_s * (1.0 - frac), "p50": tau_s, "p90": tau_s * (1.0 + frac)},
            "Q": {"p10": q_factor * (1.0 - frac), "p50": q_factor, "p90": q_factor * (1.0 + frac)},
        },
    }


def _multimode_payload(*, f220_hz: float, q220: float, f221_hz: float, q221: float, frac: float = 0.01) -> dict[str, object]:
    return {
        "modes": [
            _mode_payload("220", f220_hz, q220, frac=frac),
            _mode_payload("221", f221_hz, q221, frac=frac),
        ]
    }


def _compatible_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"compatible_geometries": rows}


def _rf_at_spin(spin: float) -> float:
    return float(kerr_ratio_curve(chi_grid=[spin])["Rf_grid"][0])


def _run(rows: list[dict[str, object]], *, spin: float, frac: float = 0.01) -> dict[str, object]:
    rf_obs = _rf_at_spin(spin)
    multimode = _multimode_payload(f220_hz=250.0, q220=4.2, f221_hz=250.0 * rf_obs, q221=1.4, frac=frac)
    return run_ratio_filter(
        run_id="run_test",
        multimode_estimates=multimode,
        compatible_payload=_compatible_payload(rows),
        ranked_all_full=None,
        estimates_sha256="sha_est",
        compatible_set_sha256="sha_cs",
        sigma_rf=2.0,
        sigma_rq=2.0,
        chi_grid_points=200,
        apply_rq=False,
    )


def test_kerr_ratio_curve_basic() -> None:
    result = kerr_ratio_curve(n_points=10)
    assert len(result["chi_grid"]) == 10
    assert len(result["Rf_grid"]) == 10
    assert len(result["RQ_grid"]) == 10
    assert result["Rf_range"]["min"] < result["Rf_range"]["max"]
    assert result["RQ_range"]["min"] < result["RQ_range"]["max"]


def test_kerr_ratio_curve_schwarzschild() -> None:
    result = kerr_ratio_curve(chi_grid=[0.0])
    assert result["Rf_grid"][0] == pytest.approx(0.927, abs=0.01)
    assert result["RQ_grid"][0] == pytest.approx(0.304, abs=0.01)


def test_kerr_ratio_curve_near_extremal() -> None:
    result = kerr_ratio_curve(chi_grid=[0.99])
    assert result["Rf_grid"][0] > 0.95


def test_kerr_ratio_curve_monotonicity() -> None:
    result = kerr_ratio_curve(chi_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    assert result["Rf_grid"] == sorted(result["Rf_grid"])


def test_filter_excludes_incompatible() -> None:
    result = _run([{"geometry_id": "kerr_a0.01", "metadata": {"spin": 0.01}}], spin=0.8, frac=0.002)
    assert result["n_excluded"] == 1
    row = result["filtering"]["per_geometry"][0]
    assert row["compatible_by_ratio"] is False
    assert row["tension_sigma"] is not None


def test_filter_keeps_compatible() -> None:
    result = _run([{"geometry_id": "kerr_a0.67", "metadata": {"spin": 0.67}}], spin=0.67)
    assert result["n_surviving"] == 1
    row = result["filtering"]["per_geometry"][0]
    assert row["compatible_by_ratio"] is True


def test_filter_no_spin_warning() -> None:
    result = _run([{"geometry_id": "alt_geom", "metadata": {"theory": "Alt"}}], spin=0.67)
    assert "NO_SPIN_IN_ATLAS" in result["diagnostics"]["warning_codes"]
    row = result["filtering"]["per_geometry"][0]
    assert row["compatible_by_ratio"] is None


def test_filter_abort_no_221() -> None:
    multimode = {"modes": [_mode_payload("220", 250.0, 4.2)]}
    with pytest.raises(ValueError, match=r"\(2,2,1\)"):
        compute_observed_ratios(multimode, sigma_rf=2.0, sigma_rq=2.0)


def test_reduction_fraction() -> None:
    rows = [
        {"geometry_id": "kerr_a0.10", "metadata": {"spin": 0.10}},
        {"geometry_id": "kerr_a0.67", "metadata": {"spin": 0.67}},
        {"geometry_id": "kerr_a0.95", "metadata": {"spin": 0.95}},
        {"geometry_id": "alt_geom", "metadata": {"theory": "Alt"}},
    ]
    result = _run(rows, spin=0.67, frac=0.002)
    assert result["diagnostics"]["reduction_fraction"] == pytest.approx(0.5)


def test_tension_sigma_calculation() -> None:
    result = _run([{"geometry_id": "kerr_a0.01", "metadata": {"spin": 0.01}}], spin=0.9, frac=0.001)
    row = result["filtering"]["per_geometry"][0]
    assert row["tension_sigma"] == pytest.approx(row["tension_Rf"])


def test_informativity_class() -> None:
    assert classify_informativity(0.6) == "HIGH"
    assert classify_informativity(0.2) == "LOW"
    assert classify_informativity(0.0) == "UNINFORMATIVE"


def test_backward_compat_contract() -> None:
    result = _run([{"geometry_id": "kerr_a0.67", "metadata": {"spin": 0.67}}], spin=0.67)
    contract = CONTRACTS["s4e_kerr_ratio_filter"]
    assert contract.name == "s4e_kerr_ratio_filter"
    assert "outputs/ratio_filter_result.json" in contract.produced_outputs
    assert {"filtering", "diagnostics", "kerr_consistency", "verdict"} <= set(result.keys())
    assert result["Rf_kerr_range"]["min"] == pytest.approx(result["kerr_reference"]["Rf_range"]["min"])
