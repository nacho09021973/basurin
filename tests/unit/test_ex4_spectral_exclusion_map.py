"""Unit tests for experiment_ex4_spectral_exclusion_map.py.

Tests operate entirely on inline payloads — no real pipeline runs required.
Run with:
    pytest -q -o "addopts=" tests/unit/test_ex4_spectral_exclusion_map.py -v
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path (parent conftest.py also does this).
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mvp.experiment_ex4_spectral_exclusion_map import (
    build_exclusion_matrix,
    build_theory_survival,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

MINI_AGGREGATE = {
    "schema_version": "mvp_aggregate_v2",
    "n_events": 3,
    "events": [
        {
            "event_id": "GW150914",
            "run_id": "run_a",
            "threshold_d2": 5.99,
            "n_atlas": 10,
            "censoring": {
                "has_221": True,
                "vote_kerr": "PASS",
                "weight_scalar": 1.0,
                "weight_vector4": 1.0,
                "reason": None,
            },
        },
        {
            "event_id": "GW151226",
            "run_id": "run_b",
            "threshold_d2": 5.99,
            "n_atlas": 10,
            "censoring": {
                "has_221": False,
                "vote_kerr": "INCONCLUSIVE",
                "weight_scalar": 1.0,
                "weight_vector4": 0.0,
                "reason": "mode_221_missing",
            },
        },
        {
            "event_id": "GW170104",
            "run_id": "run_c",
            "threshold_d2": 5.99,
            "n_atlas": 10,
            "censoring": {
                "has_221": True,
                "vote_kerr": "PASS",
                "weight_scalar": 1.0,
                "weight_vector4": 1.0,
                "reason": None,
            },
        },
    ],
    "joint_posterior": {
        "joint_ranked_all": [
            {
                "geometry_id": "kerr_220_a0.67",
                "d2_per_event": [1.2, 2.3, 1.8],
                "d2_sum": 5.3,
                "coverage": 1.0,
                "metadata": {"theory": "GR_Kerr", "mode": "(2,2,0)", "spin": 0.67},
            },
            {
                "geometry_id": "kerr_220_a0.90",
                "d2_per_event": [12.5, 0.5, 15.3],
                "d2_sum": 28.3,
                "coverage": 1.0,
                "metadata": {"theory": "GR_Kerr", "mode": "(2,2,0)", "spin": 0.90},
            },
            {
                "geometry_id": "bardeen_220_q0.3",
                "d2_per_event": [8.1, None, 7.2],
                "d2_sum": 15.3,
                "coverage": 0.67,
                "metadata": {"theory": "Bardeen", "mode": "(2,2,0)", "charge": 0.3},
            },
        ],
    },
}


def _get_matrix() -> list[dict]:
    events = MINI_AGGREGATE["events"]
    joint_ranked_all = MINI_AGGREGATE["joint_posterior"]["joint_ranked_all"]
    return build_exclusion_matrix(events, joint_ranked_all, top_k=50)


def _get_matrix_and_survival():
    matrix = _get_matrix()
    per_theory, summary = build_theory_survival(matrix, n_events=3)
    return matrix, per_theory, summary


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_compatible_geometry_detected():
    """kerr_220_a0.67: d2 < 5.99 in all 3 events → globally compatible."""
    matrix = _get_matrix()
    row = next(r for r in matrix if r["geometry_id"] == "kerr_220_a0.67")
    assert row["is_globally_compatible"] is True
    assert row["n_excluded"] == 0
    assert row["survival_fraction"] == 1.0


def test_excluded_geometry_detected():
    """kerr_220_a0.90: d2=12.5 (event 0) and d2=15.3 (event 2) exceed threshold."""
    matrix = _get_matrix()
    row = next(r for r in matrix if r["geometry_id"] == "kerr_220_a0.90")
    assert row["is_globally_compatible"] is False
    assert row["n_excluded"] == 2
    # 1 compatible, 2 excluded → survival = 1/3
    assert abs(row["survival_fraction"] - 1 / 3) < 1e-9


def test_not_evaluated_handled():
    """bardeen_220_q0.3: event index 1 has d2=None → NOT_EVALUATED."""
    matrix = _get_matrix()
    row = next(r for r in matrix if r["geometry_id"] == "bardeen_220_q0.3")
    # Event 1 (GW151226) is null → NOT_EVALUATED
    assert row["status_per_event"][1] == "NOT_EVALUATED"
    # Denominator = evaluated events only: events 0 (EXCLUDED) + 2 (EXCLUDED) = 2
    # n_compatible=0, n_excluded=2 → survival_fraction = 0/2 = 0.0
    assert row["survival_fraction"] == 0.0
    assert row["n_not_evaluated"] == 1


def test_theory_survival_grouping():
    """GR_Kerr has 2 entries: 1 globally compatible, 1 excluded at least once."""
    _, per_theory, _ = _get_matrix_and_survival()
    gr_kerr = next(t for t in per_theory if t["theory"] == "GR_Kerr")
    assert gr_kerr["n_globally_compatible"] == 1
    assert gr_kerr["n_excluded_at_least_once"] == 1


def test_excess_sigma_calculation():
    """kerr_220_a0.90 at event 0: excess_sigma = sqrt(12.5 - 5.99) ≈ 2.551."""
    matrix = _get_matrix()
    row = next(r for r in matrix if r["geometry_id"] == "kerr_220_a0.90")
    sigma = row["excess_sigma_per_event"][0]
    expected = math.sqrt(12.5 - 5.99)  # sqrt(6.51) ≈ 2.5515
    assert sigma is not None
    assert abs(sigma - expected) < 0.01


def test_spin_range_compatible():
    """Only kerr_220_a0.67 (spin=0.67) is globally compatible for GR_Kerr."""
    _, per_theory, _ = _get_matrix_and_survival()
    gr_kerr = next(t for t in per_theory if t["theory"] == "GR_Kerr")
    assert gr_kerr["spin_range_compatible"] == [0.67, 0.67]


def test_schema_fields_present(tmp_path, monkeypatch):
    """exclusion_map.json and theory_survival.json contain all required fields."""
    # Set up a fake aggregate run under tmp_path
    runs_root = tmp_path / "runs"
    run_id = "test_run_ex4_schema"

    agg_dir = runs_root / run_id / "s5_aggregate" / "outputs"
    agg_dir.mkdir(parents=True)
    (agg_dir / "aggregate.json").write_text(
        json.dumps(MINI_AGGREGATE, indent=2), encoding="utf-8"
    )

    # Create RUN_VALID (required by check_run_valid=True in contract)
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS", "created": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(sys, "argv", ["ex4", "--run", run_id])

    from mvp.experiment_ex4_spectral_exclusion_map import main as ex4_main

    result_code = ex4_main()
    assert result_code == 0

    stage_dir = runs_root / run_id / "experiment_ex4_spectral_exclusion"
    em_path = stage_dir / "outputs" / "exclusion_map.json"
    ts_path = stage_dir / "outputs" / "theory_survival.json"
    assert em_path.exists(), f"exclusion_map.json not found at {em_path}"
    assert ts_path.exists(), f"theory_survival.json not found at {ts_path}"

    em = json.loads(em_path.read_text())
    ts = json.loads(ts_path.read_text())

    # Verify exclusion_map.json schema fields
    for field in [
        "schema_version",
        "run_id",
        "n_events",
        "created",
        "aggregate_sha256",
        "parameters",
        "n_geometries_evaluated",
        "events",
        "matrix",
    ]:
        assert field in em, f"exclusion_map.json missing field: {field}"
    assert em["schema_version"] == "ex4_exclusion_map_v1"
    assert em["run_id"] == run_id
    assert em["n_events"] == 3

    # Verify theory_survival.json schema fields
    for field in [
        "schema_version",
        "run_id",
        "n_events",
        "created",
        "n_theories",
        "summary",
        "per_theory",
    ]:
        assert field in ts, f"theory_survival.json missing field: {field}"
    assert ts["schema_version"] == "ex4_theory_survival_v1"
    assert ts["run_id"] == run_id
    assert ts["n_events"] == 3

    # Verify manifest.json and stage_summary.json are produced
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()


def test_abort_on_missing_threshold(tmp_path, monkeypatch, capsys):
    """SystemExit(2) when an event is missing threshold_d2."""
    runs_root = tmp_path / "runs"
    run_id = "test_run_ex4_bad"

    # Build aggregate where first event lacks threshold_d2
    bad_aggregate = {
        "schema_version": "mvp_aggregate_v2",
        "n_events": 1,
        "events": [
            {
                "event_id": "GW150914",
                "run_id": "run_a",
                # threshold_d2 intentionally absent
                "n_atlas": 10,
                "censoring": {
                    "has_221": True,
                    "vote_kerr": "PASS",
                    "weight_scalar": 1.0,
                    "weight_vector4": 1.0,
                    "reason": None,
                },
            }
        ],
        "joint_posterior": {
            "joint_ranked_all": [
                {
                    "geometry_id": "kerr_220_a0.67",
                    "d2_per_event": [1.2],
                    "d2_sum": 1.2,
                    "coverage": 1.0,
                    "metadata": {"theory": "GR_Kerr", "spin": 0.67},
                }
            ]
        },
    }

    agg_dir = runs_root / run_id / "s5_aggregate" / "outputs"
    agg_dir.mkdir(parents=True)
    (agg_dir / "aggregate.json").write_text(
        json.dumps(bad_aggregate, indent=2), encoding="utf-8"
    )

    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS", "created": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(sys, "argv", ["ex4", "--run", run_id])

    from mvp.experiment_ex4_spectral_exclusion_map import main as ex4_main

    with pytest.raises(SystemExit) as exc_info:
        ex4_main()
    assert exc_info.value.code == 2

    # Verify the error message mentions threshold_d2
    captured = capsys.readouterr()
    assert "threshold_d2" in captured.err
