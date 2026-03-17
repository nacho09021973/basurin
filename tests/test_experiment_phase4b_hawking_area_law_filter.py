"""Regression tests for mvp/experiment_phase4b_hawking_area_law_filter.py.

Coverage:
 1. test_phase4b_requires_host_run_valid_pass (2 variants)
 2. test_phase4b_requires_phase4_upstream_present_and_pass (3 variants)
 3. test_phase4b_requires_gwtc_posteriors_dir_present
 4. test_phase4b_requires_one_posterior_file_per_event
 5. test_phase4b_rejects_missing_or_empty_samples
 6. test_phase4b_rejects_missing_required_sample_fields
 7. test_phase4b_rejects_non_numeric_or_non_finite_sample_values
 8. test_phase4b_rejects_invalid_component_spin_abs_gt_1
 9. test_phase4b_derives_initial_area_quantiles_from_samples
10. test_phase4b_uses_sample_median_as_default_initial_area_estimator
11. test_phase4b_applies_relational_filter_A_final_ge_A_initial
12. test_phase4b_preserves_phys_key_columns
13. test_phase4b_writes_only_under_runs_host_run_experiment_phase4b
14. test_phase4b_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest
15. test_phase4b_summary_declares_historical_area_theorem_runs_not_used
16. test_phase4b_does_not_require_manual_per_event_initial_area_csv
17. test_phase4b_summary_declares_discriminative_filter
18. test_phase4_current_summary_can_declare_non_discriminative_role
19. test_phase4_live_stage_summary_declares_non_discriminative
20. test_phase4b_hawking_filter_summary_schema
21. test_phase4b_hawking_filter_support_summary_schema
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from mvp.experiment_phase4b_hawking_area_law_filter import (
    REQUIRED_UNITS,
    _derive_initial_area_stats,
    _kerr_area,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_PHASE4_NAME = "phase4_hawking_area_common_support"
_PHASE4B_NAME = "phase4b_hawking_area_law_filter"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mk_run_valid(tmp_path: Path, run_id: str) -> Path:
    run_dir = tmp_path / run_id
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    return run_dir


def _hawking_area(M: float, chi: float) -> float:
    """A = 8*pi*M^2*(1+sqrt(1-chi^2)) in geom_solar_mass_sq units."""
    spin_sq = 1.0 - chi**2
    if spin_sq < 0.0:
        return float("nan")
    return 8.0 * math.pi * M**2 * (1.0 + math.sqrt(spin_sq))


def _mk_phase4_upstream(
    run_dir: Path,
    hawking_rows: list[dict[str, Any]],
    verdict: str = "PASS",
) -> Path:
    """Create all Phase4 upstream artifacts under run_dir/experiment/phase4_*."""
    phase4_dir = run_dir / "experiment" / _PHASE4_NAME
    outputs_dir = phase4_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ss = {
        "experiment_name": _PHASE4_NAME,
        "verdict": verdict,
        "host_run": run_dir.name,
        "discriminative_filter": False,
        "filter_role": "domain_admissibility_only",
    }
    (phase4_dir / "stage_summary.json").write_text(
        json.dumps(ss), encoding="utf-8"
    )
    (phase4_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "mvp_manifest_v1"}), encoding="utf-8"
    )

    hawking_fields = [
        "event_id", "family", "provenance", "M_solar", "chi", "A", "S", "hawking_pass",
    ]
    _write_csv(outputs_dir / "per_event_hawking_area.csv", hawking_rows, hawking_fields)

    # Stub remaining required Phase4 outputs
    (outputs_dir / "per_event_common_support.csv").write_text(
        "event_id,n_k_common\n", encoding="utf-8"
    )
    (outputs_dir / "common_support_summary.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (outputs_dir / "hawking_area_summary.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    return phase4_dir


def _mk_gwtc_posteriors(
    run_dir: Path,
    event_samples: dict[str, list[dict]],
) -> Path:
    """Create gwtc_posteriors/<event_id>.json files under external_inputs/."""
    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    for eid, samples in event_samples.items():
        data = {"event_id": eid, "samples": samples}
        (posteriors_dir / f"{eid}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    return posteriors_dir


def _default_samples() -> list[dict]:
    """Default IMR posterior samples for one event.

    m1=5, m2=3, chi1=0.1, chi2=0.1 →
    A_initial ≈ 8π*25*(1+√0.99) + 8π*9*(1+√0.99) ≈ 1706 M_sun²
    (much less than the final areas in _default_hawking_rows).
    """
    return [{"m1_source": 5.0, "m2_source": 3.0, "chi1": 0.1, "chi2": 0.1}]


def _default_hawking_rows() -> list[dict[str, Any]]:
    """Two rows for event E1: both should pass (A_final >> A_initial≈1706)."""
    M1, chi1 = 10.0, 0.5
    A1 = _hawking_area(M1, chi1)
    S1 = A1 / 4.0
    M2, chi2 = 15.0, 0.3
    A2 = _hawking_area(M2, chi2)
    S2 = A2 / 4.0
    return [
        {
            "event_id": "E1", "family": "kerr", "provenance": "berti",
            "M_solar": M1, "chi": chi1, "A": A1, "S": S1, "hawking_pass": True,
        },
        {
            "event_id": "E1", "family": "kerr", "provenance": "fits",
            "M_solar": M2, "chi": chi2, "A": A2, "S": S2, "hawking_pass": True,
        },
    ]


def _full_setup(
    tmp_path: Path,
    run_id: str = "run_host",
    hawking_rows: list[dict[str, Any]] | None = None,
    event_samples: dict[str, list[dict]] | None = None,
) -> Path:
    """Create a fully valid setup and return run_dir."""
    run_dir = _mk_run_valid(tmp_path, run_id)
    _mk_phase4_upstream(
        run_dir,
        hawking_rows if hawking_rows is not None else _default_hawking_rows(),
    )
    if event_samples is None:
        event_samples = {"E1": _default_samples()}
    _mk_gwtc_posteriors(run_dir, event_samples)
    return run_dir


# ---------------------------------------------------------------------------
# Test 1 – host-run gating: RUN_VALID must be PASS
# ---------------------------------------------------------------------------


def test_phase4b_requires_host_run_valid_pass_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if RUN_VALID/verdict.json does not exist."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    (tmp_path / "run_no_valid").mkdir(parents=True, exist_ok=True)

    with pytest.raises((FileNotFoundError, RuntimeError)):
        run_experiment(run_id="run_no_valid")


def test_phase4b_requires_host_run_valid_pass_fail_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if RUN_VALID verdict != PASS."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = tmp_path / "run_fail"
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "FAIL"}), encoding="utf-8")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        run_experiment(run_id="run_fail")


# ---------------------------------------------------------------------------
# Test 2 – Phase4 upstream gating
# ---------------------------------------------------------------------------


def test_phase4b_requires_phase4_upstream_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if phase4 stage_summary.json is absent."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _mk_run_valid(tmp_path, "run_no_p4")

    with pytest.raises(FileNotFoundError):
        run_experiment(run_id="run_no_p4")


def test_phase4b_requires_phase4_upstream_verdict_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if phase4 stage_summary verdict != PASS."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_p4fail")
    _mk_phase4_upstream(run_dir, _default_hawking_rows(), verdict="FAIL")
    _mk_gwtc_posteriors(run_dir, {"E1": _default_samples()})

    with pytest.raises(RuntimeError, match="not PASS"):
        run_experiment(run_id="run_p4fail")


def test_phase4b_requires_phase4_hawking_area_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if per_event_hawking_area.csv is missing from phase4 outputs."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_no_hawk")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())
    _mk_gwtc_posteriors(run_dir, {"E1": _default_samples()})
    (run_dir / "experiment" / _PHASE4_NAME / "outputs" / "per_event_hawking_area.csv").unlink()

    with pytest.raises(FileNotFoundError):
        run_experiment(run_id="run_no_hawk")


# ---------------------------------------------------------------------------
# Test 3 – gwtc_posteriors directory must be present
# ---------------------------------------------------------------------------


def test_phase4b_requires_gwtc_posteriors_dir_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if external_inputs/gwtc_posteriors/ directory is missing."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_no_post")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())
    # No posteriors directory created

    with pytest.raises(FileNotFoundError, match="gwtc_posteriors"):
        run_experiment(run_id="run_no_post")


# ---------------------------------------------------------------------------
# Test 4 – one posterior file per event required
# ---------------------------------------------------------------------------


def test_phase4b_requires_one_posterior_file_per_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if a required posterior file is missing for one event."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_miss_ev")

    # Phase4 has events E1 and E2
    rows = _default_hawking_rows()
    rows.append({
        "event_id": "E2", "family": "kerr", "provenance": "berti",
        "M_solar": 8.0, "chi": 0.4,
        "A": _hawking_area(8.0, 0.4), "S": _hawking_area(8.0, 0.4) / 4.0,
        "hawking_pass": True,
    })
    _mk_phase4_upstream(run_dir, rows)

    # Only provide posterior for E1, missing E2
    _mk_gwtc_posteriors(run_dir, {"E1": _default_samples()})

    with pytest.raises(FileNotFoundError):
        run_experiment(run_id="run_miss_ev")


# ---------------------------------------------------------------------------
# Test 5 – reject missing or empty samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_samples,exc_type", [
    ([], ValueError),
    ("not_a_list", ValueError),
])
def test_phase4b_rejects_missing_or_empty_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_samples: Any,
    exc_type: type,
) -> None:
    """Abort if samples is empty or not a list."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_empty_samp")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    (posteriors_dir / "E1.json").write_text(
        json.dumps({"event_id": "E1", "samples": bad_samples}), encoding="utf-8"
    )

    with pytest.raises((ValueError, TypeError)):
        run_experiment(run_id="run_empty_samp")


def test_phase4b_rejects_missing_samples_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if posterior JSON has no 'samples' key."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_no_samp_key")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    (posteriors_dir / "E1.json").write_text(
        json.dumps({"event_id": "E1"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="samples"):
        run_experiment(run_id="run_no_samp_key")


# ---------------------------------------------------------------------------
# Test 6 – reject missing required sample fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", ["m1_source", "m2_source", "chi1", "chi2"])
def test_phase4b_rejects_missing_required_sample_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    """Abort if any required sample field is absent."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_id = f"run_no_{missing_field}"
    run_dir = _mk_run_valid(tmp_path, run_id)
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    sample = {"m1_source": 5.0, "m2_source": 3.0, "chi1": 0.1, "chi2": 0.1}
    del sample[missing_field]

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    (posteriors_dir / "E1.json").write_text(
        json.dumps({"event_id": "E1", "samples": [sample]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=missing_field):
        run_experiment(run_id=run_id)


# ---------------------------------------------------------------------------
# Test 7 – reject non-numeric or non-finite sample values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad_val", [
    ("m1_source", "not_a_number"),
    ("m2_source", "nan"),
    ("chi1", "inf"),
    ("chi2", "-inf"),
    ("m1_source", ""),
])
def test_phase4b_rejects_non_numeric_or_non_finite_sample_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad_val: str
) -> None:
    """Abort if any sample field is non-numeric or non-finite."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    safe_val = bad_val.replace("-", "neg").replace(".", "dot").replace("_", "u")
    run_id = f"run_badval_{field}_{safe_val}"[:64]
    run_dir = _mk_run_valid(tmp_path, run_id)
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    sample: dict[str, Any] = {"m1_source": 5.0, "m2_source": 3.0, "chi1": 0.1, "chi2": 0.1}
    sample[field] = bad_val

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    (posteriors_dir / "E1.json").write_text(
        json.dumps({"event_id": "E1", "samples": [sample]}), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        run_experiment(run_id=run_id)


# ---------------------------------------------------------------------------
# Test 8 – reject invalid component spin |chi| > 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad_spin", [
    ("chi1", 1.001),
    ("chi2", -1.5),
])
def test_phase4b_rejects_invalid_component_spin_abs_gt_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad_spin: float
) -> None:
    """Abort if |chi1| > 1 or |chi2| > 1 in any sample."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_id = f"run_badspin_{field}"
    run_dir = _mk_run_valid(tmp_path, run_id)
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    sample: dict[str, Any] = {"m1_source": 5.0, "m2_source": 3.0, "chi1": 0.1, "chi2": 0.1}
    sample[field] = bad_spin

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    (posteriors_dir / "E1.json").write_text(
        json.dumps({"event_id": "E1", "samples": [sample]}), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        run_experiment(run_id=run_id)


# ---------------------------------------------------------------------------
# Test 9 – derives initial area quantiles from samples
# ---------------------------------------------------------------------------


def test_phase4b_derives_initial_area_quantiles_from_samples() -> None:
    """_derive_initial_area_stats returns correct p10/p50/p90 and n_samples."""
    # 3 samples with known A_initial values
    samples = [
        {"m1_source": 5.0, "m2_source": 3.0, "chi1": 0.0, "chi2": 0.0},  # A1+A2 = small
        {"m1_source": 10.0, "m2_source": 6.0, "chi1": 0.0, "chi2": 0.0},  # medium
        {"m1_source": 20.0, "m2_source": 12.0, "chi1": 0.0, "chi2": 0.0},  # large
    ]
    # Expected: A = 16*pi*M^2 for chi=0
    expected = sorted([
        16 * math.pi * (5**2 + 3**2),
        16 * math.pi * (10**2 + 6**2),
        16 * math.pi * (20**2 + 12**2),
    ])

    stats = _derive_initial_area_stats("E_test", samples)

    assert stats["n_samples"] == 3
    assert abs(stats["p50"] - expected[1]) < 1e-6  # median is middle value
    assert stats["p10"] <= stats["p50"] <= stats["p90"]


# ---------------------------------------------------------------------------
# Test 10 – uses sample median as default initial area estimator
# ---------------------------------------------------------------------------


def test_phase4b_uses_sample_median_as_default_initial_area_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A_initial used in filter equals p50 of the sample distribution."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    # 3 samples: sorted A_initial values are [low, mid, high]
    # Median = mid value
    samples = [
        {"m1_source": 1.0, "m2_source": 1.0, "chi1": 0.0, "chi2": 0.0},
        {"m1_source": 5.0, "m2_source": 5.0, "chi1": 0.0, "chi2": 0.0},
        {"m1_source": 20.0, "m2_source": 20.0, "chi1": 0.0, "chi2": 0.0},
    ]
    # A_initial for mid sample: 2 * 16*pi*25 = 800*pi ≈ 2513
    a_mid = 2 * 16 * math.pi * 25

    # Use a large final area so the row passes regardless
    A_final = 8 * _hawking_area(30.0, 0.1)
    run_dir = _full_setup(
        tmp_path, "run_median",
        hawking_rows=[{
            "event_id": "E1", "family": "kerr", "provenance": "fits",
            "M_solar": 30.0, "chi": 0.1, "A": A_final, "S": A_final / 4.0,
            "hawking_pass": True,
        }],
        event_samples={"E1": samples},
    )

    result = run_experiment(run_id="run_median")
    ss = result["stage_summary"]

    # stage_summary must declare estimator
    assert ss["initial_area_definition"]["estimator"] == "sample_median"

    # Check per_event_initial_area_from_posteriors.csv has p50 matching a_mid
    exp_dir = tmp_path / "run_median" / "experiment" / _PHASE4B_NAME
    derived_csv = exp_dir / "outputs" / "per_event_initial_area_from_posteriors.csv"
    rows = list(csv.DictReader(derived_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    assert abs(float(rows[0]["A_initial_p50"]) - a_mid) < 1e-3

    # hawking_filter_summary declares estimator
    hfs = result["hawking_filter_summary"]
    assert hfs["initial_area_estimator"] == "sample_median"


# ---------------------------------------------------------------------------
# Test 11 – relational filter A_final >= A_initial
# ---------------------------------------------------------------------------


def test_phase4b_applies_relational_filter_A_final_ge_A_initial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some rows pass (A_final >= A_initial) and some fail (A_final < A_initial)."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    # Row 1: M=10, chi=0.5 → A_final ≈ 4691  (PASS for A_initial≈3041)
    # Row 2: M=5,  chi=0.3 → A_final ≈ 1228  (FAIL for A_initial≈3041)
    M1, chi1 = 10.0, 0.5
    A1 = _hawking_area(M1, chi1)
    M2, chi2 = 5.0, 0.3
    A2 = _hawking_area(M2, chi2)

    # Posterior sample that gives A_initial_p50 between A2 and A1
    # m1=m2=5.5, chi=0 → A_initial = 2 * 16*pi*30.25 ≈ 3041
    a_init_p50 = 2 * 16 * math.pi * 5.5**2
    assert A1 > a_init_p50  # row1 passes
    assert A2 < a_init_p50  # row2 fails

    hawking_rows = [
        {
            "event_id": "E1", "family": "kerr", "provenance": "berti",
            "M_solar": M1, "chi": chi1,
            "A": A1, "S": A1 / 4.0, "hawking_pass": True,
        },
        {
            "event_id": "E1", "family": "kerr", "provenance": "fits",
            "M_solar": M2, "chi": chi2,
            "A": A2, "S": A2 / 4.0, "hawking_pass": True,
        },
    ]
    samples = [{"m1_source": 5.5, "m2_source": 5.5, "chi1": 0.0, "chi2": 0.0}]

    run_dir = _full_setup(
        tmp_path, "run_filter",
        hawking_rows=hawking_rows,
        event_samples={"E1": samples},
    )

    result = run_experiment(run_id="run_filter")

    hfs = result["hawking_filter_summary"]
    assert hfs["n_rows_input_common"] == 2
    assert hfs["n_rows_hawking_pass"] == 1
    assert hfs["n_rows_hawking_fail"] == 1
    assert hfs["n_rows_hawking_pass"] < hfs["n_rows_input_common"]
    assert hfs["n_rows_hawking_fail"] > 0

    exp_dir = tmp_path / "run_filter" / "experiment" / _PHASE4B_NAME
    filter_csv = exp_dir / "outputs" / "per_event_hawking_filter.csv"
    rows = list(csv.DictReader(filter_csv.open(encoding="utf-8")))
    assert len(rows) == 2

    pass_rows = [r for r in rows if r["hawking_pass"] == "True"]
    fail_rows = [r for r in rows if r["hawking_pass"] == "False"]
    assert len(pass_rows) == 1
    assert len(fail_rows) == 1
    assert abs(float(pass_rows[0]["M_solar"]) - M1) < 1e-9
    assert abs(float(fail_rows[0]["M_solar"]) - M2) < 1e-9
    assert float(pass_rows[0]["area_gap"]) > 0.0
    assert float(fail_rows[0]["area_gap"]) < 0.0


# ---------------------------------------------------------------------------
# Test 12 – phys_key columns preserved
# ---------------------------------------------------------------------------


def test_phase4b_preserves_phys_key_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output CSV must contain family, provenance, M_solar, chi for phys_key reconstruction."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_pk")
    run_experiment(run_id="run_pk")

    filter_csv = (
        tmp_path / "run_pk" / "experiment" / _PHASE4B_NAME
        / "outputs" / "per_event_hawking_filter.csv"
    )
    rows = list(csv.DictReader(filter_csv.open(encoding="utf-8")))
    assert len(rows) > 0

    for r in rows:
        for col in ("family", "provenance", "M_solar", "chi"):
            assert col in r, f"Missing phys_key column {col!r} in filter CSV"
        assert r["family"].strip()
        assert r["provenance"].strip()
        assert r["M_solar"].strip()
        assert r["chi"].strip()


# ---------------------------------------------------------------------------
# Test 13 – writes only under runs/<host_run>/experiment/phase4b...
# ---------------------------------------------------------------------------


def test_phase4b_writes_only_under_runs_host_run_experiment_phase4b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All outputs must reside under runs/<host_run>/experiment/phase4b_hawking_area_law_filter/."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_isolation")
    run_experiment(run_id="run_isolation")

    exp_dir = tmp_path / "run_isolation" / "experiment" / _PHASE4B_NAME
    assert exp_dir.is_dir()
    assert (exp_dir / "manifest.json").exists()
    assert (exp_dir / "stage_summary.json").exists()
    assert (exp_dir / "outputs" / "per_event_hawking_filter.csv").exists()
    assert (exp_dir / "outputs" / "hawking_filter_summary.json").exists()
    assert (exp_dir / "outputs" / "hawking_filter_support_summary.json").exists()
    assert (exp_dir / "outputs" / "per_event_initial_area_from_posteriors.csv").exists()

    for p in tmp_path.iterdir():
        if p.is_dir() and p.name != "run_isolation":
            assert not (p / "experiment" / _PHASE4B_NAME).exists(), (
                f"Unexpected phase4b output found under {p}"
            )


# ---------------------------------------------------------------------------
# Test 14 – entrypoint logging
# ---------------------------------------------------------------------------


def test_phase4b_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """run_experiment must print the five canonical path lines."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_log")
    run_experiment(run_id="run_log")

    out = capsys.readouterr().out
    assert "OUT_ROOT=" in out
    assert "STAGE_DIR=" in out
    assert "OUTPUTS_DIR=" in out
    assert "STAGE_SUMMARY=" in out
    assert "MANIFEST=" in out

    exp_dir = tmp_path / "run_log" / "experiment" / _PHASE4B_NAME
    assert str(exp_dir) in out


# ---------------------------------------------------------------------------
# Test 15 – stage_summary declares historical runs not used
# ---------------------------------------------------------------------------


def test_phase4b_summary_declares_historical_area_theorem_runs_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stage_summary notes must explicitly state historical runs are not used."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_hist")
    result = run_experiment(run_id="run_hist")

    notes = " ".join(result["stage_summary"].get("notes", []))
    assert "historical" in notes.lower() or "analysis_area_theorem" in notes.lower(), (
        "stage_summary.notes must mention historical runs quarantine"
    )

    # Also verify in the written JSON
    exp_dir = tmp_path / "run_hist" / "experiment" / _PHASE4B_NAME
    ss = json.loads((exp_dir / "stage_summary.json").read_text(encoding="utf-8"))
    notes_written = " ".join(ss.get("notes", []))
    assert "historical" in notes_written.lower() or "analysis_area_theorem" in notes_written.lower()


# ---------------------------------------------------------------------------
# Test 16 – does not require manual per_event_initial_area.csv
# ---------------------------------------------------------------------------


def test_phase4b_does_not_require_manual_per_event_initial_area_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run must succeed without any manual per_event_initial_area.csv present."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _full_setup(tmp_path, "run_no_manual_csv")

    # Explicitly verify the old CSV location does NOT exist
    old_csv = run_dir / "external_inputs" / "hawking_area_initial" / "per_event_initial_area.csv"
    assert not old_csv.exists(), "Test setup must not create the old manual CSV"

    # Must succeed without it
    result = run_experiment(run_id="run_no_manual_csv")
    assert result["stage_summary"]["verdict"] == "PASS"

    # The derived CSV is an output, not an input
    exp_dir = run_dir / "experiment" / _PHASE4B_NAME
    derived_csv = exp_dir / "outputs" / "per_event_initial_area_from_posteriors.csv"
    assert derived_csv.exists(), "Derived CSV must be written as output"


# ---------------------------------------------------------------------------
# Test 17 – stage_summary declares discriminative filter
# ---------------------------------------------------------------------------


def test_phase4b_summary_declares_discriminative_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stage_summary must declare discriminative_filter=True and correct filter_role."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_disc")
    run_experiment(run_id="run_disc")

    exp_dir = tmp_path / "run_disc" / "experiment" / _PHASE4B_NAME
    ss = json.loads((exp_dir / "stage_summary.json").read_text(encoding="utf-8"))

    assert ss["verdict"] == "PASS"
    assert ss["discriminative_filter"] is True
    assert "discriminative" in ss["filter_role"].lower()

    notes = " ".join(ss.get("notes", []))
    assert "discriminative" in notes.lower()

    fd = ss["filter_definition"]
    assert fd["rule"] == "A_final >= A_initial"
    assert fd["tolerance"] == 0.0
    assert fd["stable_identifier"] == "phys_key"

    # New gating fields (posteriors-based)
    for gf in [
        "host_run_valid", "phase4_upstream_present", "phase4_upstream_pass",
        "gwtc_posteriors_present", "gwtc_posteriors_schema_valid",
        "gwtc_posteriors_event_coverage_complete",
    ]:
        assert ss["gating"][gf] is True, f"gating.{gf} must be True"

    # initial_area_definition
    iad = ss["initial_area_definition"]
    assert iad["source"] == "gwtc_posteriors"
    assert iad["estimator"] == "sample_median"
    assert "8*pi" in iad["component_formula"]
    assert iad["units"] == REQUIRED_UNITS

    for mf in [
        "n_rows_input_common", "n_rows_hawking_pass", "n_rows_hawking_fail",
        "n_events_total", "n_events_with_nonempty_hawking",
        "n_events_empty_after_filter", "pass_fraction",
    ]:
        assert mf in ss["metrics"], f"Missing metrics.{mf}"


# ---------------------------------------------------------------------------
# Test 18 – Phase4 current stage_summary declares non-discriminative role
# ---------------------------------------------------------------------------


def test_phase4_current_summary_can_declare_non_discriminative_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase4 stage_summary must declare filter_role=domain_admissibility_only."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _full_setup(tmp_path, "run_p4role")

    ss_path = run_dir / "experiment" / _PHASE4_NAME / "stage_summary.json"
    ss = json.loads(ss_path.read_text(encoding="utf-8"))

    assert ss.get("discriminative_filter") is False
    assert ss.get("filter_role") == "domain_admissibility_only"


# ---------------------------------------------------------------------------
# Test 19 – live Phase4 declares non-discriminative
# ---------------------------------------------------------------------------


def test_phase4_live_stage_summary_declares_non_discriminative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live Phase4 module must write discriminative_filter=False."""
    import json as _json

    from mvp.experiment_phase4_hawking_area_common_support import (
        run_experiment as p4_run,
    )
    from tests.test_experiment_phase4_hawking_area_common_support import (
        _geom,
        _mk_batch,
        _mk_run,
        _write_compat,
        _write_csv as p4_write_csv,
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    host = "run_p4live"
    b220, b221 = "b220_live", "b221_live"
    s220, s221 = "sub220_live", "sub221_live"

    _mk_run(tmp_path, host)
    batch220 = _mk_batch(tmp_path, b220)
    batch221 = _mk_batch(tmp_path, b221)
    sub220_dir = _mk_run(tmp_path, s220)
    sub221_dir = _mk_run(tmp_path, s221)

    p4_write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": s220, "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    p4_write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": s221, "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_compat(sub220_dir, [_geom()])
    _write_compat(sub221_dir, [_geom()])

    p4_run(run_id=host, batch_220=b220, batch_221=b221)

    ss_path = tmp_path / host / "experiment" / _PHASE4_NAME / "stage_summary.json"
    ss = _json.loads(ss_path.read_text(encoding="utf-8"))

    assert ss.get("discriminative_filter") is False
    assert ss.get("filter_role") == "domain_admissibility_only"


# ---------------------------------------------------------------------------
# Test 20 – hawking_filter_summary.json schema
# ---------------------------------------------------------------------------


def test_phase4b_hawking_filter_summary_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hawking_filter_summary.json must have all required schema fields."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_schema")
    run_experiment(run_id="run_schema")

    exp_dir = tmp_path / "run_schema" / "experiment" / _PHASE4B_NAME
    hfs = json.loads(
        (exp_dir / "outputs" / "hawking_filter_summary.json").read_text(encoding="utf-8")
    )

    for f in [
        "schema_version", "host_run", "source_phase4_experiment",
        "initial_area_source", "initial_area_estimator",
        "filter_rule", "tolerance", "units",
        "n_rows_input_common", "n_rows_hawking_pass", "n_rows_hawking_fail",
        "pass_fraction", "n_events_total", "n_events_with_nonempty_hawking",
        "n_events_empty_after_filter", "area_gap_quantiles",
    ]:
        assert f in hfs, f"Missing field {f!r} in hawking_filter_summary.json"

    assert hfs["schema_version"] == "hawking_filter_summary_v1"
    assert hfs["filter_rule"] == "A_final >= A_initial"
    assert hfs["units"] == REQUIRED_UNITS
    assert hfs["initial_area_source"] == "external_inputs/gwtc_posteriors"
    assert hfs["initial_area_estimator"] == "sample_median"
    for q in ("p10", "p50", "p90"):
        assert q in hfs["area_gap_quantiles"], f"Missing quantile {q!r}"


# ---------------------------------------------------------------------------
# Test 21 – hawking_filter_support_summary.json schema
# ---------------------------------------------------------------------------


def test_phase4b_hawking_filter_support_summary_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hawking_filter_support_summary.json must have schema_version, rows, and n_samples."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    _full_setup(tmp_path, "run_supsummary")
    run_experiment(run_id="run_supsummary")

    exp_dir = tmp_path / "run_supsummary" / "experiment" / _PHASE4B_NAME
    hfss = json.loads(
        (exp_dir / "outputs" / "hawking_filter_support_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert hfss["schema_version"] == "hawking_filter_support_summary_v1"
    assert "rows" in hfss
    assert isinstance(hfss["rows"], list)

    for row in hfss["rows"]:
        for f in ["event_id", "n_input_rows", "n_pass_rows", "n_fail_rows",
                  "pass_fraction", "empty_after_filter", "A_initial", "n_samples"]:
            assert f in row, f"Missing field {f!r} in support_summary row"
        assert isinstance(row["n_samples"], int) and row["n_samples"] > 0
