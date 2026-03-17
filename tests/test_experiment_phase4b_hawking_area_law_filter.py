"""Regression tests for mvp/experiment_phase4b_hawking_area_law_filter.py.

Coverage:
 1. test_phase4b_requires_host_run_valid_pass
 2. test_phase4b_requires_phase4_upstream_present_and_pass
 3. test_phase4b_requires_initial_area_csv_with_required_columns
 4. test_phase4b_requires_all_event_ids_present
 5. test_phase4b_rejects_non_numeric_or_non_finite_A_initial
 6. test_phase4b_applies_relational_filter_A_final_ge_A_initial
 7. test_phase4b_preserves_phys_key_columns
 8. test_phase4b_writes_only_under_runs_host_run_experiment_phase4b
 9. test_phase4b_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest
10. test_phase4b_summary_declares_discriminative_filter
11. test_phase4_current_summary_can_declare_non_discriminative_role
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


def _mk_initial_area_csv(
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    ext_dir = run_dir / "external_inputs" / "hawking_area_initial"
    ext_dir.mkdir(parents=True, exist_ok=True)
    path = ext_dir / "per_event_initial_area.csv"
    fields = ["event_id", "A_initial", "source_ref", "method", "units"]
    _write_csv(path, rows, fields)
    return path


def _default_hawking_rows() -> list[dict[str, Any]]:
    """Two rows for event E1: both should pass (A_final > 1.0)."""
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


def _default_initial_rows(A_initial: float = 1.0) -> list[dict[str, Any]]:
    return [
        {
            "event_id": "E1",
            "A_initial": A_initial,
            "source_ref": "test_ref",
            "method": "test_method",
            "units": REQUIRED_UNITS,
        }
    ]


def _full_setup(
    tmp_path: Path,
    run_id: str = "run_host",
    hawking_rows: list[dict[str, Any]] | None = None,
    initial_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a fully valid setup and return run_dir."""
    run_dir = _mk_run_valid(tmp_path, run_id)
    _mk_phase4_upstream(
        run_dir,
        hawking_rows if hawking_rows is not None else _default_hawking_rows(),
    )
    _mk_initial_area_csv(
        run_dir,
        initial_rows if initial_rows is not None else _default_initial_rows(),
    )
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
    run_dir = _mk_run_valid(tmp_path, "run_no_p4")
    # No phase4 upstream created at all

    with pytest.raises(FileNotFoundError):
        run_experiment(run_id="run_no_p4")


def test_phase4b_requires_phase4_upstream_verdict_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if phase4 stage_summary verdict != PASS."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_p4fail")
    _mk_phase4_upstream(run_dir, _default_hawking_rows(), verdict="FAIL")
    _mk_initial_area_csv(run_dir, _default_initial_rows())

    with pytest.raises(RuntimeError, match="not PASS"):
        run_experiment(run_id="run_p4fail")


def test_phase4b_requires_phase4_hawking_area_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if per_event_hawking_area.csv is missing from phase4 outputs."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_no_hawk")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())
    _mk_initial_area_csv(run_dir, _default_initial_rows())
    # Remove the hawking area CSV
    (run_dir / "experiment" / _PHASE4_NAME / "outputs" / "per_event_hawking_area.csv").unlink()

    with pytest.raises(FileNotFoundError):
        run_experiment(run_id="run_no_hawk")


# ---------------------------------------------------------------------------
# Test 3 – initial area CSV schema validation
# ---------------------------------------------------------------------------


def test_phase4b_requires_initial_area_csv_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if per_event_initial_area.csv is missing entirely."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_no_init")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())
    # No initial area CSV

    with pytest.raises(FileNotFoundError):
        run_experiment(run_id="run_no_init")


@pytest.mark.parametrize("missing_col", ["event_id", "A_initial", "source_ref", "method", "units"])
def test_phase4b_requires_initial_area_csv_with_required_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_col: str
) -> None:
    """Abort if any required column is missing from per_event_initial_area.csv."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, f"run_no_{missing_col}")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    # Build row with the required column removed
    row = {
        "event_id": "E1",
        "A_initial": 1.0,
        "source_ref": "ref",
        "method": "fit",
        "units": REQUIRED_UNITS,
    }
    del row[missing_col]
    fields = [c for c in ["event_id", "A_initial", "source_ref", "method", "units"]
              if c != missing_col]
    ext_dir = run_dir / "external_inputs" / "hawking_area_initial"
    ext_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(ext_dir / "per_event_initial_area.csv", [row], fields)

    with pytest.raises(ValueError, match="missing required columns"):
        run_experiment(run_id=f"run_no_{missing_col}")


def test_phase4b_rejects_incompatible_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if units value is not REQUIRED_UNITS."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_bad_units")
    _mk_phase4_upstream(run_dir, _default_hawking_rows())
    _mk_initial_area_csv(
        run_dir,
        [{"event_id": "E1", "A_initial": 1.0, "source_ref": "r", "method": "m",
          "units": "wrong_units"}],
    )

    with pytest.raises(ValueError, match="incompatible"):
        run_experiment(run_id="run_bad_units")


# ---------------------------------------------------------------------------
# Test 4 – all event_ids must be present in initial area CSV
# ---------------------------------------------------------------------------


def test_phase4b_requires_all_event_ids_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort if any event_id in phase4 hawking rows lacks an A_initial entry."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path, "run_missing_eid")

    # Phase4 has events E1 and E2
    rows = _default_hawking_rows()
    rows.append({
        "event_id": "E2", "family": "kerr", "provenance": "berti",
        "M_solar": 8.0, "chi": 0.4,
        "A": _hawking_area(8.0, 0.4), "S": _hawking_area(8.0, 0.4) / 4.0,
        "hawking_pass": True,
    })
    _mk_phase4_upstream(run_dir, rows)

    # Initial area CSV only has E1 (missing E2)
    _mk_initial_area_csv(run_dir, _default_initial_rows())

    with pytest.raises(ValueError, match="Missing A_initial"):
        run_experiment(run_id="run_missing_eid")


# ---------------------------------------------------------------------------
# Test 5 – reject non-numeric or non-finite A_initial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_val", ["not_a_number", "nan", "inf", "-inf", ""])
def test_phase4b_rejects_non_numeric_or_non_finite_A_initial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_val: str
) -> None:
    """Abort if A_initial is non-numeric or non-finite."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_id = f"run_bad_a_{bad_val.replace('-', 'neg').replace('.', 'dot')}"
    run_dir = _mk_run_valid(tmp_path, run_id)
    _mk_phase4_upstream(run_dir, _default_hawking_rows())

    ext_dir = run_dir / "external_inputs" / "hawking_area_initial"
    ext_dir.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "A_initial", "source_ref", "method", "units"]
    _write_csv(
        ext_dir / "per_event_initial_area.csv",
        [{"event_id": "E1", "A_initial": bad_val, "source_ref": "r",
          "method": "m", "units": REQUIRED_UNITS}],
        fields,
    )

    with pytest.raises(ValueError):
        run_experiment(run_id=run_id)


# ---------------------------------------------------------------------------
# Test 6 – relational filter A_final >= A_initial
# ---------------------------------------------------------------------------


def test_phase4b_applies_relational_filter_A_final_ge_A_initial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some rows pass (A_final >= A_initial) and some fail (A_final < A_initial)."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    # Row 1: M=10, chi=0.5 → A≈4691  (large, will PASS for A_initial=3000)
    # Row 2: M=5,  chi=0.3 → A≈1228  (small, will FAIL for A_initial=3000)
    M1, chi1 = 10.0, 0.5
    A1 = _hawking_area(M1, chi1)  # ≈4691
    M2, chi2 = 5.0, 0.3
    A2 = _hawking_area(M2, chi2)  # ≈1228
    A_init = 3000.0  # between A2 and A1

    assert A1 > A_init  # row1 passes
    assert A2 < A_init  # row2 fails

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

    run_dir = _full_setup(
        tmp_path, "run_filter",
        hawking_rows=hawking_rows,
        initial_rows=[{
            "event_id": "E1", "A_initial": A_init,
            "source_ref": "ref", "method": "fit", "units": REQUIRED_UNITS,
        }],
    )

    result = run_experiment(run_id="run_filter")

    hfs = result["hawking_filter_summary"]
    assert hfs["n_rows_input_common"] == 2
    assert hfs["n_rows_hawking_pass"] == 1
    assert hfs["n_rows_hawking_fail"] == 1
    assert hfs["n_rows_hawking_pass"] < hfs["n_rows_input_common"]
    assert hfs["n_rows_hawking_fail"] > 0

    # Verify filter CSV
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

    # area_gap signs
    assert float(pass_rows[0]["area_gap"]) > 0.0   # A_final > A_initial
    assert float(fail_rows[0]["area_gap"]) < 0.0   # A_final < A_initial


# ---------------------------------------------------------------------------
# Test 7 – phys_key columns preserved
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
        # phys_key fields must be non-empty
        assert r["family"].strip()
        assert r["provenance"].strip()
        assert r["M_solar"].strip()
        assert r["chi"].strip()


# ---------------------------------------------------------------------------
# Test 8 – writes only under runs/<host_run>/experiment/phase4b...
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

    # No other directory should contain phase4b experiment outputs
    for p in tmp_path.iterdir():
        if p.is_dir() and p.name != "run_isolation":
            assert not (p / "experiment" / _PHASE4B_NAME).exists(), (
                f"Unexpected phase4b output found under {p}"
            )


# ---------------------------------------------------------------------------
# Test 9 – entrypoint logging
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
# Test 10 – stage_summary declares discriminative filter
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

    # notes must mention discriminative
    notes = " ".join(ss.get("notes", []))
    assert "discriminative" in notes.lower()

    # filter_definition contract
    fd = ss["filter_definition"]
    assert fd["rule"] == "A_final >= A_initial"
    assert fd["tolerance"] == 0.0
    assert fd["stable_identifier"] == "phys_key"

    # gating sub-fields
    for gf in [
        "host_run_valid", "phase4_upstream_present", "phase4_upstream_pass",
        "initial_area_input_present", "initial_area_schema_valid",
        "initial_area_units_compatible",
    ]:
        assert ss["gating"][gf] is True, f"gating.{gf} must be True"

    # metrics
    for mf in [
        "n_rows_input_common", "n_rows_hawking_pass", "n_rows_hawking_fail",
        "n_events_total", "n_events_with_nonempty_hawking",
        "n_events_empty_after_filter", "pass_fraction",
    ]:
        assert mf in ss["metrics"], f"Missing metrics.{mf}"


# ---------------------------------------------------------------------------
# Test 11 – Phase4 current stage_summary declares non-discriminative role
# ---------------------------------------------------------------------------


def test_phase4_current_summary_can_declare_non_discriminative_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase4 stage_summary must declare filter_role=domain_admissibility_only
    and discriminative_filter=False to distinguish itself from Phase4B.
    """
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _full_setup(tmp_path, "run_p4role")

    # Check the stubbed phase4 stage_summary (written by _mk_phase4_upstream)
    ss_path = run_dir / "experiment" / _PHASE4_NAME / "stage_summary.json"
    ss = json.loads(ss_path.read_text(encoding="utf-8"))

    assert ss.get("discriminative_filter") is False, (
        "Phase4 stage_summary must declare discriminative_filter=False"
    )
    assert ss.get("filter_role") == "domain_admissibility_only", (
        "Phase4 stage_summary must declare filter_role='domain_admissibility_only'"
    )


def test_phase4_live_stage_summary_declares_non_discriminative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live Phase4 module must write discriminative_filter=False
    and filter_role=domain_admissibility_only in its stage_summary.

    Uses a minimal invocation of Phase4's run_experiment directly.
    """
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

    ss_path = (
        tmp_path / host / "experiment" / _PHASE4_NAME / "stage_summary.json"
    )
    ss = _json.loads(ss_path.read_text(encoding="utf-8"))

    assert ss.get("discriminative_filter") is False, (
        "Live Phase4 must declare discriminative_filter=False"
    )
    assert ss.get("filter_role") == "domain_admissibility_only", (
        "Live Phase4 must declare filter_role='domain_admissibility_only'"
    )


# ---------------------------------------------------------------------------
# Additional contract: summary JSON schemas
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
        "filter_rule", "tolerance", "units",
        "n_rows_input_common", "n_rows_hawking_pass", "n_rows_hawking_fail",
        "pass_fraction", "n_events_total", "n_events_with_nonempty_hawking",
        "n_events_empty_after_filter", "area_gap_quantiles",
    ]:
        assert f in hfs, f"Missing field {f!r} in hawking_filter_summary.json"

    assert hfs["schema_version"] == "hawking_filter_summary_v1"
    assert hfs["filter_rule"] == "A_final >= A_initial"
    assert hfs["units"] == REQUIRED_UNITS
    for q in ("p10", "p50", "p90"):
        assert q in hfs["area_gap_quantiles"], f"Missing quantile {q!r}"


def test_phase4b_hawking_filter_support_summary_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hawking_filter_support_summary.json must have schema_version and rows."""
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
                  "pass_fraction", "empty_after_filter", "A_initial"]:
            assert f in row, f"Missing field {f!r} in support_summary row"
