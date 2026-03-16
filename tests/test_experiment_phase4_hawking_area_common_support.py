"""Regression tests for mvp/experiment_phase4_hawking_area_common_support.py.

Coverage:
1. test_host_run_gating_fails_if_run_valid_not_pass
2. test_batch_gating_requires_stage_summary_pass_and_results_csv
3. test_phys_key_uses_family_provenance_M_solar_chi_round6
4. test_hawking_filter_is_applied_only_after_k220_inter_k221
5. test_outputs_written_only_under_runs_host_run_experiment_phase4
6. test_entrypoint_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from mvp.experiment_phase4_hawking_area_common_support import run_experiment


# ---------------------------------------------------------------------------
# Shared fixtures helpers (mirrors phase-3 test helpers)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mk_run(tmp_path: Path, run_id: str) -> Path:
    """Create a minimal valid host-run with RUN_VALID/verdict.json = PASS."""
    run_dir = tmp_path / run_id
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    return run_dir


def _mk_batch(tmp_path: Path, batch_id: str) -> Path:
    """Create a minimal valid batch-run (offline_batch stage_summary PASS, no RUN_VALID)."""
    batch_dir = tmp_path / batch_id
    ss_dir = batch_dir / "experiment" / "offline_batch"
    ss_dir.mkdir(parents=True, exist_ok=True)
    (ss_dir / "stage_summary.json").write_text(
        json.dumps({"verdict": "PASS", "status": "PASS"}), encoding="utf-8"
    )
    return batch_dir


def _write_compat(subrun_dir: Path, geoms: list[dict[str, Any]]) -> None:
    outdir = subrun_dir / "s4_geometry_filter" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"compatible_geometries": geoms}
    (outdir / "compatible_set.json").write_text(json.dumps(payload), encoding="utf-8")


def _geom(
    geometry_id: str = "g1",
    family: str = "kerr",
    source: str = "berti",
    M_solar: float = 10.0,
    chi: float = 0.5,
) -> dict[str, Any]:
    return {
        "geometry_id": geometry_id,
        "family": family,
        "source": source,
        "M_solar": M_solar,
        "chi": chi,
    }


def _run_basic(
    tmp_path: Path,
    host: str = "run_host",
    b220: str = "batch220",
    b221: str = "batch221",
    sub220: str = "sub220",
    sub221: str = "sub221",
    geom220: dict | None = None,
    geom221: dict | None = None,
) -> dict:
    """Set up a minimal valid experiment and run it. Returns run_experiment result."""
    if geom220 is None:
        geom220 = _geom()
    if geom221 is None:
        geom221 = _geom()

    _mk_run(tmp_path, host)
    batch220 = _mk_batch(tmp_path, b220)
    batch221 = _mk_batch(tmp_path, b221)
    sub220_dir = _mk_run(tmp_path, sub220)
    sub221_dir = _mk_run(tmp_path, sub221)

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": sub220, "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": sub221, "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    _write_compat(sub220_dir, [geom220])
    _write_compat(sub221_dir, [geom221])

    return run_experiment(run_id=host, batch_220=b220, batch_221=b221)


# ---------------------------------------------------------------------------
# Test 1 – host-run gating: no RUN_VALID → abort
# ---------------------------------------------------------------------------


def test_host_run_gating_fails_if_run_valid_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host run without RUN_VALID/verdict.json=PASS must be rejected."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    # Host run directory exists but has no RUN_VALID
    (tmp_path / "run_no_valid").mkdir(parents=True, exist_ok=True)
    _mk_batch(tmp_path, "b220_gfail")
    _mk_batch(tmp_path, "b221_gfail")
    _write_csv(
        tmp_path / "b220_gfail" / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sx", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        tmp_path / "b221_gfail" / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sy", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    with pytest.raises((FileNotFoundError, RuntimeError)):
        run_experiment(
            run_id="run_no_valid",
            batch_220="b220_gfail",
            batch_221="b221_gfail",
        )


def test_host_run_gating_fails_if_verdict_is_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host run with verdict=FAIL must also be rejected."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    run_dir = tmp_path / "run_fail_verdict"
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "FAIL"}), encoding="utf-8")

    _mk_batch(tmp_path, "b220_vfail")
    _mk_batch(tmp_path, "b221_vfail")

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        run_experiment(
            run_id="run_fail_verdict",
            batch_220="b220_vfail",
            batch_221="b221_vfail",
        )


# ---------------------------------------------------------------------------
# Test 2 – batch gating: requires stage_summary PASS and results.csv
# ---------------------------------------------------------------------------


def test_batch_gating_requires_stage_summary_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch with verdict=FAIL in stage_summary must be rejected."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_bgate")

    # batch220: stage_summary verdict=FAIL
    b220_dir = tmp_path / "b220_fail"
    ss_dir = b220_dir / "experiment" / "offline_batch"
    ss_dir.mkdir(parents=True, exist_ok=True)
    (ss_dir / "stage_summary.json").write_text(
        json.dumps({"verdict": "FAIL", "status": "FAIL"}), encoding="utf-8"
    )
    _write_csv(
        b220_dir / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sx", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _mk_batch(tmp_path, "b221_bgate_ok")

    with pytest.raises(ValueError, match="FAIL"):
        run_experiment(
            run_id="run_bgate",
            batch_220="b220_fail",
            batch_221="b221_bgate_ok",
        )


def test_batch_gating_requires_results_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch with stage_summary=PASS but missing results.csv must be rejected."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_nocsv")
    # batch220: PASS stage_summary but NO results.csv
    b220_dir = tmp_path / "b220_nocsv"
    ss_dir = b220_dir / "experiment" / "offline_batch"
    ss_dir.mkdir(parents=True, exist_ok=True)
    (ss_dir / "stage_summary.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    _mk_batch(tmp_path, "b221_nocsv_ok")

    with pytest.raises(FileNotFoundError):
        run_experiment(
            run_id="run_nocsv",
            batch_220="b220_nocsv",
            batch_221="b221_nocsv_ok",
        )


# ---------------------------------------------------------------------------
# Test 3 – phys_key uses (family, provenance, M_solar, chi) rounded to 6 dec
# ---------------------------------------------------------------------------


def test_phys_key_uses_family_provenance_M_solar_chi_round6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two geometries with same phys_key after rounding collapse to 1 key in K_common."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_rnd")
    batch220 = _mk_batch(tmp_path, "b220_rnd")
    batch221 = _mk_batch(tmp_path, "b221_rnd")
    sub220 = _mk_run(tmp_path, "sub220_rnd")
    sub221 = _mk_run(tmp_path, "sub221_rnd")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_rnd", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_rnd", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    # Two geometries with M_solar differing below 6th decimal → same key after rounding
    geom_a = _geom(geometry_id="ga", M_solar=10.0000001, chi=0.5)  # rounds to 10.0
    geom_b = _geom(geometry_id="gb", M_solar=10.0000009, chi=0.5)  # rounds to 10.000001
    # geom_a rounds to (kerr, berti, 10.0, 0.5)
    # geom_b rounds to (kerr, berti, 10.000001, 0.5) → different key
    # Verify: round(10.0000001, 6) == 10.0 and round(10.0000009, 6) == 10.000001
    assert round(10.0000001, 6) == 10.0
    assert round(10.0000009, 6) == 10.000001

    _write_compat(sub220, [geom_a, geom_b])
    _write_compat(sub221, [geom_a])  # only geom_a → k_common = {geom_a phys_key}

    result = run_experiment(run_id="run_rnd", batch_220="b220_rnd", batch_221="b221_rnd")

    css = result["common_support_summary"]
    assert css["n_common_events"] == 1
    # sub220 has 2 distinct phys_keys; sub221 has 1; intersection = 1
    support_csv = (
        tmp_path / "run_rnd" / "experiment" / "phase4_hawking_area_common_support"
        / "outputs" / "per_event_common_support.csv"
    )
    rows = list(csv.DictReader(support_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    assert int(rows[0]["n_k220"]) == 2
    assert int(rows[0]["n_k221"]) == 1
    assert int(rows[0]["n_k_common"]) == 1

    # The per_event_hawking_area.csv must have 1 row (only k_common) with correct M_solar
    hawking_csv = (
        tmp_path / "run_rnd" / "experiment" / "phase4_hawking_area_common_support"
        / "outputs" / "per_event_hawking_area.csv"
    )
    h_rows = list(csv.DictReader(hawking_csv.open(encoding="utf-8")))
    assert len(h_rows) == 1
    assert float(h_rows[0]["M_solar"]) == 10.0  # rounded to 6 dec


# ---------------------------------------------------------------------------
# Test 4 – Hawking filter applied ONLY on K_common, not on K220 \ K221
# ---------------------------------------------------------------------------


def test_hawking_filter_is_applied_only_after_k220_inter_k221(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K220-only geometry must NOT appear in per_event_hawking_area.csv.

    K220 has {K_shared, K_extra}; K221 has {K_shared}.
    K_common = {K_shared}; Hawking filter runs only on K_common.
    per_event_hawking_area must have exactly 1 row (K_shared only).
    """
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_hfilt")
    b220 = _mk_batch(tmp_path, "b220_hfilt")
    b221 = _mk_batch(tmp_path, "b221_hfilt")
    sub220 = _mk_run(tmp_path, "sub220_hfilt")
    sub221 = _mk_run(tmp_path, "sub221_hfilt")

    _write_csv(
        b220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_hfilt", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        b221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_hfilt", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    geom_shared = _geom(geometry_id="g_shared", M_solar=10.0, chi=0.5)
    geom_extra = _geom(geometry_id="g_extra", M_solar=99.0, chi=0.9, source="extra_src")
    _write_compat(sub220, [geom_shared, geom_extra])
    _write_compat(sub221, [geom_shared])

    result = run_experiment(run_id="run_hfilt", batch_220="b220_hfilt", batch_221="b221_hfilt")

    exp_dir = tmp_path / "run_hfilt" / "experiment" / "phase4_hawking_area_common_support"
    hawking_csv = exp_dir / "outputs" / "per_event_hawking_area.csv"
    h_rows = list(csv.DictReader(hawking_csv.open(encoding="utf-8")))

    # Only 1 row: the shared key.  The extra key (only in K220) must be absent.
    assert len(h_rows) == 1
    assert float(h_rows[0]["M_solar"]) == 10.0

    # Support CSV: n_k_hawking should reflect only k_common
    support_csv = exp_dir / "outputs" / "per_event_common_support.csv"
    s_rows = list(csv.DictReader(support_csv.open(encoding="utf-8")))
    assert int(s_rows[0]["n_k220"]) == 2
    assert int(s_rows[0]["n_k221"]) == 1
    assert int(s_rows[0]["n_k_common"]) == 1

    # Verify Hawking area formula for K_shared (M=10, chi=0.5)
    M, chi = 10.0, 0.5
    expected_A = 8.0 * math.pi * M**2 * (1.0 + math.sqrt(1.0 - chi**2))
    expected_S = expected_A / 4.0
    assert abs(float(h_rows[0]["A"]) - expected_A) < 1e-9
    assert abs(float(h_rows[0]["S"]) - expected_S) < 1e-9
    assert h_rows[0]["hawking_pass"] == "True"


# ---------------------------------------------------------------------------
# Test 5 – outputs written only under runs/<host_run>/experiment/phase4...
# ---------------------------------------------------------------------------


def test_outputs_written_only_under_runs_host_run_experiment_phase4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All outputs must reside under runs/<host_run>/experiment/phase4_hawking_area_common_support/."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _run_basic(
        tmp_path,
        host="run_outcheck",
        b220="b220_oc",
        b221="b221_oc",
        sub220="sub220_oc",
        sub221="sub221_oc",
    )

    exp_dir = tmp_path / "run_outcheck" / "experiment" / "phase4_hawking_area_common_support"
    assert exp_dir.is_dir()

    # Required output artefacts
    assert (exp_dir / "manifest.json").exists()
    assert (exp_dir / "stage_summary.json").exists()
    assert (exp_dir / "outputs" / "per_event_common_support.csv").exists()
    assert (exp_dir / "outputs" / "per_event_hawking_area.csv").exists()
    assert (exp_dir / "outputs" / "common_support_summary.json").exists()
    assert (exp_dir / "outputs" / "hawking_area_summary.json").exists()

    # Nothing must be written outside run_outcheck
    other_runs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name != "run_outcheck"
                  and not p.name.startswith(("b220", "b221", "sub"))]
    for run_dir in other_runs:
        assert not (run_dir / "experiment" / "phase4_hawking_area_common_support").exists(), (
            f"Unexpected output found under {run_dir}"
        )

    # stage_summary must have verdict=PASS
    ss = json.loads((exp_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert ss["verdict"] == "PASS"
    assert ss["host_run"] == "run_outcheck"

    # common_support_summary contract fields
    css = json.loads((exp_dir / "outputs" / "common_support_summary.json").read_text(encoding="utf-8"))
    for field in ["host_run", "batch_220", "batch_221", "phys_key_definition",
                  "n_common_events", "K220", "K221", "K220_inter_K221",
                  "n_empty_intersection_events", "n_non_subset_cases"]:
        assert field in css, f"Missing field {field!r} in common_support_summary.json"

    # hawking_area_summary contract fields
    has = json.loads((exp_dir / "outputs" / "hawking_area_summary.json").read_text(encoding="utf-8"))
    for field in ["formula", "n_rows_input_common", "n_rows_hawking_pass",
                  "n_rows_hawking_fail", "n_events_with_nonempty_hawking",
                  "A_quantiles", "S_quantiles"]:
        assert field in has, f"Missing field {field!r} in hawking_area_summary.json"
    assert "p10" in has["A_quantiles"]
    assert "p50" in has["A_quantiles"]
    assert "p90" in has["A_quantiles"]


# ---------------------------------------------------------------------------
# Test 6 – entrypoint logs OUT_ROOT, STAGE_DIR, OUTPUTS_DIR, STAGE_SUMMARY, MANIFEST
# ---------------------------------------------------------------------------


def test_entrypoint_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """run_experiment must print the five canonical path lines via log_stage_paths."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _run_basic(
        tmp_path,
        host="run_logcheck",
        b220="b220_lc",
        b221="b221_lc",
        sub220="sub220_lc",
        sub221="sub221_lc",
    )

    captured = capsys.readouterr()
    out = captured.out

    assert "OUT_ROOT=" in out
    assert "STAGE_DIR=" in out
    assert "OUTPUTS_DIR=" in out
    assert "STAGE_SUMMARY=" in out
    assert "MANIFEST=" in out

    # STAGE_DIR must point to the phase4 experiment dir
    exp_dir = tmp_path / "run_logcheck" / "experiment" / "phase4_hawking_area_common_support"
    assert str(exp_dir) in out


# ---------------------------------------------------------------------------
# Test 7 – stage_summary.json has all required contract fields
# ---------------------------------------------------------------------------


def test_stage_summary_contract_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stage_summary.json must contain all fields required by the phase-4 contract."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _run_basic(
        tmp_path,
        host="run_ss",
        b220="b220_ss",
        b221="b221_ss",
        sub220="sub220_ss",
        sub221="sub221_ss",
    )

    exp_dir = tmp_path / "run_ss" / "experiment" / "phase4_hawking_area_common_support"
    ss = json.loads((exp_dir / "stage_summary.json").read_text(encoding="utf-8"))

    # Top-level required fields
    for f in ["experiment_name", "verdict", "host_run", "batch_220", "batch_221"]:
        assert f in ss, f"Missing top-level field {f!r}"

    # Gating sub-fields
    for f in ["host_run_valid", "batch_220_pass", "batch_221_pass",
              "results_csv_present_220", "results_csv_present_221"]:
        assert f in ss["gating"], f"Missing gating.{f}"

    # support_definition sub-fields
    sd = ss["support_definition"]
    assert sd["kind"] == "phys_key_intersection"
    assert sd["phys_key_fields"] == ["family", "provenance", "M_solar", "chi"]
    assert sd["round_decimals"] == 6
    assert sd["source_of_truth"] == "compatible_set.json"

    # hawking_area_filter sub-fields
    haf = ss["hawking_area_filter"]
    assert haf["applied_on"] == "K220_inter_K221"
    assert haf["scope"] == "per_event"
    assert haf["stable_identifier"] == "phys_key"

    # metrics sub-fields
    for f in ["n_common_events", "K220", "K221", "K220_inter_K221",
              "n_empty_intersection_events", "n_rows_hawking_pass"]:
        assert f in ss["metrics"], f"Missing metrics.{f}"


# ---------------------------------------------------------------------------
# Test 8 – Hawking area formula correctness and hawking_pass=False for chi > 1
# ---------------------------------------------------------------------------


def test_hawking_pass_false_for_unphysical_chi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geometry with |chi| > 1 must have hawking_pass=False and A=NaN."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_chi")
    b220 = _mk_batch(tmp_path, "b220_chi")
    b221 = _mk_batch(tmp_path, "b221_chi")
    sub220 = _mk_run(tmp_path, "sub220_chi")
    sub221 = _mk_run(tmp_path, "sub221_chi")

    _write_csv(
        b220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_chi", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        b221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_chi", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    # Geometry with chi > 1: unphysical
    geom_unphys = _geom(geometry_id="g_unphys", M_solar=10.0, chi=1.5)
    _write_compat(sub220, [geom_unphys])
    _write_compat(sub221, [geom_unphys])

    result = run_experiment(run_id="run_chi", batch_220="b220_chi", batch_221="b221_chi")

    exp_dir = tmp_path / "run_chi" / "experiment" / "phase4_hawking_area_common_support"
    h_rows = list(csv.DictReader(
        (exp_dir / "outputs" / "per_event_hawking_area.csv").open(encoding="utf-8")
    ))
    assert len(h_rows) == 1
    assert h_rows[0]["hawking_pass"] == "False"

    has = json.loads((exp_dir / "outputs" / "hawking_area_summary.json").read_text(encoding="utf-8"))
    assert has["n_rows_hawking_pass"] == 0
    assert has["n_rows_hawking_fail"] == 1
    assert has["n_events_with_nonempty_hawking"] == 0
