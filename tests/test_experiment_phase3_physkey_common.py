"""Regression tests for mvp/experiment_phase3_physkey_common.py.

Coverage:
1. test_base_case                             – common PASS event, non-empty phys_key intersection
2. test_fail_rows_filtered                    – FAIL rows skipped; exclusion counters in summary correct
3. test_metadata_source_and_ref               – 220 uses metadata.source, 221 uses metadata.ref → k_inter == 1
4. test_non_subset_cases                      – 220 has extra phys_key not in 221 → appears in non_subset_cases
5. test_missing_provenance_aborts             – geometry missing provenance → ValueError, no partial outputs
6. test_missing_m_solar_aborts                – geometry missing M_solar → ValueError, no partial outputs
7. test_batch_gate_passes_with_stage_summary  – batch without RUN_VALID accepted via stage_summary PASS
8. test_batch_gate_fails_when_verdict_not_pass – batch stage_summary verdict=FAIL → explicit abort
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from mvp.experiment_phase3_physkey_common import run_experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mk_run(tmp_path: Path, run_id: str) -> Path:
    """Create a minimal valid host-run directory with RUN_VALID/verdict.json = PASS."""
    run_dir = tmp_path / run_id
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    return run_dir


def _mk_batch(tmp_path: Path, batch_id: str) -> Path:
    """Create a minimal valid batch-run directory with offline_batch stage_summary PASS.

    batch_220 / batch_221 are offline_batch artefacts, not pipeline host-runs.
    Their completion contract is experiment/offline_batch/stage_summary.json verdict=PASS,
    NOT RUN_VALID (which offline_batch does not materialise).
    """
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


# ---------------------------------------------------------------------------
# Test 1 – base case
# ---------------------------------------------------------------------------


def test_base_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One common PASS event, identical phys_keys in both batches.

    Checks:
    - outputs exist (summary, csv, manifest, stage_summary)
    - n_common_events == 1
    - K220 == K221 == K220_inter_K221 (one key each)
    - per_event row: k220 == k221 == k_inter == 1
    - empty_intersection_events is empty
    - non_subset_cases is empty (k220 ⊆ k221)
    """
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_main")
    batch220 = _mk_batch(tmp_path, "batch220")
    batch221 = _mk_batch(tmp_path, "batch221")
    sub220 = _mk_run(tmp_path, "sub220_e1")
    sub221 = _mk_run(tmp_path, "sub221_e1")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_e1", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_e1", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    geom = {"geometry_id": "g1", "family": "kerr", "source": "berti", "M_solar": 10.0, "chi": 0.5}
    _write_compat(sub220, [geom])
    _write_compat(sub221, [geom])

    result = run_experiment(
        run_id="run_main",
        batch_220="batch220",
        batch_221="batch221",
    )

    exp_dir = tmp_path / "run_main" / "experiment" / "phase3_physkey_common"
    assert (exp_dir / "outputs" / "summary_physkey_common.json").exists()
    assert (exp_dir / "outputs" / "per_event_physkey_intersection.csv").exists()
    assert (exp_dir / "manifest.json").exists()
    assert (exp_dir / "stage_summary.json").exists()

    assert result["n_common_events"] == 1
    assert len(result["K220"]) == 1
    assert len(result["K221"]) == 1
    assert result["K220"] == result["K221"] == result["K220_inter_K221"]
    assert result["empty_intersection_events"] == []
    assert result["non_subset_cases"] == []

    # per-event CSV
    rows = list(
        csv.DictReader(
            (exp_dir / "outputs" / "per_event_physkey_intersection.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert len(rows) == 1
    assert rows[0]["event_id"] == "E1"
    assert int(rows[0]["k220"]) == 1
    assert int(rows[0]["k221"]) == 1
    assert int(rows[0]["k_inter"]) == 1


# ---------------------------------------------------------------------------
# Test 2 – FAIL rows are filtered
# ---------------------------------------------------------------------------


def test_fail_rows_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FAIL event in results.csv must be skipped; experiment must not abort."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_fail_filter")
    batch220 = _mk_batch(tmp_path, "batch220_ff")
    batch221 = _mk_batch(tmp_path, "batch221_ff")
    sub_pass220 = _mk_run(tmp_path, "sub_pass220")
    sub_pass221 = _mk_run(tmp_path, "sub_pass221")
    sub_fail220 = _mk_run(tmp_path, "sub_fail220")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[
            {"event_id": "E1", "subrun_id": "sub_pass220", "status": "PASS"},
            {"event_id": "E2", "subrun_id": "sub_fail220", "status": "FAIL"},
        ],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[
            {"event_id": "E1", "subrun_id": "sub_pass221", "status": "PASS"},
        ],
        fields=["event_id", "subrun_id", "status"],
    )

    geom = {"geometry_id": "g1", "family": "kerr", "source": "berti", "M_solar": 10.0, "chi": 0.5}
    _write_compat(sub_pass220, [geom])
    _write_compat(sub_pass221, [geom])

    result = run_experiment(
        run_id="run_fail_filter",
        batch_220="batch220_ff",
        batch_221="batch221_ff",
    )

    # E2 (FAIL) must be absent; only E1 counted
    assert result["n_common_events"] == 1
    assert result["n_events_valid_220"] == 1  # only E1 PASS from 220

    # Exclusion metric traceability
    assert result["n_rows_total_220"] == 2           # E1 PASS + E2 FAIL
    assert result["n_rows_skipped_status_220"] == 1  # E2 FAIL
    assert result["n_rows_skipped_missing_compatible_220"] == 0
    assert result["n_rows_total_221"] == 1
    assert result["n_rows_skipped_status_221"] == 0
    assert result["n_rows_skipped_missing_compatible_221"] == 0

    csv_rows = list(
        csv.DictReader(
            (
                tmp_path
                / "run_fail_filter"
                / "experiment"
                / "phase3_physkey_common"
                / "outputs"
                / "per_event_physkey_intersection.csv"
            ).open(encoding="utf-8")
        )
    )
    event_ids = {r["event_id"] for r in csv_rows}
    assert "E2" not in event_ids
    assert "E1" in event_ids


# ---------------------------------------------------------------------------
# Test 3 – metadata.source vs metadata.ref → same physical geometry
# ---------------------------------------------------------------------------


def test_metadata_source_and_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """220 uses metadata.source; 221 uses metadata.ref.

    Both refer to the same physical geometry → k_inter == 1.
    """
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_meta")
    batch220 = _mk_batch(tmp_path, "batch220_meta")
    batch221 = _mk_batch(tmp_path, "batch221_meta")
    sub220 = _mk_run(tmp_path, "sub220_meta")
    sub221 = _mk_run(tmp_path, "sub221_meta")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_meta", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_meta", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    # 220: source lives in metadata.source
    geom220 = {
        "geometry_id": "g220_meta",
        "metadata": {"family": "kerr", "source": "berti_2009_fit", "M_solar": 10.0, "chi": 0.3},
    }
    # 221: source lives in metadata.ref (no metadata.source key)
    geom221 = {
        "geometry_id": "g221_meta",
        "metadata": {"family": "kerr", "ref": "berti_2009_fit", "M_solar": 10.0, "chi": 0.3},
    }
    _write_compat(sub220, [geom220])
    _write_compat(sub221, [geom221])

    result = run_experiment(
        run_id="run_meta",
        batch_220="batch220_meta",
        batch_221="batch221_meta",
    )

    assert result["n_common_events"] == 1
    # The two geometries map to the same phys_key → intersection == 1
    csv_rows = list(
        csv.DictReader(
            (
                tmp_path
                / "run_meta"
                / "experiment"
                / "phase3_physkey_common"
                / "outputs"
                / "per_event_physkey_intersection.csv"
            ).open(encoding="utf-8")
        )
    )
    assert len(csv_rows) == 1
    assert int(csv_rows[0]["k_inter"]) == 1
    assert result["K220"] == result["K221"]
    assert len(result["K220_inter_K221"]) == 1


# ---------------------------------------------------------------------------
# Test 4 – non_subset_cases: 220 has extra phys_key
# ---------------------------------------------------------------------------


def test_non_subset_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """220 has one extra phys_key not present in 221.

    Contract:
    - event appears in non_subset_cases
    - K220 is NOT assumed ⊆ K221
    - k_inter < k220
    """
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_nonsub")
    batch220 = _mk_batch(tmp_path, "batch220_ns")
    batch221 = _mk_batch(tmp_path, "batch221_ns")
    sub220 = _mk_run(tmp_path, "sub220_ns")
    sub221 = _mk_run(tmp_path, "sub221_ns")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_ns", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_ns", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    # 220: two geometries (shared + extra)
    geom_shared = {"geometry_id": "gs", "family": "kerr", "source": "berti", "M_solar": 10.0, "chi": 0.5}
    geom_extra = {"geometry_id": "ge", "family": "kerr", "source": "berti", "M_solar": 99.0, "chi": 0.9}
    _write_compat(sub220, [geom_shared, geom_extra])
    # 221: only shared geometry
    _write_compat(sub221, [geom_shared])

    result = run_experiment(
        run_id="run_nonsub",
        batch_220="batch220_ns",
        batch_221="batch221_ns",
    )

    assert result["n_non_subset_cases"] == 1
    assert "E1" in result["non_subset_cases"]

    csv_rows = list(
        csv.DictReader(
            (
                tmp_path
                / "run_nonsub"
                / "experiment"
                / "phase3_physkey_common"
                / "outputs"
                / "per_event_physkey_intersection.csv"
            ).open(encoding="utf-8")
        )
    )
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert int(row["k220"]) == 2
    assert int(row["k221"]) == 1
    assert int(row["k_inter"]) == 1


# ---------------------------------------------------------------------------
# Test 5 – contract error: missing provenance in compatible_set.json
# ---------------------------------------------------------------------------


def test_missing_provenance_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A geometry with no source/ref must trigger an explicit ValueError (abort)."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_abort")
    batch220 = _mk_batch(tmp_path, "batch220_ab")
    batch221 = _mk_batch(tmp_path, "batch221_ab")
    sub220 = _mk_run(tmp_path, "sub220_ab")
    sub221 = _mk_run(tmp_path, "sub221_ab")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_ab", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_ab", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    # geometry missing provenance (no source, no ref, no metadata.source, no metadata.ref)
    bad_geom = {"geometry_id": "g_bad", "family": "kerr", "M_solar": 10.0, "chi": 0.5}
    _write_compat(sub220, [bad_geom])
    # 221 is valid but irrelevant – 220 will fail first
    good_geom = {"geometry_id": "g_ok", "family": "kerr", "source": "berti", "M_solar": 10.0, "chi": 0.5}
    _write_compat(sub221, [good_geom])

    with pytest.raises(ValueError, match="provenance"):
        run_experiment(
            run_id="run_abort",
            batch_220="batch220_ab",
            batch_221="batch221_ab",
        )

    # No outputs should have been written (atomic write never committed)
    exp_dir = tmp_path / "run_abort" / "experiment" / "phase3_physkey_common"
    assert not exp_dir.exists()


# ---------------------------------------------------------------------------
# Test 6 – missing M_solar aborts
# ---------------------------------------------------------------------------


def test_missing_m_solar_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A geometry with no M_solar must trigger an explicit ValueError."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_abort_m")
    batch220 = _mk_batch(tmp_path, "batch220_abm")
    batch221 = _mk_batch(tmp_path, "batch221_abm")
    sub220 = _mk_run(tmp_path, "sub220_abm")
    sub221 = _mk_run(tmp_path, "sub221_abm")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_abm", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_abm", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    bad_geom = {"geometry_id": "g_bad", "family": "kerr", "source": "berti", "chi": 0.5}
    _write_compat(sub220, [bad_geom])
    good_geom = {"geometry_id": "g_ok", "family": "kerr", "source": "berti", "M_solar": 10.0, "chi": 0.5}
    _write_compat(sub221, [good_geom])

    with pytest.raises(ValueError, match="M_solar"):
        run_experiment(
            run_id="run_abort_m",
            batch_220="batch220_abm",
            batch_221="batch221_abm",
        )

    # Atomic write must not have committed partial outputs
    exp_dir = tmp_path / "run_abort_m" / "experiment" / "phase3_physkey_common"
    assert not exp_dir.exists()


# ---------------------------------------------------------------------------
# Test 7 – batch gating: stage_summary.json PASS (no RUN_VALID on batch)
# ---------------------------------------------------------------------------


def test_batch_gate_passes_with_stage_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch runs without RUN_VALID are accepted when offline_batch stage_summary is PASS.

    This reflects the real contract: batch_220/221 are offline_batch artefacts, not
    pipeline host-runs. require_run_valid must NOT be called on them.
    """
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_host")
    # Batches have NO RUN_VALID – only stage_summary PASS
    batch220 = _mk_batch(tmp_path, "batch220_gate")
    batch221 = _mk_batch(tmp_path, "batch221_gate")
    sub220 = _mk_run(tmp_path, "sub220_gate")
    sub221 = _mk_run(tmp_path, "sub221_gate")

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_gate", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_gate", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    geom = {"geometry_id": "g1", "family": "kerr", "source": "berti", "M_solar": 10.0, "chi": 0.5}
    _write_compat(sub220, [geom])
    _write_compat(sub221, [geom])

    # Must not raise – batch gating uses stage_summary, not RUN_VALID
    result = run_experiment(
        run_id="run_host",
        batch_220="batch220_gate",
        batch_221="batch221_gate",
    )
    assert result["n_common_events"] == 1


# ---------------------------------------------------------------------------
# Test 8 – batch gating: stage_summary.json verdict != PASS → abort
# ---------------------------------------------------------------------------


def test_batch_gate_fails_when_verdict_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch with stage_summary verdict != PASS must abort with an explicit error."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    _mk_run(tmp_path, "run_host_fail")
    batch220_dir = tmp_path / "batch220_fail_verdict"
    ss_dir = batch220_dir / "experiment" / "offline_batch"
    ss_dir.mkdir(parents=True, exist_ok=True)
    (ss_dir / "stage_summary.json").write_text(
        json.dumps({"verdict": "FAIL", "status": "FAIL"}), encoding="utf-8"
    )
    # results.csv present but irrelevant – verdict check fires first
    _write_csv(
        batch220_dir / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub_x", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    batch221 = _mk_batch(tmp_path, "batch221_fail_verdict_ok")
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub_y", "status": "PASS"}],
        fields=["event_id", "subrun_id", "status"],
    )

    with pytest.raises(ValueError, match="FAIL"):
        run_experiment(
            run_id="run_host_fail",
            batch_220="batch220_fail_verdict",
            batch_221="batch221_fail_verdict_ok",
        )
