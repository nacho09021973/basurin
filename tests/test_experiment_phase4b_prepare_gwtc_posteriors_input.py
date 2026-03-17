"""Tests for mvp/experiment_phase4b_prepare_gwtc_posteriors_input.py.

Coverage:
 1. test_prepare_requires_host_run_valid_pass
 2. test_prepare_requires_phase4_upstream_present_and_pass
 3. test_prepare_extracts_required_event_ids_from_phase4_upstream
 4. test_write_placeholders_creates_missing_json_files_without_overwriting_existing
 5. test_validate_only_fails_if_coverage_incomplete
 6. test_validate_only_fails_if_schema_missing_required_fields
 7. test_validate_only_accepts_placeholder_schema_without_numeric_requirement
 8. test_validate_only_fails_with_require_numeric_samples_when_values_are_tofill
 9. test_validate_only_passes_with_numeric_samples
10. test_prepare_writes_only_under_runs_host_run_experiment_phase4b_prepare_gwtc_posteriors_input_and_external_inputs_gwtc_posteriors
11. test_prepare_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest
12. test_stage_summary_explains_chain_phase3_phase4a_plus_external_imr_to_phase4b
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from mvp.experiment_phase4b_prepare_gwtc_posteriors_input import (
    EXPERIMENT_NAME,
    PHASE4_UPSTREAM_NAME,
    _inventory_posteriors,
    _placeholder_payload,
    _read_required_event_ids_from_phase4,
    _validate_numeric_samples,
    _validate_posterior_schema,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_RUN_ID = "testrun_prepare"

_HAWKING_FIELDS = [
    "event_id", "family", "provenance", "M_solar", "chi", "A", "S", "hawking_pass",
]


def _mk_run_valid(tmp_path: Path, run_id: str = _RUN_ID) -> Path:
    run_dir = tmp_path / run_id
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    return run_dir


def _mk_phase4_upstream(
    run_dir: Path,
    hawking_rows: list[dict[str, Any]],
    verdict: str = "PASS",
) -> Path:
    """Create all Phase4 upstream artifacts under run_dir/experiment/phase4_*."""
    phase4_dir = run_dir / "experiment" / PHASE4_UPSTREAM_NAME
    outputs_dir = phase4_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ss = {
        "experiment_name": PHASE4_UPSTREAM_NAME,
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

    csv_path = outputs_dir / "per_event_hawking_area.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HAWKING_FIELDS)
        writer.writeheader()
        writer.writerows(hawking_rows)

    return phase4_dir


def _mk_posterior(run_dir: Path, event_id: str, samples: list[dict]) -> Path:
    """Write a single posterior JSON under external_inputs/gwtc_posteriors/."""
    p_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    p_dir.mkdir(parents=True, exist_ok=True)
    path = p_dir / f"{event_id}.json"
    path.write_text(
        json.dumps({"event_id": event_id, "samples": samples}), encoding="utf-8"
    )
    return path


def _numeric_sample(
    m1: float = 30.0,
    m2: float = 25.0,
    chi1: float = 0.3,
    chi2: float = 0.2,
) -> dict:
    return {"m1_source": m1, "m2_source": m2, "chi1": chi1, "chi2": chi2}


def _hawking_row(event_id: str, M: float = 60.0, chi: float = 0.5) -> dict:
    import math
    A = 8 * math.pi * M**2 * (1 + math.sqrt(1 - chi**2))
    return {
        "event_id": event_id, "family": "kerr", "provenance": "nr",
        "M_solar": M, "chi": chi, "A": A, "S": A / 4, "hawking_pass": True,
    }


# ---------------------------------------------------------------------------
# Test 1: requires host run valid PASS
# ---------------------------------------------------------------------------


def test_prepare_requires_host_run_valid_pass(tmp_path, monkeypatch):
    """Abort if RUN_VALID verdict.json is absent or not PASS."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))

    # Case A: verdict.json missing entirely
    run_dir = tmp_path / _RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    # no RUN_VALID dir
    with pytest.raises(FileNotFoundError, match="RUN_VALID"):
        run_experiment(_RUN_ID, mode="write_placeholders")

    # Case B: verdict is FAIL
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "FAIL"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FAIL"):
        run_experiment(_RUN_ID, mode="write_placeholders")


# ---------------------------------------------------------------------------
# Test 2: requires Phase4 upstream present and PASS
# ---------------------------------------------------------------------------


def test_prepare_requires_phase4_upstream_present_and_pass(tmp_path, monkeypatch):
    """Abort if Phase4 upstream dir is missing or verdict != PASS."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)

    # Case A: no phase4 upstream at all
    with pytest.raises(FileNotFoundError, match=PHASE4_UPSTREAM_NAME):
        run_experiment(_RUN_ID, mode="write_placeholders")

    # Case B: phase4 upstream present but verdict is FAIL
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")], verdict="FAIL")
    with pytest.raises(RuntimeError, match="not PASS"):
        run_experiment(_RUN_ID, mode="write_placeholders")


# ---------------------------------------------------------------------------
# Test 3: extracts required event_ids exclusively from Phase4 upstream CSV
# ---------------------------------------------------------------------------


def test_prepare_extracts_required_event_ids_from_phase4_upstream(
    tmp_path, monkeypatch
):
    """Required event_ids come only from per_event_hawking_area.csv, sorted."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    events = ["GW200225", "GW151012", "GW170814"]
    rows = [_hawking_row(e) for e in events]
    _mk_phase4_upstream(run_dir, rows)

    # Invoke write_placeholders so it runs through
    result = run_experiment(_RUN_ID, mode="write_placeholders")

    inv = result["inventory"]
    assert inv["required_event_ids"] == sorted(events)
    assert inv["n_required"] == 3


# ---------------------------------------------------------------------------
# Test 4: --write-placeholders creates missing, does NOT overwrite existing
# ---------------------------------------------------------------------------


def test_write_placeholders_creates_missing_json_files_without_overwriting_existing(
    tmp_path, monkeypatch
):
    """Write placeholders for missing events; existing files are never overwritten."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    events = ["GW150914", "GW151012", "GW170814"]
    _mk_phase4_upstream(run_dir, [_hawking_row(e) for e in events])

    # Pre-populate GW150914 with real numeric data
    existing_sample = [_numeric_sample()]
    _mk_posterior(run_dir, "GW150914", existing_sample)
    original_content = (
        run_dir / "external_inputs" / "gwtc_posteriors" / "GW150914.json"
    ).read_text(encoding="utf-8")

    result = run_experiment(_RUN_ID, mode="write_placeholders")

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"

    # All required files now exist
    for eid in events:
        assert (posteriors_dir / f"{eid}.json").exists()

    # Existing file was NOT overwritten
    current_content = (posteriors_dir / "GW150914.json").read_text(encoding="utf-8")
    assert current_content == original_content

    # Two new files created, one already present
    vs = result["validation_summary"]
    assert vs["files_created"] == 2
    assert vs["files_already_present"] == 1

    # New placeholders have schema structure with TO_FILL
    for eid in ("GW151012", "GW170814"):
        data = json.loads((posteriors_dir / f"{eid}.json").read_text(encoding="utf-8"))
        assert data["event_id"] == eid
        assert data["samples"][0]["m1_source"] == "TO_FILL"


# ---------------------------------------------------------------------------
# Test 5: --validate-only aborts if coverage incomplete
# ---------------------------------------------------------------------------


def test_validate_only_fails_if_coverage_incomplete(tmp_path, monkeypatch):
    """Abort with RuntimeError if any required posterior file is missing."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    events = ["GW150914", "GW170814"]
    _mk_phase4_upstream(run_dir, [_hawking_row(e) for e in events])

    # Only provide one of the two required files
    _mk_posterior(run_dir, "GW150914", [_numeric_sample()])

    with pytest.raises(RuntimeError, match="Coverage incomplete"):
        run_experiment(_RUN_ID, mode="validate_only")


# ---------------------------------------------------------------------------
# Test 6: --validate-only aborts if schema missing required fields
# ---------------------------------------------------------------------------


def test_validate_only_fails_if_schema_missing_required_fields(
    tmp_path, monkeypatch
):
    """Abort with RuntimeError if posterior JSON is missing required fields."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")])

    # Write a posterior missing chi2
    bad_sample = {"m1_source": 30.0, "m2_source": 25.0, "chi1": 0.3}  # no chi2
    _mk_posterior(run_dir, "GW150914", [bad_sample])

    with pytest.raises(RuntimeError, match="Schema invalid"):
        run_experiment(_RUN_ID, mode="validate_only")


# ---------------------------------------------------------------------------
# Test 7: --validate-only accepts placeholder "TO_FILL" without --require-numeric
# ---------------------------------------------------------------------------


def test_validate_only_accepts_placeholder_schema_without_numeric_requirement(
    tmp_path, monkeypatch
):
    """Placeholder schema (TO_FILL values) passes validate_only without --require-numeric."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")])

    placeholder = _placeholder_payload("GW150914")
    p_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    p_dir.mkdir(parents=True, exist_ok=True)
    (p_dir / "GW150914.json").write_text(
        json.dumps(placeholder), encoding="utf-8"
    )

    result = run_experiment(_RUN_ID, mode="validate_only", require_numeric_samples=False)

    vs = result["validation_summary"]
    assert vs["coverage_complete"] is True
    assert vs["schema_valid"] is True
    assert vs["files_invalid_schema"] == []
    assert result["stage_summary"]["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Test 8: --validate-only + --require-numeric-samples fails on TO_FILL
# ---------------------------------------------------------------------------


def test_validate_only_fails_with_require_numeric_samples_when_values_are_tofill(
    tmp_path, monkeypatch
):
    """Abort with RuntimeError when --require-numeric-samples and values are TO_FILL."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")])

    placeholder = _placeholder_payload("GW150914")
    p_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    p_dir.mkdir(parents=True, exist_ok=True)
    (p_dir / "GW150914.json").write_text(
        json.dumps(placeholder), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Non-numeric"):
        run_experiment(_RUN_ID, mode="validate_only", require_numeric_samples=True)


# ---------------------------------------------------------------------------
# Test 9: --validate-only passes with real numeric samples
# ---------------------------------------------------------------------------


def test_validate_only_passes_with_numeric_samples(tmp_path, monkeypatch):
    """validate_only with --require-numeric-samples passes when all values are numeric."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    events = ["GW150914", "GW170814"]
    _mk_phase4_upstream(run_dir, [_hawking_row(e) for e in events])

    for eid in events:
        _mk_posterior(run_dir, eid, [_numeric_sample()])

    result = run_experiment(
        _RUN_ID, mode="validate_only", require_numeric_samples=True
    )

    vs = result["validation_summary"]
    assert vs["coverage_complete"] is True
    assert vs["schema_valid"] is True
    assert vs["numeric_samples_valid"] is True
    assert vs["files_invalid_numeric"] == []
    assert result["stage_summary"]["verdict"] == "PASS"
    assert result["stage_summary"]["validation"]["numeric_samples_valid"] is True


# ---------------------------------------------------------------------------
# Test 10: writes only under correct paths
# ---------------------------------------------------------------------------


def test_prepare_writes_only_under_runs_host_run_experiment_phase4b_prepare_gwtc_posteriors_input_and_external_inputs_gwtc_posteriors(
    tmp_path, monkeypatch
):
    """All writes stay within runs/<host_run>/experiment/<name> and external_inputs."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")])

    run_experiment(_RUN_ID, mode="write_placeholders")

    exp_dir = run_dir / "experiment" / EXPERIMENT_NAME
    assert exp_dir.is_dir()
    assert (exp_dir / "stage_summary.json").exists()
    assert (exp_dir / "manifest.json").exists()
    assert (exp_dir / "outputs" / "required_event_ids.txt").exists()
    assert (exp_dir / "outputs" / "gwtc_posteriors_inventory.json").exists()
    assert (exp_dir / "outputs" / "gwtc_posteriors_validation_summary.json").exists()

    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"
    assert posteriors_dir.is_dir()
    assert (posteriors_dir / "GW150914.json").exists()

    # Nothing written outside run_dir
    assert not (tmp_path / "external_inputs").exists()
    assert not (tmp_path / "experiment").exists()


# ---------------------------------------------------------------------------
# Test 11: logs OUT_ROOT, STAGE_DIR, OUTPUTS_DIR, STAGE_SUMMARY, MANIFEST
# ---------------------------------------------------------------------------


def test_prepare_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest(
    tmp_path, monkeypatch, capsys
):
    """Entrypoint must print canonical OUT_ROOT/STAGE_DIR/OUTPUTS_DIR/STAGE_SUMMARY/MANIFEST."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")])

    run_experiment(_RUN_ID, mode="write_placeholders")

    out = capsys.readouterr().out
    assert "OUT_ROOT=" in out
    assert "STAGE_DIR=" in out
    assert "OUTPUTS_DIR=" in out
    assert "STAGE_SUMMARY=" in out
    assert "MANIFEST=" in out

    # Verify paths are under the expected directories
    exp_dir = run_dir / "experiment" / EXPERIMENT_NAME
    assert str(exp_dir) in out
    assert str(exp_dir / "outputs") in out
    assert str(exp_dir / "stage_summary.json") in out
    assert str(exp_dir / "manifest.json") in out


# ---------------------------------------------------------------------------
# Test 12: stage_summary explains the Phase3 → Phase4A + external IMR → Phase4B chain
# ---------------------------------------------------------------------------


def test_stage_summary_explains_chain_phase3_phase4a_plus_external_imr_to_phase4b(
    tmp_path, monkeypatch
):
    """stage_summary notes must explain why external IMR input is needed."""
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_dir = _mk_run_valid(tmp_path)
    _mk_phase4_upstream(run_dir, [_hawking_row("GW150914")])

    result = run_experiment(_RUN_ID, mode="write_placeholders")

    ss = result["stage_summary"]

    # Experiment identity
    assert ss["experiment_name"] == EXPERIMENT_NAME

    # Chain explanation present in notes
    notes_text = " ".join(ss["notes"])
    assert "phase4b_hawking_area_law_filter" in notes_text
    assert "A_initial" in notes_text
    assert "Phase3" in notes_text or "Phase4A" in notes_text or "external" in notes_text

    # Gating block present
    assert ss["gating"]["host_run_valid"] is True
    assert ss["gating"]["phase4_upstream_present"] is True
    assert ss["gating"]["phase4_upstream_pass"] is True

    # External input definition present
    ext = ss["external_input_definition"]
    assert "gwtc_posteriors" in ext["path"]
    assert "m1_source" in ext["required_sample_fields"]
    assert "m2_source" in ext["required_sample_fields"]
    assert "chi1" in ext["required_sample_fields"]
    assert "chi2" in ext["required_sample_fields"]

    # Inventory and validation blocks present
    assert "n_required" in ss["inventory"]
    assert "coverage_complete" in ss["validation"]
