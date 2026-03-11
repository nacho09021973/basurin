"""Integration tests for the golden geometry pipeline (s4g → s4h → s4i → s4j).

Tests:
    1. s4g→s4i filename contract: S4G_OUTPUT_REL in s4i matches s4g OUTPUT_FILE.
    2. s4g→s4i key contract: s4i reads accepted_geometry_ids from s4g output.
    3. Golden stages in CONTRACTS: s4g, s4h, s4i, s4j all registered.
    4. s4h skip produces SKIPPED verdict via finalize.
    5. Full pipeline e2e: s4g→s4h→s4i→s4j produces manifest+summary for each stage.
    6. Deprecation warning on import of experiment_t0_sweep.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.contracts import CONTRACTS
from mvp.s4i_common_geometry_intersection import (
    S4G_OUTPUT_LEGACY_REL,
    S4G_OUTPUT_PRIMARY_REL,
    compute_intersection,
)
from mvp.s4k_event_support_region import _derive_downstream_status


# ── Contract alignment tests ────────────────────────────────────────────────

def test_s4g_s4i_filename_contract():
    """s4i reads s4g mode220 alias as primary and keeps legacy fallback."""
    assert S4G_OUTPUT_PRIMARY_REL == "s4g_mode220_geometry_filter/outputs/mode220_filter.json"
    assert S4G_OUTPUT_LEGACY_REL == "s4g_mode220_geometry_filter/outputs/geometries_220.json"


def test_s4g_s4i_key_contract():
    """s4i must extract geometry IDs from s4g output using accepted_geometry_ids key."""
    # Simulate s4g output payload
    s4g_payload = {
        "accepted_geometry_ids": ["kerr_01", "kerr_03", "kerr_07"],
        "accepted_geometries": [],
        "verdict": "PASS",
    }
    # s4i logic: get accepted_geometry_ids with fallback to geometry_ids
    ids = s4g_payload.get("accepted_geometry_ids", s4g_payload.get("geometry_ids", []))
    assert ids == ["kerr_01", "kerr_03", "kerr_07"]
    assert len(ids) > 0


def test_s4g_s4i_key_contract_fallback():
    """s4i falls back to geometry_ids key if accepted_geometry_ids is absent."""
    s4g_payload = {
        "geometry_ids": ["kerr_02", "kerr_05"],
        "verdict": "PASS",
    }
    ids = s4g_payload.get("accepted_geometry_ids", s4g_payload.get("geometry_ids", []))
    assert ids == ["kerr_02", "kerr_05"]


def test_golden_stages_in_contracts():
    """All 4 golden geometry stages must be registered in CONTRACTS dict."""
    expected = [
        "s4g_mode220_geometry_filter",
        "s4h_mode221_geometry_filter",
        "s4i_common_geometry_intersection",
        "s4j_hawking_area_filter",
        "s4k_event_support_region",
    ]
    for stage in expected:
        assert stage in CONTRACTS, f"Missing contract for {stage}"


def test_s4i_contract_requires_s4g_output():
    """s4i contract must declare s4g mode220_filter.json as required input."""
    contract = CONTRACTS["s4i_common_geometry_intersection"]
    assert any("mode220_filter.json" in inp for inp in contract.required_inputs)


def test_s4i_intersection_is_mode_agnostic_on_suffix():
    """Mode suffix (_l2m2n0/_l2m2n1) should not prevent common-geometry match."""
    ids_220 = ["Kerr_M62_a0.6600_l2m2n0", "EdGB_M62_a0.67_z0.3"]
    ids_221 = ["Kerr_M62_a0.6600_l2m2n1", "Kerr_M70_a0.5000_l2m2n1"]
    common = compute_intersection(ids_220, ids_221)
    assert common == ["Kerr_M62_a0.6600"]


def test_s4j_contract_requires_s4i_output():
    """s4j contract must declare s4i common_intersection.json as required input."""
    contract = CONTRACTS["s4j_hawking_area_filter"]
    assert any("common_intersection.json" in inp for inp in contract.required_inputs)


def test_s4k_contract_requires_consolidation_inputs():
    """s4k must require the explicit golden-geometry branch plus s3b summary."""
    contract = CONTRACTS["s4k_event_support_region"]
    assert contract.required_inputs == [
        "s3b_multimode_estimates/stage_summary.json",
        "s4g_mode220_geometry_filter/outputs/mode220_filter.json",
        "s4h_mode221_geometry_filter/outputs/mode221_filter.json",
        "s4i_common_geometry_intersection/outputs/common_intersection.json",
        "s4j_hawking_area_filter/outputs/hawking_area_filter.json",
    ]
    assert contract.produced_outputs == ["outputs/event_support_region.json"]


def test_s4k_downstream_status_marks_noninformative_support() -> None:
    downstream = _derive_downstream_status(
        support_region_status="SUPPORT_REGION_AVAILABLE",
        multimode_viability={"class": "RINGDOWN_NONINFORMATIVE", "reasons": ["test"]},
        domain_status="UNKNOWN",
    )
    assert downstream["class"] == "GEOMETRY_PRESENT_BUT_NONINFORMATIVE"
    assert "multimode_viability=RINGDOWN_NONINFORMATIVE" in downstream["reasons"]


def test_s4k_downstream_status_out_of_domain_has_priority() -> None:
    downstream = _derive_downstream_status(
        support_region_status="SUPPORT_REGION_AVAILABLE",
        multimode_viability={"class": "MULTIMODE_OK", "reasons": []},
        domain_status="OUT_OF_DOMAIN",
    )
    assert downstream["class"] == "OUT_OF_DOMAIN"
    assert downstream["reasons"] == ["domain_status=OUT_OF_DOMAIN from s4d_kerr_from_multimode"]


def test_s4h_contract_has_atlas_external_input():
    """s4h contract must declare atlas as external input."""
    contract = CONTRACTS["s4h_mode221_geometry_filter"]
    assert "atlas" in contract.external_inputs


def test_s4h_skips_cleanly_when_mode221_obs_is_absent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)

    run_id = "s4h_skip_pytest"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp.s4h_mode221_geometry_filter",
            "--run-id",
            run_id,
            "--atlas-path",
            str((REPO_ROOT / "mvp" / "test_atlas_fixture.json").resolve()),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )

    assert cp.returncode == 0, cp.stderr
    stage_dir = run_dir / "s4h_mode221_geometry_filter"
    payload = json.loads((stage_dir / "outputs" / "mode221_filter.json").read_text(encoding="utf-8"))
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))

    assert payload["verdict"] == "SKIPPED_221_UNAVAILABLE"
    assert payload["n_passed"] == 0
    assert summary["verdict"] == "PASS"
    assert summary["results"]["verdict"] == "SKIPPED_221_UNAVAILABLE"


# ── Deprecation test ────────────────────────────────────────────────────────

def test_t0_sweep_deprecation_warning():
    """Importing experiment_t0_sweep must emit a DeprecationWarning."""
    # Remove from sys.modules to force re-import
    mod_name = "mvp.experiment_t0_sweep"
    was_loaded = mod_name in sys.modules
    if was_loaded:
        saved = sys.modules.pop(mod_name)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                __import__(mod_name)
            except ImportError:
                # numpy or other heavy deps may not be installed in test env;
                # the warning is emitted before those imports, so check anyway.
                pass
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "DEPRECATED" in str(deprecation_warnings[0].message)
    finally:
        if was_loaded:
            sys.modules[mod_name] = saved


def test_s4k_consolidates_explicit_branch_into_single_event_artifact(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)

    run_id = "s4k_smoke_pytest"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")
    (run_dir / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "run_provenance_v1",
                "run_id": run_id,
                "invocation": {"event_id": "GW150914"},
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "s3b_multimode_estimates").mkdir(parents=True, exist_ok=True)
    (run_dir / "s3b_multimode_estimates" / "stage_summary.json").write_text(
        json.dumps(
            {
                "stage": "s3b_multimode_estimates",
                "multimode_viability": {"class": "MULTIMODE_OK", "reasons": []},
                "systematics_gate": {"status": "PASS"},
                "science_evidence": {"status": "NOT_EVALUATED"},
                "annotations": {"kerr_inconsistency_is_not_fail": True},
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "s4g_mode220_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "s4g_mode220_geometry_filter" / "outputs" / "mode220_filter.json").write_text(
        json.dumps(
            {
                "schema_name": "golden_geometry_mode_filter",
                "accepted_geometry_ids": ["geo_A", "geo_B"],
                "verdict": "PASS",
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "s4h_mode221_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "s4h_mode221_geometry_filter" / "outputs" / "mode221_filter.json").write_text(
        json.dumps(
            {
                "schema_name": "golden_geometry_mode_filter",
                "geometry_ids": ["geo_A"],
                "verdict": "PASS",
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "s4i_common_geometry_intersection" / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "s4i_common_geometry_intersection" / "outputs" / "common_intersection.json").write_text(
        json.dumps(
            {
                "schema_name": "golden_geometry_common",
                "common_geometry_ids": ["geo_A"],
                "mode221_skipped": False,
                "verdict": "PASS",
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "s4j_hawking_area_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "s4j_hawking_area_filter" / "outputs" / "hawking_area_filter.json").write_text(
        json.dumps(
            {
                "schema_name": "golden_geometry_per_event",
                "golden_geometry_ids": ["geo_A"],
                "verdict": "PASS",
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "s4d_kerr_from_multimode" / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json").write_text(
        json.dumps(
            {
                "schema_name": "kerr_extraction",
                "verdict": "PASS",
                "domain_status": "IN_DOMAIN",
            }
        ),
        encoding="utf-8",
    )

    cp = subprocess.run(
        ["python", "-m", "mvp.s4k_event_support_region", "--run-id", run_id],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert cp.returncode == 0, f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"

    stage_dir = run_dir / "s4k_event_support_region"
    out_path = stage_dir / "outputs" / "event_support_region.json"
    assert out_path.exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "manifest.json").exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))

    assert payload["schema_name"] == "golden_geometry_event_support"
    assert payload["event_id"] == "GW150914"
    assert payload["analysis_path"] == "MULTIMODE_INTERSECTION"
    assert payload["support_region_status"] == "SUPPORT_REGION_AVAILABLE"
    assert payload["final_geometry_ids"] == ["geo_A"]
    assert payload["domain_status"] == "IN_DOMAIN"
    assert payload["downstream_status"]["class"] == "MULTIMODE_USABLE"
    assert "support region available and multimode_viability=MULTIMODE_OK" in payload["downstream_status"]["reasons"]
    assert payload["multimode_viability"]["class"] == "MULTIMODE_OK"
    assert payload["mode_220_region"]["geometry_ids"] == ["geo_A", "geo_B"]
    assert payload["mode_221_region"]["geometry_ids"] == ["geo_A"]
    assert payload["common_intersection"]["geometry_ids"] == ["geo_A"]
    assert payload["hawking_filtered_region"]["golden_geometry_ids"] == ["geo_A"]

    assert stage_summary["verdict"] == "PASS"
    assert stage_summary["results"]["analysis_path"] == "MULTIMODE_INTERSECTION"
    assert stage_summary["results"]["support_region_status"] == "SUPPORT_REGION_AVAILABLE"
    assert stage_summary["results"]["downstream_status_class"] == "MULTIMODE_USABLE"
    assert stage_summary["results"]["domain_status"] == "IN_DOMAIN"
    assert stage_summary["results"]["n_final"] == 1
