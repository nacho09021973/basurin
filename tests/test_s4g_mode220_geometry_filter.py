"""Tests for s4g_mode220_geometry_filter.

Tests:
    1. filter_mode220 returns empty lists when no atlas entry has mode-220 predictions.
    2. filter_mode220 accepts an entry with chi2 < threshold, rejects chi2 >= threshold.
    3. filter_mode220 returns sorted accepted_geometry_ids and parallel accepted_geometries.
    4. filter_mode220 supports unified atlas format (mode_220 sub-dict).
    5. filter_mode220 supports existing atlas format (metadata.mode == "(2,2,0)").
    6. load_atlas_entries accepts a JSON list.
    7. load_atlas_entries accepts {"entries": [...]} dict.
    8. load_atlas_entries accepts {"atlas": [...]} dict.
    9. load_atlas_entries raises ValueError for unknown dict keys.
    10. Contract: CONTRACTS has s4g_mode220_geometry_filter with correct produced output.
    11. Contract: required_inputs includes mode220_obs.json path.
    12. Contract: external_inputs includes "atlas".
    13. CLI end-to-end: PASS verdict, geometries_220.json written, canonical paths printed.
    14. CLI: missing obs file returns exit code 2.
    15. CLI: missing atlas file returns exit code 2.
    16. Payload schema: all required fields present with correct types.
    17. Payload: n_geometries_accepted == len(accepted_geometry_ids).
    18. Payload: n_geometries_scanned counts only entries with mode-220 predictions.
    19. Payload: mode field is exactly "220".
    20. Oracle gating: n_geometries_accepted == 0 → FAIL (verdict NO_COMMON_GEOMETRIES).
    21. Oracle gating: n_geometries_accepted > 1 → PASS.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s4g_mode220_geometry_filter import (
    extract_mode220_predictions,
    filter_mode220,
    load_atlas_entries,
    STAGE,
    OUTPUT_FILE,
)
from mvp.contracts import CONTRACTS
from mvp.golden_geometry_spec import (
    DEFAULT_MODE_CHI2_THRESHOLD_90,
    MODE_220,
    VERDICT_PASS,
    VERDICT_NO_COMMON_GEOMETRIES,
)


# ── Atlas entry builders ───────────────────────────────────────────────────

def _unified_entry(gid: str, f_hz: float, tau_s: float) -> dict[str, Any]:
    """Build an atlas entry in unified golden format."""
    return {"geometry_id": gid, "mode_220": {"f_hz": f_hz, "tau_s": tau_s}}


def _legacy_entry(gid: str, f_hz: float, tau_s: float) -> dict[str, Any]:
    """Build an atlas entry in existing (metadata.mode) format."""
    return {
        "geometry_id": gid,
        "metadata": {"mode": "(2,2,0)"},
        "f_hz": f_hz,
        "tau_s": tau_s,
    }


def _no_mode_entry(gid: str) -> dict[str, Any]:
    """Build an atlas entry with no mode-220 prediction."""
    return {"geometry_id": gid, "mode_221": {"f_hz": 100.0, "tau_s": 0.01}}


# ── filter_mode220 unit tests ─────────────────────────────────────────────

def test_filter_empty_when_no_mode220_entries():
    ids, entries = filter_mode220(
        obs_f_hz=250.0,
        obs_tau_s=0.004,
        sigma_f_hz=10.0,
        sigma_tau_s=0.001,
        atlas_entries=[_no_mode_entry("g1"), _no_mode_entry("g2")],
        chi2_threshold=DEFAULT_MODE_CHI2_THRESHOLD_90,
    )
    assert ids == []
    assert entries == []


def test_filter_accepts_below_threshold():
    # Entry at exactly the observed point → chi2 == 0 → passes.
    entry = _unified_entry("g1", f_hz=250.0, tau_s=0.004)
    ids, entries = filter_mode220(
        obs_f_hz=250.0,
        obs_tau_s=0.004,
        sigma_f_hz=10.0,
        sigma_tau_s=0.001,
        atlas_entries=[entry],
        chi2_threshold=DEFAULT_MODE_CHI2_THRESHOLD_90,
    )
    assert ids == ["g1"]
    assert entries == [entry]


def test_filter_rejects_at_threshold():
    # chi2 == threshold is rejected (strict <).
    import math
    threshold = DEFAULT_MODE_CHI2_THRESHOLD_90
    # chi2 = (df/sigma_f)^2; set df = sqrt(threshold)*sigma_f for chi2 exactly threshold.
    sigma_f = 10.0
    sigma_tau = 0.001
    df = math.sqrt(threshold) * sigma_f
    entry = _unified_entry("g1", f_hz=250.0 + df, tau_s=0.004)
    ids, _ = filter_mode220(
        obs_f_hz=250.0,
        obs_tau_s=0.004,
        sigma_f_hz=sigma_f,
        sigma_tau_s=sigma_tau,
        atlas_entries=[entry],
        chi2_threshold=threshold,
    )
    assert ids == []


def test_filter_returns_sorted_ids():
    entries = [
        _unified_entry("g3", f_hz=250.0, tau_s=0.004),
        _unified_entry("g1", f_hz=250.0, tau_s=0.004),
        _unified_entry("g2", f_hz=250.0, tau_s=0.004),
    ]
    ids, accepted = filter_mode220(
        obs_f_hz=250.0,
        obs_tau_s=0.004,
        sigma_f_hz=10.0,
        sigma_tau_s=0.001,
        atlas_entries=entries,
        chi2_threshold=DEFAULT_MODE_CHI2_THRESHOLD_90,
    )
    assert ids == ["g1", "g2", "g3"]
    assert [e["geometry_id"] for e in accepted] == ["g1", "g2", "g3"]


def test_filter_unified_format():
    entry = _unified_entry("kerr_01", f_hz=260.0, tau_s=0.005)
    ids, _ = filter_mode220(
        obs_f_hz=260.0,
        obs_tau_s=0.005,
        sigma_f_hz=20.0,
        sigma_tau_s=0.002,
        atlas_entries=[entry],
        chi2_threshold=DEFAULT_MODE_CHI2_THRESHOLD_90,
    )
    assert "kerr_01" in ids


def test_filter_legacy_format():
    entry = _legacy_entry("legacy_01", f_hz=260.0, tau_s=0.005)
    ids, _ = filter_mode220(
        obs_f_hz=260.0,
        obs_tau_s=0.005,
        sigma_f_hz=20.0,
        sigma_tau_s=0.002,
        atlas_entries=[entry],
        chi2_threshold=DEFAULT_MODE_CHI2_THRESHOLD_90,
    )
    assert "legacy_01" in ids


# ── load_atlas_entries tests ───────────────────────────────────────────────

def test_load_atlas_list(tmp_path):
    data = [{"geometry_id": "g1"}, {"geometry_id": "g2"}]
    p = tmp_path / "atlas.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = load_atlas_entries(p)
    assert result == data


def test_load_atlas_entries_key(tmp_path):
    data = {"entries": [{"geometry_id": "g1"}]}
    p = tmp_path / "atlas.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert load_atlas_entries(p) == data["entries"]


def test_load_atlas_atlas_key(tmp_path):
    data = {"atlas": [{"geometry_id": "g1"}]}
    p = tmp_path / "atlas.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert load_atlas_entries(p) == data["atlas"]


def test_load_atlas_unknown_key_raises(tmp_path):
    data = {"unknown_key": [{"geometry_id": "g1"}]}
    p = tmp_path / "atlas.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_atlas_entries(p)


# ── Contract tests ─────────────────────────────────────────────────────────

def test_contract_registered():
    assert STAGE in CONTRACTS


def test_contract_produced_outputs():
    contract = CONTRACTS[STAGE]
    assert "outputs/geometries_220.json" in contract.produced_outputs


def test_contract_required_inputs():
    contract = CONTRACTS[STAGE]
    assert any("mode220_obs.json" in inp for inp in contract.required_inputs)


def test_contract_external_inputs():
    contract = CONTRACTS[STAGE]
    assert "atlas" in contract.external_inputs


# ── CLI end-to-end tests ───────────────────────────────────────────────────

def _make_run(tmp_path: Path, obs: dict, atlas: list, run_id: str = "test-s4g-run") -> tuple[Path, Path, str]:
    """Create a minimal run directory structure and return (runs_root, atlas_path, run_id)."""
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    # RUN_VALID marker
    run_valid_dir = run_dir / "RUN_VALID"
    run_valid_dir.mkdir(parents=True)
    verdict_file = run_valid_dir / "verdict.json"
    verdict_file.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    # Obs file
    obs_dir = run_dir / "s4g_mode220_geometry_filter" / "inputs"
    obs_dir.mkdir(parents=True)
    (obs_dir / "mode220_obs.json").write_text(json.dumps(obs), encoding="utf-8")
    # Atlas
    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_text(json.dumps(atlas), encoding="utf-8")
    return runs_root, atlas_path, run_id


def _run_cli(runs_root: Path, atlas_path: Path, run_id: str, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    import os
    cmd = [
        sys.executable, "-m", "mvp.s4g_mode220_geometry_filter",
        "--run-id", run_id,
        "--atlas-path", str(atlas_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)},
        cwd=str(REPO_ROOT),
    )


def _make_obs(f_hz: float = 250.0, tau_s: float = 0.004) -> dict:
    return {"obs_f_hz": f_hz, "obs_tau_s": tau_s, "sigma_f_hz": 15.0, "sigma_tau_s": 0.001}


def test_cli_pass_verdict(tmp_path):
    obs = _make_obs()
    atlas = [
        _unified_entry("g1", f_hz=250.0, tau_s=0.004),
        _unified_entry("g2", f_hz=252.0, tau_s=0.004),
    ]
    runs_root, atlas_path, run_id = _make_run(tmp_path, obs, atlas)
    result = _run_cli(runs_root, atlas_path, run_id)
    assert result.returncode == 0, result.stderr
    out_file = runs_root / run_id / STAGE / "outputs" / OUTPUT_FILE
    assert out_file.exists(), f"Expected {out_file} to exist"
    payload = json.loads(out_file.read_text())
    assert payload["verdict"] == VERDICT_PASS
    stdout = result.stdout
    assert "OUT_ROOT=" in stdout
    assert "STAGE_DIR=" in stdout
    assert "OUTPUTS_DIR=" in stdout
    assert "STAGE_SUMMARY=" in stdout
    assert "MANIFEST=" in stdout


def test_cli_missing_obs_file(tmp_path):
    run_id = "test-s4g-no-obs"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_text(json.dumps([_unified_entry("g1", 250.0, 0.004)]), encoding="utf-8")
    result = _run_cli(runs_root, atlas_path, run_id)
    assert result.returncode == 2


def test_cli_missing_atlas(tmp_path):
    obs = _make_obs()
    runs_root, _, run_id = _make_run(tmp_path, obs, [])
    missing_atlas = tmp_path / "nonexistent_atlas.json"
    result = _run_cli(runs_root, missing_atlas, run_id)
    assert result.returncode == 2


# ── Payload schema tests ───────────────────────────────────────────────────

def test_payload_required_fields(tmp_path):
    obs = _make_obs()
    atlas = [_unified_entry("g1", 250.0, 0.004)]
    runs_root, atlas_path, run_id = _make_run(tmp_path, obs, atlas)
    result = _run_cli(runs_root, atlas_path, run_id)
    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (runs_root / run_id / STAGE / "outputs" / OUTPUT_FILE).read_text()
    )
    required = [
        "schema_name", "schema_version", "run_id", "stage", "mode",
        "n_geometries_scanned", "n_geometries_accepted",
        "accepted_geometry_ids", "accepted_geometries", "verdict",
    ]
    for field in required:
        assert field in payload, f"Missing field: {field}"


def test_payload_mode_is_220(tmp_path):
    obs = _make_obs()
    atlas = [_unified_entry("g1", 250.0, 0.004)]
    runs_root, atlas_path, run_id = _make_run(tmp_path, obs, atlas)
    _run_cli(runs_root, atlas_path, run_id)
    payload = json.loads(
        (runs_root / run_id / STAGE / "outputs" / OUTPUT_FILE).read_text()
    )
    assert payload["mode"] == MODE_220


def test_payload_counts_consistent(tmp_path):
    obs = _make_obs()
    atlas = [
        _unified_entry("g1", 250.0, 0.004),
        _unified_entry("g2", 252.0, 0.004),
        _no_mode_entry("g3"),  # no mode-220 prediction → not scanned
    ]
    runs_root, atlas_path, run_id = _make_run(tmp_path, obs, atlas)
    _run_cli(runs_root, atlas_path, run_id)
    payload = json.loads(
        (runs_root / run_id / STAGE / "outputs" / OUTPUT_FILE).read_text()
    )
    assert payload["n_geometries_accepted"] == len(payload["accepted_geometry_ids"])
    assert payload["n_geometries_scanned"] == 2  # only g1, g2 have mode-220


# ── Oracle gating tests ────────────────────────────────────────────────────

def test_oracle_gating_fail_zero_accepted(tmp_path):
    """n_geometries_accepted == 0 → verdict FAIL (NO_COMMON_GEOMETRIES)."""
    obs = _make_obs(f_hz=250.0, tau_s=0.004)
    # Atlas entry far from observation → chi2 >> threshold
    atlas = [_unified_entry("g1", f_hz=9999.0, tau_s=9.999)]
    runs_root, atlas_path, run_id = _make_run(tmp_path, obs, atlas)
    _run_cli(runs_root, atlas_path, run_id)
    payload = json.loads(
        (runs_root / run_id / STAGE / "outputs" / OUTPUT_FILE).read_text()
    )
    assert payload["n_geometries_accepted"] == 0
    assert payload["verdict"] == VERDICT_NO_COMMON_GEOMETRIES


def test_oracle_gating_pass_multiple_accepted(tmp_path):
    """n_geometries_accepted > 1 → verdict PASS."""
    obs = _make_obs()
    atlas = [
        _unified_entry("g1", 250.0, 0.004),
        _unified_entry("g2", 251.0, 0.004),
        _unified_entry("g3", 252.0, 0.004),
    ]
    runs_root, atlas_path, run_id = _make_run(tmp_path, obs, atlas, run_id="test-s4g-multi")
    _run_cli(runs_root, atlas_path, run_id)
    payload = json.loads(
        (runs_root / run_id / STAGE / "outputs" / OUTPUT_FILE).read_text()
    )
    assert payload["n_geometries_accepted"] > 1
    assert payload["verdict"] == VERDICT_PASS
