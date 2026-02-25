"""Unit tests for mvp/s4c_kerr_consistency.py.

Gaps addressed (from test_coverage_proposal.md Gap 5):
  - _read_json:                valid JSON dict, non-dict JSON raises ValueError
  - Stage output schema:       kerr_consistency.json has required keys
  - kerr_consistent logic:     n_compatible > 0 → True; n_compatible == 0 → False
  - Missing optional inputs:   runs without compatible_set.json (optional path)
  - CLI abort:                 missing required inputs → non-zero exit code

Note: _mode_row() and _infer_censoring() referenced in the proposal do not
exist in the current s4c_kerr_consistency.py implementation; the business
logic is instead encoded directly in main(). These tests cover the
equivalent observable behaviour at the function and subprocess level.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s4c_kerr_consistency import _read_json


# ---------------------------------------------------------------------------
# _read_json
# ---------------------------------------------------------------------------


def test_read_json_valid_dict(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text('{"key": "value", "num": 42}', encoding="utf-8")
    result = _read_json(f)
    assert result == {"key": "value", "num": 42}


def test_read_json_non_dict_raises(tmp_path: Path) -> None:
    f = tmp_path / "list.json"
    f.write_text('[1, 2, 3]', encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        _read_json(f)


def test_read_json_null_raises(tmp_path: Path) -> None:
    f = tmp_path / "null.json"
    f.write_text("null", encoding="utf-8")
    with pytest.raises(ValueError):
        _read_json(f)


def test_read_json_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _read_json(tmp_path / "nonexistent.json")


def test_read_json_malformed_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text("{not json}", encoding="utf-8")
    with pytest.raises(Exception):
        _read_json(f)


# ---------------------------------------------------------------------------
# Helpers for subprocess-level tests
# ---------------------------------------------------------------------------


def _make_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text('{"verdict": "PASS"}', encoding="utf-8")


def _write_estimates(runs_root: Path, run_id: str, payload: dict | None = None) -> Path:
    out = runs_root / run_id / "s3_ringdown_estimates" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "estimates.json"
    if payload is None:
        payload = {
            "schema_version": "mvp_estimates_v2",
            "event_id": "GW150914",
            "method": "spectral_lorentzian",
            "combined": {"f_hz": 250.0, "tau_s": 0.05, "Q": 39.3},
            "combined_uncertainty": {
                "sigma_f_hz": 2.0, "sigma_tau_s": 0.002, "sigma_Q": 1.0,
                "sigma_lnf": 0.008, "sigma_lnQ": 0.025, "r": 0.0,
                "cov_logf_logQ": 0.0, "sigma_logf": 0.008, "sigma_logQ": 0.025,
            },
            "per_detector": {},
            "n_detectors_valid": 1,
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    (runs_root / run_id / "s3_ringdown_estimates" / "stage_summary.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates", "verdict": "PASS"}), encoding="utf-8"
    )
    return path


def _write_multimode(runs_root: Path, run_id: str, payload: dict | None = None) -> Path:
    out = runs_root / run_id / "s3b_multimode_estimates" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "multimode_estimates.json"
    if payload is None:
        payload = {
            "schema_version": "mvp_multimode_v1",
            "run_id": run_id,
            "results": {"verdict": "PASS", "modes": {}},
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    (runs_root / run_id / "s3b_multimode_estimates" / "stage_summary.json").write_text(
        json.dumps({"stage": "s3b_multimode_estimates", "verdict": "PASS"}), encoding="utf-8"
    )
    return path


def _write_compatible_set(
    runs_root: Path, run_id: str, n_compatible: int, d2_min: float | None = None
) -> Path:
    out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "compatible_set.json"
    payload: dict = {"n_compatible": n_compatible}
    if d2_min is not None:
        payload["d2_min"] = d2_min
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_s4c(runs_root: Path, run_id: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(MVP_DIR / "s4c_kerr_consistency.py"),
        "--run-id", run_id,
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _load_output(runs_root: Path, run_id: str) -> dict:
    path = runs_root / run_id / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


def test_s4c_output_has_required_schema_keys(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_schema"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)
    _write_compatible_set(runs_root, run_id, n_compatible=5, d2_min=0.02)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode == 0, f"s4c failed (rc={r.returncode}):\n{r.stderr}"

    out = _load_output(runs_root, run_id)
    for key in ("schema_version", "run_id", "event_id", "kerr_consistent", "d2_min", "chi_best", "source"):
        assert key in out, f"Missing key in kerr_consistency.json: {key!r}"


def test_s4c_schema_version_correct(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_ver"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode == 0, f"s4c failed:\n{r.stderr}"

    out = _load_output(runs_root, run_id)
    assert out["schema_version"] == "mvp_kerr_consistency_v1"


# ---------------------------------------------------------------------------
# kerr_consistent logic
# ---------------------------------------------------------------------------


def test_s4c_kerr_consistent_true_when_n_compatible_positive(tmp_path: Path) -> None:
    """n_compatible > 0 → kerr_consistent = True."""
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_consistent_true"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)
    _write_compatible_set(runs_root, run_id, n_compatible=10, d2_min=0.01)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode == 0, f"s4c failed:\n{r.stderr}"

    out = _load_output(runs_root, run_id)
    assert out["kerr_consistent"] is True


def test_s4c_kerr_consistent_false_when_n_compatible_zero(tmp_path: Path) -> None:
    """n_compatible == 0 → kerr_consistent = False."""
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_consistent_false"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)
    _write_compatible_set(runs_root, run_id, n_compatible=0, d2_min=None)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode == 0, f"s4c failed:\n{r.stderr}"

    out = _load_output(runs_root, run_id)
    assert out["kerr_consistent"] is False


def test_s4c_d2_min_propagated_from_compatible_set(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_d2min"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)
    _write_compatible_set(runs_root, run_id, n_compatible=3, d2_min=0.0512)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode == 0, f"s4c failed:\n{r.stderr}"

    out = _load_output(runs_root, run_id)
    assert abs(out["d2_min"] - 0.0512) < 1e-9


# ---------------------------------------------------------------------------
# Missing optional inputs (compatible_set is optional)
# ---------------------------------------------------------------------------


def test_s4c_succeeds_without_compatible_set(tmp_path: Path) -> None:
    """compatible_set.json is optional — stage must succeed without it."""
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_no_compat"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)
    # Deliberately do NOT write compatible_set.json

    r = _run_s4c(runs_root, run_id)
    assert r.returncode == 0, f"s4c failed without compatible_set.json:\n{r.stderr}"

    out = _load_output(runs_root, run_id)
    # Without compatible set: n_compatible is None → kerr_consistent = False
    assert out["kerr_consistent"] is False
    assert out["d2_min"] is None


# ---------------------------------------------------------------------------
# Missing required inputs → non-zero exit
# ---------------------------------------------------------------------------


def test_s4c_aborts_if_s3_estimates_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_no_s3"
    _make_run_valid(runs_root, run_id)
    # Write multimode but NOT estimates
    _write_multimode(runs_root, run_id)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode != 0, "Expected non-zero exit when s3 estimates are missing"


def test_s4c_aborts_if_multimode_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_no_s3b"
    _make_run_valid(runs_root, run_id)
    # Write estimates but NOT multimode
    _write_estimates(runs_root, run_id)

    r = _run_s4c(runs_root, run_id)
    assert r.returncode != 0, "Expected non-zero exit when multimode estimates are missing"


def test_s4c_no_python_traceback_on_missing_inputs(tmp_path: Path) -> None:
    """Abort paths must not produce unhandled Python tracebacks."""
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_no_crash"
    _make_run_valid(runs_root, run_id)
    # No inputs at all

    r = _run_s4c(runs_root, run_id)
    assert "Traceback (most recent call last)" not in r.stderr, (
        f"Unhandled Python traceback:\n{r.stderr}"
    )


# ---------------------------------------------------------------------------
# Source metadata in output
# ---------------------------------------------------------------------------


def test_s4c_source_records_compatible_set_present(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_source_present"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)
    _write_compatible_set(runs_root, run_id, n_compatible=1)

    _run_s4c(runs_root, run_id)
    out = _load_output(runs_root, run_id)
    assert out["source"]["compatible_set_present"] is True


def test_s4c_source_records_compatible_set_absent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_source_absent"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)
    _write_multimode(runs_root, run_id)

    _run_s4c(runs_root, run_id)
    out = _load_output(runs_root, run_id)
    assert out["source"]["compatible_set_present"] is False


def test_s4c_event_id_propagated_from_estimates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s4c_eventid"
    _make_run_valid(runs_root, run_id)
    _write_estimates(runs_root, run_id)  # default payload has event_id="GW150914"
    _write_multimode(runs_root, run_id)

    _run_s4c(runs_root, run_id)
    out = _load_output(runs_root, run_id)
    assert out["event_id"] == "GW150914"
