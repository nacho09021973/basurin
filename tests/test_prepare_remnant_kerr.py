"""Tests for mvp/tools/prepare_remnant_kerr.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from basurin_io import write_json_atomic
from mvp.tools.prepare_remnant_kerr import af_nr_fit, resolve_remnant, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    return runs_root


def _write_provenance(runs_root: Path, run_id: str, event_id: str) -> None:
    write_json_atomic(
        runs_root / run_id / "run_provenance.json",
        {
            "schema_version": "run_provenance_v1",
            "run_id": run_id,
            "invocation": {"event_id": event_id},
        },
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Unit tests for af_nr_fit
# ---------------------------------------------------------------------------

def test_af_nr_fit_equal_mass_nonspinning():
    """Equal-mass non-spinning: known NR result ~0.686."""
    af = af_nr_fit(30.0, 30.0, chi_eff=0.0)
    assert 0.65 < af < 0.72


def test_af_nr_fit_gw150914_approx():
    """GW150914 progenitors → close to curated 0.67."""
    af = af_nr_fit(36.0, 29.0, chi_eff=0.0)
    assert abs(af - 0.67) < 0.03


def test_af_nr_fit_gw190521_approx():
    """GW190521 progenitors → close to curated 0.72."""
    af = af_nr_fit(85.0, 66.0, chi_eff=0.1)
    assert abs(af - 0.72) < 0.03


def test_af_nr_fit_output_range():
    """Output always in valid Kerr range [0, 0.998]."""
    for m1, m2, chi_eff in [(10, 10, 0), (100, 1, 0.9), (50, 50, -0.9), (1, 100, -0.5)]:
        af = af_nr_fit(m1, m2, chi_eff)
        assert 0.0 <= af <= 0.998


# ---------------------------------------------------------------------------
# Unit tests for resolve_remnant
# ---------------------------------------------------------------------------

def test_resolve_remnant_curated_event():
    """GW150914 has curated af → method=curated."""
    rem = resolve_remnant("GW150914")
    assert rem is not None
    assert rem["af_method"] == "curated"
    assert 0.6 < rem["af"] < 0.75
    assert rem["Mf"] > 50.0


def test_resolve_remnant_gw190521_curated():
    """GW190521 is in legacy catalog → curated."""
    rem = resolve_remnant("GW190521")
    assert rem is not None
    assert rem["af_method"] == "curated"
    assert abs(rem["af"] - 0.72) < 0.01


def test_resolve_remnant_unknown_event_returns_none():
    """Unknown event ID returns None without raising."""
    assert resolve_remnant("GW999999_999999") is None


# ---------------------------------------------------------------------------
# Integration tests for run()
# ---------------------------------------------------------------------------

def test_run_writes_remnant_kerr_json(tmp_path, monkeypatch):
    """run() creates remnant_kerr.json with valid Mf/af for a curated event."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_GW150914_001"
    _write_provenance(runs_root, run_id, "GW150914")

    result = run(run_id)

    assert result["status"] == "written"
    assert result["event_id"] == "GW150914"
    assert result["Mf"] is not None and result["Mf"] > 0
    assert result["af"] is not None and 0.0 <= result["af"] <= 0.998

    out_path = runs_root / run_id / "external_inputs" / "remnant_kerr.json"
    assert out_path.exists()

    payload = _read_json(out_path)
    assert payload["Mf"] == result["Mf"]
    assert payload["af"] == result["af"]
    assert payload["event_id"] == "GW150914"
    assert payload["af_method"] == "curated"
    assert "citation" in payload
    assert "created_utc" in payload


def test_run_skips_if_exists(tmp_path, monkeypatch):
    """run() is a no-op if remnant_kerr.json already exists (without --force)."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_GW150914_002"
    _write_provenance(runs_root, run_id, "GW150914")

    run(run_id)
    result = run(run_id)
    assert result["status"] == "skipped_already_exists"


def test_run_force_overwrites(tmp_path, monkeypatch):
    """run(force=True) overwrites an existing remnant_kerr.json."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_GW150914_003"
    _write_provenance(runs_root, run_id, "GW150914")

    out_path = runs_root / run_id / "external_inputs" / "remnant_kerr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('{"Mf": 999, "af": 999}', encoding="utf-8")

    result = run(run_id, force=True)
    assert result["status"] == "written"
    payload = _read_json(out_path)
    assert payload["Mf"] != 999


def test_run_event_id_override(tmp_path, monkeypatch):
    """--event-id overrides what run_provenance says."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_override_001"
    _write_provenance(runs_root, run_id, "GW_WRONG_EVENT")

    result = run(run_id, event_id_override="GW150914")
    assert result["status"] == "written"
    assert result["event_id"] == "GW150914"


def test_run_no_provenance_without_override_raises(tmp_path, monkeypatch):
    """run() raises RuntimeError if no provenance and no --event-id."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_noprov_001"
    (runs_root / run_id).mkdir(parents=True)

    with pytest.raises(RuntimeError, match="event_id"):
        run(run_id)


def test_run_unknown_event_raises(tmp_path, monkeypatch):
    """run() raises RuntimeError for an event not in the catalog."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_unknown_001"
    _write_provenance(runs_root, run_id, "GW999999_999999")

    with pytest.raises(RuntimeError, match="not found"):
        run(run_id)


def test_run_nr_fit_event(tmp_path, monkeypatch):
    """Events not in _LEGACY_CHI_FINAL get af via NR fit, still valid."""
    from mvp.gwtc_events import GWTC_EVENTS

    candidate = next(
        (eid for eid, entry in GWTC_EVENTS.items()
         if entry.get("chi_final") is None
         and entry.get("m_final_msun") is not None
         and entry.get("m1_source") is not None
         and entry.get("m2_source") is not None),
        None,
    )
    if candidate is None:
        pytest.skip("No non-curated event with m1/m2 available")

    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_nrfit_001"
    _write_provenance(runs_root, run_id, candidate)

    result = run(run_id)
    assert result["status"] == "written"
    assert result["af_method"] == "nr_fit_barausse_rezzolla_2009"
    assert 0.0 <= result["af"] <= 0.998
    assert result["Mf"] > 0.0


def test_payload_schema(tmp_path, monkeypatch):
    """Written JSON contains all required schema fields."""
    runs_root = _setup_runs_root(tmp_path, monkeypatch)
    run_id = "test_run_schema_001"
    _write_provenance(runs_root, run_id, "GW150914")
    run(run_id)

    out_path = runs_root / run_id / "external_inputs" / "remnant_kerr.json"
    payload = _read_json(out_path)

    required = {"Mf", "af", "af_method", "event_id", "run_id", "created_utc", "citation", "provenance"}
    assert required.issubset(payload.keys()), f"Missing fields: {required - payload.keys()}"
    assert payload["provenance"]["schema"] == "remnant_kerr_v1"
