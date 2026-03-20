"""Regression tests for FIX-B: when s1_fetch_strain aborts, RUN_VALID must be FAIL.

Contract: any run that is aborted (any stage exits non-zero) must NOT leave
RUN_VALID/verdict.json with verdict=PASS. The pipeline writes PASS optimistically
at start; each failure path must overwrite it to FAIL before returning.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, call
import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def tmp_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


def _read_verdict(runs: Path, run_id: str) -> str | None:
    p = runs / run_id / "RUN_VALID" / "verdict.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["verdict"]


# ---------------------------------------------------------------------------
# run_single_event: s1 failure must write FAIL to RUN_VALID
# ---------------------------------------------------------------------------

def test_single_event_s1_abort_sets_run_valid_fail(tmp_runs: Path, monkeypatch):
    """When s1_fetch_strain returns non-zero, RUN_VALID must be FAIL (not PASS)."""
    import mvp.pipeline as pipeline

    monkeypatch.setattr(pipeline, "resolve_out_root", lambda _: tmp_runs)

    # Stub _write_run_provenance and _write_timeline to be no-ops
    monkeypatch.setattr(pipeline, "_write_run_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_write_timeline", lambda *a, **kw: None)

    call_count = {"n": 0}

    def _fake_run_stage(script, args, label, out_root, run_id, timeline, timeout):
        call_count["n"] += 1
        if "s0_oracle" in script:
            return 0  # s0 passes
        if "s1_fetch_strain" in script:
            return 2  # s1 aborts
        return 0

    monkeypatch.setattr(pipeline, "_build_s0_oracle_args", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_build_s1_fetch_args", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_run_stage", _fake_run_stage)
    monkeypatch.setattr(pipeline, "_resolve_adaptive_dt_start", lambda *a, **kw: (0.003, "test"))

    run_id = "test-s1-abort-single"
    rc, returned_run_id = pipeline.run_single_event(
        event_id="GW150914",
        atlas_path="/fake/atlas.json",
        run_id=run_id,
    )

    assert rc != 0, "run_single_event should return non-zero when s1 fails"

    verdict = _read_verdict(tmp_runs, run_id)
    assert verdict == "FAIL", (
        f"RUN_VALID verdict must be FAIL when s1 aborts, got {verdict!r}. "
        "FIX-B regression: optimistic PASS was not overwritten."
    )


# ---------------------------------------------------------------------------
# run_multimode_event: s1 failure must write FAIL to RUN_VALID
# ---------------------------------------------------------------------------

def test_multimode_event_s1_abort_sets_run_valid_fail(tmp_runs: Path, monkeypatch):
    """Same contract for run_multimode_event: s1 abort → RUN_VALID=FAIL."""
    import mvp.pipeline as pipeline

    monkeypatch.setattr(pipeline, "resolve_out_root", lambda _: tmp_runs)
    monkeypatch.setattr(pipeline, "_write_run_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_write_timeline", lambda *a, **kw: None)

    def _fake_run_stage(script, args, label, out_root, run_id, timeline, timeout):
        if "s0_oracle" in script:
            return 0
        if "s1_fetch_strain" in script:
            return 2
        return 0

    monkeypatch.setattr(pipeline, "_build_s0_oracle_args", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_build_s1_fetch_args", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_run_stage", _fake_run_stage)
    monkeypatch.setattr(pipeline, "_resolve_adaptive_dt_start", lambda *a, **kw: (0.003, "test"))

    run_id = "test-s1-abort-multimode"
    rc, returned_run_id = pipeline.run_multimode_event(
        event_id="GW150914",
        atlas_path="/fake/atlas.json",
        run_id=run_id,
    )

    assert rc != 0
    verdict = _read_verdict(tmp_runs, run_id)
    assert verdict == "FAIL", (
        f"RUN_VALID verdict must be FAIL when s1 aborts in multimode, got {verdict!r}. "
        "FIX-B regression."
    )


# ---------------------------------------------------------------------------
# Control: successful run must NOT overwrite to FAIL
# ---------------------------------------------------------------------------

def test_s0_success_does_not_set_fail(tmp_runs: Path, monkeypatch):
    """A run that passes s0 but stops at s1 only sets FAIL on s1 non-zero, not s0."""
    import mvp.pipeline as pipeline

    monkeypatch.setattr(pipeline, "resolve_out_root", lambda _: tmp_runs)
    monkeypatch.setattr(pipeline, "_write_run_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_write_timeline", lambda *a, **kw: None)

    def _fake_run_stage(script, args, label, out_root, run_id, timeline, timeout):
        if "s0_oracle" in script:
            return 2  # s0 fails → FAIL is explicitly set for s0
        return 0

    monkeypatch.setattr(pipeline, "_build_s0_oracle_args", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_build_s1_fetch_args", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_run_stage", _fake_run_stage)
    monkeypatch.setattr(pipeline, "_resolve_adaptive_dt_start", lambda *a, **kw: (0.003, "test"))

    run_id = "test-s0-fail"
    rc, _ = pipeline.run_single_event(
        event_id="GW150914",
        atlas_path="/fake/atlas.json",
        run_id=run_id,
    )

    assert rc != 0
    verdict = _read_verdict(tmp_runs, run_id)
    # s0 failure also sets FAIL (this was already implemented before FIX-B)
    assert verdict == "FAIL"
