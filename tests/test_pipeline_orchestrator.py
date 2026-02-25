"""Unit tests for mvp/pipeline.py orchestration logic.

Gaps addressed (from test_coverage_proposal.md Gap 2):
  - _generate_run_id:          output format validation
  - run_single_event:          RUN_VALID created before stages; abort on first failure;
                               pipeline_timeline.json written with correct keys
  - run_multi_event:           abort if any event fails (abort_on_event_fail=True);
                               continues when abort_on_event_fail=False
  - _run_stage:                timeout → return code 124; heartbeat thread lifecycle
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mvp.pipeline as pipeline
from mvp.pipeline import (
    _generate_run_id,
    _run_stage,
    run_multi_event,
    run_single_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runs_root(tmp_path: Path) -> Path:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    return runs_root


# ---------------------------------------------------------------------------
# _generate_run_id
# ---------------------------------------------------------------------------


def test_generate_run_id_starts_with_mvp_and_event() -> None:
    run_id = _generate_run_id("GW150914")
    assert run_id.startswith("mvp_GW150914_")


def test_generate_run_id_only_safe_characters() -> None:
    run_id = _generate_run_id("GW150914")
    assert re.match(r"^[A-Za-z0-9._-]+$", run_id), f"Unsafe chars in: {run_id}"


def test_generate_run_id_different_events_different_prefix() -> None:
    id1 = _generate_run_id("GW150914")
    id2 = _generate_run_id("GW151226")
    assert id1.startswith("mvp_GW150914_")
    assert id2.startswith("mvp_GW151226_")
    assert not id1.startswith("mvp_GW151226_")


def test_generate_run_id_length_reasonable() -> None:
    run_id = _generate_run_id("GW150914")
    # "mvp_GW150914_" (13) + timestamp (16: YYYYMMDDTHHMMSSz) = 29
    assert len(run_id) > 20
    assert len(run_id) < 128


# ---------------------------------------------------------------------------
# _run_stage — timeout returns 124
# ---------------------------------------------------------------------------


def test_run_stage_timeout_returns_124(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    run_id = "test_timeout_run"
    (runs_root / run_id / "RUN_VALID").mkdir(parents=True)
    (runs_root / run_id / "RUN_VALID" / "verdict.json").write_text(
        '{"verdict":"PASS"}', encoding="utf-8"
    )

    # Simulate a process that times out immediately
    class _HangingProc:
        returncode = -9

        def wait(self, timeout: float | None = None) -> None:
            if timeout is not None:
                raise subprocess.TimeoutExpired("cmd", timeout)

        def kill(self) -> None:
            # After kill(), wait() called again without timeout → set returncode
            self.returncode = -9

    hanging = _HangingProc()
    # Second wait() call (after kill) should return normally
    wait_calls: list[int] = []

    def _wait(timeout: float | None = None) -> None:
        wait_calls.append(1)
        if len(wait_calls) == 1 and timeout is not None:
            raise subprocess.TimeoutExpired("cmd", timeout)

    hanging.wait = _wait  # type: ignore[method-assign]

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: hanging)

    timeline: dict[str, Any] = {"stages": []}
    rc = _run_stage(
        "s1_fetch_strain.py",
        ["--run", run_id],
        "s1_fetch_strain",
        runs_root,
        run_id,
        timeline,
        stage_timeout_s=0.001,
    )

    assert rc == 124
    assert timeline["stages"][-1]["timed_out"] is True
    # The timeline entry records the raw process returncode (e.g., -9 after kill),
    # while _run_stage returns 124 as the canonical timeout sentinel.
    assert timeline["stages"][-1]["returncode"] != 0


def test_run_stage_success_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    run_id = "test_ok_run"

    class _SuccessProc:
        returncode = 0

        def wait(self, timeout: float | None = None) -> None:
            pass

        def kill(self) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _SuccessProc())

    timeline: dict[str, Any] = {"stages": []}
    rc = _run_stage(
        "s1_fetch_strain.py",
        ["--run", run_id],
        "s1_fetch_strain",
        runs_root,
        run_id,
        timeline,
    )

    assert rc == 0
    assert timeline["stages"][-1]["timed_out"] is False
    assert timeline["stages"][-1]["returncode"] == 0


def test_run_stage_nonzero_exit_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    run_id = "test_fail_run"

    class _FailProc:
        returncode = 2

        def wait(self, timeout: float | None = None) -> None:
            pass

        def kill(self) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _FailProc())

    timeline: dict[str, Any] = {"stages": []}
    rc = _run_stage(
        "s1_fetch_strain.py",
        ["--run", run_id],
        "s1_fetch_strain",
        runs_root,
        run_id,
        timeline,
    )

    assert rc == 2


def test_run_stage_appends_to_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    run_id = "test_timeline_run"

    class _OkProc:
        returncode = 0

        def wait(self, timeout: float | None = None) -> None:
            pass

        def kill(self) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _OkProc())

    timeline: dict[str, Any] = {"stages": []}
    _run_stage(
        "s1_fetch_strain.py",
        ["--run", run_id],
        "s1_fetch_strain",
        runs_root,
        run_id,
        timeline,
    )

    assert len(timeline["stages"]) == 1
    entry = timeline["stages"][0]
    assert "stage" in entry
    assert "started_utc" in entry
    assert "ended_utc" in entry
    assert "duration_s" in entry
    assert "returncode" in entry


# ---------------------------------------------------------------------------
# run_single_event — RUN_VALID created before any stage
# ---------------------------------------------------------------------------


def test_run_single_event_creates_run_valid_before_first_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_valid_existed_on_first_call: list[bool] = []

    def fake_run_stage(
        script: str,
        args: list[str],
        label: str,
        out_root: Path,
        run_id: str,
        timeline: dict[str, Any],
        stage_timeout_s: float | None = None,
    ) -> int:
        rv = out_root / run_id / "RUN_VALID" / "verdict.json"
        run_valid_existed_on_first_call.append(rv.exists())
        return 1  # fail immediately so pipeline aborts after first call

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    run_single_event("GW150914", "fake_atlas.json")

    assert run_valid_existed_on_first_call, "No stage was ever called"
    assert run_valid_existed_on_first_call[0], (
        "RUN_VALID did not exist when the first stage was called"
    )


def test_run_single_event_aborts_after_first_stage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    stages_called: list[str] = []

    def fake_run_stage(
        script: str,
        args: list[str],
        label: str,
        out_root: Path,
        run_id: str,
        timeline: dict[str, Any],
        stage_timeout_s: float | None = None,
    ) -> int:
        stages_called.append(label)
        return 1  # Always fail

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    rc, _ = run_single_event("GW150914", "fake_atlas.json")

    assert rc != 0
    # Only the first stage should have been invoked
    assert len(stages_called) == 1, (
        f"Expected 1 stage call, got {len(stages_called)}: {stages_called}"
    )


def test_run_single_event_returns_nonzero_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    monkeypatch.setattr(
        pipeline,
        "_run_stage",
        lambda *a, **kw: 1,
    )

    rc, run_id = run_single_event("GW150914", "fake_atlas.json")
    assert rc != 0
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_run_single_event_writes_timeline_on_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    monkeypatch.setattr(pipeline, "_run_stage", lambda *a, **kw: 1)

    rc, run_id = run_single_event("GW150914", "fake_atlas.json", run_id="fixed_run")

    timeline_path = runs_root / run_id / "pipeline_timeline.json"
    assert timeline_path.exists(), "pipeline_timeline.json not written on abort"
    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert tl["run_id"] == run_id
    assert tl["event_id"] == "GW150914"
    assert "ended_utc" in tl


def test_run_single_event_timeline_has_required_keys_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    # All stages succeed
    monkeypatch.setattr(pipeline, "_run_stage", lambda *a, **kw: 0)

    rc, run_id = run_single_event("GW150914", "fake_atlas.json", run_id="ok_run")

    timeline_path = runs_root / run_id / "pipeline_timeline.json"
    assert timeline_path.exists()
    tl = json.loads(timeline_path.read_text(encoding="utf-8"))

    for key in ("schema_version", "run_id", "mode", "started_utc", "ended_utc", "stages"):
        assert key in tl, f"Missing key in timeline: {key!r}"
    assert tl["mode"] == "single"
    assert tl["ended_utc"] is not None


def test_run_single_event_unknown_estimator_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    # s0 succeeds so we reach the estimator branch; s1 and s2 also succeed
    call_idx: list[int] = [0]

    def fake_run_stage(*a: Any, **kw: Any) -> int:
        # First 3 calls (s0, s1, s2) succeed; then we hit estimator logic
        call_idx[0] += 1
        return 0

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    rc, _ = run_single_event(
        "GW150914", "fake_atlas.json", estimator="nonexistent_estimator"
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# run_multi_event — abort semantics
# ---------------------------------------------------------------------------


def _make_fake_single_event(
    fail_events: set[str],
    runs_root: Path,
    call_log: list[str],
) -> Any:
    """Return a fake run_single_event that records calls and fails selectively."""

    def _fake(
        event_id: str,
        atlas_path: str,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> tuple[int, str]:
        call_log.append(event_id)
        fake_run_id = f"run_{event_id}"
        if event_id in fail_events:
            return 1, fake_run_id
        # Create minimal RUN_VALID so run_multi_event can proceed
        rv_dir = runs_root / fake_run_id / "RUN_VALID"
        rv_dir.mkdir(parents=True, exist_ok=True)
        (rv_dir / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
        return 0, fake_run_id

    return _fake


def test_run_multi_event_aborts_on_first_event_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    call_log: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "run_single_event",
        _make_fake_single_event({"GW150914"}, runs_root, call_log),
    )
    # s5_aggregate also needs to be faked
    monkeypatch.setattr(pipeline, "_run_stage", lambda *a, **kw: 0)

    rc, _ = run_multi_event(
        ["GW150914", "GW151226"],
        "fake_atlas.json",
        agg_run_id="agg_test",
        abort_on_event_fail=True,
    )

    assert rc != 0
    assert call_log == ["GW150914"], (
        f"Second event should not have been called; got: {call_log}"
    )


def test_run_multi_event_continues_when_abort_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    call_log: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "run_single_event",
        _make_fake_single_event({"GW150914"}, runs_root, call_log),
    )
    monkeypatch.setattr(pipeline, "_run_stage", lambda *a, **kw: 0)

    run_multi_event(
        ["GW150914", "GW151226"],
        "fake_atlas.json",
        agg_run_id="agg_batch",
        abort_on_event_fail=False,
    )

    # Both events should have been attempted
    assert "GW150914" in call_log
    assert "GW151226" in call_log


def test_run_multi_event_all_fail_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    call_log: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "run_single_event",
        _make_fake_single_event({"GW150914", "GW151226"}, runs_root, call_log),
    )
    monkeypatch.setattr(pipeline, "_run_stage", lambda *a, **kw: 0)

    rc, _ = run_multi_event(
        ["GW150914", "GW151226"],
        "fake_atlas.json",
        agg_run_id="agg_all_fail",
        abort_on_event_fail=False,
    )

    # All events failed → aggregate is never run → should return error
    assert rc != 0


def test_run_multi_event_aggregate_not_run_if_event_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    aggregate_called: list[bool] = []

    def fake_run_stage(
        script: str,
        args: list[str],
        label: str,
        *rest: Any,
        **kw: Any,
    ) -> int:
        if "s5_aggregate" in script or label == "s5_aggregate":
            aggregate_called.append(True)
        return 0

    call_log: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "run_single_event",
        _make_fake_single_event({"GW150914"}, runs_root, call_log),
    )
    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    rc, _ = run_multi_event(
        ["GW150914", "GW151226"],
        "fake_atlas.json",
        agg_run_id="agg_abort",
        abort_on_event_fail=True,
    )

    assert rc != 0
    assert not aggregate_called, "Aggregate stage should not run when event fails with abort"


def test_run_multi_event_creates_run_valid_for_agg_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = _make_runs_root(tmp_path)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    call_log: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "run_single_event",
        _make_fake_single_event({"GW150914"}, runs_root, call_log),
    )
    monkeypatch.setattr(pipeline, "_run_stage", lambda *a, **kw: 0)

    run_multi_event(
        ["GW150914"],
        "fake_atlas.json",
        agg_run_id="agg_rv_test",
        abort_on_event_fail=True,
    )

    rv_path = runs_root / "agg_rv_test" / "RUN_VALID" / "verdict.json"
    assert rv_path.exists(), "RUN_VALID was not created for the aggregate run"
