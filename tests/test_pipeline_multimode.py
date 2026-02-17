from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp import pipeline

ATLAS_FIXTURE = Path("mvp/test_atlas_fixture.json")


def test_single_with_t0_sweep_missing_script_is_best_effort(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
        timeline["stages"].append(
            {
                "stage": label,
                "script": script,
                "command": [script] + list(args),
                "started_utc": "now",
                "ended_utc": "now",
                "duration_s": 0.0,
                "returncode": 0,
                "timed_out": False,
            }
        )
        pipeline._write_timeline(out_root, run_id, timeline)
        return 0

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)
    monkeypatch.setattr(pipeline, "MVP_DIR", tmp_path / "missing_mvp")

    rc, run_id = pipeline.run_single_event(
        event_id="GW150914",
        atlas_path=str(ATLAS_FIXTURE),
        synthetic=True,
        duration_s=4.0,
        with_t0_sweep=True,
    )

    assert rc == 0
    timeline_path = runs_root / run_id / "pipeline_timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))

    exp_entries = [s for s in timeline["stages"] if s["stage"] == "experiment_t0_sweep"]
    assert len(exp_entries) == 1
    assert exp_entries[0]["best_effort"] is True
    assert exp_entries[0]["label"] == "experiment_t0_sweep"
    assert exp_entries[0]["script"] == "mvp/experiment_t0_sweep.py"
    assert exp_entries[0]["status"] == "SKIPPED"
    assert exp_entries[0]["returncode"] is None
    assert exp_entries[0]["duration_s"] == 0.0
    assert exp_entries[0]["message"] == "missing script"


def test_multimode_writes_multimode_results_and_stages(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def fake_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
        stage_dir = out_root / run_id / label / "outputs"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if label == "s3b_multimode_estimates":
            payload = {"results": {"verdict": "INSUFFICIENT_DATA"}}
            (stage_dir / "multimode_estimates.json").write_text(json.dumps(payload), encoding="utf-8")
        if label == "s4c_kerr_consistency":
            payload = {"consistent_kerr_95": True, "chi_best_fit": 0.69, "d2_min": 1.2}
            (stage_dir / "kerr_consistency.json").write_text(json.dumps(payload), encoding="utf-8")

        timeline["stages"].append(
            {
                "stage": label,
                "script": script,
                "command": [script] + list(args),
                "started_utc": "now",
                "ended_utc": "now",
                "duration_s": 0.0,
                "returncode": 0,
                "timed_out": False,
            }
        )
        pipeline._write_timeline(out_root, run_id, timeline)
        return 0

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    rc, run_id = pipeline.run_multimode_event(
        event_id="GW150914",
        atlas_path=str(ATLAS_FIXTURE),
        synthetic=True,
        duration_s=4.0,
    )

    assert rc == 0
    timeline = json.loads((runs_root / run_id / "pipeline_timeline.json").read_text(encoding="utf-8"))
    assert timeline["mode"] == "multimode"

    stage_names = [s["stage"] for s in timeline["stages"]]
    assert stage_names == [
        "s1_fetch_strain",
        "s2_ringdown_window",
        "s3_ringdown_estimates",
        "s3b_multimode_estimates",
        "s4_geometry_filter",
        "s4c_kerr_consistency",
    ]

    assert timeline["multimode_results"] == {
        "kerr_consistent": True,
        "chi_best": 0.69,
        "d2_min": 1.2,
        "extraction_quality": "INSUFFICIENT_DATA",
    }


def test_multi_forwards_with_t0_sweep_to_each_event(monkeypatch):
    calls = []

    def fake_single_event(**kwargs):
        calls.append(kwargs)
        return 0, f"run_{kwargs['event_id']}"

    monkeypatch.setattr(pipeline, "run_single_event", fake_single_event)

    def fake_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
        timeline["stages"].append(
            {
                "stage": label,
                "script": script,
                "command": [script] + list(args),
                "started_utc": "now",
                "ended_utc": "now",
                "duration_s": 0.0,
                "returncode": 0,
                "timed_out": False,
            }
        )
        pipeline._write_timeline(out_root, run_id, timeline)
        return 0

    monkeypatch.setattr(pipeline, "_run_stage", fake_stage)

    rc, _ = pipeline.run_multi_event(
        events=["GW1", "GW2"],
        atlas_path=str(ATLAS_FIXTURE),
        synthetic=True,
        with_t0_sweep=True,
    )

    assert rc == 0
    assert len(calls) == 2
    assert all(c["with_t0_sweep"] is True for c in calls)
