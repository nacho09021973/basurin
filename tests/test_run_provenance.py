from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mvp import pipeline


REQUIRED_KEYS = {
    "run_id",
    "started_utc",
    "pipeline_cmdline",
    "git_commit",
    "git_dirty",
    "python_version",
    "platform",
    "contracts_sha256",
    "pipeline_sha256",
    "deps_freeze_sha256",
}


def _failing_run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s=None):
    timeline["stages"].append(
        {
            "stage": label,
            "script": script,
            "command": [script] + list(args),
            "started_utc": "now",
            "ended_utc": "now",
            "duration_s": 0.0,
            "returncode": 1,
            "timed_out": False,
        }
    )
    pipeline._write_timeline(out_root, run_id, timeline)
    return 1


def test_run_provenance_written_and_has_required_keys(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "isolated_runs"
    run_id = "prov_required_keys"

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(pipeline, "_run_stage", _failing_run_stage)

    rc, produced_run_id = pipeline.run_single_event(
        event_id="GW150914",
        atlas_path="mvp/test_atlas_fixture.json",
        run_id=run_id,
        synthetic=True,
    )

    assert rc != 0
    assert produced_run_id == run_id

    provenance_path = runs_root / run_id / "run_provenance.json"
    assert provenance_path.exists()

    timeline = json.loads((runs_root / run_id / "pipeline_timeline.json").read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert REQUIRED_KEYS.issubset(provenance.keys())
    assert provenance["started_utc"] == timeline["started_utc"]

    contracts_path = Path(__file__).resolve().parents[1] / "mvp" / "contracts.py"
    contracts_sha256 = hashlib.sha256(contracts_path.read_bytes()).hexdigest()
    assert provenance["contracts_sha256"] == contracts_sha256


def test_no_repo_runs_written(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "other_root"
    run_id = "prov_no_repo_runs"

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(pipeline, "_run_stage", _failing_run_stage)

    rc, _ = pipeline.run_single_event(
        event_id="GW150914",
        atlas_path="mvp/test_atlas_fixture.json",
        run_id=run_id,
        synthetic=True,
    )

    assert rc != 0
    assert (runs_root / run_id / "run_provenance.json").exists()

    repo_runs_dir = Path(__file__).resolve().parents[1] / "runs" / run_id
    assert not repo_runs_dir.exists()
