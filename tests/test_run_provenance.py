from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from mvp import pipeline


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


def test_provenance_schema_keys(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    run_id = "prov_schema"

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def _fake_create_run_valid(out_root: Path, rid: str) -> None:
        (out_root / rid / "RUN_VALID").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "_create_run_valid", _fake_create_run_valid)
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

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert set(provenance.keys()) == {
        "schema_version",
        "run_id",
        "created_utc",
        "environment",
        "invocation",
        "dependencies",
    }
    assert provenance["schema_version"] == "run_provenance_v1"
    assert provenance["invocation"]["mode"] == "single"
    assert provenance["invocation"]["event_id"] == "GW150914"


def test_provenance_git_sha_format(tmp_path) -> None:
    out_root = tmp_path / "runs"
    run_id = "git_sha_case"

    pipeline._write_run_provenance(out_root, run_id, mode="single", event_id="GW150914")

    payload = json.loads((out_root / run_id / "run_provenance.json").read_text(encoding="utf-8"))
    git_sha = payload["environment"]["git_sha"]

    assert git_sha == "UNKNOWN" or bool(re.fullmatch(r"[0-9a-f]{40}", git_sha))


def test_provenance_atlas_hash_matches(tmp_path) -> None:
    out_root = tmp_path / "runs"
    run_id = "atlas_hash_case"
    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_text('{"atlas": "demo"}\n', encoding="utf-8")

    pipeline._write_run_provenance(
        out_root,
        run_id,
        mode="single",
        event_id="GW150914",
        atlas_path=str(atlas_path),
    )

    payload = json.loads((out_root / run_id / "run_provenance.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    assert payload["invocation"]["atlas_sha256"] == expected_hash


def test_provenance_no_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "top-secret")
    monkeypatch.setenv("API_KEY", "abc123")

    out_root = tmp_path / "runs"
    run_id = "no_secrets_case"
    pipeline._write_run_provenance(out_root, run_id, mode="multi", events=["GW150914", "GW151226"])

    payload = json.loads((out_root / run_id / "run_provenance.json").read_text(encoding="utf-8"))
    env_keys = set(payload["environment"].keys())
    assert env_keys == {
        "git_sha",
        "git_dirty",
        "git_branch",
        "python_version",
        "platform",
        "basurin_runs_root",
        "basurin_losc_root",
    }

    as_text = json.dumps(payload).lower()
    for needle in ("secret_token", "api_key", "password", "token"):
        assert needle not in as_text
