from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "tools/basurin_run_real.py", *args]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run_valid(run_dir: Path) -> None:
    _write_json(run_dir / "RUN_VALID" / "outputs" / "run_valid.json", {"verdict": "PASS"})


def _make_stage_summary(run_dir: Path, stage_name: str) -> Path:
    path = run_dir / stage_name / "stage_summary.json"
    _write_json(path, {"verdict": "PASS"})
    return path


def test_runner_fails_without_run_valid(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_missing"
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_cli(["--run", run_id], env)

    assert result.returncode != 0


def test_runner_dry_run_prints_stage_names(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_ok"
    run_dir = runs_root / run_id
    _make_run_valid(run_dir)

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_cli(["--run", run_id, "--dry-run"], env)

    assert result.returncode == 0
    assert "ringdown_real_ringdown_window_v1__dt0000ms__dur0250ms" in result.stdout
    assert "ringdown_real_observables_v0__dt0000ms__dur0250ms" in result.stdout
    assert "ringdown_real_features_v0__dt0000ms__dur0250ms" in result.stdout
    assert "ringdown_real_inference_v0__dt0000ms__dur0250ms" in result.stdout
    assert not (run_dir / "REAL_PIPELINE_SUMMARY.json").exists()


def test_runner_dry_run_skips_real_v0_when_output_exists(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_real_v0_ready"
    run_dir = runs_root / run_id
    _make_run_valid(run_dir)

    real_v0_output = (
        run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json"
    )
    real_v0_output.parent.mkdir(parents=True, exist_ok=True)
    real_v0_output.write_text("[]", encoding="utf-8")

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_cli(["--run", run_id, "--dry-run"], env)

    assert result.returncode == 0
    assert "ringdown_real_v0_stage.py" not in result.stdout


def test_runner_writes_summary_when_stages_exist(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_summary"
    run_dir = runs_root / run_id
    _make_run_valid(run_dir)

    _make_stage_summary(run_dir, "ringdown_real_v0")
    real_v0_output = (
        run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json"
    )
    real_v0_output.parent.mkdir(parents=True, exist_ok=True)
    real_v0_output.write_text("[]", encoding="utf-8")

    suffix = "dt0000ms__dur0250ms"
    window_stage = f"ringdown_real_ringdown_window_v1__{suffix}"
    observables_stage = f"ringdown_real_observables_v0__{suffix}"
    features_stage = f"ringdown_real_features_v0__{suffix}"
    inference_stage = f"ringdown_real_inference_v0__{suffix}"

    window_summary = _make_stage_summary(run_dir, window_stage)
    observables_summary = _make_stage_summary(run_dir, observables_stage)
    features_summary = _make_stage_summary(run_dir, features_stage)
    inference_summary = _make_stage_summary(run_dir, inference_stage)

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_cli(["--run", run_id], env)

    assert result.returncode == 0

    summary_path = run_dir / "REAL_PIPELINE_SUMMARY.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["run_id"] == run_id
    assert summary["params"]["band_hz"] == [150.0, 400.0]

    stages = summary["stages"]
    assert stages["RUN_VALID"]["path"] == "RUN_VALID/outputs/run_valid.json"

    window_artifacts = stages["window"]["artifacts"]
    assert window_artifacts[0]["path"] == str(window_summary.relative_to(run_dir))
    assert window_artifacts[0]["sha256"] == sha256_file(window_summary)

    observables_artifacts = stages["observables"]["artifacts"]
    assert observables_artifacts[0]["sha256"] == sha256_file(observables_summary)

    features_artifacts = stages["features"]["artifacts"]
    assert features_artifacts[0]["sha256"] == sha256_file(features_summary)

    inference_artifacts = stages["inference"]["artifacts"]
    assert inference_artifacts[0]["sha256"] == sha256_file(inference_summary)

    assert summary["final_verdict"] == "PASS"
