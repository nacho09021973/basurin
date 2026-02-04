import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_tool(repo_root: Path, env: dict[str, str], args: list[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, str(repo_root / "tools" / "basurin_run_real.py"), *args]
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)


def test_dry_run_stage_names(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "dry-run-stage-names"
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    _write_json(run_valid_path, {"run": run_id, "verdict": "PASS"})

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_tool(
        repo_root,
        env,
        [
            "--run",
            run_id,
            "--dt-start-s",
            "0.0123",
            "--duration-s",
            "0.25",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "dt0012ms__dur0250ms" in result.stdout
    assert "ringdown_real_ringdown_window_v1__dt0012ms__dur0250ms" in result.stdout
    assert "ringdown_real_observables_v0__dt0012ms__dur0250ms" in result.stdout
    assert "ringdown_real_features_v0__dt0012ms__dur0250ms" in result.stdout
    assert "ringdown_real_inference_v0__dt0012ms__dur0250ms" in result.stdout

    summary_path = runs_root / run_id / "REAL_PIPELINE_SUMMARY.json"
    assert not summary_path.exists()


def test_abort_without_run_valid(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "missing-run-valid"
    (runs_root / run_id).mkdir(parents=True)

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_tool(repo_root, env, ["--run", run_id, "--dry-run"])

    assert result.returncode != 0


def test_abort_when_run_valid_not_pass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "run-valid-fail"
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    _write_json(run_valid_path, {"run": run_id, "verdict": "FAIL"})

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    result = _run_tool(repo_root, env, ["--run", run_id, "--dry-run"])

    assert result.returncode != 0
