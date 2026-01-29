import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _run_stage(repo_root: Path, runs_root: Path, run_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "stages" / "stage_run_index.py"),
            "--run",
            run_id,
            "--runs-root",
            str(runs_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_abort_when_run_valid_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "missing-run-valid"
    (runs_root / run_id).mkdir(parents=True)

    result = _run_stage(repo_root, runs_root, run_id)

    assert result.returncode != 0
    assert not (runs_root / run_id / "RUN_INDEX").exists()


def test_abort_when_run_valid_not_pass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "run-valid-fail"
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    run_valid_path.parent.mkdir(parents=True)
    _write_json(run_valid_path, {"run": run_id, "verdict": "FAIL"})

    result = _run_stage(repo_root, runs_root, run_id)

    assert result.returncode != 0
    assert not (runs_root / run_id / "RUN_INDEX").exists()


def test_deterministic_index(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "deterministic-run"
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    run_valid_path.parent.mkdir(parents=True)
    _write_json(run_valid_path, {"run": run_id, "verdict": "PASS"})

    stage_dir = runs_root / run_id / "spectrum"
    stage_dir.mkdir(parents=True)
    _write_json(stage_dir / "stage_summary.json", {"stage": "spectrum", "verdict": "PASS"})
    _write_json(stage_dir / "manifest.json", {"stage": "spectrum", "files": {}})

    first = _run_stage(repo_root, runs_root, run_id)
    assert first.returncode == 0

    index_path = runs_root / run_id / "RUN_INDEX" / "outputs" / "index.json"
    first_bytes = index_path.read_bytes()
    first_hash = hashlib.sha256(first_bytes).hexdigest()

    second = _run_stage(repo_root, runs_root, run_id)
    assert second.returncode == 0

    second_bytes = index_path.read_bytes()
    second_hash = hashlib.sha256(second_bytes).hexdigest()

    assert first_bytes == second_bytes
    assert first_hash == second_hash


def test_entries_include_stage_and_experiment(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "with-experiment"
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    run_valid_path.parent.mkdir(parents=True)
    _write_json(run_valid_path, {"run": run_id, "verdict": "PASS"})

    stage_dir = runs_root / run_id / "dictionary"
    stage_dir.mkdir(parents=True)
    _write_json(stage_dir / "stage_summary.json", {"stage": "dictionary", "verdict": "PASS"})
    _write_json(stage_dir / "manifest.json", {"stage": "dictionary", "files": {}})

    exp_dir = runs_root / run_id / "experiment" / "exp_alpha"
    exp_dir.mkdir(parents=True)
    _write_json(exp_dir / "stage_summary.json", {"stage": "exp_alpha", "results": {"verdict": "FAIL"}})
    _write_json(exp_dir / "manifest.json", {"stage": "exp_alpha", "files": {}})

    result = _run_stage(repo_root, runs_root, run_id)
    assert result.returncode == 0

    index_path = runs_root / run_id / "RUN_INDEX" / "outputs" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))

    entries = {(entry["name"], entry["kind"]): entry for entry in payload["entries"]}
    assert ("dictionary", "stage") in entries
    assert ("exp_alpha", "experiment") in entries
    assert entries[("dictionary", "stage")]["verdict"] == "PASS"
    assert entries[("exp_alpha", "experiment")]["verdict"] == "FAIL"
