from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


STAGE = "stages/bayes_validation_v1_stage.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _setup_run_valid(run_dir: Path, verdict: str = "PASS") -> None:
    _write_json(run_dir / "RUN_VALID" / "outputs" / "run_valid.json", {"overall_verdict": verdict})


def _setup_dictionary_input(run_dir: Path, content: bytes = b"dummy-spectrum") -> Path:
    path = run_dir / "dictionary" / "outputs" / "spectrum.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _run_stage(runs_root: Path, run_id: str, extra_args: list[str] | None = None, env: dict[str, str] | None = None):
    cmd = ["python", STAGE, "--run", run_id]
    if extra_args:
        cmd.extend(extra_args)
    run_env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    if env:
        run_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, check=False, env=run_env)


def test_gate_run_valid(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_gate"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir, verdict="FAIL")

    res = _run_stage(runs_root, run_id)

    assert res.returncode == 2
    assert not (run_dir / "bayes_validation_v1").exists()


def test_output_contract_files_exist(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_contract"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_dictionary_input(run_dir)

    res = _run_stage(runs_root, run_id)

    assert res.returncode == 0, res.stderr
    stage_dir = run_dir / "bayes_validation_v1"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "outputs" / "bayes_validation.json").exists()


def test_determinism_same_seed_same_output(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_determinism"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_dictionary_input(run_dir, content=b"deterministic-spectrum")

    res_a = _run_stage(runs_root, run_id, ["--stage-name", "bayes_validation_a", "--seed", "123"])
    res_b = _run_stage(runs_root, run_id, ["--stage-name", "bayes_validation_b", "--seed", "123"])

    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr

    out_a = run_dir / "bayes_validation_a" / "outputs" / "bayes_validation.json"
    out_b = run_dir / "bayes_validation_b" / "outputs" / "bayes_validation.json"
    hash_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    hash_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert hash_a == hash_b


def test_scipy_missing_killswitch(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_no_scipy"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_dictionary_input(run_dir)

    fake_pkg = tmp_path / "fakepkg"
    fake_pkg.mkdir(parents=True, exist_ok=True)
    (fake_pkg / "scipy.py").write_text("raise ImportError('simulated scipy missing')\n", encoding="utf-8")

    env = {"PYTHONPATH": f"{fake_pkg}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    res = _run_stage(runs_root, run_id, env=env)

    assert res.returncode == 2
    assert "scipy_missing" in (res.stderr + res.stdout)
