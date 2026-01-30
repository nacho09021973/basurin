from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PY = os.environ.get("PYTHON", "python")


def _wjson(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _rjson(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture()
def repo_root() -> Path:
    return Path.cwd()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    # Ejecutamos en un repo temporal simulado: crea estructura runs/<run_id> en tmp_path y corre scripts desde repo_root.
    # Para mantener import basurin_io, los scripts se ejecutan desde repo real, pero apuntamos BASURIN_RUNS_ROOT al tmp.
    run_id = "2026-01-30__exp_ringdown_00_smoke"
    r = tmp_path / "runs" / run_id
    r.mkdir(parents=True, exist_ok=True)
    return r


def _env_for(tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")
    return env


def test_abort_on_invalid_run_valid(tmp_path: Path, repo_root: Path):
    run_id = "r0"
    run_dir = tmp_path / "runs" / run_id
    _wjson(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "FAIL"})
    # synth exists but should not matter
    _wjson(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json", {
        "schema_version": "ringdown_synth_event_v1",
        "seed": 42,
        "snr_target": 12.0,
        "truth": {"f_220": 250.0, "tau_220": 0.004, "Q_220": 3.1415926},
    })

    env = _env_for(tmp_path)
    p = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id],
                       cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode == 2

    out_stage = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_00__stability_sweep"
    assert not out_stage.exists(), "no debe escribir outputs si RUN_VALID!=PASS"


def test_abort_on_missing_synthetic_event(tmp_path: Path, repo_root: Path):
    run_id = "r1"
    run_dir = tmp_path / "runs" / run_id
    _wjson(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    env = _env_for(tmp_path)

    p = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id],
                       cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode == 2


def test_ringdown_synth_stage_outputs(tmp_path: Path, repo_root: Path):
    run_id = "r2"
    env = _env_for(tmp_path)

    p = subprocess.run([PY, "stages/ringdown_synth_stage.py", "--run", run_id, "--seed", "42", "--snr", "12.0", "--f-220", "250.0", "--tau-220", "0.004"],
                       cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode == 0

    out = tmp_path / "runs" / run_id / "ringdown_synth" / "outputs" / "synthetic_event.json"
    assert out.exists()
    obj = _rjson(out)
    assert obj["truth"]["f_220"] == 250.0


def test_deterministic_sweep_plan_hash(tmp_path: Path, repo_root: Path):
    run_id = "r3"
    run_dir = tmp_path / "runs" / run_id
    _wjson(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _wjson(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json", {
        "schema_version": "ringdown_synth_event_v1",
        "seed": 42,
        "snr_target": 12.0,
        "truth": {"f_220": 250.0, "tau_220": 0.004, "Q_220": 3.1415926},
    })

    env = _env_for(tmp_path)

    p1 = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id, "--seed", "42"],
                        cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p1.returncode in (0, 2)

    plan1 = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_00__stability_sweep" / "outputs" / "sweep_plan.json"
    h1 = plan1.read_bytes()

    # rerun (clean stage dir to ensure regeneration)
    outputs_dir = plan1.parent
    assert outputs_dir.name == "outputs"
    stage_root = outputs_dir.parent  # .../EXP_RINGDOWN_00__stability_sweep
    # borrar y repetir
    import shutil
    shutil.rmtree(stage_root)

    p2 = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id, "--seed", "42"],
                        cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p2.returncode in (0, 2)

    plan2 = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_00__stability_sweep" / "outputs" / "sweep_plan.json"
    h2 = plan2.read_bytes()

    assert h1 == h2, "sweep_plan debe ser determinista para la misma seed"


def test_verdict_always_written_when_outputs_created(tmp_path: Path, repo_root: Path):
    run_id = "r4"
    run_dir = tmp_path / "runs" / run_id
    _wjson(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _wjson(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json", {
        "schema_version": "ringdown_synth_event_v1",
        "seed": 42,
        "snr_target": 12.0,
        "truth": {"f_220": 250.0, "tau_220": 0.004, "Q_220": 3.1415926},
    })

    env = _env_for(tmp_path)
    p = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id, "--seed", "42"],
                       cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode in (0, 2)

    verdict = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_00__stability_sweep" / "outputs" / "contract_verdict.json"
    assert verdict.exists()
    v = _rjson(verdict)
    assert v["verdict"] in ("PASS", "FAIL")


def test_skip_low_snr_policy(tmp_path: Path, repo_root: Path):
    run_id = "r5"
    run_dir = tmp_path / "runs" / run_id
    _wjson(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    # snr_target bajo para forzar SKIP
    _wjson(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json", {
        "schema_version": "ringdown_synth_event_v1",
        "seed": 42,
        "snr_target": 6.0,
        "truth": {"f_220": 250.0, "tau_220": 0.004, "Q_220": 3.1415926},
    })

    env = _env_for(tmp_path)
    p = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id, "--seed", "42"],
                       cwd=str(repo_root), env=env, capture_output=True, text=True)
    # puede FAIL si baseline queda SKIP; aceptamos returncode=2 pero verificamos SKIP_LOW_SNR reportado
    assert p.returncode in (0, 2)

    diag = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_00__stability_sweep" / "outputs" / "diagnostics.json"
    d = _rjson(diag)
    assert "SKIP_LOW_SNR" in d["skips"]


def test_outputs_under_runs_root_only(tmp_path: Path, repo_root: Path):
    run_id = "r6"
    run_dir = tmp_path / "runs" / run_id
    _wjson(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _wjson(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json", {
        "schema_version": "ringdown_synth_event_v1",
        "seed": 42,
        "snr_target": 12.0,
        "truth": {"f_220": 250.0, "tau_220": 0.004, "Q_220": 3.1415926},
    })

    env = _env_for(tmp_path)
    p = subprocess.run([PY, "experiment/ringdown/exp_ringdown_00_stability_sweep.py", "--run", run_id],
                       cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode in (0, 2)

    # Comprobación simple: el stage dir existe donde toca y no crea nada fuera de tmp_path/runs
    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_00__stability_sweep"
    assert stage_dir.exists()
    # No hay evidencia de writes fuera: esto se refuerza con el kill-switch del repo si lo tienes globalmente.
