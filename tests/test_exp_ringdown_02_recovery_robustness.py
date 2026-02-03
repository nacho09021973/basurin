import json
from pathlib import Path

import numpy as np
import pytest


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs))
    return runs


def make_run_valid(run_dir: Path) -> None:
    write_json(
        run_dir / "RUN_VALID" / "stage_summary.json",
        {"results": {"overall_verdict": "PASS"}},
    )


def make_synth_case(run_dir: Path) -> Path:
    out_dir = run_dir / "ringdown_synth" / "outputs"
    case_dir = out_dir / "cases" / "case000"
    case_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 0.1, 4096)
    strain = np.sin(2 * np.pi * 200.0 * t) * np.exp(-t / 0.02)
    np.savez(case_dir / "strain_H1.npz", strain=strain, t=t)
    events = [
        {
            "case_id": "case000",
            "strain_npz": "cases/case000/strain_H1.npz",
            "truth": {"f_220": 220.0, "tau_220": 0.02},
        }
    ]
    write_json(out_dir / "synthetic_events.json", events)
    return out_dir / "synthetic_events.json"


def make_exp01_baseline(run_dir: Path) -> None:
    write_json(
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_01_injection_recovery"
        / "outputs"
        / "contract_verdict.json",
        {"overall_verdict": "PASS"},
    )


def test_exp_ringdown_02_pass(monkeypatch, runs_root):
    run = "test_ringdown02_pass"
    run_dir = runs_root / run
    make_run_valid(run_dir)
    make_synth_case(run_dir)
    make_exp01_baseline(run_dir)

    import experiment.ringdown.exp_ringdown_02_recovery_robustness as mod

    def fake_core(_x, _t, _dt):
        return {"f_220_hat": 220.0, "tau_220_hat": 0.02, "Q_220_hat": 13.8}

    def fake_npz(_path):
        return fake_core(None, None, None)

    monkeypatch.setattr(mod, "recover_ringdown_from_series", fake_core)
    monkeypatch.setattr(mod, "recover_ringdown_npz", fake_npz)

    argv = [
        "prog",
        "--run",
        run,
    ]
    monkeypatch.setattr(mod.sys, "argv", argv)
    assert mod.main() == 0

    out_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_02_recovery_robustness"
    )
    contract_path = out_dir / "outputs" / "contract_verdict.json"
    assert contract_path.exists()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["overall_verdict"] == "PASS"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "stage_summary.json").exists()
    per_case = out_dir / "outputs" / "per_case.jsonl"
    assert per_case.exists()
    assert len(per_case.read_text(encoding="utf-8").strip().splitlines()) == 1
