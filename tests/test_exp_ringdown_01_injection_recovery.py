import json
from pathlib import Path
import pytest

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def make_min_run_valid(run_dir: Path, verdict="PASS"):
    write_json(
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": verdict}
    )
def make_events(run_dir: Path, n=30, bias=0.0):
    out = run_dir / "ringdown_synth" / "outputs"
    cases = []
    for i in range(n):
        cid = f"snr12_seed{i}"
        # layout esperado por el experimento
        strain_rel = f"cases/{cid}/strain.npz"
        (out / "cases" / cid).mkdir(parents=True, exist_ok=True)
        # no necesitamos npz real en este test: el recover va monkeypatch, pero el path debe existir
        (out / "cases" / cid / "strain.npz").write_bytes(b"FAKE")
        cases.append({
            "case_id": cid,
            "snr": 12,
            "seed": i,
            "truth": {"f_220": 220.0, "tau_220": 0.004, "Q_220": 10.0},
            "strain_npz": strain_rel,
            "_test_bias": bias,
        })
    write_json(out / "synthetic_events.json", cases)
    return out / "synthetic_events.json"

@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    rr = tmp_path / "runs"
    rr.mkdir()
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(rr))
    return rr

def test_exp_ringdown_01_pass(monkeypatch, runs_root):
    run = "test_ringdown01_pass"
    run_dir = runs_root / run
    make_min_run_valid(run_dir, "PASS")
    events = make_events(run_dir, n=30, bias=0.0)

    import experiment.ringdown.exp_ringdown_01_injection_recovery as mod

    def fake_recover(_path):
        # estimación sin sesgo
        return {"f_220_hat": 220.0, "tau_220_hat": 0.004, "Q_220_hat": 10.0}

    monkeypatch.setattr(mod, "recover_ringdown", fake_recover)

    argv = [
        "prog",
        "--run", run,
        "--out-root", "runs",
        "--events-json", str(events),
        "--min-cases", "24",
        "--bias-p50-max", "0.01",
        "--bias-p90-max", "0.02",
    ]
    monkeypatch.setattr(mod.sys, "argv", argv)
    assert mod.main() == 0

    out_dir = run_dir / "experiment" / "ringdown_01_injection_recovery"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "stage_summary.json").exists()
    assert (out_dir / "outputs" / "recovery_contract.json").exists()

def test_exp_ringdown_01_fail_exit2(monkeypatch, runs_root):
    run = "test_ringdown01_fail"
    run_dir = runs_root / run
    make_min_run_valid(run_dir, "PASS")
    events = make_events(run_dir, n=30, bias=0.2)

    import experiment.ringdown.exp_ringdown_01_injection_recovery as mod

    def biased_recover(_path):
        # 20% sesgo (debería fallar umbrales)
        return {"f_220_hat": 220.0 * 1.2, "tau_220_hat": 0.004 * 1.2, "Q_220_hat": 10.0}

    monkeypatch.setattr(mod, "recover_ringdown", biased_recover)

    argv = [
        "prog",
        "--run", run,
        "--out-root", "runs",
        "--events-json", str(events),
        "--min-cases", "24",
        "--bias-p50-max", "0.03",
        "--bias-p90-max", "0.08",
    ]
    monkeypatch.setattr(mod.sys, "argv", argv)

    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 2
