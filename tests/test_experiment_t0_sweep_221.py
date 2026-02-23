from __future__ import annotations

import json
from pathlib import Path

from mvp import experiment_t0_sweep_221 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setup_base_run(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    (run_dir / "s1_fetch_strain" / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "s1_fetch_strain" / "outputs" / "strain.npz").write_bytes(b"npz-placeholder")
    _write_json(run_dir / "s1_fetch_strain" / "manifest.json", {"schema_version": "mvp_manifest_v1"})

    _write_json(run_dir / "s2_ringdown_window" / "manifest.json", {"schema_version": "mvp_manifest_v1"})

    s3_out = run_dir / "s3_ringdown_estimates" / "outputs"
    s3_out.mkdir(parents=True, exist_ok=True)
    _write_json(s3_out / "estimates.json", {"m_final_msun": 65.0})
    _write_json(run_dir / "s3_ringdown_estimates" / "manifest.json", {"schema_version": "mvp_manifest_v1"})
    return run_dir


class _CP:
    def __init__(self, code: int = 0):
        self.returncode = code
        self.stdout = ""
        self.stderr = ""


def test_experiment_outputs_and_manifest(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    base_run_id = "base_run"
    exp_run_id = "exp_run"
    _setup_base_run(runs_root, base_run_id)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def fake_run_stage(cmd: list[str], env: dict[str, str]):
        subruns_root = Path(env["BASURIN_RUNS_ROOT"])
        if any("s3b_multimode_estimates.py" in part for part in cmd):
            run_id = cmd[cmd.index("--run-id") + 1]
            out = subruns_root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
            if run_id.endswith("-1"):
                payload = {
                    "results": {"quality_flags": ["221_valid_fraction_low"]},
                    "modes": [
                        {"label": "220", "ln_f": 1.0, "Sigma": [[1, 0], [0, 1]], "fit": {"stability": {}}},
                        {"label": "221", "ln_f": None, "Sigma": None, "fit": {"stability": {"valid_fraction": 0.2}}},
                    ],
                }
            else:
                payload = {
                    "results": {"quality_flags": []},
                    "modes": [
                        {"label": "220", "ln_f": 1.0, "Sigma": [[1, 0], [0, 1]], "fit": {"stability": {}}},
                        {"label": "221", "ln_f": 2.0, "Sigma": [[1, 0], [0, 1]], "fit": {"stability": {"valid_fraction": 0.95}}},
                    ],
                }
            _write_json(out, payload)
        return _CP(0)

    monkeypatch.setattr(mod, "_run_stage_cmd", fake_run_stage)

    rc = mod.main(
        [
            "--event-id",
            "GW250114",
            "--base-run-id",
            base_run_id,
            "--exp-run-id",
            exp_run_id,
            "--t0-grid=-1,0,1",
            "--units",
            "M",
        ]
    )
    assert rc == 0

    exp_dir = runs_root / exp_run_id / "experiment" / "t0_sweep_221"
    outputs_dir = exp_dir / "outputs"
    assert outputs_dir.exists()
    table = json.loads((outputs_dir / "t0_sweep_table.json").read_text(encoding="utf-8"))
    assert len(table) == 3
    required = {"event_id", "t0", "units", "has_221", "valid_fraction_221", "reason", "flags", "subrun_path"}
    assert required.issubset(set(table[0]))

    best = json.loads((outputs_dir / "best_t0.json").read_text(encoding="utf-8"))
    assert best["best_t0"]["has_221"] is True
    assert best["best_t0"]["valid_fraction_221"] == 0.95

    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["hashes"]
    assert "outputs/t0_sweep_table.json" in hashes
    assert "outputs/t0_sweep_table.csv" in hashes
    assert "outputs/best_t0.json" in hashes
    assert "outputs/diagnostics.json" in hashes

    assert (runs_root / exp_run_id / "experiment" / "t0_sweep_221").exists()
    # Integration-lite: todas las escrituras relevantes caen bajo BASURIN_RUNS_ROOT.
    assert not (tmp_path / "experiment").exists()
