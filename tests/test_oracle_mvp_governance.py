from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s0_oracle_mvp import main as oracle_main

S2_SCRIPT = REPO_ROOT / "mvp" / "s2_ringdown_window.py"


def _write_run_valid(runs_root: Path, run_id: str, verdict: str) -> None:
    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(
        json.dumps({"verdict": verdict}, indent=2) + "\n", encoding="utf-8"
    )


def test_oracle_offline_requires_local_inputs_and_never_touches_network(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    run_id = "oracle_offline_fail"
    _write_run_valid(runs_root, run_id, "PASS")

    network_called = {"value": False}

    def _deny_network(*args, **kwargs):
        network_called["value"] = True
        raise AssertionError("network should not be called")

    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    rc = oracle_main(["--run", run_id, "--require-offline"])

    assert rc == 2
    assert network_called["value"] is False

    summary = json.loads((runs_root / run_id / "s0_oracle_mvp" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["oracle"]["verdict"] == "FAIL"
    assert any("Offline policy active" in r for r in summary["oracle"]["reasons"])


def test_downstream_stage_aborts_immediately_when_run_valid_not_pass(tmp_path):
    runs_root = tmp_path / "runs"
    run_id = "downstream_gate"
    _write_run_valid(runs_root, run_id, "FAIL")

    proc = subprocess.run(
        [sys.executable, str(S2_SCRIPT), "--run", run_id, "--event-id", "GW150914"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)},
        check=False,
    )

    assert proc.returncode == 2
    assert "RUN_VALID check failed" in (proc.stdout + proc.stderr)
    assert not (runs_root / run_id / "s2_ringdown_window").exists()


def test_oracle_pass_writes_contract_artifacts_with_sha256(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    run_id = "oracle_pass"
    _write_run_valid(runs_root, run_id, "PASS")

    local_h5 = tmp_path / "H1_local.h5"
    local_h5.write_bytes(b"local hdf5 placeholder")

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = oracle_main([
        "--run", run_id,
        "--require-offline",
        "--local-hdf5", f"H1={local_h5}",
    ])

    assert rc == 0

    stage_dir = runs_root / run_id / "s0_oracle_mvp"
    assert (stage_dir / "outputs").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "manifest.json").exists()

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hashes"]["oracle_metrics"]
    assert manifest["hashes"]["stage_summary"]
