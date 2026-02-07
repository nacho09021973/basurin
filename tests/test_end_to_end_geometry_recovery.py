from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basurin_io import compute_sha256, read_json


def _mk_run_valid(root: Path, run_id: str) -> None:
    run_valid = root / "runs" / run_id / "RUN_VALID" / "verdict.json"
    run_valid.parent.mkdir(parents=True, exist_ok=True)
    run_valid.write_text('{"verdict":"PASS"}\n', encoding="utf-8")


def _run_sweep(root: Path, run_id: str, ns: str, sigmas: str) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "experiment" / "ringdown" / "e2e_sweep_v0.py"),
        "--run",
        run_id,
        "--root",
        str(root),
        "--Ns",
        ns,
        "--sigmas",
        sigmas,
        "--seed-base",
        "12345",
        "--topk",
        "3",
        "--thesis-top1",
        "0.70",
        "--thesis-top3",
        "0.95",
    ]
    subprocess.run(cmd, check=True)
    return root / "runs" / run_id / "experiment" / "ringdown" / "e2e_sweep_v0"


def _assert_hashes(manifest: dict, stage_dir: Path) -> None:
    artifacts = manifest.get("artifacts", {})
    assert "manifest.json" in artifacts
    assert "stage_summary.json" in artifacts
    assert "outputs/sweep_results.json" in artifacts
    assert "outputs/degeneracy_report.json" in artifacts
    assert "outputs/verdict.json" in artifacts
    for rel, info in artifacts.items():
        p = stage_dir / rel
        assert p.exists(), f"missing artifact {rel}"
        if rel == "manifest.json":
            continue
        assert info["sha256"] == compute_sha256(p)


def _validate_contract(stage_dir: Path) -> None:
    outputs = stage_dir / "outputs"
    sweep = outputs / "sweep_results.json"
    deg = outputs / "degeneracy_report.json"
    ver = outputs / "verdict.json"
    assert sweep.exists()
    assert deg.exists()
    assert ver.exists()

    summary = read_json(stage_dir / "stage_summary.json")
    assert summary["model"]["family"] == "phi_phenomenological_v0"
    assert summary["model"]["epistemic_status"] == "conjectural/phenomenological"

    manifest = read_json(stage_dir / "manifest.json")
    _assert_hashes(manifest, stage_dir)

    verdict = read_json(ver)
    assert isinstance(verdict["thesis"]["thesis_gate_pass"], bool)
    assert "Nmax_70_at_5pct" in verdict

    degeneracy = read_json(deg)
    assert isinstance(degeneracy["grid"], list)
    for entry in degeneracy["grid"]:
        assert "confusions" in entry and isinstance(entry["confusions"], dict)
        assert "top_pairs" in entry and isinstance(entry["top_pairs"], list)


def test_e2e_sweep_smoke_contracts(tmp_path: Path):
    run_id = "run_smoke"
    _mk_run_valid(tmp_path, run_id)
    stage_dir = _run_sweep(tmp_path, run_id, ns="8,16,32", sigmas="0,0.05")
    _validate_contract(stage_dir)


def test_e2e_sweep_heavy_optional(tmp_path: Path):
    if os.environ.get("BASURIN_E2E_HEAVY") != "1":
        return
    run_id = "run_heavy"
    _mk_run_valid(tmp_path, run_id)
    stage_dir = _run_sweep(tmp_path, run_id, ns="8,16,32,64,128", sigmas="0,0.05")
    _validate_contract(stage_dir)
    verdict = read_json(stage_dir / "outputs" / "verdict.json")
    assert isinstance(verdict["thesis"]["thesis_gate_pass"], bool)
    assert "interpretation" in verdict
