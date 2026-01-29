from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py

from tests._helpers_features import write_minimal_canonical_features


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = REPO_ROOT / "stages" / "stage_hsc_input.py"


def _write_run_valid(run_dir: Path, verdict: str = "PASS") -> None:
    outputs_dir = run_dir / "RUN_VALID" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run": run_dir.name, "verdict": verdict}
    (outputs_dir / "run_valid.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_spectrum(run_dir: Path, values: list[float]) -> None:
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spectrum_path, "w") as h5:
        h5.create_dataset("delta_uv", data=values)


def _run_stage(runs_root: Path, run_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(STAGE_PATH),
            "--run",
            run_id,
            "--runs-root",
            str(runs_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_hsc_input_stage_builds_input_json(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "hsc-input"
    run_dir = runs_root / run_id

    _write_run_valid(run_dir, verdict="PASS")
    write_minimal_canonical_features(run_dir, n=2, dx=3, dy=4, feature_key="t", seed=123)
    _write_spectrum(run_dir, [1.2, 2.4, 3.6])

    result = _run_stage(runs_root, run_id)
    assert result.returncode == 0, result.stderr

    input_path = run_dir / "HSC_INPUT" / "outputs" / "input.json"
    assert input_path.exists()
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    assert "metadata" in payload
    assert "features" in payload
    assert "spectrum" in payload
    operators = payload["spectrum"].get("operators", [])
    assert operators


def test_hsc_input_stage_accepts_run_valid_v1(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "hsc-input-v1"
    run_dir = runs_root / run_id

    outputs_dir = run_dir / "RUN_VALID" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run": run_dir.name, "schema_version": "run_valid_v1", "overall_verdict": "PASS"}
    (outputs_dir / "run_valid.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_minimal_canonical_features(run_dir, n=1, dx=2, dy=2, feature_key="t", seed=42)
    _write_spectrum(run_dir, [1.0, 2.0])

    result = _run_stage(runs_root, run_id)
    assert result.returncode == 0, result.stderr

    input_path = run_dir / "HSC_INPUT" / "outputs" / "input.json"
    assert input_path.exists()
