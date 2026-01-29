from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py

from tests._helpers_features import write_minimal_canonical_features


REPO_ROOT = Path(__file__).resolve().parents[1]
HSC_INPUT_STAGE = REPO_ROOT / "stages" / "stage_hsc_input.py"
HSC_DETECTOR_STAGE = REPO_ROOT / "experiment" / "hsc_detector" / "stage_hsc_detector.py"
OPE_STAGE = (
    REPO_ROOT
    / "experiment"
    / "ope_coefficients_bulk_overlap"
    / "stage_ope_coefficients_bulk_overlap.py"
)


def _write_run_valid(run_dir: Path, verdict: str = "PASS") -> None:
    outputs_dir = run_dir / "RUN_VALID" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run": run_dir.name, "verdict": verdict}
    (outputs_dir / "run_valid.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_spectrum_with_phi(run_dir: Path) -> None:
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spectrum_path, "w") as h5:
        h5.create_dataset("delta_uv", data=[1.1])
        h5.create_dataset("z_grid", data=[0.0, 1.0, 2.0])
        h5.create_dataset("phi", data=[[1.0, 1.0, 1.0]])


def _run_stage(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_hsc_input_injects_ope_coefficients(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "hsc-with-ope"
    run_dir = runs_root / run_id

    _write_run_valid(run_dir, verdict="PASS")
    write_minimal_canonical_features(run_dir, n=2, dx=2, dy=2, feature_key="t", seed=321)
    _write_spectrum_with_phi(run_dir)

    result_ope = _run_stage(
        [
            sys.executable,
            str(OPE_STAGE),
            "--run",
            run_id,
            "--runs-root",
            str(runs_root),
            "--n-light",
            "1",
            "--n-tower",
            "1",
        ]
    )
    assert result_ope.returncode == 0, result_ope.stderr

    result_input = _run_stage(
        [
            sys.executable,
            str(HSC_INPUT_STAGE),
            "--run",
            run_id,
            "--runs-root",
            str(runs_root),
        ]
    )
    assert result_input.returncode == 0, result_input.stderr

    input_path = run_dir / "HSC_INPUT" / "outputs" / "input.json"
    assert input_path.exists()

    result_detector = _run_stage(
        [
            sys.executable,
            str(HSC_DETECTOR_STAGE),
            "--run",
            run_id,
            "--input",
            str(input_path),
            "--out-root",
            str(runs_root),
        ]
    )
    assert result_detector.returncode == 0, result_detector.stderr

    verdict_path = (
        run_dir / "experiment" / "hsc_detector" / "outputs" / "verdict.json"
    )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["missing_inputs"] is None
