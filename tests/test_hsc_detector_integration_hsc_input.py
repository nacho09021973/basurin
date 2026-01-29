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


def _run_hsc_input(runs_root: Path, run_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(HSC_INPUT_STAGE),
            "--run",
            run_id,
            "--runs-root",
            str(runs_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_hsc_detector(runs_root: Path, run_id: str, input_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(HSC_DETECTOR_STAGE),
            "--run",
            run_id,
            "--input",
            str(input_path),
            "--out-root",
            str(runs_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_hsc_input_feeds_hsc_detector_missing_only_ope(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "hsc-input-detector"
    run_dir = runs_root / run_id

    _write_run_valid(run_dir, verdict="PASS")
    write_minimal_canonical_features(run_dir, n=2, dx=2, dy=2, feature_key="t", seed=321)
    _write_spectrum(run_dir, [1.1, 2.2])

    result_input = _run_hsc_input(runs_root, run_id)
    assert result_input.returncode == 0, result_input.stderr

    input_path = run_dir / "HSC_INPUT" / "outputs" / "input.json"
    assert input_path.exists()

    result_detector = _run_hsc_detector(runs_root, run_id, input_path)
    assert result_detector.returncode == 0, result_detector.stderr

    verdict_path = (
        run_dir / "experiment" / "hsc_detector" / "outputs" / "verdict.json"
    )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["missing_inputs"] == ["ope_coefficients"]
