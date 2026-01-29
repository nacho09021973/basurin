from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
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


def _write_spectrum_no_phi(run_dir: Path) -> None:
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spectrum_path, "w") as h5:
        h5.create_dataset("delta_uv", data=[1.1, 2.2])
        h5.create_dataset("z_grid", data=[0.0, 1.0, 2.0])


def _write_spectrum_with_phi(run_dir: Path) -> None:
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spectrum_path, "w") as h5:
        h5.create_dataset("delta_uv", data=[1.1])
        h5.create_dataset("z_grid", data=[0.0, 1.0, 2.0])
        h5.create_dataset("phi", data=[[1.0, 1.0, 1.0]])


def _run_stage(runs_root: Path, run_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(STAGE_PATH),
            "--run",
            run_id,
            "--runs-root",
            str(runs_root),
            "--n-light",
            "1",
            "--n-tower",
            "2",
            "--integration",
            "trapz",
            "--measure",
            "flat",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage_under_determined_without_phi(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "ope-no-phi"
    run_dir = runs_root / run_id

    _write_run_valid(run_dir, verdict="PASS")
    _write_spectrum_no_phi(run_dir)

    result = _run_stage(runs_root, run_id)
    assert result.returncode == 0, result.stderr

    verdict_path = (
        run_dir
        / "experiment"
        / "ope_coefficients_bulk_overlap"
        / "outputs"
        / "verdict.json"
    )
    assert verdict_path.exists()
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["overall_verdict"] == "UNDERDETERMINED"
    assert verdict["missing_inputs"] == ["phi"]


def test_stage_pass_with_phi(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "ope-with-phi"
    run_dir = runs_root / run_id

    _write_run_valid(run_dir, verdict="PASS")
    _write_spectrum_with_phi(run_dir)

    result = _run_stage(runs_root, run_id)
    assert result.returncode == 0, result.stderr

    output_path = (
        run_dir
        / "experiment"
        / "ope_coefficients_bulk_overlap"
        / "outputs"
        / "ope_coefficients.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    coeffs = payload["ope_coefficients"]
    assert coeffs
    assert coeffs["op_000_op_000_op_000"] == 2.0
    assert coeffs["op_000_op_000_[op_000 op_000]_0"] == 2.0
    assert coeffs["op_000_op_000_[op_000 op_000]_1"] == 1.0
