import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HSC_STAGE = REPO_ROOT / "experiment" / "hsc_detector" / "stage_hsc_detector.py"


def _run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False)


def _prepare_run(run_id: str) -> None:
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    _run_cmd(["python", "01_genera_ads_puro.py", "--run", run_id]).check_returncode()
    _run_cmd(["python", "03_sturm_liouville.py", "--run", run_id]).check_returncode()
    _run_cmd(
        [
            "python",
            "04_diccionario.py",
            "--run",
            run_id,
            "--k-features",
            "2",
            "--n-bootstrap",
            "0",
        ]
    ).check_returncode()


def _prepare_spectrum_only_run(run_id: str) -> None:
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    _run_cmd(
        [
            "python",
            "01_genera_neutrino_sandbox.py",
            "--run",
            run_id,
            "--n-delta",
            "8",
            "--n-modes",
            "3",
            "--noise-rel",
            "0.0",
        ]
    ).check_returncode()
    _run_cmd(
        [
            "python",
            "04_diccionario.py",
            "--run",
            run_id,
            "--k-features",
            "2",
            "--n-bootstrap",
            "0",
        ]
    ).check_returncode()


def test_contract_run_valid_pass_smoke() -> None:
    run_id = "2026-01-22__pytest_smoke"
    _prepare_run(run_id)
    result = _run_cmd(["python", "tools/contract_run_valid.py", "--run", run_id])
    assert result.returncode == 0

    report_path = Path("runs") / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["verdict"] == "PASS"


def test_contract_run_valid_pass_spectrum_only() -> None:
    run_id = "2026-01-22__pytest_spectrum_only"
    _prepare_spectrum_only_run(run_id)
    result = _run_cmd(["python", "tools/contract_run_valid.py", "--run", run_id])
    assert result.returncode == 0

    report_path = Path("runs") / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["verdict"] == "PASS"


def test_contract_run_valid_fails_missing_spectrum() -> None:
    run_id = "2026-01-22__pytest_missing_spectrum"
    _prepare_run(run_id)
    spectrum_path = Path("runs") / run_id / "spectrum" / "outputs" / "spectrum.h5"
    if spectrum_path.exists():
        spectrum_path.unlink()
    result = _run_cmd(["python", "tools/contract_run_valid.py", "--run", run_id])
    assert result.returncode == 2

    report_path = Path("runs") / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["verdict"] == "FAIL"


def _write_run_valid(run_dir: Path, verdict: str = "PASS") -> None:
    outputs_dir = run_dir / "RUN_VALID" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"overall_verdict": verdict}
    (outputs_dir / "run_valid.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _write_hsc_input(run_dir: Path) -> Path:
    input_dir = run_dir / "inputs" / "hsc"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_payload = {
        "metadata": {
            "schema_version": "hsc_input_v1",
            "theory_name": "Test Theory",
            "d": 3.0,
            "conventions": {
                "light_ops": ["sigma", "epsilon"],
                "tower_ops_prefix": ["[sigma sigma]_"],
            },
        },
        "spectrum": {
            "operators": [
                {"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1},
            ]
        },
        "ope_coefficients": {"sigma_sigma_sigma": 0.5},
    }
    input_path = input_dir / "input.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    return input_path


def _run_hsc_stage(run_id: str, out_root: Path, input_path: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(HSC_STAGE),
        "--run",
        run_id,
        "--input",
        str(input_path),
        "--out-root",
        str(out_root),
    ]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_hsc_detector_aborts_when_run_valid_missing(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_valid_missing"
    run_dir = out_root / run_id
    input_path = _write_hsc_input(run_dir)

    result = _run_hsc_stage(run_id, out_root, input_path)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "RUN_VALID missing" in combined


def test_hsc_detector_runs_when_run_valid_pass(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_valid_pass"
    run_dir = out_root / run_id
    input_path = _write_hsc_input(run_dir)
    _write_run_valid(run_dir, verdict="PASS")

    result = _run_hsc_stage(run_id, out_root, input_path)
    assert result.returncode == 0
