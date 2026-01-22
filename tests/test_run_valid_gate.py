import json
import shutil
import subprocess
from pathlib import Path


def _run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False)


def _prepare_run(run_id: str) -> None:
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    _run_cmd(["python", "01_genera_ads_puro.py", "--run", run_id]).check_returncode()
    _run_cmd(["python", "03_sturm_liouville.py", "--run", run_id]).check_returncode()
    _run_cmd(["python", "04_diccionario.py", "--run", run_id]).check_returncode()


def test_contract_run_valid_pass_smoke() -> None:
    run_id = "2026-01-22__pytest_smoke"
    _prepare_run(run_id)
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
