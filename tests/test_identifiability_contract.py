import json
import subprocess
import sys
from pathlib import Path


def _run_contract(tmp_path: Path, run_id: str) -> subprocess.CompletedProcess:
    script_path = Path(__file__).resolve().parents[1] / "tools" / "contract_identifiability.py"
    return subprocess.run(
        [sys.executable, str(script_path), "--run", run_id],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_input(tmp_path: Path, run_id: str, payload: dict) -> Path:
    input_path = (
        tmp_path
        / "runs"
        / run_id
        / "bridge_f4_1_alignment"
        / "outputs"
        / "degeneracy_per_point.json"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    return input_path


def test_identifiability_under_determined(tmp_path: Path) -> None:
    run_id = "run-under-determined"
    _write_input(
        tmp_path,
        run_id,
        {
            "degeneracy_index": 500.0,
            "stability_score": 0.95,
        },
    )

    result = _run_contract(tmp_path, run_id)
    assert result.returncode == 0, result.stderr

    output_path = (
        tmp_path
        / "runs"
        / run_id
        / "IDENTIFIABILITY"
        / "outputs"
        / "identifiability.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["verdict"] == "UNDERDETERMINED"
    assert payload["scientific_status"] == "NOT_IDENTIFIABLE_UNDER_CURRENT_OBSERVABLES"
    assert payload["pipeline_status"] == "OK"
    assert payload["evidence"]["degeneracy_index"]["value"] == 500.0
    assert payload["evidence"]["stability_score"]["value"] == 0.95


def test_identifiability_pass(tmp_path: Path) -> None:
    run_id = "run-pass"
    _write_input(
        tmp_path,
        run_id,
        {
            "degeneracy_index": 10.0,
            "stability_score": 0.2,
        },
    )

    result = _run_contract(tmp_path, run_id)
    assert result.returncode == 0, result.stderr

    output_path = (
        tmp_path
        / "runs"
        / run_id
        / "IDENTIFIABILITY"
        / "outputs"
        / "identifiability.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["verdict"] == "PASS"
    assert payload["scientific_status"] == "IDENTIFIABLE_UNDER_CURRENT_OBSERVABLES"
