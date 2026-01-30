import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "experiment" / "run_valid" / "stage_run_valid.py"


def test_run_valid_emits_contract_and_passes(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_id = "rv_ok"
    (out_root / run_id).mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [sys.executable, str(STAGE), "--run", run_id, "--out-root", str(out_root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    stage_dir = out_root / run_id / "RUN_VALID"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    outp = stage_dir / "outputs" / "run_valid.json"
    assert (stage_dir / "verdict.json").exists()
    assert outp.exists()
    data = json.loads(outp.read_text(encoding="utf-8"))
    assert data["overall_verdict"] == "PASS"


def test_run_valid_fails_on_missing_required_path(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_id = "rv_fail"
    (out_root / run_id).mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [
            sys.executable,
            str(STAGE),
            "--run",
            run_id,
            "--out-root",
            str(out_root),
            "--require",
            "inputs/does_not_exist.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
