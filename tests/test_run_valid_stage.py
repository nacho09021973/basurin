from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_valid_creates_verdict_and_manifest(tmp_path: Path):
    # repo-like layout
    work = tmp_path / "work"
    work.mkdir()
    (work / "runs").mkdir()
    (work / "stages").mkdir()

    # minimal geometry stage to satisfy RUN_VALID
    run = "test_run"
    g = work / "runs" / run / "geometry"
    (g / "outputs").mkdir(parents=True)
    (g / "outputs" / "dummy.h5").write_bytes(b"H5")

    (g / "manifest.json").write_text("{}", encoding="utf-8")
    (g / "stage_summary.json").write_text("{}", encoding="utf-8")

    # write stage file under test env (copied from repo in real usage)
    stage_py = Path("experiment/run_valid/stage_run_valid.py").resolve()
    assert stage_py.exists(), "missing experiment/run_valid/stage_run_valid.py in repo"

    env = dict(**{**dict(), "BASURIN_RUNS_ROOT": str(work / "runs")})
    p = subprocess.run(
        [sys.executable, str(stage_py), "--run", run],
        cwd=str(work),
        env={**env, **dict()},
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr

    rv = work / "runs" / run / "RUN_VALID"
    assert (rv / "verdict.json").exists()
    assert (rv / "manifest.json").exists()
    assert (rv / "stage_summary.json").exists()
    assert (rv / "outputs" / "run_valid.json").exists()

    verdict = json.loads((rv / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "PASS"
