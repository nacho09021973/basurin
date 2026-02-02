from __future__ import annotations

import json
import subprocess
from pathlib import Path

def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def test_basurin_where_reports_missing_run_dir(tmp_path: Path) -> None:
    # Repo fake: tool usa CWD; creamos estructura mínima
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    # Copia el tool desde el repo real (asumimos ejecución desde repo real en CI local)
    # En tu entorno, este test corre dentro del repo real, así que invocamos el tool por ruta relativa.
    res = subprocess.run(
        ["python", "tools/basurin_where.py", "--run", "no_such_run", "--out-root", str(tmp_path / "runs"), "--ringdown-min"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2
    assert "STATUS: MISSING_RUN_DIR" in res.stdout

def test_basurin_where_ready_when_run_valid_and_synth_present(tmp_path: Path) -> None:
    run_id = "2026-02-01__unit_test__where"
    run_dir = tmp_path / "runs" / run_id

    # RUN_VALID
    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _write_json(run_dir / "RUN_VALID" / "stage_summary.json", {"results": {"overall_verdict": "PASS"}})
    (run_dir / "RUN_VALID" / "manifest.json").write_text("{}", encoding="utf-8")

    # ringdown_synth
    _write_json(run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json", {"truth": {"f_220": 250.0, "tau_220": 0.02}})
    _write_json(run_dir / "ringdown_synth" / "stage_summary.json", {"stage": "ringdown_synth"})
    (run_dir / "ringdown_synth" / "manifest.json").write_text("{}", encoding="utf-8")

    res = subprocess.run(
        ["python", "tools/basurin_where.py", "--run", run_id, "--out-root", str(tmp_path / "runs"), "--ringdown-min"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "READY: YES" in res.stdout
