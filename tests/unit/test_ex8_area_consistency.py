from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mvp.experiment_ex8_area_consistency import (
    classify_consistency,
    compute_horizon_area,
    invert_kerr_qnm,
    main,
    propagate_uncertainties,
)
from mvp.kerr_qnm_fits import MSUN_S, kerr_Q, kerr_omega_dimless


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _create_mock_run(tmp_path: Path, *, two_mode_preferred: bool = True, include_221: bool = True):
    run_id = "test_ex8_run"
    run_dir = tmp_path / "runs" / run_id

    _write_json(
        run_dir / "RUN_VALID" / "verdict.json",
        {"verdict": "PASS", "run_id": run_id, "created": "2026-01-01T00:00:00Z"},
    )

    _write_json(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "schema_version": "mvp_estimates_v2",
            "event_id": "GWTEST",
            "combined": {"f_hz": 251.0, "Q": 4.3},
            "combined_uncertainty": {"sigma_f_hz": 5.0, "sigma_Q": 0.8},
        },
    )

    mc = {
        "schema_version": "model_comparison_v1",
        "two_mode_preferred": two_mode_preferred,
        "decision": {"two_mode_preferred": two_mode_preferred},
    }
    if include_221:
        mc.update(
            {
                "f_hz_221": 280.0,
                "Q_221": 2.1,
                "sigma_f_hz_221": 30.0,
                "sigma_Q_221": 1.5,
            }
        )
    _write_json(run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json", mc)

    return run_id, str(tmp_path / "runs")


def _run_cli(run_id: str, runs_root: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = runs_root
    return subprocess.run(
        [sys.executable, "mvp/experiment_ex8_area_consistency.py", "--run", run_id],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_schwarzschild_area():
    assert abs(compute_horizon_area(10.0, 0.0) - (16.0 * math.pi * 100.0)) < 1e-6


def test_area_monotonicity_in_spin():
    vals = [compute_horizon_area(50.0, c) for c in (0.0, 0.3, 0.6, 0.9, 0.99)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_inversion_schwarzschild():
    q0 = kerr_Q(0.0, (2, 2, 0))
    inv = invert_kerr_qnm(250.0, q0, 2, 2, 0)
    assert inv["converged"] is True
    assert abs(inv["chi"] - 0.0) < 0.01


def test_inversion_roundtrip():
    chi = 0.67
    M = 68.5
    f = kerr_omega_dimless(chi, (2, 2, 0)) / (2.0 * math.pi * M * MSUN_S)
    Q = kerr_Q(chi, (2, 2, 0))
    inv = invert_kerr_qnm(f, Q, 2, 2, 0)
    assert inv["converged"] is True
    assert abs(inv["M_solar"] - M) < 0.01
    assert abs(inv["chi"] - chi) < 0.001


def test_inversion_out_of_range():
    inv = invert_kerr_qnm(250.0, 100000.0, 2, 2, 0)
    assert inv["converged"] is False


def test_consistency_classification():
    assert classify_consistency(0.5, (2.0, 3.0)) == "CONSISTENT"
    assert classify_consistency(2.5, (2.0, 3.0)) == "TENSION"
    assert classify_consistency(4.0, (2.0, 3.0)) == "INCONSISTENT"


def test_mode_221_not_viable(tmp_path: Path):
    run_id, runs_root = _create_mock_run(tmp_path, two_mode_preferred=False)
    cp = _run_cli(run_id, runs_root)
    assert cp.returncode == 0

    payload = json.loads(
        (Path(runs_root) / run_id / "experiment_ex8_area_consistency" / "outputs" / "area_consistency.json").read_text(encoding="utf-8")
    )
    assert payload["viability"]["mode_221_viable"] is False
    assert payload["consistency"]["status"] == "MODE_221_NOT_VIABLE"


def test_determinism(tmp_path: Path):
    run_id, runs_root = _create_mock_run(tmp_path)
    cp1 = _run_cli(run_id, runs_root)
    assert cp1.returncode == 0
    p = Path(runs_root) / run_id / "experiment_ex8_area_consistency" / "outputs" / "area_consistency.json"
    a = json.loads(p.read_text(encoding="utf-8"))

    cp2 = _run_cli(run_id, runs_root)
    assert cp2.returncode == 0
    b = json.loads(p.read_text(encoding="utf-8"))

    a.pop("created", None)
    b.pop("created", None)
    assert a == b


def test_schema_fields_present(tmp_path: Path):
    run_id, runs_root = _create_mock_run(tmp_path)
    cp = _run_cli(run_id, runs_root)
    assert cp.returncode == 0
    payload = json.loads(
        (Path(runs_root) / run_id / "experiment_ex8_area_consistency" / "outputs" / "area_consistency.json").read_text(encoding="utf-8")
    )
    for k in ("schema_version", "run_id", "mode_220", "mode_221", "consistency", "hawking_area", "viability", "created"):
        assert k in payload
    for k in ("f_hz", "Q", "M_solar", "chi", "area_GM2", "sigma_f_hz", "sigma_Q", "sigma_M_solar", "sigma_chi", "sigma_area_GM2", "inversion_converged"):
        assert k in payload["mode_220"]


def test_abort_on_missing_estimates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    run_id = "test_ex8_run"
    run_dir = tmp_path / "runs" / run_id
    _write_json(
        run_dir / "RUN_VALID" / "verdict.json",
        {"verdict": "PASS", "run_id": run_id, "created": "2026-01-01T00:00:00Z"},
    )
    _write_json(
        run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json",
        {"schema_version": "model_comparison_v1", "two_mode_preferred": False},
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path / "runs"))
    with pytest.raises(SystemExit) as ex:
        main(["--run", run_id])
    assert ex.value.code == 2
    captured = capsys.readouterr()
    assert "estimates.json" in captured.err

