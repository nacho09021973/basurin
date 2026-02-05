from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

from experiment.uldm_laser.exp_uldm_laser_00_coherence_gate import main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_minimal_run(tmp_path: Path, run_id: str, run_valid: bool = True) -> Path:
    run_dir = tmp_path / "runs" / run_id
    spec_dir = run_dir / "spectrum" / "outputs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(spec_dir / "spectrum.h5", "w") as h5:
        h5.create_dataset("delta_uv", data=[1.2, 1.8, 2.5, 3.0, 3.4])

    if run_valid:
        _write_json(
            run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
            {"verdict": "PASS"},
        )
    return run_dir


def test_abort_if_run_invalid(tmp_path: Path, monkeypatch) -> None:
    run_id = "invalid-run"
    _make_minimal_run(tmp_path, run_id, run_valid=False)
    monkeypatch.chdir(tmp_path)

    try:
        main(["--run", run_id])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit(2)")

    assert not (tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00").exists()


def test_determinism_hash(tmp_path: Path, monkeypatch) -> None:
    run_id = "det-run"
    _make_minimal_run(tmp_path, run_id, run_valid=True)
    monkeypatch.chdir(tmp_path)

    rc1 = main(["--run", run_id, "--seed", "123"])
    assert rc1 == 0
    stats_path = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00" / "outputs" / "stats.json"
    h1 = _sha256(stats_path)

    rc2 = main(["--run", run_id, "--seed", "123"])
    assert rc2 == 0
    h2 = _sha256(stats_path)
    assert h1 == h2


def test_detection_gap(tmp_path: Path, monkeypatch) -> None:
    run_id = "gap-run"
    _make_minimal_run(tmp_path, run_id, run_valid=True)
    monkeypatch.chdir(tmp_path)
    stats_path = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00" / "outputs" / "stats.json"

    rc0 = main([
        "--run",
        run_id,
        "--seed",
        "7",
        "--a-inj",
        "0.0",
    ])
    assert rc0 == 0
    stats0 = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats0["null"]["p99"] < stats0["thresholds"]["threshold_null_max"]

    rc1 = main([
        "--run",
        run_id,
        "--seed",
        "7",
        "--a-inj",
        "3.0",
    ])
    assert rc1 == 0
    stats1 = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats1["inj"]["tpr_at_fpr_1pct"] > 0.9
