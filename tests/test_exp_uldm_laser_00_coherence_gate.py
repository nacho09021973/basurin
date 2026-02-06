from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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

    rc1 = main(["--run", run_id, "--seed", "123", "--tag", "det-tag"])
    assert rc1 == 0
    stats_path = (
        tmp_path
        / "runs"
        / run_id
        / "experiment"
        / "uldm_laser_00"
        / "det-tag"
        / "outputs"
        / "stats.json"
    )
    h1 = _sha256(stats_path)

    rc2 = main(["--run", run_id, "--seed", "123", "--tag", "det-tag"])
    assert rc2 == 0
    h2 = _sha256(stats_path)
    assert h1 == h2


def test_detection_gap(tmp_path: Path, monkeypatch) -> None:
    run_id = "gap-run"
    _make_minimal_run(tmp_path, run_id, run_valid=True)
    monkeypatch.chdir(tmp_path)
    stats_path = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00" / "same-tag" / "outputs" / "stats.json"

    rc0 = main([
        "--run",
        run_id,
        "--seed",
        "7",
        "--a-inj",
        "0.0",
        "--tag",
        "same-tag",
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
        "--tag",
        "same-tag",
    ])
    assert rc1 == 0
    stats1 = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats1["inj"]["tpr_at_fpr_1pct"] > 0.9


def test_entrypoint_help_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiment" / "uldm_laser" / "exp_uldm_laser_00_coherence_gate.py"
    p = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "--run" in (p.stdout + p.stderr) or "usage" in (p.stdout + p.stderr).lower()


def test_tag_creates_isolated_outputs(tmp_path: Path, monkeypatch) -> None:
    run_id = "tagged-run"
    _make_minimal_run(tmp_path, run_id, run_valid=True)
    monkeypatch.chdir(tmp_path)

    assert main(["--run", run_id, "--seed", "13", "--tag", "t1"]) == 0
    assert main(["--run", run_id, "--seed", "17", "--tag", "t2"]) == 0

    p1 = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00" / "t1" / "outputs" / "stats.json"
    p2 = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00" / "t2" / "outputs" / "stats.json"
    assert p1.exists()
    assert p2.exists()
    assert _sha256(p1) != _sha256(p2)


def test_auto_tag_default_does_not_overwrite(tmp_path: Path, monkeypatch) -> None:
    run_id = "auto-tag-run"
    _make_minimal_run(tmp_path, run_id, run_valid=True)
    monkeypatch.chdir(tmp_path)

    assert main(["--run", run_id, "--seed", "1", "--a-inj", "2.0"]) == 0
    assert main(["--run", run_id, "--seed", "2", "--a-inj", "2.0"]) == 0

    base = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00"
    auto_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("auto_")])
    assert len(auto_dirs) == 2
    assert auto_dirs[0].name != auto_dirs[1].name
    for d in auto_dirs:
        assert (d / "outputs" / "stats.json").exists()


def test_invalid_tag_aborts_without_writing(tmp_path: Path, monkeypatch) -> None:
    run_id = "invalid-tag-run"
    _make_minimal_run(tmp_path, run_id, run_valid=True)
    monkeypatch.chdir(tmp_path)

    for bad in ("bad tag", "bad/seg", "bad..seg"):
        try:
            main(["--run", run_id, "--tag", bad])
        except SystemExit as exc:
            code = exc.code[0] if isinstance(exc.code, tuple) else exc.code
            assert code == 2
        else:
            raise AssertionError("Expected SystemExit(2)")

    stage_base = tmp_path / "runs" / run_id / "experiment" / "uldm_laser_00"
    assert not stage_base.exists()
