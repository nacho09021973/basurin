from __future__ import annotations

import hashlib
import json
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.s4_geometry_filter import main as s4_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_valid(run_dir: Path) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})


def _prepare_inputs(base_dir: Path, run_id: str) -> tuple[Path, Path]:
    run_dir = base_dir / run_id
    _write_run_valid(run_dir)

    atlas_path = base_dir / "atlas.json"
    write_json_atomic(
        atlas_path,
        [
            {"geometry_id": "g1", "f_hz": 250.0, "Q": 3.14},
            {"geometry_id": "g2", "f_hz": 300.0, "Q": 4.00},
        ],
    )

    estimates_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    write_json_atomic(
        estimates_path,
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
        },
    )

    return atlas_path, estimates_path


def test_s4_stage_name_prevents_overwrite(monkeypatch, tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s4_stage_name"
    run_dir = out_root / run_id
    atlas_path, _ = _prepare_inputs(out_root, run_id)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--atlas-path",
            str(atlas_path),
            "--epsilon",
            "0.6",
        ],
    )
    rc = s4_main()
    assert rc == 0

    baseline_path = run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    assert baseline_path.exists()
    baseline_sha = _sha256(baseline_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--stage-name",
            "s4_spectral_geometry_filter",
            "--atlas-path",
            str(atlas_path),
            "--epsilon",
            "0.7",
        ],
    )
    rc = s4_main()
    assert rc == 0

    spectral_path = run_dir / "s4_spectral_geometry_filter" / "outputs" / "compatible_set.json"
    assert spectral_path.exists()
    assert _sha256(baseline_path) == baseline_sha

    baseline_summary = json.loads((run_dir / "s4_geometry_filter" / "stage_summary.json").read_text(encoding="utf-8"))
    assert baseline_summary["outputs"][0]["sha256"] == baseline_sha


def test_s4_invalid_stage_name_fails_before_writing(monkeypatch, tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s4_invalid_stage"
    run_dir = out_root / run_id
    atlas_path, _ = _prepare_inputs(out_root, run_id)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--stage-name",
            "invalid_name",
            "--atlas-path",
            str(atlas_path),
        ],
    )

    rc = s4_main()
    assert rc != 0
    assert not (run_dir / "invalid_name").exists()
