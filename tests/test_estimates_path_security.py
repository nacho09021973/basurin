from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from basurin_io import write_json_atomic
from mvp.s4_geometry_filter import main as s4_main


def _write_run_valid(run_dir: Path) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})


def _install_numpy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    if "numpy" in sys.modules:
        return
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))


def test_s6_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s6_reject_traversal"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    write_json_atomic(run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json", {"compatible_geometries": []})

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s6_information_geometry.py",
            "--run",
            run_id,
            "--estimates-path",
            "../../evil.json",
        ],
    )

    _install_numpy_stub(monkeypatch)
    from mvp.s6_information_geometry import main as s6_main

    with pytest.raises(SystemExit) as exc:
        s6_main()

    assert exc.value.code == 2
    verdict = json.loads((run_dir / "s6_information_geometry" / "stage_summary.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL"
    assert "escapes run directory" in verdict["error"]


def test_s4_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s4_reject_traversal"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    atlas_path = tmp_path / "atlas.json"
    write_json_atomic(atlas_path, [{"geometry_id": "g1", "f_hz": 250.0, "Q": 3.14}])

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--atlas-path",
            str(atlas_path),
            "--estimates-path",
            "../../evil.json",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        s4_main()

    assert exc.value.code == 2
    verdict = json.loads((run_dir / "s4_geometry_filter" / "stage_summary.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL"
    assert "escapes run directory" in verdict["error"]


def test_s6_rejects_absolute_path_outside_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s6_reject_absolute"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    write_json_atomic(run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json", {"compatible_geometries": []})
    outside_path = tmp_path / "outside_estimates.json"
    write_json_atomic(outside_path, {"combined": {"f_hz": 1.0, "Q": 1.0}})

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s6_information_geometry.py",
            "--run",
            run_id,
            "--estimates-path",
            str(outside_path),
        ],
    )

    _install_numpy_stub(monkeypatch)
    from mvp.s6_information_geometry import main as s6_main

    with pytest.raises(SystemExit) as exc:
        s6_main()

    assert exc.value.code == 2
    verdict = json.loads((run_dir / "s6_information_geometry" / "stage_summary.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL"
    assert "escapes run directory" in verdict["error"]


def test_s6_estimates_override_records_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s6_override_sha"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    estimates_rel = Path("s3_spectral_estimates/outputs/spectral_estimates.json")
    estimates_path = run_dir / estimates_rel
    write_json_atomic(
        estimates_path,
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 3.14, "snr_peak": 12.0},
            "per_detector": {},
        },
    )
    expected_sha = hashlib.sha256(estimates_path.read_bytes()).hexdigest()

    write_json_atomic(
        run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        {"compatible_geometries": [{"geometry_id": "g1", "f_hz": 260.0, "Q": 3.5}]},
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s6_information_geometry.py",
            "--run",
            run_id,
            "--estimates-path",
            str(estimates_rel),
        ],
    )

    _install_numpy_stub(monkeypatch)
    from mvp.s6_information_geometry import main as s6_main

    rc = s6_main()
    assert rc == 0

    summary = json.loads((run_dir / "s6_information_geometry" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["estimates_path_override"] == str(estimates_rel)
    inputs = {row["label"]: row for row in summary["inputs"]}
    assert inputs["estimates"]["path"] == str(estimates_rel)
    assert inputs["estimates"]["sha256"] == expected_sha


def test_s4_estimates_override_records_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_s4_override_sha"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    atlas_path = tmp_path / "atlas.json"
    write_json_atomic(atlas_path, [{"geometry_id": "g1", "f_hz": 250.0, "Q": 3.14}])

    # s3 default output is now a required_inputs contract field; create it so the
    # contract check passes even when an override is provided.
    default_estimates = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    write_json_atomic(
        default_estimates,
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
        },
    )

    estimates_rel = Path("s3_spectral_estimates/outputs/spectral_estimates.json")
    estimates_path = run_dir / estimates_rel
    write_json_atomic(
        estimates_path,
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
        },
    )
    expected_sha = hashlib.sha256(estimates_path.read_bytes()).hexdigest()

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--atlas-path",
            str(atlas_path),
            "--estimates-path",
            str(estimates_rel),
        ],
    )

    rc = s4_main()
    assert rc == 0

    summary = json.loads((run_dir / "s4_geometry_filter" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["estimates_path_override"] == str(estimates_rel)
    inputs = {row["label"]: row for row in summary["inputs"]}
    assert inputs["estimates"]["path"] == str(estimates_rel)
    assert inputs["estimates"]["sha256"] == expected_sha
