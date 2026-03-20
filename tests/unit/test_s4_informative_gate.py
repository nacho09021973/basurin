from __future__ import annotations

import json
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.s4_geometry_filter import (
    _compute_acceptance_fraction,
    _compute_informative,
    main as s4_main,
)


def _write_run_valid(run_dir: Path) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})


def _prepare_inputs(
    out_root: Path,
    run_id: str,
    *,
    atlas_entries: list[dict[str, object]],
) -> Path:
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    atlas_path = out_root / f"{run_id}_atlas.json"
    write_json_atomic(atlas_path, atlas_entries)

    estimates_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    write_json_atomic(
        estimates_path,
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
        },
    )
    return atlas_path


def _run_s4(
    monkeypatch,
    tmp_path: Path,
    *,
    run_id: str,
    atlas_entries: list[dict[str, object]],
    extra_args: list[str] | None = None,
) -> dict[str, object]:
    out_root = tmp_path / "runs"
    atlas_path = _prepare_inputs(out_root, run_id, atlas_entries=atlas_entries)

    argv = [
        "s4_geometry_filter.py",
        "--run",
        run_id,
        "--atlas-path",
        str(atlas_path),
        "--epsilon",
        "5.991",
    ]
    if extra_args:
        argv.extend(extra_args)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr("sys.argv", argv)

    rc = s4_main()
    assert rc == 0

    summary_path = out_root / run_id / "s4_geometry_filter" / "stage_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_acceptance_fraction_computation() -> None:
    result = _compute_acceptance_fraction(730, 800)
    assert abs(result - 0.9125) < 1e-4


def test_informative_true_when_below_threshold() -> None:
    acceptance_fraction = _compute_acceptance_fraction(200, 800)
    informative = _compute_informative(acceptance_fraction, 0.80)
    assert informative is True


def test_informative_false_when_saturated() -> None:
    acceptance_fraction = _compute_acceptance_fraction(730, 800)
    informative = _compute_informative(acceptance_fraction, 0.80)
    assert informative is False


def test_acceptance_fraction_in_stage_summary(monkeypatch, tmp_path: Path) -> None:
    atlas_entries = [
        {"geometry_id": f"g{i:03d}", "f_hz": 250.0, "Q": 3.14}
        for i in range(730)
    ] + [
        {"geometry_id": f"far{i:03d}", "f_hz": 500.0, "Q": 20.0}
        for i in range(70)
    ]

    summary = _run_s4(
        monkeypatch,
        tmp_path,
        run_id="run_s4_informative_summary",
        atlas_entries=atlas_entries,
    )

    results = summary["results"]
    assert "acceptance_fraction" in results
    assert "informative" in results
    assert "informative_threshold" in results
    assert isinstance(results["acceptance_fraction"], float)
    assert isinstance(results["informative"], bool)
    assert isinstance(results["informative_threshold"], float)


def test_backward_compatible_without_flag(monkeypatch, tmp_path: Path) -> None:
    atlas_entries = [
        {"geometry_id": f"g{i:03d}", "f_hz": 250.0, "Q": 3.14}
        for i in range(730)
    ] + [
        {"geometry_id": f"far{i:03d}", "f_hz": 500.0, "Q": 20.0}
        for i in range(70)
    ]

    summary = _run_s4(
        monkeypatch,
        tmp_path,
        run_id="run_s4_informative_default",
        atlas_entries=atlas_entries,
    )

    results = summary["results"]
    assert abs(results["acceptance_fraction"] - 0.9125) < 1e-4
    assert results["informative"] is False
    assert results["informative_threshold"] == 0.80
    assert results["n_atlas"] == 800
    assert results["n_compatible"] == 730


def test_filter_status_ok(monkeypatch, tmp_path: Path) -> None:
    atlas_entries = [
        {"geometry_id": f"g{i:03d}", "f_hz": 250.0, "Q": 3.14}
        for i in range(200)
    ] + [
        {"geometry_id": f"far{i:03d}", "f_hz": 500.0, "Q": 20.0}
        for i in range(600)
    ]

    summary = _run_s4(
        monkeypatch,
        tmp_path,
        run_id="run_s4_filter_status_ok",
        atlas_entries=atlas_entries,
    )

    results = summary["results"]
    assert results["filter_status"] == "OK"


def test_filter_status_saturated(monkeypatch, tmp_path: Path) -> None:
    atlas_entries = [
        {"geometry_id": f"g{i:03d}", "f_hz": 250.0, "Q": 3.14}
        for i in range(730)
    ] + [
        {"geometry_id": f"far{i:03d}", "f_hz": 500.0, "Q": 20.0}
        for i in range(70)
    ]

    summary = _run_s4(
        monkeypatch,
        tmp_path,
        run_id="run_s4_filter_status_saturated",
        atlas_entries=atlas_entries,
    )

    results = summary["results"]
    assert results["filter_status"] == "SATURATED"


def test_filter_status_empty(monkeypatch, tmp_path: Path) -> None:
    atlas_entries = [
        {"geometry_id": f"far{i:03d}", "f_hz": 500.0, "Q": 20.0}
        for i in range(800)
    ]

    summary = _run_s4(
        monkeypatch,
        tmp_path,
        run_id="run_s4_filter_status_empty",
        atlas_entries=atlas_entries,
    )

    results = summary["results"]
    assert results["n_compatible"] == 0
    assert results["filter_status"] == "EMPTY"


def test_filter_status_edge_at_threshold(monkeypatch, tmp_path: Path) -> None:
    atlas_entries = [
        {"geometry_id": f"g{i:03d}", "f_hz": 250.0, "Q": 3.14}
        for i in range(640)
    ] + [
        {"geometry_id": f"far{i:03d}", "f_hz": 500.0, "Q": 20.0}
        for i in range(160)
    ]

    summary = _run_s4(
        monkeypatch,
        tmp_path,
        run_id="run_s4_filter_status_edge",
        atlas_entries=atlas_entries,
    )

    results = summary["results"]
    assert abs(results["acceptance_fraction"] - 0.80) < 1e-9
    assert results["informative"] is True
    assert results["filter_status"] == "OK"
