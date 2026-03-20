from __future__ import annotations

import json
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.s4_geometry_filter import compute_compatible_set, main as s4_main


def _atlas_with_target_d2(d2_values: list[float]) -> list[dict[str, object]]:
    atlas: list[dict[str, object]] = []
    for i, d2 in enumerate(d2_values):
        atlas.append(
            {
                "geometry_id": f"g{i}",
                "phi_atlas": [d2 ** 0.5, 0.0],
            }
        )
    return atlas


def _compute_fixture(
    *,
    threshold_mode: str,
    epsilon: float = 2500.0,
    delta_lnl: float = 3.0,
) -> dict[str, object]:
    atlas = _atlas_with_target_d2([100, 105, 110, 120, 150, 200, 500, 800, 1000, 2000])
    return compute_compatible_set(
        1.0,
        1.0,
        atlas,
        epsilon,
        metric="mahalanobis_log",
        metric_params={"sigma_logf": 1.0, "sigma_logQ": 1.0, "cov_logf_logQ": 0.0},
        threshold_mode=threshold_mode,
        threshold_params={"delta_lnL": delta_lnl} if threshold_mode == "delta_lnL" else None,
    )


def _write_run_valid(run_dir: Path) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})


def _run_stage_summary(
    monkeypatch,
    tmp_path: Path,
    *,
    threshold_mode: str,
    delta_lnl: float = 3.0,
) -> dict[str, object]:
    out_root = tmp_path / "runs"
    run_id = f"run_s4_{threshold_mode}"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    atlas_path = tmp_path / f"{run_id}_atlas.json"
    write_json_atomic(atlas_path, _atlas_with_target_d2([100, 105, 110, 120, 150, 200, 500, 800, 1000, 2000]))
    write_json_atomic(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "event_id": "EVT",
            "combined": {"f_hz": 1.0, "Q": 1.0, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 1.0, "sigma_logQ": 1.0, "cov_logf_logQ": 0.0},
        },
    )

    argv = [
        "s4_geometry_filter.py",
        "--run",
        run_id,
        "--atlas-path",
        str(atlas_path),
        "--threshold-mode",
        threshold_mode,
    ]
    if threshold_mode == "d2":
        argv.extend(["--epsilon", "2500"])
    else:
        argv.extend(["--delta-lnL", str(delta_lnl)])

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr("sys.argv", argv)

    rc = s4_main()
    assert rc == 0
    summary_path = run_dir / "s4_geometry_filter" / "stage_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_delta_lnL_desaturates() -> None:
    out = _compute_fixture(threshold_mode="delta_lnL", delta_lnl=3.0)
    assert out["d2_min"] == 100.0
    assert out["n_compatible"] == 2
    compatible_ids = [row["geometry_id"] for row in out["compatible_geometries"]]
    assert compatible_ids == ["g0", "g1"]


def test_d2_mode_unchanged() -> None:
    out_default = _compute_fixture(threshold_mode="d2", epsilon=2500.0)
    out_explicit = _compute_fixture(threshold_mode="d2", epsilon=2500.0)

    assert out_default["n_compatible"] == 10
    assert json.dumps(out_default, sort_keys=True) == json.dumps(out_explicit, sort_keys=True)


def test_delta_lnL_always_includes_best() -> None:
    out = _compute_fixture(threshold_mode="delta_lnL", delta_lnl=0.0)
    compatible_ids = {row["geometry_id"] for row in out["compatible_geometries"]}
    assert "g0" in compatible_ids


def test_delta_lnL_registered_in_stage_summary(monkeypatch, tmp_path: Path) -> None:
    summary = _run_stage_summary(
        monkeypatch,
        tmp_path,
        threshold_mode="delta_lnL",
        delta_lnl=5.0,
    )

    results = summary["results"]
    assert results["threshold_mode"] == "delta_lnL"
    assert results["delta_lnL_threshold"] == 5.0
    assert results["d2_min"] == 100.0
    assert results["d2_at_best"] == 100.0


def test_delta_lnL_field_present_even_in_d2_mode() -> None:
    out = _compute_fixture(threshold_mode="d2", epsilon=2500.0)
    assert all("delta_lnL" in row for row in out["compatible_geometries"])
