from __future__ import annotations

import json
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.s4_geometry_filter import compute_compatible_set, main as s4_main
from mvp.schemas import validate_compatible_set


def _minimal_fixture() -> dict:
    return {
        "schema_version": "mvp_compatible_set_v1",
        "observables": {"f_hz": 250.0, "Q": 4.0},
        "metric": "euclidean_log",
        "metric_params": {},
        "epsilon": 0.3,
        "n_atlas": 2,
        "n_compatible": 1,
        "compatible_geometries": [{"geometry_id": "g1", "distance": 0.0, "compatible": True}],
        "ranked_all": [{"geometry_id": "g1", "distance": 0.0, "compatible": True}],
        "bits_excluded": 1.0,
        "bits_kl": 1.0,
        "likelihood_stats": None,
    }


def test_validate_minimal_fixture() -> None:
    ok, errors = validate_compatible_set(_minimal_fixture())
    assert ok is True
    assert errors == []


def test_validate_missing_key() -> None:
    data = _minimal_fixture()
    data.pop("metric")

    ok, errors = validate_compatible_set(data)

    assert ok is False
    assert "missing required key: metric" in errors


def test_validate_strict_mahalanobis() -> None:
    data = _minimal_fixture()
    data["metric"] = "mahalanobis_log"

    ok, errors = validate_compatible_set(data, strict_mahalanobis=True)

    assert ok is False
    assert "missing mahalanobis key: threshold_d2" in errors
    assert "missing mahalanobis key: d2_min" in errors
    assert "missing mahalanobis key: distance" in errors
    assert "missing mahalanobis key: covariance_logspace" in errors


def test_validate_real_output() -> None:
    atlas = [
        {"geometry_id": "g1", "f_hz": 250.0, "Q": 4.0},
        {"geometry_id": "g2", "f_hz": 310.0, "Q": 6.0},
    ]
    out = compute_compatible_set(
        250.0,
        4.0,
        atlas,
        5.991,
        metric="mahalanobis_log",
        metric_params={"sigma_logf": 0.1, "sigma_logQ": 0.2},
    )

    ok, errors = validate_compatible_set(out, strict_mahalanobis=True)
    assert ok is True
    assert errors == []


def test_s4_self_check_warns(monkeypatch, tmp_path: Path, capsys) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_schema_warn"
    run_dir = out_root / run_id

    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    atlas_path = tmp_path / "atlas.json"
    write_json_atomic(atlas_path, [{"geometry_id": "g1", "f_hz": 250.0, "Q": 4.0}])
    write_json_atomic(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 4.0, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
        },
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "mvp.s4_geometry_filter.validate_compatible_set",
        lambda data, strict_mahalanobis=False: (False, ["forced self-check error"]),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--atlas-path",
            str(atlas_path),
        ],
    )

    rc = s4_main()
    captured = capsys.readouterr()

    assert rc == 0
    assert "WARNING: compatible_set self-check failed" in captured.err
    assert "forced self-check error" in captured.err


def test_diagnostics_empty_status() -> None:
    atlas = [
        {"geometry_id": "g1", "f_hz": 260.0, "Q": 4.5},
        {"geometry_id": "g2", "f_hz": 270.0, "Q": 4.8},
    ]
    out = compute_compatible_set(
        250.0,
        4.0,
        atlas,
        1e-9,
        metric="mahalanobis_log",
        metric_params={"sigma_logf": 0.1, "sigma_logQ": 0.2},
    )

    diagnostics = out["diagnostics"]
    assert out["n_compatible"] == 0
    assert diagnostics["acceptance_fraction"] == 0.0
    assert diagnostics["informative_status"] == "EMPTY"
    assert diagnostics["d2_iqr"] is not None
    assert diagnostics["d2_range"] is not None


def test_diagnostics_saturated_status() -> None:
    atlas = [
        {"geometry_id": "g1", "f_hz": 250.0, "Q": 4.0},
        {"geometry_id": "g2", "f_hz": 250.2, "Q": 4.1},
        {"geometry_id": "g3", "f_hz": 249.9, "Q": 3.9},
        {"geometry_id": "g4", "f_hz": 250.1, "Q": 4.0},
        {"geometry_id": "g5", "f_hz": 249.8, "Q": 4.2},
    ]
    out = compute_compatible_set(
        250.0,
        4.0,
        atlas,
        100.0,
        metric="mahalanobis_log",
        metric_params={"sigma_logf": 0.1, "sigma_logQ": 0.2},
    )

    diagnostics = out["diagnostics"]
    assert out["n_atlas"] == 5
    assert out["n_compatible"] == 5
    assert diagnostics["acceptance_fraction"] > 0.80
    assert diagnostics["informative_status"] == "SATURATED"


def test_diagnostics_ok_status_and_determinism() -> None:
    atlas = [
        {"geometry_id": "g1", "f_hz": 250.0, "Q": 4.0},
        {"geometry_id": "g2", "f_hz": 268.0, "Q": 4.3},
        {"geometry_id": "g3", "f_hz": 360.0, "Q": 8.0},
        {"geometry_id": "g4", "f_hz": 390.0, "Q": 10.0},
        {"geometry_id": "g5", "f_hz": 420.0, "Q": 12.0},
    ]
    kwargs = {
        "f_obs": 250.0,
        "Q_obs": 4.0,
        "atlas": atlas,
        "epsilon": 1.6,
        "metric": "mahalanobis_log",
        "metric_params": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
    }
    out_a = compute_compatible_set(**kwargs)
    out_b = compute_compatible_set(**kwargs)

    diagnostics = out_a["diagnostics"]
    assert out_a["n_atlas"] == 5
    assert out_a["n_compatible"] == 2
    assert diagnostics["acceptance_fraction"] == 0.4
    assert diagnostics["informative_status"] == "OK"
    assert diagnostics["d2_quantiles"]["p10"] is not None
    assert diagnostics["d2_quantiles"]["p90"] is not None

    assert json.dumps(out_a, sort_keys=True) == json.dumps(out_b, sort_keys=True)
