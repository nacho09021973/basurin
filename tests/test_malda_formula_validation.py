from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "malda" / "12_validate_formula_candidates.py"
_SPEC = importlib.util.spec_from_file_location("malda_12_validate_formula_candidates", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_feature_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "is_bbh",
        "m1_src",
        "m2_src",
        "q",
        "eta",
        "chi_eff",
        "E_rad_frac",
        "af",
        "F_220_dimless",
        "f_ratio_221_220",
    ]
    rows = []
    for idx in range(6):
        m1 = 35.0 + idx
        m2 = 24.0 + idx
        q = m2 / m1
        eta = (m1 * m2) / ((m1 + m2) ** 2)
        chi_eff = -0.1 + 0.05 * idx
        rows.append(
            {
                "event_id": f"E{idx}",
                "is_bbh": "0" if idx == 0 else "1",
                "m1_src": f"{m1:.8f}",
                "m2_src": f"{m2:.8f}",
                "q": f"{q:.8f}",
                "eta": f"{eta:.8f}",
                "chi_eff": f"{chi_eff:.8f}",
                "E_rad_frac": f"{(eta / (2.973317 - chi_eff)) - 0.036214575:.8f}",
                "af": f"{np.sqrt(eta) * (chi_eff + 1.324642):.8f}",
                "F_220_dimless": f"{(np.exp(chi_eff) * eta) + 0.27300668:.8f}",
                "f_ratio_221_220": f"{(chi_eff * ((eta ** 2) - 0.020066187)) + 0.97551227:.8f}",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_discovery_summary(path: Path) -> None:
    _write_json(
        path,
        [
            {
                "target": "E_rad_frac",
                "description": "Fraction of mass radiated in GWs",
                "n_valid": 5,
                "best_equation": "(eta / (2.973317 - chi_eff)) + -0.036214575",
                "complexity": 7,
                "loss": 3.4046468e-06,
                "kan_top_features": ["q", "eta", "chi_eff"],
            },
            {
                "target": "af",
                "description": "Final BH spin",
                "n_valid": 5,
                "best_equation": "sqrt(eta) * (chi_eff + 1.324642)",
                "complexity": 7,
                "loss": 3.7857735e-05,
                "kan_top_features": ["q", "eta", "chi_eff"],
            },
            {
                "target": "F_220_dimless",
                "description": "Dimensionless QNM frequency",
                "n_valid": 5,
                "best_equation": "(exp(chi_eff) * eta) + 0.27300668",
                "complexity": 8,
                "loss": 2.8401008e-05,
                "kan_top_features": ["q", "eta", "chi_eff"],
            },
            {
                "target": "f_ratio_221_220",
                "description": "Frequency ratio",
                "n_valid": 5,
                "best_equation": "(chi_eff * (square(eta) + -0.020066187)) - -0.97551227",
                "complexity": 8,
                "loss": 1.1495182e-06,
                "kan_top_features": ["q", "eta", "chi_eff"],
            },
        ],
    )


def test_evaluate_formula_supports_derived_log_variables() -> None:
    columns = {
        "m1_src": np.asarray([30.0, 35.0], dtype=np.float64),
        "m2_src": np.asarray([20.0, 24.0], dtype=np.float64),
        "chi_eff": np.asarray([0.1, 0.2], dtype=np.float64),
    }

    prediction, variable_names = _MODULE.evaluate_formula(
        "abs((log_m2_src * -0.050042927) / ((chi_eff / 0.47722834) - log_m1_src))",
        columns,
    )

    expected = np.abs(
        (np.log(columns["m2_src"]) * -0.050042927)
        / ((columns["chi_eff"] / 0.47722834) - np.log(columns["m1_src"]))
    )
    assert variable_names == ["chi_eff", "log_m1_src", "log_m2_src"]
    assert np.allclose(prediction, expected)


def test_evaluate_formula_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="not available"):
        _MODULE.evaluate_formula("foo + 1", {"eta": np.asarray([0.25], dtype=np.float64)})


def test_main_writes_contract_artifacts_and_infers_bbh_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runsroot"
    run_id = "malda_formula_validation_smoke"
    run_dir = runs_root / run_id
    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    feature_table_path = run_dir / "experiment" / "malda_feature_table" / "outputs" / "event_features.csv"
    discovery_summary_path = run_dir / "experiment" / "malda_discovery" / "outputs" / "discovery_summary.json"
    discovery_stage_summary_path = run_dir / "experiment" / "malda_discovery" / "stage_summary.json"

    _write_feature_table(feature_table_path)
    _write_discovery_summary(discovery_summary_path)
    _write_json(
        discovery_stage_summary_path,
        {
            "stage": "malda_discovery",
            "verdict": "PASS",
            "config": {
                "bbh_only": True,
            },
        },
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(
        [
            "--run-id",
            run_id,
            "--bootstrap-samples",
            "16",
        ]
    )

    stage_dir = run_dir / "experiment" / "malda_formula_validation"
    outputs_dir = stage_dir / "outputs"
    validation_json = outputs_dir / "formula_validation.json"
    metrics_csv = outputs_dir / "formula_validation_metrics.csv"
    recommendations_json = outputs_dir / "formula_recommendations.json"
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    payload = json.loads(validation_json.read_text(encoding="utf-8"))
    metrics_rows = list(csv.DictReader(metrics_csv.open("r", encoding="utf-8")))
    stdout = capsys.readouterr().out

    assert rc == 0
    assert stage_dir.exists()
    assert validation_json.exists()
    assert metrics_csv.exists()
    assert recommendations_json.exists()
    assert (stage_dir / "manifest.json").exists()
    assert stage_summary["verdict"] == "PASS"
    assert stage_summary["config"]["bbh_only"] is True
    assert stage_summary["config"]["bbh_only_source"] == "discovery_stage_summary"
    assert stage_summary["results"]["n_targets_evaluated"] == 4
    assert stage_summary["hashes"]["formula_validation"]
    assert payload["targets"]["E_rad_frac"]["metrics"]["n_target_valid"] == 5
    assert payload["targets"]["E_rad_frac"]["recommendation"]["label"] == "EXPERIMENTAL_PRIOR_CANDIDATE"
    assert payload["targets"]["af"]["metrics"]["fit"]["r2"] > 0.999999
    assert all(Path(path).is_relative_to(run_dir) for path in stage_summary["outputs"].values())
    assert {row["target"] for row in metrics_rows} == {
        "E_rad_frac",
        "af",
        "F_220_dimless",
        "f_ratio_221_220",
    }
    for key in ("OUT_ROOT=", "STAGE_DIR=", "OUTPUTS_DIR=", "STAGE_SUMMARY=", "MANIFEST="):
        assert key in stdout


def test_main_requires_run_valid_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runsroot_gate"
    run_id = "malda_formula_validation_gate"
    run_dir = runs_root / run_id

    _write_feature_table(run_dir / "experiment" / "malda_feature_table" / "outputs" / "event_features.csv")
    _write_discovery_summary(run_dir / "experiment" / "malda_discovery" / "outputs" / "discovery_summary.json")

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(["--run-id", run_id])
    stderr = capsys.readouterr().err

    assert rc == 1
    assert "RUN_VALID check failed" in stderr
    assert not (run_dir / "experiment" / "malda_formula_validation" / "stage_summary.json").exists()
