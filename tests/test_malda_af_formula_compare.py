from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "malda" / "13_compare_af_formula_candidates.py"
_SPEC = importlib.util.spec_from_file_location("malda_13_compare_af_formula_candidates", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_feature_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "is_bbh", "eta", "chi_eff", "af"]
    rows = []
    for idx, (eta, chi_eff, is_bbh) in enumerate(
        [
            (0.200, -0.20, "0"),
            (0.205, -0.10, "1"),
            (0.212, -0.02, "1"),
            (0.220, 0.05, "1"),
            (0.228, 0.12, "1"),
            (0.236, 0.20, "1"),
            (0.242, 0.30, "1"),
            (0.246, 0.38, "1"),
        ]
    ):
        af = ((eta + 0.4833311) ** 2) * (chi_eff + 1.237785)
        rows.append(
            {
                "event_id": f"E{idx}",
                "is_bbh": is_bbh,
                "eta": f"{eta:.8f}",
                "chi_eff": f"{chi_eff:.8f}",
                "af": f"{af:.8f}",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_main_compares_af_formulas_and_writes_contract_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runsroot"
    run_id = "malda_af_formula_compare_smoke"
    run_dir = runs_root / run_id
    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    feature_table_path = run_dir / "experiment" / "malda_feature_table" / "outputs" / "event_features.csv"
    _write_feature_table(feature_table_path)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(
        [
            "--run-id",
            run_id,
            "--n-splits",
            "8",
            "--test-fraction",
            "0.33",
            "--seed",
            "7",
        ]
    )

    stage_dir = run_dir / "experiment" / "malda_af_formula_compare"
    outputs_dir = stage_dir / "outputs"
    compare_json = outputs_dir / "formula_compare.json"
    metrics_csv = outputs_dir / "formula_compare_split_metrics.csv"
    ranking_json = outputs_dir / "formula_compare_ranking.json"
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    ranking = json.loads(ranking_json.read_text(encoding="utf-8"))
    metrics_rows = list(csv.DictReader(metrics_csv.open("r", encoding="utf-8")))
    stdout = capsys.readouterr().out

    assert rc == 0
    assert compare_json.exists()
    assert metrics_csv.exists()
    assert ranking_json.exists()
    assert (stage_dir / "manifest.json").exists()
    assert stage_summary["verdict"] == "PASS"
    assert stage_summary["results"]["winner_by_test_nrmse_std_mean"] == "multiplicative"
    assert payload["results"]["winner_by_test_nrmse_std_mean"] == "multiplicative"
    assert ranking["winner_by_test_nrmse_std_mean"] == "multiplicative"
    assert len(metrics_rows) == 2 * 2 * 8
    assert stage_summary["results"]["n_rows_compared"] == 7
    assert stage_summary["hashes"]["formula_compare"]
    assert all(Path(path).is_relative_to(run_dir) for path in stage_summary["outputs"].values())
    for key in ("OUT_ROOT=", "STAGE_DIR=", "OUTPUTS_DIR=", "STAGE_SUMMARY=", "MANIFEST="):
        assert key in stdout


def test_main_requires_run_valid_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runsroot_gate"
    run_id = "malda_af_formula_compare_gate"
    run_dir = runs_root / run_id
    _write_feature_table(run_dir / "experiment" / "malda_feature_table" / "outputs" / "event_features.csv")

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(["--run-id", run_id])
    stderr = capsys.readouterr().err

    assert rc == 1
    assert "RUN_VALID check failed" in stderr
    assert not (run_dir / "experiment" / "malda_af_formula_compare" / "manifest.json").exists()
