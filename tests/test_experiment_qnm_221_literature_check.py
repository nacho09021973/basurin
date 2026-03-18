from __future__ import annotations

import json
from pathlib import Path

import pytest

from basurin_io import write_json_atomic

import mvp.experiment_qnm_221_literature_check as mod

REQUIRED_SUMMARY_FIELDS = {
    "run_id",
    "source_estimates_path",
    "mf_source",
    "af_source",
    "f221_measured",
    "tau221_measured",
    "f221_kerr",
    "tau221_kerr",
    "rel_err_f",
    "rel_err_tau",
    "stable_t0_fraction",
    "model_selection_metric_name",
    "model_selection_metric_value",
    "verdict",
    "verdict_reason",
}


def _write_run_valid(root: Path, run_id: str, verdict: str = "PASS") -> None:
    write_json_atomic(root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


def _write_minimal_s3(root: Path, run_id: str) -> None:
    write_json_atomic(
        root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {"combined": {"f_hz": 250.0, "Q": 10.0}},
    )


def _write_minimal_s3b(root: Path, run_id: str) -> None:
    write_json_atomic(
        root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "modes": [{"label": "221", "f_hz": 250.0, "Q": 5.0}],
            "joint_3d": {
                "valid": True,
                "Sigma_3d": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "point_estimate": [0.0, 0.0, 0.0],
            },
        },
    )


def _write_model_comp(root: Path, run_id: str) -> None:
    write_json_atomic(
        root / run_id / "s3b_multimode_estimates" / "outputs" / "model_comparison.json",
        {
            "schema_version": "model_comparison_v1",
            "delta_bic": -3.1,
            "decision": {"two_mode_preferred": True},
            "thresholds": {"two_mode_preferred_delta_bic": 2.0},
        },
    )


def _write_remnant(root: Path, run_id: str) -> None:
    write_json_atomic(root / run_id / "external_inputs" / "remnant_kerr.json", {"Mf": 1.0, "af": 0.7})


def test_abort_if_run_valid_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "r_missing"
    _write_minimal_s3(runs_root, run_id)
    _write_minimal_s3b(runs_root, run_id)

    with pytest.raises(FileNotFoundError):
        mod.main(["--run-id", run_id])


def test_no_write_outside_experiment_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "r_ok"
    _write_run_valid(runs_root, run_id, "PASS")
    _write_minimal_s3(runs_root, run_id)
    _write_minimal_s3b(runs_root, run_id)
    _write_model_comp(runs_root, run_id)
    _write_remnant(runs_root, run_id)

    before = {p.relative_to(runs_root / run_id) for p in (runs_root / run_id).rglob("*") if p.is_file()}
    monkeypatch.setattr(mod, "_predict_kerr_221", lambda mf, af: (250.0, 5.0 / (3.141592653589793 * 250.0), {"tool": "fake"}))
    rc = mod.main(["--run-id", run_id])
    assert rc == 0

    run_root = runs_root / run_id
    after = {p.relative_to(run_root) for p in run_root.rglob("*") if p.is_file()}
    new_files = after - before
    assert new_files
    assert all(str(path).startswith("experiment/qnm_221_literature_check/") for path in new_files)


def test_summary_contains_required_fields_and_valid_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "r_fields"
    _write_run_valid(runs_root, run_id, "PASS")
    _write_minimal_s3(runs_root, run_id)
    _write_minimal_s3b(runs_root, run_id)
    _write_model_comp(runs_root, run_id)
    _write_remnant(runs_root, run_id)

    monkeypatch.setattr(mod, "_predict_kerr_221", lambda mf, af: (250.0, 5.0 / (3.141592653589793 * 250.0), {"tool": "fake"}))
    rc = mod.main(["--run-id", run_id])
    assert rc == 0

    summary = json.loads(
        (runs_root / run_id / "experiment" / "qnm_221_literature_check" / "outputs" / "summary_221_validation.json").read_text(encoding="utf-8")
    )
    assert REQUIRED_SUMMARY_FIELDS.issubset(summary.keys())
    assert summary["verdict"] in mod.ALLOWED_VERDICTS


def test_extract_221_from_real_s3b_schema_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "qnm_221_literature_check"
        / "multimode_estimates.real_schema.json"
    )
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    s3_estimates = {"combined": {"f_hz": 250.0, "Q": 10.0}}

    f221, tau221, policy = mod._extract_221_from_multimode(data, s3_estimates)

    assert policy in {
        "mode_221.fit.stability.p50",
        "explicit_mode_221",
        "joint_3d_ln_ratio_times_s3_f220",
    }
    assert f221 is None or f221 > 0.0
    assert tau221 is None or tau221 > 0.0
