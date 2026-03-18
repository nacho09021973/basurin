from __future__ import annotations

import json
import math
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


def _write_run_provenance(root: Path, run_id: str, event_id: str) -> None:
    write_json_atomic(
        root / run_id / "run_provenance.json",
        {
            "schema_version": "run_provenance_v1",
            "run_id": run_id,
            "invocation": {"event_id": event_id},
        },
    )


def _write_minimal_s3(root: Path, run_id: str) -> None:
    write_json_atomic(
        root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {"event_id": "GW150914", "combined": {"f_hz": 250.0, "Q": 10.0}},
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


def _write_s3b_221_fit(
    root: Path,
    run_id: str,
    *,
    verdict: str = "PASS",
    quality_flags: list[str] | None = None,
) -> None:
    write_json_atomic(
        root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "schema_version": "multimode_estimates_v1",
            "results": {
                "verdict": verdict,
                "quality_flags": [] if quality_flags is None else quality_flags,
            },
            "modes": [
                {
                    "label": "221",
                    "mode": [2, 2, 1],
                    "fit": {
                        "stability": {
                            "lnf_p50": math.log(250.0),
                            "lnQ_p50": math.log(5.0),
                        }
                    },
                }
            ],
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


def _write_gwtc_remnant_h5(
    root: Path,
    run_id: str,
    *,
    event_id: str,
    mixed_rows: list[tuple[float, float]],
    other_rows: list[tuple[float, float]] | None = None,
    mixed_fields: tuple[str, ...] = ("final_mass", "final_spin"),
) -> Path:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    raw_dir = root / run_id / "external_inputs" / "gwtc_posteriors" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"IGWN-GWTC3p0-v2-{event_id}_PEDataRelease_mixed_cosmo.h5"

    with h5py.File(str(path), "w") as h5:
        mixed_dtype = np.dtype([(field, "f8") for field in mixed_fields])
        h5.create_dataset("C01:Mixed/posterior_samples", data=np.array(mixed_rows, dtype=mixed_dtype))
        if other_rows is not None:
            other_dtype = np.dtype([("final_mass", "f8"), ("final_spin", "f8")])
            h5.create_dataset("C01:IMRPhenomXPHM/posterior_samples", data=np.array(other_rows, dtype=other_dtype))

    return path


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


def test_main_returns_insufficient_data_when_upstream_221_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "r_221_flagged"
    _write_run_valid(runs_root, run_id, "PASS")
    _write_minimal_s3(runs_root, run_id)
    _write_s3b_221_fit(
        runs_root,
        run_id,
        verdict="INSUFFICIENT_DATA",
        quality_flags=[
            "220_lnQ_span_explosive",
            "221_cv_Q_explosive",
            "221_lnQ_span_explosive",
        ],
    )
    _write_model_comp(runs_root, run_id)
    _write_remnant(runs_root, run_id)

    def _should_not_predict(_: float, __: float) -> tuple[float, float, dict[str, str]]:
        raise AssertionError("blocked upstream 221 must not reach Gate A Kerr prediction")

    monkeypatch.setattr(mod, "_predict_kerr_221", _should_not_predict)
    rc = mod.main(["--run-id", run_id])
    assert rc == 0

    summary = json.loads(
        (runs_root / run_id / "experiment" / "qnm_221_literature_check" / "outputs" / "summary_221_validation.json").read_text(encoding="utf-8")
    )
    stage_summary = json.loads(
        (runs_root / run_id / "experiment" / "qnm_221_literature_check" / "stage_summary.json").read_text(encoding="utf-8")
    )

    reason = "upstream_221_quality_flags:221_cv_Q_explosive,221_lnQ_span_explosive"
    assert summary["f221_measured"] is None
    assert summary["tau221_measured"] is None
    assert summary["extraction_policy"] == reason
    assert summary["verdict"] == "INSUFFICIENT_DATA"
    assert summary["verdict_reason"] == reason
    assert summary["gates"]["A"]["status"] == "NOT_AVAILABLE"
    assert summary["gates"]["A"]["reason"] == "missing_measured_f_or_tau_221"
    assert stage_summary["verdict"] == "PASS"
    assert stage_summary["results"]["verdict"] == "INSUFFICIENT_DATA"
    assert stage_summary["results"]["verdict_reason"] == reason


def test_main_uses_stability_p50_when_upstream_passes_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "r_221_pass"
    _write_run_valid(runs_root, run_id, "PASS")
    _write_minimal_s3(runs_root, run_id)
    _write_s3b_221_fit(runs_root, run_id, verdict="PASS", quality_flags=[])
    _write_model_comp(runs_root, run_id)
    _write_remnant(runs_root, run_id)

    expected_tau = 5.0 / (math.pi * 250.0)
    monkeypatch.setattr(mod, "_predict_kerr_221", lambda mf, af: (250.0, expected_tau, {"tool": "fake"}))
    rc = mod.main(["--run-id", run_id])
    assert rc == 0

    summary = json.loads(
        (runs_root / run_id / "experiment" / "qnm_221_literature_check" / "outputs" / "summary_221_validation.json").read_text(encoding="utf-8")
    )

    assert summary["extraction_policy"] == "mode_221.fit.stability.p50"
    assert summary["f221_measured"] == pytest.approx(250.0)
    assert summary["tau221_measured"] == pytest.approx(expected_tau)
    assert summary["gates"]["A"]["status"] == "PASS"


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

    assert f221 is None
    assert tau221 is None
    assert policy == "upstream_221_quality_flags:221_Sigma_invalid,221_lnQ_span_explosive,221_no_point_estimate"


def test_extract_mf_af_from_external_h5_prefers_mixed_dataset(tmp_path: Path) -> None:
    event_id = "GW200129_065458"
    path = _write_gwtc_remnant_h5(
        tmp_path,
        "r_h5",
        event_id=event_id,
        mixed_rows=[(55.0, 0.62), (60.0, 0.70), (65.0, 0.78)],
        other_rows=[(80.0, 0.20), (90.0, 0.30), (100.0, 0.40)],
    )

    mf, af, meta = mod._extract_mf_af_from_h5(path)

    assert mf == pytest.approx(60.0)
    assert af == pytest.approx(0.70)
    assert meta["dataset_path"] == "C01:Mixed/posterior_samples"
    assert meta["reason"] is None


def test_summary_uses_external_h5_remnant_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "mvp_GW200129_065458_20260318T080000Z"
    event_id = "GW200129_065458"
    _write_run_valid(runs_root, run_id, "PASS")
    _write_run_provenance(runs_root, run_id, event_id)
    _write_minimal_s3(runs_root, run_id)
    _write_minimal_s3b(runs_root, run_id)
    _write_model_comp(runs_root, run_id)
    _write_gwtc_remnant_h5(
        runs_root,
        run_id,
        event_id=event_id,
        mixed_rows=[(58.0, 0.66), (60.0, 0.70), (63.0, 0.73)],
    )

    seen: dict[str, float] = {}

    def _fake_predict(mf: float, af: float) -> tuple[float, float, dict[str, str]]:
        seen["mf"] = mf
        seen["af"] = af
        return 250.0, 5.0 / (3.141592653589793 * 250.0), {"tool": "fake"}

    monkeypatch.setattr(mod, "_predict_kerr_221", _fake_predict)
    rc = mod.main(["--run-id", run_id])
    assert rc == 0

    summary = json.loads(
        (runs_root / run_id / "experiment" / "qnm_221_literature_check" / "outputs" / "summary_221_validation.json").read_text(encoding="utf-8")
    )

    assert seen["mf"] == pytest.approx(60.0)
    assert seen["af"] == pytest.approx(0.70)
    assert summary["mf_source"].endswith(".h5")
    assert summary["af_source"].endswith(".h5")
    assert summary["remnant_extraction_reason"] is None
    assert summary["f221_kerr"] == pytest.approx(250.0)
    assert summary["tau221_kerr"] == pytest.approx(5.0 / (3.141592653589793 * 250.0))


def test_external_h5_without_final_spin_degrades_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "mvp_GW200129_065458_20260318T090000Z"
    event_id = "GW200129_065458"
    _write_run_valid(runs_root, run_id, "PASS")
    _write_run_provenance(runs_root, run_id, event_id)
    _write_minimal_s3(runs_root, run_id)
    _write_minimal_s3b(runs_root, run_id)
    _write_model_comp(runs_root, run_id)
    _write_gwtc_remnant_h5(
        runs_root,
        run_id,
        event_id=event_id,
        mixed_rows=[(58.0,), (60.0,), (63.0,)],
        mixed_fields=("final_mass",),
    )

    rc = mod.main(["--run-id", run_id])
    assert rc == 0

    summary = json.loads(
        (runs_root / run_id / "experiment" / "qnm_221_literature_check" / "outputs" / "summary_221_validation.json").read_text(encoding="utf-8")
    )

    assert summary["mf_source"].endswith(".h5")
    assert summary["af_source"].endswith(".h5")
    assert summary["f221_kerr"] is None
    assert summary["tau221_kerr"] is None
    assert summary["verdict"] == "INSUFFICIENT_DATA"
    assert summary["remnant_extraction_reason"] == "no_posterior_samples_with_final_mass_and_final_spin"
