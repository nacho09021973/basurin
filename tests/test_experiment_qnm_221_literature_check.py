from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mvp import experiment_qnm_221_literature_check as mod
from mvp.kerr_qnm_fits import kerr_qnm


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mode_payload(*, label: str, mode: list[int], f_hz: float, tau_s: float, valid_fraction: float = 1.0) -> dict:
    ln_f = math.log(f_hz)
    ln_q = math.log(math.pi * f_hz * tau_s)
    return {
        "label": label,
        "mode": mode,
        "ln_f": ln_f,
        "ln_Q": ln_q,
        "Sigma": [[1.0e-4, 0.0], [0.0, 1.0e-4]],
        "fit": {
            "stability": {
                "valid_fraction": valid_fraction,
                "lnf_p10": math.log(f_hz * 0.98),
                "lnf_p50": ln_f,
                "lnf_p90": math.log(f_hz * 1.02),
                "lnQ_p10": math.log(math.pi * (f_hz * 0.98) * (tau_s * 0.98)),
                "lnQ_p50": ln_q,
                "lnQ_p90": math.log(math.pi * (f_hz * 1.02) * (tau_s * 1.02)),
            }
        },
    }


def _seed_base_run(runs_root: Path, run_id: str, *, run_valid_verdict: str = "PASS") -> dict[str, float]:
    run_dir = runs_root / run_id
    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": run_valid_verdict})
    _write_json(
        run_dir / "run_provenance.json",
        {
            "schema_version": "run_provenance_v1",
            "run_id": run_id,
            "invocation": {"event_id": "GW150914"},
        },
    )

    m_final = 68.0
    chi_final = 0.69
    q220 = kerr_qnm(m_final, chi_final, (2, 2, 0))
    q221 = kerr_qnm(m_final, chi_final, (2, 2, 1))

    _write_json(
        run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "schema_version": "multimode_estimates_v1",
            "run_id": run_id,
            "results": {"quality_flags": [], "messages": [], "verdict": "OK"},
            "modes": [
                _mode_payload(label="220", mode=[2, 2, 0], f_hz=q220.f_hz, tau_s=q220.tau_s),
                _mode_payload(label="221", mode=[2, 2, 1], f_hz=q221.f_hz, tau_s=q221.tau_s),
            ],
        },
    )
    _write_json(
        run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json",
        {
            "schema_version": "model_comparison_v1",
            "delta_bic": -6.0,
            "thresholds": {"two_mode_preferred_delta_bic": 2.0},
            "decision": {"two_mode_preferred": True},
            "bic_1mode": 100.0,
            "bic_2mode": 94.0,
            "rss_1mode": 10.0,
            "rss_2mode": 5.0,
        },
    )
    _write_json(
        run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json",
        {
            "schema_name": "kerr_extraction",
            "schema_version": "mvp_kerr_extraction_v1",
            "verdict": "PASS",
            "M_final_Msun": m_final,
            "chi_final": chi_final,
        },
    )
    return {"m_final": m_final, "chi_final": chi_final}


def _fake_run_t0_point_factory(*, m_final: float, chi_final: float):
    q220 = kerr_qnm(m_final, chi_final, (2, 2, 0))
    q221 = kerr_qnm(m_final, chi_final, (2, 2, 1))

    def _fake_run_t0_point(
        *,
        event_id: str,
        base_run_dir: Path,
        exp_stage_dir: Path,
        t0_value: float,
        units: str,
        mass_msun: float,
        python_exe: str,
        out_root: Path,
    ) -> dict[str, object]:
        subrun_id = f"t0_{int(t0_value)}"
        subrun_dir = exp_stage_dir / "subruns" / subrun_id
        if int(t0_value) == 4:
            f221 = q221.f_hz * 1.30
            tau221 = q221.tau_s * 0.70
        else:
            f221 = q221.f_hz * (1.0 + (0.01 * float(t0_value)))
            tau221 = q221.tau_s * (1.0 + (0.01 * float(t0_value)))

        _write_json(
            subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
            {
                "schema_version": "multimode_estimates_v1",
                "run_id": subrun_id,
                "results": {"quality_flags": [], "messages": [], "verdict": "OK"},
                "modes": [
                    _mode_payload(label="220", mode=[2, 2, 0], f_hz=q220.f_hz, tau_s=q220.tau_s),
                    _mode_payload(label="221", mode=[2, 2, 1], f_hz=f221, tau_s=tau221),
                ],
            },
        )
        _write_json(
            subrun_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json",
            {
                "schema_version": "model_comparison_v1",
                "delta_bic": -3.0,
                "thresholds": {"two_mode_preferred_delta_bic": 2.0},
                "decision": {"two_mode_preferred": True},
            },
        )
        return {
            "event_id": event_id,
            "t0": float(t0_value),
            "units": units,
            "has_221": True,
            "valid_fraction_221": 1.0,
            "reason": None,
            "flags": [],
            "subrun_path": str(subrun_dir.relative_to(out_root)) + "/",
        }

    return _fake_run_t0_point


def test_qnm_221_literature_check_writes_only_under_experiment_dir(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    params = _seed_base_run(runs_root, "run_qnm_221")
    monkeypatch.setattr(mod, "_run_t0_point", _fake_run_t0_point_factory(**params))

    run_dir = runs_root / "run_qnm_221"
    before = {p.relative_to(run_dir) for p in run_dir.rglob("*") if p.is_file()}
    rc = mod.main(["--run-id", "run_qnm_221", "--t0-grid-ms", "0,2,4"])
    assert rc == 0

    after = {p.relative_to(run_dir) for p in run_dir.rglob("*") if p.is_file()}
    new_files = after - before
    assert new_files
    assert all(str(path).startswith("experiment/qnm_221_literature_check/") for path in new_files)

    exp_dir = run_dir / "experiment" / "qnm_221_literature_check"
    assert (exp_dir / "outputs" / "kerr_oracle_221.json").exists()
    assert (exp_dir / "outputs" / "t0_stability_221.csv").exists()
    assert (exp_dir / "outputs" / "model_selection_220_vs_220221.csv").exists()
    assert (exp_dir / "outputs" / "summary_221_validation.json").exists()
    assert (exp_dir / "stage_summary.json").exists()
    assert (exp_dir / "manifest.json").exists()
    assert not (tmp_path / "experiment").exists()


def test_qnm_221_literature_check_aborts_when_run_valid_is_not_pass(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _seed_base_run(runs_root, "run_invalid", run_valid_verdict="FAIL")

    with pytest.raises(RuntimeError, match="RUN_VALID verdict is not PASS"):
        mod.main(["--run-id", "run_invalid"])

    assert not (runs_root / "run_invalid" / "experiment" / "qnm_221_literature_check").exists()


def test_qnm_221_literature_check_summary_has_required_fields_and_allowed_verdict(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    params = _seed_base_run(runs_root, "run_summary")
    monkeypatch.setattr(mod, "_run_t0_point", _fake_run_t0_point_factory(**params))

    rc = mod.main(["--run-id", "run_summary", "--t0-grid-ms", "0,2,4"])
    assert rc == 0

    summary = json.loads(
        (runs_root / "run_summary" / "experiment" / "qnm_221_literature_check" / "outputs" / "summary_221_validation.json").read_text(encoding="utf-8")
    )
    required = {
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
    assert required.issubset(summary.keys())
    assert summary["verdict"] in set(mod.ALLOWED_VERDICTS)
