from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from basurin_io import sha256_file, write_json_atomic
from mvp.contracts import finalize, init_stage
from mvp.kerr_qnm_fits import kerr_qnm
import mvp.s7_beyond_kerr_deviation_score as s7


def _mode_payload(
    *,
    label: str,
    mode: list[int],
    f_p10: float,
    f_p50: float,
    f_p90: float,
    tau_p10: float,
    tau_p50: float,
    tau_p90: float,
) -> dict[str, Any]:
    def _lnq(f_hz: float, tau_s: float) -> float:
        return math.log(math.pi * f_hz * tau_s)

    return {
        "mode": mode,
        "label": label,
        "ln_f": math.log(f_p50),
        "ln_Q": _lnq(f_p50, tau_p50),
        "Sigma": [[1.0e-4, 0.0], [0.0, 1.0e-4]],
        "fit": {
            "method": "unit_test",
            "n_bootstrap": 64,
            "bootstrap_seed": 7,
            "stability": {
                "valid_fraction": 1.0,
                "n_successful": 64,
                "n_failed": 0,
                "lnf_p10": math.log(f_p10),
                "lnf_p50": math.log(f_p50),
                "lnf_p90": math.log(f_p90),
                "lnQ_p10": _lnq(f_p10, tau_p10),
                "lnQ_p50": _lnq(f_p50, tau_p50),
                "lnQ_p90": _lnq(f_p90, tau_p90),
            },
        },
    }


def _seed_upstream(
    run_dir: Path,
    *,
    kerr_verdict: str = "PASS",
    m_final: float | None = 68.0,
    chi_final: float | None = 0.69,
    f_scale: float = 1.0,
    tau_scale: float = 1.0,
    zero_iqr_221: bool = False,
) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    write_json_atomic(
        run_dir / "s4d_kerr_from_multimode" / "stage_summary.json",
        {"stage": "s4d_kerr_from_multimode", "verdict": "PASS"},
    )
    write_json_atomic(
        run_dir / "s3b_multimode_estimates" / "stage_summary.json",
        {"stage": "s3b_multimode_estimates", "verdict": "PASS"},
    )

    if m_final is not None and chi_final is not None:
        pred = kerr_qnm(m_final, chi_final, (2, 2, 1))
        f_221 = pred.f_hz * f_scale
        tau_221 = pred.tau_s * tau_scale
    else:
        # unused in skip case, but still valid positive values for schema completeness
        f_221 = 200.0
        tau_221 = 0.01

    if zero_iqr_221:
        f_221_p10 = f_221_p50 = f_221_p90 = f_221
        tau_221_p10 = tau_221_p50 = tau_221_p90 = tau_221
    else:
        f_221_p10, f_221_p50, f_221_p90 = f_221 * 0.98, f_221, f_221 * 1.02
        tau_221_p10, tau_221_p50, tau_221_p90 = tau_221 * 0.98, tau_221, tau_221 * 1.02

    write_json_atomic(
        run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json",
        {
            "schema_name": "kerr_extraction",
            "schema_version": "mvp_kerr_extraction_v1",
            "verdict": kerr_verdict,
            "M_final_Msun": m_final,
            "chi_final": chi_final,
        },
    )

    multimode_payload = {
        "schema_version": "multimode_estimates_v1",
        "run_id": run_dir.name,
        "source": {"stage": "s2_ringdown_window", "window": None},
        "results": {"verdict": "OK", "quality_flags": [], "messages": []},
        "modes_target": [
            {"mode": [2, 2, 0], "label": "220"},
            {"mode": [2, 2, 1], "label": "221"},
        ],
        "modes": [
            _mode_payload(
                label="220",
                mode=[2, 2, 0],
                f_p10=170.0,
                f_p50=175.0,
                f_p90=180.0,
                tau_p10=0.0038,
                tau_p50=0.0040,
                tau_p90=0.0042,
            ),
            _mode_payload(
                label="221",
                mode=[2, 2, 1],
                f_p10=f_221_p10,
                f_p50=f_221_p50,
                f_p90=f_221_p90,
                tau_p10=tau_221_p10,
                tau_p50=tau_221_p50,
                tau_p90=tau_221_p90,
            ),
        ],
    }
    write_json_atomic(
        run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        multimode_payload,
    )


def _run_execute_and_finalize(run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    ctx = init_stage(run_id, s7.STAGE, params={})
    artifacts = s7._execute(ctx)
    output_path = artifacts["beyond_kerr_score"]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    finalize(
        ctx,
        artifacts=artifacts,
        verdict="PASS",
        results={"verdict": payload.get("verdict"), "chi2_kerr_2dof": payload.get("chi2_kerr_2dof")},
    )
    return ctx.stage_dir, artifacts, payload


def test_gr_consistent_synthetic(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_consistent"
    _seed_upstream(runs_root / run_id, f_scale=1.0, tau_scale=1.0)

    _, _, payload = _run_execute_and_finalize(run_id)
    assert payload["verdict"] == "GR_CONSISTENT"
    assert payload["chi2_kerr_2dof"] < 4.605


def test_gr_inconsistent_large_deviation(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_inconsistent"
    _seed_upstream(runs_root / run_id, f_scale=1.50, tau_scale=0.50)

    _, _, payload = _run_execute_and_finalize(run_id)
    assert payload["verdict"] == "GR_INCONSISTENT"
    assert payload["chi2_kerr_2dof"] > 9.210


def test_skips_gracefully_when_s4d_gated(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_skip"
    _seed_upstream(
        runs_root / run_id,
        kerr_verdict="SKIPPED_MULTIMODE_GATE",
        m_final=None,
        chi_final=None,
    )

    _, _, payload = _run_execute_and_finalize(run_id)
    assert payload["verdict"] == "SKIPPED_S4D_GATE"
    assert payload["chi2_kerr_2dof"] is None
    assert payload["epsilon_f"] is None
    assert payload["gr_threshold_90pct"] == 4.605
    assert payload["gr_threshold_99pct"] == 9.210


def test_contract_outputs_exist_after_run(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_contract_outputs"
    _seed_upstream(runs_root / run_id)

    stage_dir, artifacts, _ = _run_execute_and_finalize(run_id)
    output_path = artifacts["beyond_kerr_score"]
    assert (stage_dir / "outputs" / "beyond_kerr_score.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "manifest.json").exists()

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "hashes" in manifest
    assert manifest["hashes"]["beyond_kerr_score"] == sha256_file(output_path)


def test_chi2_cdf_2dof_boundary_values() -> None:
    assert s7._chi2_cdf_2dof(0.0) == 0.0
    assert s7._chi2_cdf_2dof(float("inf")) == 1.0


def test_sigma_floor_when_iqr_is_zero(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_sigma_floor"
    _seed_upstream(
        runs_root / run_id,
        f_scale=1.001,
        tau_scale=0.999,
        zero_iqr_221=True,
    )

    _, _, payload = _run_execute_and_finalize(run_id)
    assert math.isfinite(payload["delta_f_norm"])
    assert math.isfinite(payload["delta_tau_norm"])
    assert math.isfinite(payload["chi2_kerr_2dof"])
