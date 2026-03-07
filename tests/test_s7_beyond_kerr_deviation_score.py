from __future__ import annotations

import json
from pathlib import Path

from mvp.contracts import finalize, init_stage
from mvp.kerr_qnm_fits import kerr_qnm
import mvp.s7_beyond_kerr_deviation_score as s7


def _seed_upstream(
    run_dir: Path,
    *,
    kerr_verdict: str = "PASS",
    m_final: float | None = 68.0,
    chi_final: float | None = 0.69,
    f_scale: float = 1.0,
    tau_scale: float = 1.0,
) -> None:
    (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")

    s4d_out = run_dir / "s4d_kerr_from_multimode" / "outputs"
    s4d_out.mkdir(parents=True, exist_ok=True)

    if m_final is not None and chi_final is not None:
        pred = kerr_qnm(m_final, chi_final, (2, 2, 1))
        f_221 = pred.f_hz * f_scale
        tau_221 = pred.tau_s * tau_scale
    else:
        # values unused when gate skips
        f_221 = 200.0
        tau_221 = 0.01

    (s4d_out / "kerr_extraction.json").write_text(
        json.dumps(
            {
                "schema_name": "kerr_extraction",
                "schema_version": "mvp_kerr_extraction_v1",
                "verdict": kerr_verdict,
                "M_final_Msun": m_final,
                "chi_final": chi_final,
            }
        ),
        encoding="utf-8",
    )

    s3b_out = run_dir / "s3b_multimode_estimates" / "outputs"
    s3b_out.mkdir(parents=True, exist_ok=True)
    (s3b_out / "multimode_estimates.json").write_text(
        json.dumps(
            {
                "estimates": {
                    "per_mode": {
                        "221": {
                            "f_hz": {"p10": f_221 * 0.98, "p50": f_221, "p90": f_221 * 1.02},
                            "tau_s": {"p10": tau_221 * 0.98, "p50": tau_221, "p90": tau_221 * 1.02},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _run_execute_and_finalize(run_id: str) -> tuple[Path, dict]:
    ctx = init_stage(run_id, s7.STAGE, params={})
    artifacts = s7._execute(ctx)
    payload = json.loads(artifacts["beyond_kerr_score"].read_text(encoding="utf-8"))
    finalize(
        ctx,
        artifacts=artifacts,
        verdict="PASS",
        results={"verdict": payload.get("verdict"), "chi2_kerr_2dof": payload.get("chi2_kerr_2dof")},
    )
    return ctx.run_dir, payload


def test_gr_consistent_synthetic(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_consistent"
    run_dir = runs_root / run_id
    _seed_upstream(run_dir, f_scale=1.0, tau_scale=1.0)

    _, payload = _run_execute_and_finalize(run_id)
    assert payload["verdict"] == "GR_CONSISTENT"
    assert payload["chi2_kerr_2dof"] < 4.605


def test_gr_inconsistent_large_deviation(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_inconsistent"
    run_dir = runs_root / run_id
    _seed_upstream(run_dir, f_scale=1.5, tau_scale=0.5)

    _, payload = _run_execute_and_finalize(run_id)
    assert payload["verdict"] == "GR_INCONSISTENT"
    assert payload["chi2_kerr_2dof"] >= 9.210


def test_skips_gracefully_when_s4d_gated(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_skip"
    run_dir = runs_root / run_id
    _seed_upstream(
        run_dir,
        kerr_verdict="SKIPPED_MULTIMODE_GATE",
        m_final=None,
        chi_final=None,
    )

    _, payload = _run_execute_and_finalize(run_id)
    assert payload["verdict"] == "SKIPPED_S4D_GATE"
    assert payload["chi2_kerr_2dof"] is None
    assert payload["epsilon_f"] is None


def test_contract_outputs_exist_after_run(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    run_id = "s7_contract_outputs"
    run_dir = runs_root / run_id
    _seed_upstream(run_dir)

    run_dir_after, _ = _run_execute_and_finalize(run_id)
    stage_dir = run_dir_after / s7.STAGE
    assert (stage_dir / "outputs" / "beyond_kerr_score.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "manifest.json").exists()


def test_chi2_cdf_2dof_boundary_values() -> None:
    assert s7._chi2_cdf_2dof(-1.0) == 0.0
    assert s7._chi2_cdf_2dof(0.0) == 0.0
    assert 0.989 < s7._chi2_cdf_2dof(9.210) < 1.0
