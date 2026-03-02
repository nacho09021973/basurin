import json
from pathlib import Path

from mvp.contracts import init_stage
import mvp.s4d_kerr_from_multimode as s4d


def _sigma_ln_f_ln_q() -> list[list[float]]:
    return [[0.04, 0.01], [0.01, 0.09]]


def test_s4d_fails_without_extra_boundary_artifact_on_spin_grid_saturation(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "s4d_boundary_audit_pytest"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")

    s3b_out = run_dir / "s3b_multimode_estimates" / "outputs"
    s3b_out.mkdir(parents=True)
    multimode = {
        "estimates": {
            "per_mode": {
                "220": {"f_hz": {"p10": 150.0, "p50": 160.0, "p90": 170.0}, "tau_s": {"p10": 0.003, "p50": 0.004, "p90": 0.005}},
                "221": {"f_hz": {"p10": 240.0, "p50": 250.0, "p90": 260.0}, "tau_s": {"p10": 0.002, "p50": 0.0025, "p90": 0.003}},
            }
        },
        "modes": [
            {"label": "220", "Sigma": _sigma_ln_f_ln_q()},
            {"label": "221", "Sigma": _sigma_ln_f_ln_q()},
        ],
    }
    (s3b_out / "multimode_estimates.json").write_text(json.dumps(multimode) + "\n", encoding="utf-8")

    def _fake_build_grid():
        n = 8
        grid_m = [100.0] * n
        grid_a = [s4d.A_MAX] * n
        lnf_220 = [0.0] * n
        lntau_220 = [0.0] * n
        lnf_221 = [0.0] * n
        lntau_221 = [0.0] * n
        return grid_m, grid_a, lnf_220, lntau_220, lnf_221, lntau_221

    monkeypatch.setattr(s4d, "_build_grid", _fake_build_grid)

    ctx = init_stage(run_id, s4d.STAGE)
    try:
        s4d._execute(ctx)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit(2) on boundary saturation")

    stage_dir = run_dir / "s4d_kerr_from_multimode"
    out_dir = stage_dir / "outputs"

    audit_path = out_dir / "boundary_audit.json"
    assert not audit_path.exists()

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    params = summary.get("parameters", {})
    assert params.get("A_MAX") == s4d.A_MAX
    assert params.get("GRID_A_SIZE") == s4d.GRID_A_SIZE
    assert params.get("matching_metric") == "mahalanobis_sigma"
    assert params.get("sigma_space_input") == "ln_f_ln_Q"
    assert params.get("sigma_space_matching") == "ln_f_ln_tau"
    assert params.get("sigma_transform_applied") is True
    assert params.get("boundary_fraction", 0.0) >= 1.0
    assert params.get("boundary_hits", 0) > 0
    assert "KERR_GRID_SATURATION: median_spin_on_grid_edge" in summary["error"]
    output_paths = {item["path"] for item in summary["outputs"]}
    assert output_paths == set()

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == {"stage_summary": "stage_summary.json"}
    assert "boundary_audit" not in manifest["artifacts"]
    assert "boundary_audit" not in manifest["hashes"]


def test_sigma_transform_preserves_symmetric_positive_definite() -> None:
    sigma_x = ((0.04, 0.01), (0.01, 0.09))
    sigma_y = s4d._sigma_lnf_lnq_to_lnf_lntau(sigma_x)
    assert sigma_y[0][1] == sigma_y[1][0]
    det = sigma_y[0][0] * sigma_y[1][1] - sigma_y[0][1] * sigma_y[1][0]
    assert det > 0.0


def test_best_idx_joint_accepts_sigma_weighting() -> None:
    obs = {"220": {"f_hz": 10.0, "tau_s": 1.0}, "221": {"f_hz": 20.0, "tau_s": 2.0}}
    lnf_220 = [2.30, 2.35]
    lntau_220 = [0.02, -0.02]
    lnf_221 = [2.99, 3.02]
    lntau_221 = [0.70, 0.69]
    inv_sigma = s4d._invert_2x2_sigma(s4d._sigma_lnf_lnq_to_lnf_lntau(((0.03, 0.01), (0.01, 0.05))))
    idx = s4d._best_idx_joint(obs, lnf_220, lntau_220, lnf_221, lntau_221, inv_sigma, inv_sigma)
    assert idx in (0, 1)
