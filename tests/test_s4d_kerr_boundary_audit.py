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
    assert params.get("sigma_space_matching") == "ln_f_ln_Q"
    assert params.get("sigma_transform_applied") is False
    assert params.get("boundary_fraction", 0.0) >= 1.0
    assert params.get("boundary_hits", 0) > 0
    assert "KERR_GRID_SATURATION: median_spin_on_grid_edge" in summary["error"]
    output_paths = {item["path"] for item in summary["outputs"]}
    assert output_paths == set()

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == {"stage_summary": "stage_summary.json"}
    assert "boundary_audit" not in manifest["artifacts"]
    assert "boundary_audit" not in manifest["hashes"]


def test_lnq_reconstruction_matches_lnf_lntau_identity() -> None:
    lnf = [2.30, 2.35]
    lntau = [-3.10, -3.00]
    ln_pi = s4d.math.log(s4d.math.pi)
    lnq = [lf + lt + ln_pi for lf, lt in zip(lnf, lntau)]

    for i in range(len(lnf)):
        assert lnq[i] == lnf[i] + lntau[i] + ln_pi


def test_best_idx_joint_accepts_sigma_weighting() -> None:
    obs = {"220": {"f_hz": 10.0, "tau_s": 1.0}, "221": {"f_hz": 20.0, "tau_s": 2.0}}
    lnf_220 = [2.30, 2.35]
    lnq_220 = [s4d.math.log(obs["220"]["f_hz"]) + s4d.math.log(obs["220"]["tau_s"]) + s4d.math.log(s4d.math.pi), 2.60]
    lnf_221 = [2.99, 3.02]
    lnq_221 = [s4d.math.log(obs["221"]["f_hz"]) + s4d.math.log(obs["221"]["tau_s"]) + s4d.math.log(s4d.math.pi), 3.80]
    inv_sigma = s4d._invert_2x2_sigma(((0.03, 0.01), (0.01, 0.05)))
    idx = s4d._best_idx_joint(obs, lnf_220, lnq_220, lnf_221, lnq_221, inv_sigma, inv_sigma)
    assert idx in (0, 1)


def test_regularize_and_invert_sigma_near_singular_is_finite() -> None:
    sigma_near_singular = ((0.07361859, 0.14723593), (0.14723593, 0.29446937))
    inv_sigma, diag = s4d._regularize_and_invert_2x2_sigma(sigma_near_singular)

    assert diag["det_after"] > 0.0
    assert diag["jitter_used"] > 0.0
    assert diag["det_after"] >= diag["det_before"]
    assert all(value == value for row in inv_sigma for value in row)


def test_should_abort_for_boundary_allows_spin_saturation_at_physical_floor() -> None:
    should_abort, reason, warning = s4d._should_abort_for_boundary(
        a_p50=s4d.A_MIN,
        m_p50=100.0,
        boundary_fraction=1.0,
        a_min=s4d.A_MIN,
        a_max=s4d.A_MAX,
        m_min=s4d.M_MIN,
        m_max=s4d.M_MAX,
        threshold=s4d.BOUNDARY_FRACTION_THRESHOLD,
    )

    assert should_abort is False
    assert reason is None
    assert warning is True


def test_should_abort_for_boundary_fails_when_spin_saturates_at_a_max() -> None:
    should_abort, reason, warning = s4d._should_abort_for_boundary(
        a_p50=s4d.A_MAX,
        m_p50=100.0,
        boundary_fraction=1.0,
        a_min=s4d.A_MIN,
        a_max=s4d.A_MAX,
        m_min=s4d.M_MIN,
        m_max=s4d.M_MAX,
        threshold=s4d.BOUNDARY_FRACTION_THRESHOLD,
    )

    assert should_abort is True
    assert reason == "median_spin_on_grid_edge"
    assert warning is False
