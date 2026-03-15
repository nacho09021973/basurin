import json
from pathlib import Path

from mvp.contracts import init_stage
import mvp.s4d_kerr_from_multimode as s4d

def _sigma_ln_f_ln_q() -> list[list[float]]:
    return [[0.04, 0.01], [0.01, 0.09]]

def _write_compatible_set(run_dir: Path, *, n_compatible: int = 1) -> None:
    payload = {
        "schema_version": "mvp_compatible_set_v1",
        "event_id": run_dir.name,
        "metric": "mahalanobis_log",
        "n_atlas": 4,
        "n_compatible": int(max(0, n_compatible)),
        "ranked_all": [{"geometry_id": f"g{i}", "d2": float(i + 1)} for i in range(4)],
        "compatible_geometries": [{"geometry_id": f"g{i}"} for i in range(max(0, n_compatible))],
    }
    out_dir = run_dir / "s4_geometry_filter" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compatible_set.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")

def test_s4d_fails_without_extra_boundary_artifact_on_spin_grid_saturation(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "s4d_boundary_audit_pytest"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")
    _write_compatible_set(run_dir)

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
    (run_dir / "s3b_multimode_estimates" / "stage_summary.json").write_text(
        json.dumps({
            "multimode_viability": {
                "class": "MULTIMODE_OK",
                "reasons": [],
                "metrics": {"boundary_fraction": None, "valid_fraction": {"220": 1.0, "221": 1.0}},
            }
        }),
        encoding="utf-8",
    )

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

def test_s4d_skips_multimode_when_viability_gate_blocks(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    run_id = "s4d_gate_skip_pytest"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")
    _write_compatible_set(run_dir)
    s3b_out = run_dir / "s3b_multimode_estimates" / "outputs"
    s3b_out.mkdir(parents=True)
    (s3b_out / "multimode_estimates.json").write_text(json.dumps({"modes": []}) + "\n", encoding="utf-8")
    (run_dir / "s3b_multimode_estimates" / "stage_summary.json").write_text(
        json.dumps({
            "multimode_viability": {
                "class": "SINGLEMODE_ONLY",
                "reasons": ["BOUNDARY_FRACTION_HIGH"],
                "metrics": {"boundary_fraction": 1.0, "valid_fraction": {"220": 0.9, "221": 0.2}},
            }
        }),
        encoding="utf-8",
    )

    ctx = init_stage(run_id, s4d.STAGE)
    artifacts = s4d._execute(ctx)
    assert set(artifacts.keys()) == {"kerr_from_multimode", "kerr_from_multimode_diagnostics", "kerr_extraction"}

    diag = json.loads((run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_from_multimode_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["diagnostics"]["multimode_evaluated"] is False
    assert "MULTIMODE_UNAVAILABLE_221" in diag["diagnostics"]["skips"]

    extraction = json.loads((run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json").read_text(encoding="utf-8"))
    assert extraction["verdict"] == "SKIPPED_MULTIMODE_GATE"
    assert extraction["M_final_Msun"] is None
    assert extraction["skip_reason_code"] == "MULTIMODE_UNAVAILABLE_221"
    assert extraction["multimode_fallback"]["program_classification"] == "SINGLE_MODE_CONSTRAINED_PROGRAM"

def test_invert_kerr_from_freqs_recovers_high_spin_branch() -> None:
    from mvp.kerr_qnm_fits import kerr_qnm

    q220 = kerr_qnm(65.0, 0.99, (2, 2, 0))
    q221 = kerr_qnm(65.0, 0.99, (2, 2, 1))

    m_final, chi_final = s4d._invert_kerr_from_freqs(q220.f_hz, q221.f_hz)

    assert abs(m_final - 65.0) < 0.5
    assert chi_final > 0.97

def test_extract_kerr_with_covariance_core_uses_grid_inverter(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_invert(
        f_220_hz: float,
        f_221_hz: float,
        grid_m: list[float],
        grid_a: list[float],
        lnf_220: list[float],
        lnf_221: list[float],
    ) -> tuple[float, float]:
        calls["n"] += 1
        return (70.0, 0.6)

    monkeypatch.setattr(s4d, "_invert_kerr_from_freqs_grid", _fake_invert)

    out = s4d._extract_kerr_with_covariance_core(
        f_220_hz=200.0,
        f_221_hz=300.0,
        sigma_f220=2.0,
        sigma_f221=3.0,
        grid_m=[70.0, 71.0],
        grid_a=[0.6, 0.61],
        lnf_220=[1.0, 1.1],
        lnf_221=[1.2, 1.3],
    )

    assert calls["n"] >= 5
    assert len(out) == 5
    assert out[0] == 70.0
    assert out[1] == 0.6
