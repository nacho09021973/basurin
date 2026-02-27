import json
from pathlib import Path

from mvp.contracts import init_stage
import mvp.s4d_kerr_from_multimode as s4d


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
        }
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
    assert "KERR_GRID_SATURATION: median_spin_on_grid_edge" in summary["error"]
    output_paths = {item["path"] for item in summary["outputs"]}
    assert output_paths == set()

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == {"stage_summary": "stage_summary.json"}
    assert "boundary_audit" not in manifest["artifacts"]
    assert "boundary_audit" not in manifest["hashes"]
