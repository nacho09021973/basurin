from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mvp.tools import compare_221_strategies as module



def _write_multimode_artifact(run_dir: Path, *, usable: bool, reason: str, valid_fraction: float, cv_f: float, cv_q: float, lnf_span: float, lnq_span: float) -> None:
    path = run_dir / "s3b_multimode_estimates" / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode_221_usable": usable,
        "mode_221_usable_reason": reason,
        "results": {"verdict": "OK", "quality_flags": []},
        "source": {
            "mode_221_residual": {"strategy": "refit_220_each_iter"},
            "band_strategy": {"method": "default_split_60_40"},
        },
        "modes": [
            {"label": "220", "fit": {"stability": {}}},
            {
                "label": "221",
                "fit": {
                    "stability": {
                        "valid_fraction": valid_fraction,
                        "cv_f": cv_f,
                        "cv_Q": cv_q,
                        "lnf_span": lnf_span,
                        "lnQ_span": lnq_span,
                    }
                },
            },
        ],
    }
    (path / "multimode_estimates.json").write_text(json.dumps(payload), encoding="utf-8")



def test_build_run_id_is_deterministic() -> None:
    run_id = module._build_run_id(
        "mvp_GW250114_082203_compare221",
        "hilbert_peakband",
        "refit_220_each_iter",
        "coherent_harmonic_band",
    )
    assert run_id == (
        "mvp_GW250114_082203_compare221"
        "__m-hilbert_peakband__r-refit_220_each_iter__b-coherent_harmonic_band"
    )



def test_main_writes_auditable_outputs_under_runs_root(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], text: bool, capture_output: bool) -> subprocess.CompletedProcess[str]:
        assert text is True
        assert capture_output is True
        run_id = cmd[cmd.index("--run-id") + 1]
        calls.append(cmd)
        _write_multimode_artifact(
            runs_root / run_id,
            usable=True,
            reason="ok",
            valid_fraction=0.75,
            cv_f=0.12,
            cv_q=0.21,
            lnf_span=0.34,
            lnq_span=0.56,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main([
        "--event-id",
        "GW250114_082203",
        "--base-run-prefix",
        "mvp_GW250114_082203_compare221",
    ])

    assert rc == 0
    assert len(calls) == 8

    compare_run = runs_root / "mvp_GW250114_082203_compare221__compare221_audit" / module.EXPERIMENT_STAGE
    summary = json.loads((compare_run / "stage_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((compare_run / "manifest.json").read_text(encoding="utf-8"))
    matrix = json.loads((compare_run / "outputs" / module.MATRIX_JSON).read_text(encoding="utf-8"))
    rows = json.loads((compare_run / "outputs" / module.RESULTS_JSON).read_text(encoding="utf-8"))["rows"]
    csv_text = (compare_run / "outputs" / module.SUMMARY_CSV).read_text(encoding="utf-8")

    assert summary["n_runs"] == 8
    assert summary["all_subruns_passed"] is True
    assert manifest["artifacts"][module.RESULTS_JSON] == f"outputs/{module.RESULTS_JSON}"
    assert matrix["compare_run_id"] == "mvp_GW250114_082203_compare221__compare221_audit"
    assert len(rows) == 8
    assert "mode_221_usable_reason" in csv_text
    assert all(str(path).startswith(str(runs_root)) for path in [compare_run])



def test_evaluate_runs_improved_when_candidate_crosses_usability_gate() -> None:
    baseline = {
        "run_id": "base",
        "mode_221_usable": False,
        "mode_221_usable_reason": "221_lnQ_span_explosive",
        "valid_fraction": 0.42,
        "cv_Q": 1.20,
        "lnQ_span": 1.30,
        "cv_f": 0.30,
        "lnf_span": 0.40,
    }
    candidate = {
        "run_id": "cand",
        "mode_221_usable": True,
        "mode_221_usable_reason": "ok",
        "valid_fraction": 0.61,
        "cv_Q": 0.80,
        "lnQ_span": 0.70,
        "cv_f": 0.20,
        "lnf_span": 0.20,
    }

    result = __import__("mvp.tools.evaluate_221_strategy", fromlist=["*"]).evaluate_runs(
        baseline,
        candidate,
        valid_fraction_delta=0.05,
        spread_delta=0.05,
    )

    assert result["verdict"] == "improved"
    assert "candidate crossed the main usability gate" in " ".join(result["trace"])



def test_evaluate_runs_neutral_when_both_unusable_but_candidate_metrics_improve() -> None:
    baseline = {
        "run_id": "base",
        "mode_221_usable": False,
        "mode_221_usable_reason": "221_valid_fraction_low",
        "valid_fraction": 0.20,
        "cv_Q": 1.10,
        "lnQ_span": 1.40,
        "cv_f": 0.40,
        "lnf_span": 0.60,
    }
    candidate = {
        "run_id": "cand",
        "mode_221_usable": False,
        "mode_221_usable_reason": "221_valid_fraction_low",
        "valid_fraction": 0.28,
        "cv_Q": 0.90,
        "lnQ_span": 1.10,
        "cv_f": 0.30,
        "lnf_span": 0.50,
    }

    result = __import__("mvp.tools.evaluate_221_strategy", fromlist=["*"]).evaluate_runs(
        baseline,
        candidate,
        valid_fraction_delta=0.05,
        spread_delta=0.05,
    )

    assert result["verdict"] == "neutral"
    assert result["summary"]["better_metrics"]
    assert result["rules"]["still_unusable_policy"].startswith("if both runs")



def test_evaluate_runs_degraded_when_both_usable_and_candidate_is_worse() -> None:
    baseline = {
        "run_id": "base",
        "mode_221_usable": True,
        "mode_221_usable_reason": "ok",
        "valid_fraction": 0.72,
        "cv_Q": 0.40,
        "lnQ_span": 0.45,
        "cv_f": 0.10,
        "lnf_span": 0.15,
    }
    candidate = {
        "run_id": "cand",
        "mode_221_usable": True,
        "mode_221_usable_reason": "ok",
        "valid_fraction": 0.60,
        "cv_Q": 0.55,
        "lnQ_span": 0.60,
        "cv_f": 0.18,
        "lnf_span": 0.22,
    }

    result = __import__("mvp.tools.evaluate_221_strategy", fromlist=["*"]).evaluate_runs(
        baseline,
        candidate,
        valid_fraction_delta=0.05,
        spread_delta=0.05,
    )

    assert result["verdict"] == "degraded"
    assert "valid_fraction" in result["summary"]["worse_metrics"]



def test_evaluate_cli_reads_runs_from_runs_root(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    for run_id, usable, reason, vf, cvf, cvq, lnf, lnq in [
        ("baseline_run", False, "221_lnQ_span_explosive", 0.35, 0.20, 1.10, 0.25, 1.40),
        ("candidate_run", True, "ok", 0.60, 0.10, 0.60, 0.10, 0.70),
    ]:
        run_dir = runs_root / run_id
        rv = run_dir / "RUN_VALID"
        rv.mkdir(parents=True, exist_ok=True)
        (rv / "verdict.json").write_text('{"verdict": "PASS"}', encoding="utf-8")
        _write_multimode_artifact(
            run_dir,
            usable=usable,
            reason=reason,
            valid_fraction=vf,
            cv_f=cvf,
            cv_q=cvq,
            lnf_span=lnf,
            lnq_span=lnq,
        )

    module_eval = __import__("mvp.tools.evaluate_221_strategy", fromlist=["*"])
    from io import StringIO
    import contextlib

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        rc = module_eval.main([
            "--baseline-run",
            "baseline_run",
            "--candidate-run",
            "candidate_run",
        ])

    payload = json.loads(buf.getvalue())
    assert rc == 0
    assert payload["verdict"] == "improved"
    assert payload["baseline"]["run_id"] == "baseline_run"
    assert payload["candidate"]["run_id"] == "candidate_run"
