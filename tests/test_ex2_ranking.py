from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.experiment.ex2_ranking import rank_events

REPO_ROOT = Path(__file__).resolve().parent.parent


def _prepare_run(tmp_path: Path) -> tuple[Path, str, Path]:
    runs_root = tmp_path / "runs_tmp"
    run_id = "it_ex2_ranking"
    run_dir = runs_root / run_id

    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    write_json_atomic(
        run_dir / "s6c_brunete_psd_curvature" / "outputs" / "brunete_metrics.json",
        {
            "schema_version": "brunete_metrics_v1",
            "run_id": run_id,
            "metrics": [
                {
                    "event_id": "EV_A",
                    "detector": "H1",
                    "kappa": 2.0,
                    "sigma": 0.2,
                    "chi_psd": 0.5,
                    "warnings": [],
                },
                {
                    "event_id": "EV_A",
                    "detector": "L1",
                    "kappa": 1.0,
                    "sigma": 0.4,
                    "chi_psd": 0.2,
                    "warnings": [],
                },
                {
                    "event_id": "EV_B",
                    "detector": "H1",
                    "kappa": 0.8,
                    "sigma": 0.2,
                    "chi_psd": 2.0,
                    "warnings": ["PSD_POLYFIT_INSUFFICIENT_POINTS"],
                },
                {
                    "event_id": "EV_C",
                    "detector": "H1",
                    "kappa": 0.5,
                    "sigma": 0.0,
                    "chi_psd": 0.1,
                    "warnings": [],
                },
            ],
        },
    )
    return runs_root, run_id, run_dir


def test_rank_events_golden_order_and_schema() -> None:
    rows = [
        {"event_id": "E2", "kappa": 1.0, "sigma": 0.5, "chi_psd": 0.0, "warnings": []},
        {"event_id": "E1", "kappa": 2.0, "sigma": 0.2, "chi_psd": 0.0, "warnings": []},
        {"event_id": "E3", "kappa": 0.1, "sigma": 1.0, "chi_psd": 0.0, "warnings": ["W"]},
    ]
    ranking = rank_events(rows)
    assert [x["event_id"] for x in ranking] == ["E1", "E2", "E3"]

    required_keys = {"event_id", "D", "n_det_ok", "n_det_total", "n_warnings", "features"}
    allowed_feature_keys = {
        "score_formula",
        "kappa_abs_median",
        "sigma_median",
        "chi_psd_median",
    }
    for item in ranking:
        assert set(item.keys()) == required_keys
        assert set(item["features"].keys()) == allowed_feature_keys


def test_cli_deterministic_byte_output_and_no_write_outside_runs_root(tmp_path: Path) -> None:
    runs_root, run_id, run_dir = _prepare_run(tmp_path)
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [sys.executable, "-m", "mvp.experiment.ex2_ranking", "--run-id", run_id]

    p1 = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    assert p1.returncode == 0, p1.stderr
    out_json = run_dir / "experiment" / "ex2_ranking" / "outputs" / "event_ranking.json"
    out_csv = run_dir / "experiment" / "ex2_ranking" / "outputs" / "event_ranking.csv"
    assert out_json.exists()
    assert out_csv.exists()

    content_first = out_json.read_bytes()

    p2 = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    assert p2.returncode == 0, p2.stderr
    content_second = out_json.read_bytes()

    assert content_first == content_second

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    ranked_ids = [row["event_id"] for row in payload["ranking"]]
    assert ranked_ids == ["EV_A", "EV_B", "EV_C"]

    assert not (REPO_ROOT / "runs" / run_id).exists()


def test_ex2_ranking_missing_required_keys_error_message_is_actionable(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_tmp"
    run_id = "it_ex2_ranking_missing_keys"
    run_dir = runs_root / run_id
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    write_json_atomic(
        run_dir / "s6c_brunete_psd_curvature" / "outputs" / "brunete_metrics.json",
        {
            "metrics": [
                {
                    "event_id": "EV_FAIL",
                    "detector": "H1",
                    "kappa": 0.7,
                    "chi_psd": 0.1,
                }
            ]
        },
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [sys.executable, "-m", "mvp.experiment.ex2_ranking", "--run-id", run_id]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)

    assert proc.returncode != 0
    err = proc.stderr
    assert "event_id=EV_FAIL" in err
    assert "detector=H1" in err
    assert "missing_key=sigma" in err
    assert "available_keys=['chi_psd', 'detector', 'event_id', 'kappa']" in err
