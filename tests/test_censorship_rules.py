from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_run(tmp_path: Path, run_id: str) -> Path:
    runs_root = tmp_path / "runs"
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    return runs_root


def test_s6c_fails_on_hard_censorship_rule_violation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "agg_violation"
    runs_root = _base_run(tmp_path, run_id)

    _write_json(
        runs_root / run_id / "s5_aggregate" / "outputs" / "aggregate.json",
        {
            "schema_version": "mvp_aggregate_v2",
            "events": [
                {
                    "event_id": "GWX",
                    "censoring": {
                        "has_221": False,
                        "vote_kerr": "PASS",
                        "weight_scalar": 1.0,
                        "weight_vector4": 1.0,
                        "reason": "221_valid_fraction_low",
                    },
                    "scalar": {"kerr_tension": 0.2, "kerr_tension_pvalue": 0.7},
                }
            ],
        },
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    proc = subprocess.run(
        [sys.executable, str(repo_root / "mvp" / "s6c_population_geometry.py"), "--run", run_id],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode != 0
    assert "Hard censorship rule violation" in proc.stderr


def test_s6c_writes_outputs_and_hashes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "agg_ok"
    runs_root = _base_run(tmp_path, run_id)

    _write_json(
        runs_root / run_id / "s5_aggregate" / "outputs" / "aggregate.json",
        {
            "schema_version": "mvp_aggregate_v2",
            "events": [
                {
                    "event_id": "GW1",
                    "censoring": {"has_221": False, "vote_kerr": "INCONCLUSIVE", "weight_scalar": 1.0, "weight_vector4": 0.0, "reason": "221_valid_fraction_low"},
                    "scalar": {"kerr_tension": 0.2, "kerr_tension_pvalue": 0.6},
                },
                {
                    "event_id": "GW2",
                    "censoring": {"has_221": True, "vote_kerr": "PASS", "weight_scalar": 1.0, "weight_vector4": 1.0, "reason": None},
                    "scalar": {"kerr_tension": 0.1, "kerr_tension_pvalue": 0.8},
                },
            ],
        },
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    proc = subprocess.run(
        [sys.executable, str(repo_root / "mvp" / "s6c_population_geometry.py"), "--run", run_id],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / run_id / "s6c_population_geometry"
    claim = stage_dir / "outputs" / "population_scalar_claim.json"
    diag = stage_dir / "outputs" / "population_diagnostics.json"
    manifest = stage_dir / "manifest.json"
    assert claim.exists()
    assert diag.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["artifacts"]["population_scalar_claim"] == "outputs/population_scalar_claim.json"
    assert data["artifacts"]["population_diagnostics"] == "outputs/population_diagnostics.json"
    assert len(data["hashes"]["population_scalar_claim"]) == 64
    assert len(data["hashes"]["population_diagnostics"]) == 64


def test_s6b_coords_3d_contains_censoring_weight(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "run_s6b"
    runs_root = _base_run(tmp_path, run_id)

    _write_json(
        runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {"event_id": "GW150914", "combined": {"f_hz": 250.0, "Q": 10.0}},
    )
    _write_json(
        runs_root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
        {
            "results": {"quality_flags": ["221_valid_fraction_low"]},
            "modes": [
                {"label": "220", "ln_f": 5.0, "ln_Q": 2.0},
                {"label": "221", "ln_f": 5.1, "ln_Q": 2.1},
            ],
        },
    )
    _write_json(
        runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        {"ranked_all": [{"geometry_id": "GR", "distance": 0.2}]},
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    proc = subprocess.run(
        [sys.executable, str(repo_root / "mvp" / "s6b_information_geometry_3d.py"), "--run", run_id],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr

    coords = json.loads((runs_root / run_id / "s6b_information_geometry_3d" / "outputs" / "coords_3d.json").read_text(encoding="utf-8"))
    assert coords["censoring"]["has_221"] is False
    assert coords["censoring"]["weight"] == 0.0
