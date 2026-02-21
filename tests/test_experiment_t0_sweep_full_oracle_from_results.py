from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mvp import experiment_t0_sweep_full as exp


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mk_seed_results(scan_root: Path, *, seed: int, t0_values: tuple[int, ...], ln_f: float, ln_q: float, include_221: bool = True) -> None:
    points = []
    for t0 in t0_values:
        s3b = {
            "verdict": "OK",
            "has_221": include_221,
            "ln_f_220": ln_f,
            "ln_Q_220": ln_q,
            "ln_f_221": (ln_f + 0.01) if include_221 else None,
            "ln_Q_221": (ln_q + 0.01) if include_221 else None,
        }
        points.append({"t0_ms": t0, "status": "OK", "s3b": s3b, "s4c": {"verdict": "OK"}})

    seed_dir = scan_root / f"t0_sweep_full_seed{seed}"
    _write_json(
        seed_dir / "outputs" / "t0_sweep_full_results.json",
        {
            "schema_version": "experiment_t0_sweep_full_v1",
            "run_id": "BASE_RUN",
            "summary": {"n_points": len(points)},
            "points": points,
        },
    )
    _write_json(seed_dir / "stage_summary.json", {"stage": "experiment/t0_sweep_full", "verdict": "PASS"})
    _write_json(seed_dir / "manifest.json", {"schema_version": "mvp_manifest_v1"})


def test_unit_sigma_floor_and_scale_floor_for_cv() -> None:
    scan_root = Path("/tmp/non-used")
    artifacts = [
        {
            "seed_dir": scan_root / "t0_sweep_full_seed101",
            "results": {
                "points": [
                    {
                        "t0_ms": 0,
                        "status": "OK",
                        "s3b": {
                            "ln_f_220": 0.0,
                            "ln_Q_220": 0.0,
                            "has_221": False,
                            "ln_f_221": None,
                            "ln_Q_221": None,
                        },
                    }
                ]
            },
        }
    ]
    windows, rows = exp._build_windows_from_sweep_results(
        artifacts,
        sigma_floor_f=0.5,
        sigma_floor_tau=0.25,
        scale_floor_f=10.0,
        scale_floor_tau=20.0,
        gate_221=False,
    )
    assert len(windows) == 1
    assert rows[0]["f_sigma"] == 0.5
    assert rows[0]["tau_sigma"] == 0.25
    assert rows[0]["cv_f"] == 0.05
    assert rows[0]["cv_tau"] == 0.0125


def test_integration_finalize_uses_seed_results_contract(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "BASE_RUN"
    scan_root = runs_root / run_id / "experiment"
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    for seed in (101, 202, 303):
        _mk_seed_results(scan_root, seed=seed, t0_values=(0, 2, 4), ln_f=4.7, ln_q=3.2)

    args = SimpleNamespace(
        run_id=run_id,
        runs_root=str(runs_root),
        scan_root=str(scan_root),
        inventory_seeds="101,202,303",
        t0_grid_ms="0,2,4",
        t0_start_ms=0,
        t0_stop_ms=0,
        t0_step_ms=1,
        seed=101,
        phase="finalize",
        max_missing_abs=999,
        max_missing_frac=1.0,
        max_retries_per_pair=2,
        sigma_floor_f=1e-6,
        sigma_floor_tau=1e-6,
        scale_floor_f=1e-3,
        scale_floor_tau=1e-3,
        gate_221=False,
    )

    payload = exp.run_finalize_phase(args)
    assert payload["oracle_report"]["final_verdict"] == "PASS"

    derived = runs_root / run_id / "experiment" / "derived"
    inv = json.loads((derived / "t0_sweep_inventory.json").read_text(encoding="utf-8"))
    assert inv["n_windows_loaded"] == 3


def test_snapshot_golden_oracle_report_from_seed_results(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "BASE_RUN"
    scan_root = runs_root / run_id / "experiment"
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    for seed in (101, 202, 303):
        _mk_seed_results(scan_root, seed=seed, t0_values=(0, 2, 4), ln_f=4.7, ln_q=3.2)

    args = SimpleNamespace(
        run_id=run_id,
        runs_root=str(runs_root),
        scan_root=str(scan_root),
        inventory_seeds="101,202,303",
        t0_grid_ms="0,2,4",
        t0_start_ms=0,
        t0_stop_ms=0,
        t0_step_ms=1,
        seed=101,
        phase="finalize",
        max_missing_abs=999,
        max_missing_frac=1.0,
        max_retries_per_pair=2,
        sigma_floor_f=1e-6,
        sigma_floor_tau=1e-6,
        scale_floor_f=1e-3,
        scale_floor_tau=1e-3,
        gate_221=False,
    )

    exp.run_finalize_phase(args)
    oracle = json.loads((runs_root / run_id / "experiment" / "derived" / "oracle_report.json").read_text(encoding="utf-8"))

    assert oracle["final_verdict"] == "PASS"
    assert oracle["chosen_t0"] == 0.0
    assert oracle["thresholds"]["k_plateau"] == 3
    assert len(oracle["windows_summary"]) == 3
