from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from mvp import experiment_t0_sweep_full as exp
from mvp.oracles.oracle_v1_plateau import WindowMetrics, run_oracle_v1


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mk_subrun(scan_root: Path, *, seed: int, t0_ms: int, f_hz: float, tau_s: float) -> None:
    subrun = scan_root / f"t0_sweep_full_seed{seed}" / f"segment__t0ms{t0_ms:04d}"

    _write_json(subrun / "s2_ringdown_window" / "manifest.json", {"stage": "s2_ringdown_window", "t0": t0_ms})
    _write_json(
        subrun / "s2_ringdown_window" / "stage_summary.json",
        {
            "stage": "s2_ringdown_window",
            "results": {"t0_offset_ms": t0_ms, "duration_s": 0.1, "n_samples": 256},
        },
    )

    _write_json(subrun / "s3_ringdown_estimates" / "manifest.json", {"stage": "s3_ringdown_estimates"})
    _write_json(
        subrun / "s3_ringdown_estimates" / "stage_summary.json",
        {
            "stage": "s3_ringdown_estimates",
            "results": {
                "combined": {"f_hz": f_hz, "tau_s": tau_s},
                "combined_uncertainty": {"sigma_f_hz": 0.2, "sigma_tau_s": 0.02},
            },
        },
    )

    _write_json(subrun / "s3b_multimode_estimates" / "manifest.json", {"stage": "s3b_multimode_estimates"})
    _write_json(subrun / "s3b_multimode_estimates" / "stage_summary.json", {"stage": "s3b_multimode_estimates"})
    _write_json(subrun / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json", {})

    _write_json(subrun / "s4c_kerr_consistency" / "manifest.json", {"stage": "s4c_kerr_consistency"})
    _write_json(
        subrun / "s4c_kerr_consistency" / "stage_summary.json",
        {
            "stage": "s4c_kerr_consistency",
            "results": {"cond_number": 10.0, "delta_bic": 20.0, "p_ljungbox": 0.4, "n_samples": 256, "snr": 10.0},
        },
    )


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_finalize_regression_golden_manifest_and_oracle(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "BASE_RUN"
    scan_root = runs_root / run_id / "experiment"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    t0_grid_ms = (0, 2, 4, 6, 8)
    for idx, t0_ms in enumerate(t0_grid_ms):
        _mk_subrun(scan_root, seed=101, t0_ms=t0_ms, f_hz=100.0 + (idx * 0.05), tau_s=0.80 + (idx * 0.01))

    args = SimpleNamespace(
        run_id=run_id,
        runs_root=str(runs_root),
        scan_root=str(scan_root),
        inventory_seeds="101",
        t0_grid_ms=",".join(str(x) for x in t0_grid_ms),
        t0_start_ms=0,
        t0_stop_ms=0,
        t0_step_ms=1,
        seed=101,
        phase="finalize",
        max_missing_abs=0,
        max_missing_frac=0.0,
        max_retries_per_pair=2,
    )

    payload = exp.run_finalize_phase(args)

    derived = runs_root / run_id / "experiment" / "derived"
    oracle_report_path = derived / "oracle_report.json"
    assert oracle_report_path.exists()

    golden = run_oracle_v1(
        [
            WindowMetrics(
                t0=float(t0_ms),
                T=0.1,
                f_median=100.0 + (idx * 0.05),
                f_sigma=0.2,
                tau_median=0.80 + (idx * 0.01),
                tau_sigma=0.02,
                cond_number=10.0,
                delta_bic=20.0,
                p_ljungbox=0.4,
                n_samples=256,
                snr=10.0,
            )
            for idx, t0_ms in enumerate(t0_grid_ms)
        ]
    )
    assert len(golden["windows_summary"]) == 5
    assert golden["final_verdict"] == "PASS"
    assert golden["fail_global_reason"] is None
    assert golden["chosen_t0"] is not None
    assert _canonical_bytes(json.loads(oracle_report_path.read_text(encoding="utf-8"))) == _canonical_bytes(golden)

    manifest = json.loads((runs_root / run_id / "experiment" / "manifest.json").read_text(encoding="utf-8"))
    for artifact_name, artifact_path in manifest["artifacts"].items():
        path = Path(artifact_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["hashes"][artifact_name] == digest

    stage_summary = json.loads((runs_root / run_id / "experiment" / "stage_summary.json").read_text(encoding="utf-8"))
    assert stage_summary["results"]["final_verdict"] == payload["oracle_report"]["final_verdict"]
    assert stage_summary["results"]["fail_global_reason"] == payload["oracle_report"]["fail_global_reason"]
    assert stage_summary["results"]["chosen_t0"] == payload["oracle_report"]["chosen_t0"]
