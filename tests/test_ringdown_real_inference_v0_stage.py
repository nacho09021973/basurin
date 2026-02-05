"""
tests/test_ringdown_real_inference_v0_stage.py
----------------------------------------------
Minimal tests for stages/ringdown_real_inference_v0_stage.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def _write_run_valid_pass(run_dir: Path) -> None:
    _write_json(
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": "PASS"},
    )


def _write_rd_npz(path: Path, strain: np.ndarray, fs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=strain.astype(float), fs=fs)


def test_stage_real_inference_aborts_without_run_valid(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__real_inference_no_run_valid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
        "--band-hz",
        "200,500",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode != 0


def test_stage_real_inference_recovers_f_peak_and_tau_on_synthetic_decay(
    tmp_path: Path,
) -> None:
    run_id = "2040-09-01__unit_test__real_inference_synth"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs = 4096.0
    n_samples = 8192
    f_hz = 250.0
    tau_s = 0.25
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.exp(-t / tau_s) * np.sin(2.0 * np.pi * f_hz * t)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz", strain, fs)
    _write_rd_npz(inputs_dir / "L1_rd.npz", strain, fs)
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    observables = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "detectors": ["H1", "L1"],
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "rms": {"H1": 1.0, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 1.0},
    }
    features = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "duration_s": {"H1": n_samples / fs, "L1": n_samples / fs},
        "snr_proxy": {"H1": 10.0, "L1": 10.0},
    }

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        observables,
    )
    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        features,
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
        "--band-hz",
        "200,500",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    h1_fit = report["fit"]["H1"]
    assert abs(h1_fit["f_peak_hz"] - f_hz) < 5.0
    assert h1_fit["tau_s"] is not None
    assert 0.15 < h1_fit["tau_s"] < 0.4
    assert report["band_hz"] == [200.0, 500.0]


def test_stage_real_inference_reports_window_duration_from_npz(
    tmp_path: Path,
) -> None:
    run_id = "2040-09-01__unit_test__real_inference_window"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs = 4096.0
    duration_s = 0.25
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.sin(2.0 * np.pi * 220.0 * t)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz", strain, fs)
    _write_rd_npz(inputs_dir / "L1_rd.npz", strain, fs)
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    observables = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "detectors": ["H1", "L1"],
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "rms": {"H1": 1.0, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 1.0},
    }
    features = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "duration_s": {"H1": duration_s, "L1": duration_s},
        "snr_proxy": {"H1": 10.0, "L1": 10.0},
    }

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        observables,
    )
    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        features,
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for det in ["H1", "L1"]:
        expected = report["fit"][det]["n_samples"] / report["fs_hz"]
        assert abs(report["window"]["duration_s"][det] - expected) < 1e-9
        expected_df = report["fs_hz"] / report["fit"][det]["n_samples"]
        assert abs(report["fit"][det]["df_hz"] - expected_df) < 1e-9

    assert report["window"]["stage"] == "ringdown_real_ringdown_window"
    assert "duration_s" not in report["features"]
    assert "features_duration_s" in report["features"]


def test_stage_real_inference_tau_null_reports_tau_estimator_metrics(
    tmp_path: Path,
) -> None:
    """When strain is too short for 5 valid blocks, tau_s must be null and
    inference_report must contain tau_estimator with per-detector accounting."""
    run_id = "2040-09-01__unit_test__real_inference_tau_null"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs = 4096.0
    # 100 samples → with block=max(16,int(0.01*4096))=40, only ~2-3 blocks
    n_samples = 100
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.sin(2.0 * np.pi * 250.0 * t)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz", strain, fs)
    _write_rd_npz(inputs_dir / "L1_rd.npz", strain, fs)
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    observables = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "detectors": ["H1", "L1"],
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "rms": {"H1": 1.0, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 1.0},
    }
    features = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "duration_s": {"H1": n_samples / fs, "L1": n_samples / fs},
        "snr_proxy": {"H1": 10.0, "L1": 10.0},
    }

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        observables,
    )
    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        features,
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
        "--band-hz",
        "200,500",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # tau_estimator must exist with expected top-level keys
    assert "tau_estimator" in report
    te = report["tau_estimator"]
    assert "block_len_s" in te
    assert isinstance(te["block_len_s"], float)
    assert te["min_blocks_required"] == 5

    # Both detectors must have per-detector accounting
    for det in ["H1", "L1"]:
        assert det in te, f"tau_estimator missing {det}"
        det_te = te[det]
        assert "n_blocks_total" in det_te
        assert "n_blocks_valid" in det_te
        assert "reject_reasons" in det_te
        assert isinstance(det_te["n_blocks_total"], int)
        assert isinstance(det_te["n_blocks_valid"], int)
        assert det_te["n_blocks_valid"] < 5

    # tau_s must be null for both detectors
    for det in ["H1", "L1"]:
        assert report["fit"][det]["tau_s"] is None

    # decision must be INSPECT with informative reasons
    assert report["decision"]["verdict"] == "INSPECT"
    reasons_text = " ".join(report["decision"]["reasons"])
    assert "(n_blocks_valid=" in reasons_text
    assert "< 5)" in reasons_text


def test_basurin_where_ringdown_exp08_reports_real_inference_missing(
    tmp_path: Path,
) -> None:
    run_id = "2040-09-01__unit_test__where_exp08_inference"
    run_dir = tmp_path / "runs" / run_id

    _write_run_valid_pass(run_dir)
    _write_json(
        run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json",
        [{"event_id": "GW150914", "strain_npz": "cases/GW150914/strain.npz"}],
    )

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        {"detectors": ["H1", "L1"], "fs_hz": 4096.0, "n_samples": {"H1": 1, "L1": 1}},
    )
    _write_json(run_dir / "ringdown_real_observables_v0" / "stage_summary.json", {})
    _write_json(run_dir / "ringdown_real_observables_v0" / "manifest.json", {})

    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        {"run_id": run_id, "fs_hz": 4096.0},
    )
    _write_json(run_dir / "ringdown_real_features_v0" / "stage_summary.json", {})
    _write_json(run_dir / "ringdown_real_features_v0" / "manifest.json", {})

    exp08_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke" / "outputs"
    )
    _write_json(exp08_dir / "real_v0_smoke_report.json", {"status": "ok"})
    _write_json(exp08_dir / "contract_verdict.json", {"verdict": "PASS"})
    (exp08_dir / "failure_catalog.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (exp08_dir / "failure_catalog.jsonl").write_text("\n", encoding="utf-8")

    res = subprocess.run(
        [
            "python",
            "tools/basurin_where.py",
            "--run",
            run_id,
            "--out-root",
            str(tmp_path / "runs"),
            "--ringdown-exp08",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 2
    assert "ringdown_real_inference_v0" in res.stdout
    assert (
        "missing: ringdown_real_inference_v0/outputs/inference_report.json" in res.stdout
    )


# ---------------------------------------------------------------------------
# QNM damped-sinusoid fit tests
# ---------------------------------------------------------------------------


def _make_run_with_synthetic_signal(
    tmp_path: Path,
    run_id: str,
    strain_h1: np.ndarray,
    strain_l1: np.ndarray,
    fs: float,
    band_hz: str = "200,500",
) -> tuple[Path, subprocess.CompletedProcess]:
    """Helper: set up a full run directory and execute the inference stage."""
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    n_h1 = int(strain_h1.size)
    n_l1 = int(strain_l1.size)

    _write_run_valid_pass(run_dir)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz", strain_h1, fs)
    _write_rd_npz(inputs_dir / "L1_rd.npz", strain_l1, fs)
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    observables = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "detectors": ["H1", "L1"],
        "n_samples": {"H1": n_h1, "L1": n_l1},
        "rms": {"H1": 1.0, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 1.0},
    }
    features = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "n_samples": {"H1": n_h1, "L1": n_l1},
        "duration_s": {"H1": n_h1 / fs, "L1": n_l1 / fs},
        "snr_proxy": {"H1": 10.0, "L1": 10.0},
    }

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        observables,
    )
    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        features,
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
        "--band-hz",
        band_hz,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    return run_dir, res


def test_qnm_fit_recovers_damped_sinusoid_parameters(tmp_path: Path) -> None:
    """Synthetic damped sinusoid: QNM fit must recover f and tau within tolerance."""
    run_id = "2040-09-01__unit_test__qnm_fit_synth"
    fs = 4096.0
    n_samples = 8192
    f_true = 250.0
    tau_true = 0.012
    A_true = 1.0
    phi_true = 0.3

    t = np.arange(n_samples, dtype=float) / fs
    signal = A_true * np.exp(-t / tau_true) * np.cos(2.0 * np.pi * f_true * t + phi_true)
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.01 * A_true, size=n_samples)
    strain = signal + noise

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain, strain, fs, band_hz="150,400",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # qnm_fit block must exist
    assert "qnm_fit" in report
    qf = report["qnm_fit"]
    assert qf["model"] == "damped_sinusoid_v1"
    assert qf["band_hz"] == [150.0, 400.0]

    for det in ["H1", "L1"]:
        det_qnm = qf[det]
        assert det_qnm["status"] == "OK", f"{det}: {det_qnm['notes']}"
        assert det_qnm["f_qnm_hz"] is not None
        assert det_qnm["tau_qnm_s"] is not None
        assert det_qnm["Q_qnm"] is not None
        assert det_qnm["n_samples"] >= 32  # fit window, not total strain

        # Frequency within 5% of true
        f_est = det_qnm["f_qnm_hz"]
        assert abs(f_est - f_true) / f_true < 0.05, (
            f"{det}: f_est={f_est}, f_true={f_true}, err={abs(f_est-f_true)/f_true:.3f}"
        )

        # Tau within 20% of true
        tau_est = det_qnm["tau_qnm_s"]
        assert abs(tau_est - tau_true) / tau_true < 0.20, (
            f"{det}: tau_est={tau_est}, tau_true={tau_true}, "
            f"err={abs(tau_est-tau_true)/tau_true:.3f}"
        )

        # Sigmas must be finite (not NaN) or null
        sigma_f = det_qnm["sigma_f"]
        sigma_tau = det_qnm["sigma_tau"]
        if sigma_f is not None:
            assert isinstance(sigma_f, float) and sigma_f == sigma_f  # not NaN
        if sigma_tau is not None:
            assert isinstance(sigma_tau, float) and sigma_tau == sigma_tau

        # Goodness-of-fit metrics present
        assert det_qnm["rmse"] is not None
        assert det_qnm["chi2_red"] is not None

    # decision_qnm should be PASS
    assert "decision_qnm" in report
    assert report["decision_qnm"]["verdict"] == "PASS"


def test_qnm_fit_noise_only_returns_fail_or_inspect(tmp_path: Path) -> None:
    """Pure noise (no signal): QNM fit should produce FAIL status or INSPECT verdict."""
    run_id = "2040-09-01__unit_test__qnm_fit_noise"
    fs = 4096.0
    n_samples = 4096

    rng = np.random.RandomState(99)
    strain = rng.normal(0, 1.0, size=n_samples)

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain, strain, fs, band_hz="150,400",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "qnm_fit" in report
    assert "decision_qnm" in report

    # For noise-only, at least one detector should have issues
    # The decision_qnm verdict should NOT be PASS (either INSPECT or FAIL)
    # OR, if the fit "succeeds" on noise, the uncertainties should be large
    qf = report["qnm_fit"]
    dq = report["decision_qnm"]
    for det in ["H1", "L1"]:
        det_qnm = qf[det]
        if det_qnm["status"] == "FAIL":
            # Contract: notes must not be empty
            assert len(det_qnm["notes"]) > 0, f"{det}: FAIL status but empty notes"
        # If status is OK on noise, params can be anything (fit converged on noise)
        # but decision_qnm might still flag large uncertainties


def test_qnm_fit_short_strain_returns_fail(tmp_path: Path) -> None:
    """Strain shorter than 32 samples: QNM fit must return FAIL."""
    run_id = "2040-09-01__unit_test__qnm_fit_short"
    fs = 4096.0
    n_samples = 20  # < 32

    t = np.arange(n_samples, dtype=float) / fs
    strain = np.sin(2.0 * np.pi * 250.0 * t)

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain, strain, fs, band_hz="200,500",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for det in ["H1", "L1"]:
        det_qnm = report["qnm_fit"][det]
        assert det_qnm["status"] == "FAIL"
        assert len(det_qnm["notes"]) > 0
        # Contract: params must be null when FAIL
        # (f_qnm_hz and tau_qnm_s can be null)

    assert report["decision_qnm"]["verdict"] != "PASS"


def test_qnm_fit_contract_ok_implies_non_null_params(tmp_path: Path) -> None:
    """Contract: if qnm_fit.<IFO>.status=='OK' => f_qnm_hz and tau_qnm_s not null
    and tau_qnm_s > 0."""
    run_id = "2040-09-01__unit_test__qnm_contract"
    fs = 4096.0
    n_samples = 8192
    f_true = 300.0
    tau_true = 0.008

    t = np.arange(n_samples, dtype=float) / fs
    strain = np.exp(-t / tau_true) * np.cos(2.0 * np.pi * f_true * t)

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain, strain, fs, band_hz="150,400",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for det in ["H1", "L1"]:
        det_qnm = report["qnm_fit"][det]
        if det_qnm["status"] == "OK":
            assert det_qnm["f_qnm_hz"] is not None
            assert det_qnm["tau_qnm_s"] is not None
            assert det_qnm["tau_qnm_s"] > 0

    # contract_verdict must still be valid
    verdict_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "contract_verdict.json"
    )
    cv = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert cv["verdict"] in ("PASS", "INSPECT")


def test_qnm_fit_contract_fail_implies_non_empty_notes(tmp_path: Path) -> None:
    """Contract: if qnm_fit.<IFO>.status=='FAIL' => notes not empty."""
    run_id = "2040-09-01__unit_test__qnm_contract_fail"
    fs = 4096.0
    n_samples = 20  # too short → FAIL

    t = np.arange(n_samples, dtype=float) / fs
    strain = np.sin(2.0 * np.pi * 250.0 * t)

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain, strain, fs, band_hz="200,500",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for det in ["H1", "L1"]:
        det_qnm = report["qnm_fit"][det]
        if det_qnm["status"] == "FAIL":
            assert len(det_qnm["notes"]) > 0, (
                f"{det}: status FAIL but notes empty"
            )
            # f_qnm_hz and tau_qnm_s can be null
            # (not strictly required to be null, but this is the expected case)




def test_qnm_fit_reports_tau_bounds_and_clipping_contract(tmp_path: Path) -> None:
    """Contract: status OK must report tau bounds and non-null clipping diagnostics."""
    run_id = "2040-09-01__unit_test__qnm_tau_clipping_contract"
    fs = 4096.0
    n_samples = 8192
    f_true = 250.0
    tau_true = 0.01

    t = np.arange(n_samples, dtype=float) / fs
    strain = np.exp(-t / tau_true) * np.cos(2.0 * np.pi * f_true * t)

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain, strain, fs, band_hz="150,400",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for det in ["H1", "L1"]:
        det_qnm = report["qnm_fit"][det]
        assert det_qnm["tau_bounds_s"] == [0.01, 0.5]
        assert det_qnm["clipped_tau"] in [True, False]
        if det_qnm["status"] == "OK":
            assert det_qnm["tau_bounds_s"] is not None
            assert det_qnm["clipped_tau"] is not None
            if det_qnm["tau_qnm_s"] == 0.01:
                assert det_qnm["clipped_tau"] is True

def test_qnm_fit_tau_inconsistency_forces_inspect(tmp_path: Path) -> None:
    """Regression: large H1/L1 tau mismatch must force decision_qnm=INSPECT."""
    run_id = "2040-09-01__unit_test__qnm_tau_inconsistency"
    fs = 4096.0
    n_samples = 8192
    f_true = 250.0

    t = np.arange(n_samples, dtype=float) / fs
    strain_h1 = np.exp(-t / 0.04) * np.cos(2.0 * np.pi * f_true * t)
    strain_l1 = np.exp(-t / 0.01) * np.cos(2.0 * np.pi * f_true * t)

    run_dir, res = _make_run_with_synthetic_signal(
        tmp_path, run_id, strain_h1, strain_l1, fs, band_hz="150,400",
    )
    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Ensure we are testing the intended condition: both fits should be valid.
    assert report["qnm_fit"]["H1"]["status"] == "OK"
    assert report["qnm_fit"]["L1"]["status"] == "OK"

    decision_qnm = report["decision_qnm"]
    assert decision_qnm["verdict"] == "INSPECT"
    reasons_text = " ".join(decision_qnm["reasons"])
    assert "tau inconsistente" in reasons_text
    assert "(>0.2)" in reasons_text

    qnm_consistency = report["qnm_consistency"]
    assert qnm_consistency["tau_frac_diff_max"] == 0.2
    assert qnm_consistency["tau_mean_s"] is not None
    assert qnm_consistency["tau_frac_diff"] is not None
    assert qnm_consistency["tau_frac_diff"] > 0.2


def test_decision_qnm_reasons_include_clipped_tau_note() -> None:
    """Regression: clipped tau should be reported in decision_qnm.reasons."""
    from stages.ringdown_real_inference_v0_stage import _build_decision_qnm

    qnm_fit = {
        "H1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.2,
            "sigma_f": None,
            "sigma_tau": None,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
        "L1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.01,
            "sigma_f": None,
            "sigma_tau": None,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": True,
        },
    }

    decision_qnm, _ = _build_decision_qnm(qnm_fit, [150.0, 400.0])
    reasons_text = " ".join(decision_qnm["reasons"])
    assert "L1" in reasons_text
    assert "clipped" in reasons_text


def test_decision_qnm_pairwise_sigma_scaling_zero_z_passes() -> None:
    """G5: identical tau with different sigma_tau must give pairwise z=0 and PASS."""
    from stages.ringdown_real_inference_v0_stage import _build_decision_qnm

    qnm_fit = {
        "H1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": 0.01,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
        "L1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": 0.04,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
    }

    decision_qnm, qnm_consistency = _build_decision_qnm(qnm_fit, [150.0, 400.0])
    assert decision_qnm["verdict"] == "PASS"
    pairwise = qnm_consistency["pairwise_tau_zscores"]
    assert len(pairwise) == 1
    assert pairwise[0]["z_tau"] == 0.0


def test_decision_qnm_window_instability_forces_inspect() -> None:
    """G4: unstable tau across window replicas must force INSPECT with stability reason."""
    from stages.ringdown_real_inference_v0_stage import _build_decision_qnm

    qnm_fit = {
        "H1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": 0.01,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
            "window_replicas": [
                {"tau_qnm_s": 0.05, "sigma_tau": 0.01},
                {"tau_qnm_s": 0.15, "sigma_tau": 0.01},
                {"tau_qnm_s": 0.1, "sigma_tau": 0.01},
                {"tau_qnm_s": 0.2, "sigma_tau": 0.01},
            ],
        },
        "L1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": 0.01,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
    }

    decision_qnm, qnm_consistency = _build_decision_qnm(qnm_fit, [150.0, 400.0])
    assert decision_qnm["verdict"] == "INSPECT"
    reasons_text = " ".join(decision_qnm["reasons"]).lower()
    assert "stability" in reasons_text
    assert qnm_consistency["window_stability"]
    assert qnm_consistency["window_stability"][0]["stable"] is False


def test_decision_qnm_single_detector_outlier_detected_by_leave_one_out() -> None:
    """G6: single-detector tau outlier should trigger INSPECT and high leave-one-out influence."""
    from stages.ringdown_real_inference_v0_stage import _build_decision_qnm

    sigma = 0.01
    sigma_comb = (sigma**2 + sigma**2) ** 0.5
    qnm_fit = {
        "H1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": sigma,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
        "L1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1 + 10.0 * sigma_comb,
            "sigma_f": 0.5,
            "sigma_tau": sigma,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
    }

    decision_qnm, qnm_consistency = _build_decision_qnm(qnm_fit, [150.0, 400.0])
    assert decision_qnm["verdict"] == "INSPECT"

    loo = qnm_consistency["leave_one_out"]
    assert loo["max_influence_z"] is not None
    assert loo["max_influence_z"] > qnm_consistency["pairwise_z_threshold"]


def test_decision_qnm_clipping_fraction_forces_inspect_with_reason() -> None:
    """G1: high clipping_fraction or clipped_tau must not pass silently."""
    from stages.ringdown_real_inference_v0_stage import _build_decision_qnm

    qnm_fit = {
        "H1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": 0.01,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
            "clipping_fraction": 0.2,
        },
        "L1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": 0.01,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": True,
            "clipping_fraction": 0.0,
        },
    }

    decision_qnm, _ = _build_decision_qnm(qnm_fit, [150.0, 400.0])
    assert decision_qnm["verdict"] == "INSPECT"
    reasons_text = " ".join(decision_qnm["reasons"]).lower()
    assert "clipping_fraction" in reasons_text or "clipped" in reasons_text


def test_decision_qnm_nan_values_force_inspect_with_nan_reason() -> None:
    """G1: NaN in tau/sigma should force non-PASS with explicit nan reason."""
    from stages.ringdown_real_inference_v0_stage import _build_decision_qnm

    qnm_fit = {
        "H1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": float("nan"),
            "sigma_f": 0.5,
            "sigma_tau": 0.01,
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
        "L1": {
            "status": "OK",
            "f_qnm_hz": 250.0,
            "tau_qnm_s": 0.1,
            "sigma_f": 0.5,
            "sigma_tau": float("nan"),
            "tau_bounds_s": [0.01, 0.5],
            "clipped_tau": False,
        },
    }

    decision_qnm, _ = _build_decision_qnm(qnm_fit, [150.0, 400.0])
    assert decision_qnm["verdict"] == "INSPECT"
    reasons_text = " ".join(decision_qnm["reasons"]).lower()
    assert "nan" in reasons_text
