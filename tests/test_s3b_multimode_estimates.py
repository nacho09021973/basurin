from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "mvp" / "s3b_multimode_estimates.py"
_SPEC = importlib.util.spec_from_file_location("mvp_s3b_multimode_estimates", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_results_payload = _MODULE.build_results_payload
covariance_gate = _MODULE.covariance_gate
compute_robust_stability = _MODULE.compute_robust_stability
evaluate_mode = _MODULE.evaluate_mode
_discover_s2_npz = _MODULE._discover_s2_npz
_load_signal_from_npz = _MODULE._load_signal_from_npz

_discover_s2_window_meta = _MODULE._discover_s2_window_meta


def test_mode_keeps_point_estimate_when_gates_fail() -> None:
    signal = np.linspace(-1.0, 1.0, 4096)

    def estimator(_signal: np.ndarray, _fs: float) -> dict[str, float]:
        return {"f_hz": 220.0, "Q": 12.0, "tau_s": 12.0 / (np.pi * 220.0)}

    explosive_samples = np.array(
        [[np.log(220.0), x] for x in np.linspace(np.log(2.0), np.log(120.0), 80)],
        dtype=float,
    )

    original = _MODULE._bootstrap_mode_log_samples
    _MODULE._bootstrap_mode_log_samples = lambda *_args, **_kwargs: (explosive_samples, 0)
    try:
        mode, flags, ok = evaluate_mode(
            signal,
            4096.0,
            label="220",
            mode=[2, 2, 0],
            estimator=estimator,
            n_bootstrap=80,
            seed=9,
            min_valid_fraction=0.95,
            max_lnf_span=0.5,
            max_lnq_span=0.2,
            min_point_samples=50,
            min_point_valid_fraction=0.5,
        )
    finally:
        _MODULE._bootstrap_mode_log_samples = original

    assert not ok
    assert "220_lnQ_span_explosive" in flags
    assert mode["ln_f"] is not None
    assert mode["ln_Q"] is not None


def test_discover_s2_window_meta_from_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_x"
    stage_dir = run_dir / "s2_ringdown_window"
    out_dir = stage_dir / "outputs"
    out_dir.mkdir(parents=True)
    meta_path = out_dir / "window_meta.json"
    meta_path.write_text(json.dumps({"sample_rate_hz": 4096.0}), encoding="utf-8")
    (stage_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"window_meta": "s2_ringdown_window/outputs/window_meta.json"}}),
        encoding="utf-8",
    )

    assert _discover_s2_window_meta(run_dir) == meta_path


def test_main_populates_source_window_and_avoids_missing_flag(tmp_path: Path) -> None:
    run_id = "subrun_003"
    tmp_runs = tmp_path / "subruns_root"
    _write_minimal_s2_inputs(tmp_runs, run_id)
    run_dir = tmp_runs / run_id

    meta = {"sample_rate_hz": 4096.0, "offset_ms": 0}
    meta_path = run_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "s2_ringdown_window" / "manifest.json").write_text(
        json.dumps({
            "artifacts": {
                "H1_rd": "s2_ringdown_window/outputs/H1_rd.npz",
                "window_meta": "s2_ringdown_window/outputs/window_meta.json",
            }
        }),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "mvp" / "s3b_multimode_estimates.py"),
        "--runs-root",
        str(tmp_runs),
        "--run-id",
        run_id,
        "--n-bootstrap",
        "8",
        "--seed",
        "101",
    ]

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    output_path = tmp_runs / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"]["window"] == meta
    assert "missing_window_meta" not in payload["results"]["quality_flags"]


def _stable_estimator(signal: np.ndarray, fs: float) -> dict[str, float]:
    _ = signal, fs
    return {"f_hz": 250.0, "Q": 12.0, "tau_s": 12.0 / (np.pi * 250.0)}


def _sometimes_failing_estimator(signal: np.ndarray, fs: float) -> dict[str, float]:
    if float(np.mean(signal)) > 0.0:
        raise ValueError("unstable")
    return _stable_estimator(signal, fs)


def test_gates_reject_singular_sigma() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    ok, reasons = covariance_gate(singular)
    assert not ok
    assert "Sigma_not_invertible" in reasons


def test_determinism_seed_on_bootstrap() -> None:
    signal = np.cos(np.linspace(0, 8 * np.pi, 4000))
    mode_a, flags_a, ok_a = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=40,
        seed=12345,
    )
    mode_b, flags_b, ok_b = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=40,
        seed=12345,
    )

    assert ok_a and ok_b
    assert flags_a == flags_b
    assert mode_a["fit"]["stability"] == mode_b["fit"]["stability"]
    assert mode_a["ln_f"] == mode_b["ln_f"]
    assert mode_a["ln_Q"] == mode_b["ln_Q"]
    assert mode_a["Sigma"] == mode_b["Sigma"]


def test_verdict_insufficient_when_missing_221() -> None:
    signal = np.ones(2048)
    mode_220, flags_220, ok_220 = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=20,
        seed=12345,
    )
    mode_221, flags_221, ok_221 = evaluate_mode(
        signal,
        4096.0,
        label="221",
        mode=[2, 2, 1],
        estimator=_sometimes_failing_estimator,
        n_bootstrap=20,
        seed=12346,
        min_valid_fraction=0.8,
        cv_threshold=1.0,
    )

    payload = build_results_payload(
        run_id="run_x",
        window_meta=None,
        mode_220=mode_220,
        mode_220_ok=ok_220,
        mode_221=mode_221,
        mode_221_ok=ok_221,
        flags=flags_220 + flags_221,
    )

    assert payload["results"]["verdict"] == "INSUFFICIENT_DATA"
    assert payload["modes"][1]["label"] == "221"
    assert payload["modes"][1]["ln_f"] is None
    assert "221_Sigma_invalid" in payload["results"]["quality_flags"]


def test_discover_s2_npz_prefers_h1_then_l1(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_x"
    stage_dir = run_dir / "s2_ringdown_window"
    out_dir = stage_dir / "outputs"
    out_dir.mkdir(parents=True)
    h1 = out_dir / "H1_rd.npz"
    l1 = out_dir / "L1_rd.npz"
    np.savez(h1, strain=np.ones(32), sample_rate_hz=np.array([4096.0]))
    np.savez(l1, strain=np.ones(32), sample_rate_hz=np.array([4096.0]))
    (stage_dir / "manifest.json").write_text(
        json.dumps({
            "artifacts": {
                "H1_rd": "s2_ringdown_window/outputs/H1_rd.npz",
                "L1_rd": "s2_ringdown_window/outputs/L1_rd.npz",
            }
        }),
        encoding="utf-8",
    )

    assert _discover_s2_npz(run_dir) == h1


def test_load_signal_from_npz_uses_window_meta_sample_rate(tmp_path: Path) -> None:
    npz_path = tmp_path / "H1_rd.npz"
    signal = np.linspace(-1.0, 1.0, 64)
    np.savez(npz_path, strain=signal)

    loaded, fs = _load_signal_from_npz(npz_path, window_meta={"sample_rate_hz": 2048.0})
    assert np.allclose(loaded, signal)
    assert fs == 2048.0


def test_evaluate_mode_short_window_flags_bootstrap_high_nonpositive() -> None:
    signal = np.ones(32, dtype=float)
    mode, flags, ok = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=20,
        seed=12345,
    )

    assert not ok
    assert "bootstrap_high_nonpositive" in flags
    assert mode["fit"]["stability"]["message"] == "window too short after offset"


def test_evaluate_mode_invalid_block_size_flags_bootstrap_block_invalid() -> None:
    signal = np.ones(130, dtype=float)

    original = _MODULE._bootstrap_block_size
    _MODULE._bootstrap_block_size = lambda _: 130
    try:
        mode, flags, ok = evaluate_mode(
            signal,
            4096.0,
            label="220",
            mode=[2, 2, 0],
            estimator=_stable_estimator,
            n_bootstrap=20,
            seed=12345,
        )
    finally:
        _MODULE._bootstrap_block_size = original

    assert not ok
    assert "bootstrap_block_invalid" in flags
    assert mode["fit"]["stability"]["message"] == "window too short after offset"


def test_compute_robust_stability_quantiles_are_deterministic() -> None:
    samples = [
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (5.0, 6.0),
    ]
    stability = compute_robust_stability(samples)

    assert np.isclose(stability["lnf_p10"], 1.4)
    assert np.isclose(stability["lnf_p50"], 3.0)
    assert np.isclose(stability["lnf_p90"], 4.6)
    assert np.isclose(stability["lnQ_p10"], 2.4)
    assert np.isclose(stability["lnQ_p50"], 4.0)
    assert np.isclose(stability["lnQ_p90"], 5.6)
    assert np.isclose(stability["lnf_span"], 3.2)
    assert np.isclose(stability["lnQ_span"], 3.2)


def test_cv_q_high_but_lnq_span_moderate_does_not_invalidate() -> None:
    signal = np.linspace(-1.0, 1.0, 2048)

    def estimator(_signal: np.ndarray, _fs: float) -> dict[str, float]:
        return {"f_hz": 220.0, "Q": 12.0, "tau_s": 12.0 / (np.pi * 220.0)}

    outlier_samples = np.array(
        [
            [np.log(220.0), np.log(8.0)],
            [np.log(220.5), np.log(8.2)],
            [np.log(221.0), np.log(8.4)],
            [np.log(219.5), np.log(8.6)],
            [np.log(220.2), np.log(60.0)],
        ],
        dtype=float,
    )

    original = _MODULE._bootstrap_mode_log_samples
    _MODULE._bootstrap_mode_log_samples = lambda *_args, **_kwargs: (outlier_samples, 0)
    try:
        mode, flags, ok = evaluate_mode(
            signal,
            4096.0,
            label="220",
            mode=[2, 2, 0],
            estimator=estimator,
            n_bootstrap=5,
            seed=7,
            cv_threshold=0.5,
            max_lnf_span=0.1,
            max_lnq_span=2.5,
        )
    finally:
        _MODULE._bootstrap_mode_log_samples = original

    assert ok
    assert "220_cv_Q_explosive" in flags
    assert mode["ln_Q"] is not None


def test_sigma_pathological_invalidates_mode_and_reports_flag() -> None:
    signal = np.linspace(-1.0, 1.0, 2048)

    def estimator(_signal: np.ndarray, _fs: float) -> dict[str, float]:
        return {"f_hz": 220.0, "Q": 12.0, "tau_s": 12.0 / (np.pi * 220.0)}

    singular_samples = np.array(
        [
            [np.log(220.0), np.log(12.0)],
            [np.log(221.0), np.log(13.0)],
            [np.log(222.0), np.log(14.0)],
            [np.log(223.0), np.log(15.0)],
        ],
        dtype=float,
    )
    singular_samples[:, 1] = singular_samples[:, 0] + 0.5

    original = _MODULE._bootstrap_mode_log_samples
    _MODULE._bootstrap_mode_log_samples = lambda *_args, **_kwargs: (singular_samples, 0)
    try:
        mode, flags, ok = evaluate_mode(
            signal,
            4096.0,
            label="220",
            mode=[2, 2, 0],
            estimator=estimator,
            n_bootstrap=4,
            seed=8,
            max_lnf_span=5.0,
            max_lnq_span=5.0,
        )
    finally:
        _MODULE._bootstrap_mode_log_samples = original

    assert not ok
    assert "220_Sigma_invalid" in flags
    assert mode["ln_f"] is None
    assert mode["Sigma"] is None
    assert mode["fit"]["stability"]["lnf_p50"] is not None


def _write_minimal_s2_inputs(run_root: Path, run_id: str) -> None:
    run_dir = run_root / run_id
    rv_dir = run_dir / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    s2_outputs = run_dir / "s2_ringdown_window" / "outputs"
    s2_outputs.mkdir(parents=True, exist_ok=True)
    signal = np.cos(np.linspace(0, 8 * np.pi, 2048))
    np.savez(s2_outputs / "H1_rd.npz", strain=signal, sample_rate_hz=np.array([4096.0]))

    (run_dir / "s2_ringdown_window" / "manifest.json").write_text(
        json.dumps({"artifacts": {"H1_rd": "s2_ringdown_window/outputs/H1_rd.npz"}}),
        encoding="utf-8",
    )


def test_cli_runs_root_writes_under_explicit_root_only(tmp_path: Path) -> None:
    run_id = "subrun_001"
    tmp_runs = tmp_path / "subruns_root"
    _write_minimal_s2_inputs(tmp_runs, run_id)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "mvp" / "s3b_multimode_estimates.py"),
        "--runs-root",
        str(tmp_runs),
        "--run-id",
        run_id,
        "--n-bootstrap",
        "8",
        "--seed",
        "101",
    ]
    env = os.environ.copy()
    env.pop("BASURIN_RUNS_ROOT", None)

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr

    stage_dir = tmp_runs / run_id / "s3b_multimode_estimates"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "outputs" / "multimode_estimates.json").exists()

    assert not (repo_root / "runs" / run_id / "s3b_multimode_estimates").exists()


def test_cli_runs_root_still_enforces_run_valid_gate(tmp_path: Path) -> None:
    run_id = "subrun_002"
    tmp_runs = tmp_path / "subruns_root"
    run_dir = tmp_runs / run_id
    run_dir.mkdir(parents=True)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "mvp" / "s3b_multimode_estimates.py"),
        "--runs-root",
        str(tmp_runs),
        "--run-id",
        run_id,
        "--n-bootstrap",
        "4",
    ]
    env = os.environ.copy()
    env.pop("BASURIN_RUNS_ROOT", None)

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "RUN_VALID" in result.stderr
