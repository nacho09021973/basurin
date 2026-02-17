from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "mvp" / "s3b_multimode_estimates.py"
_SPEC = importlib.util.spec_from_file_location("mvp_s3b_multimode_estimates", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_results_payload = _MODULE.build_results_payload
covariance_gate = _MODULE.covariance_gate
evaluate_mode = _MODULE.evaluate_mode
_discover_s2_npz = _MODULE._discover_s2_npz
_load_signal_from_npz = _MODULE._load_signal_from_npz


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
    assert mode_a == mode_b


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
