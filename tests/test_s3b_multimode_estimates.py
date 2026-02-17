from __future__ import annotations

import numpy as np

from mvp.s3b_multimode_estimates import build_results_payload, covariance_gate, evaluate_mode


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
