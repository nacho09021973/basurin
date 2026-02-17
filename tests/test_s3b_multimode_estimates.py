from __future__ import annotations

import numpy as np

from mvp.s3b_multimode_estimates import (
    build_results_payload,
    compute_covariance,
    covariance_gate,
    evaluate_mode,
)


def _stable_estimator(signal: np.ndarray, fs: float) -> dict[str, float]:
    _ = signal, fs
    return {"f_hz": 250.0, "Q": 12.0, "tau_s": 12.0 / (np.pi * 250.0)}


def test_extract_from_compatible_signal_schema_insufficient() -> None:
    signal = np.sin(np.linspace(0, 8 * np.pi, 2048))
    mode_220, flags_220 = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=40,
        seed=12345,
    )

    payload = build_results_payload("run_x", None, mode_220, None, flags_220)
    assert payload["results"]["verdict"] == "INSUFFICIENT_DATA"
    assert mode_220 is not None


def test_sigma_gates_reject_singular() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    ok, reasons = covariance_gate(singular)
    assert not ok
    assert "Sigma_not_invertible" in reasons


def test_determinism_seed() -> None:
    signal = np.cos(np.linspace(0, 10 * np.pi, 3000))
    mode_a, _ = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=50,
        seed=777,
    )
    mode_b, _ = evaluate_mode(
        signal,
        4096.0,
        label="220",
        mode=[2, 2, 0],
        estimator=_stable_estimator,
        n_bootstrap=50,
        seed=777,
    )
    assert mode_a is not None and mode_b is not None
    assert mode_a["ln_f"] == mode_b["ln_f"]
    assert mode_a["ln_Q"] == mode_b["ln_Q"]
    assert mode_a["Sigma"] == mode_b["Sigma"]

    samples = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.8]], dtype=float)
    cov = compute_covariance(samples)
    assert cov.shape == (2, 2)
