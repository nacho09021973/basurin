from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - env dependent
    raise unittest.SkipTest("integration requires numpy")

import mvp.s3b_multimode_estimates as s3b


def test_evaluate_mode_lnq_span_gate(monkeypatch) -> None:
    n = 200
    rng = np.random.default_rng(0)
    lnf = rng.normal(loc=5.0, scale=0.01, size=n)
    lnq = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 3.0)])
    samples = np.column_stack([lnf, lnq]).astype(float)

    def fake_bootstrap(signal, fs, estimator, *, n_bootstrap, rng):
        _ = signal, fs, estimator, rng
        return samples[:n_bootstrap], 0

    monkeypatch.setattr(s3b, "_bootstrap_mode_log_samples", fake_bootstrap)

    signal = np.ones(4096, dtype=float)
    fs = 4096.0

    _, flags_strict, ok_strict = s3b.evaluate_mode(
        signal,
        fs,
        label="221",
        mode=[2, 2, 1],
        estimator=lambda *_: {"ln_f": 0.0, "ln_Q": 0.0},
        n_bootstrap=200,
        seed=123,
        min_valid_fraction=0.0,
        max_lnf_span=1.0,
        max_lnq_span=1.0,
        min_point_samples=50,
        min_point_valid_fraction=0.5,
        cv_threshold=None,
    )
    assert ok_strict is False
    assert "221_lnQ_span_explosive" in flags_strict

    _, flags_relaxed, ok_relaxed = s3b.evaluate_mode(
        signal,
        fs,
        label="221",
        mode=[2, 2, 1],
        estimator=lambda *_: {"ln_f": 0.0, "ln_Q": 0.0},
        n_bootstrap=200,
        seed=123,
        min_valid_fraction=0.0,
        max_lnf_span=1.0,
        max_lnq_span=4.0,
        min_point_samples=50,
        min_point_valid_fraction=0.5,
        cv_threshold=None,
    )
    assert ok_relaxed is True
    assert "221_lnQ_span_explosive" not in flags_relaxed
