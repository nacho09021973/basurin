from __future__ import annotations

from mvp.oracles.oracle_v1_plateau import (
    FAIL_COH,
    FAIL_COV_BAD,
    FAIL_LOW_SNR,
    FAIL_WHITE,
    WARN_221_UNSTABLE,
    WARN_COH_UNAVAILABLE,
    WARN_COND_UNAVAILABLE,
    WARN_WHITE_LOW_POWER,
    WARN_WHITE_UNAVAILABLE,
    GateConfig,
    WindowMetrics,
    evaluate_gates,
)


def _window(**overrides: object) -> WindowMetrics:
    base = dict(
        t0=0.0,
        T=1.0,
        f_median=250.0,
        f_sigma=0.5,
        tau_median=0.004,
        tau_sigma=0.0001,
        cond_number=10.0,
        delta_bic=20.0,
        p_ljungbox=0.5,
        n_samples=512,
        snr=20.0,
        chi2_coh=2.0,
        flags_221=[],
    )
    base.update(overrides)
    return WindowMetrics(**base)


def test_reason_fail_low_snr() -> None:
    ok, fails, _ = evaluate_gates(_window(snr=2.0), GateConfig(snr_min=8.0))
    assert not ok
    assert FAIL_LOW_SNR in fails


def test_reason_fail_cov_bad() -> None:
    ok, fails, _ = evaluate_gates(_window(cond_number=999.0), GateConfig(cond_max=100.0))
    assert not ok
    assert FAIL_COV_BAD in fails


def test_reason_fail_white() -> None:
    ok, fails, _ = evaluate_gates(_window(p_ljungbox=1e-6, n_samples=200), GateConfig(p_white_min=0.01))
    assert not ok
    assert FAIL_WHITE in fails


def test_reason_fail_coh() -> None:
    ok, fails, _ = evaluate_gates(_window(chi2_coh=50.0), GateConfig(chi2_max=10.0))
    assert not ok
    assert FAIL_COH in fails


def test_reason_warn_cond_unavailable() -> None:
    ok, _, warns = evaluate_gates(_window(cond_number=None), GateConfig())
    assert ok
    assert WARN_COND_UNAVAILABLE in warns


def test_reason_warn_white_unavailable() -> None:
    ok, _, warns = evaluate_gates(_window(p_ljungbox=None, n_samples=200), GateConfig())
    assert ok
    assert WARN_WHITE_UNAVAILABLE in warns


def test_reason_warn_white_low_power() -> None:
    ok, _, warns = evaluate_gates(_window(n_samples=10, p_ljungbox=None), GateConfig(n_low=100))
    assert ok
    assert WARN_WHITE_LOW_POWER in warns


def test_reason_warn_coh_unavailable() -> None:
    ok, _, warns = evaluate_gates(_window(chi2_coh=None), GateConfig())
    assert ok
    assert WARN_COH_UNAVAILABLE in warns


def test_reason_warn_221_unstable_without_gating() -> None:
    ok, fails, warns = evaluate_gates(_window(flags_221=["221_cv_Q_explosive"]), GateConfig(gate_221=False))
    assert ok
    assert not fails
    assert WARN_221_UNSTABLE in warns


def test_reason_warn_221_unstable_with_gating_enabled() -> None:
    ok, fails, warns = evaluate_gates(_window(flags_221=["221_cv_Q_explosive"]), GateConfig(gate_221=True))
    assert not ok
    assert FAIL_COV_BAD in fails
    assert WARN_221_UNSTABLE in warns
