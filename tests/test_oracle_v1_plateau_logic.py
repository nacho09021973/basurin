from __future__ import annotations

import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.oracles.oracle_v1_plateau import WindowMetrics, oracle_v1_plateau_report


def _wm(*, t0: float, f: float, tau: float, cond: float, bic: float, p: float | None, n: int = 128, snr: float = 20.0) -> WindowMetrics:
    return WindowMetrics(
        t0=t0,
        T=0.1,
        f_median=f,
        f_sigma=1.0,
        tau_median=tau,
        tau_sigma=1.0,
        cond_number=cond,
        delta_bic=bic,
        p_ljungbox=p,
        n_samples=n,
        snr=snr,
        chi2_coh=1.0,
    )


def _stable_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_no_valid_window_all_gates_fail() -> None:
    windows = [
        _wm(t0=0, f=100, tau=5, cond=150, bic=1, p=0.001),
        _wm(t0=1, f=101, tau=6, cond=130, bic=2, p=0.0005),
        _wm(t0=2, f=102, tau=7, cond=120, bic=3, p=0.0001),
    ]

    report = oracle_v1_plateau_report(windows)

    assert report["final_verdict"] == "FAIL"
    assert report["fail_global_reason"] == "NO_VALID_WINDOW"
    assert report["candidates"] == []
    assert report["chosen_t0"] is None


def test_no_plateau_has_pass_but_not_stable_consecutive() -> None:
    windows = [
        _wm(t0=0, f=10, tau=10, cond=10, bic=20, p=0.5),
        _wm(t0=1, f=20, tau=20, cond=10, bic=20, p=0.5),
        _wm(t0=2, f=11, tau=11, cond=10, bic=20, p=0.5),
        _wm(t0=3, f=21, tau=21, cond=10, bic=20, p=0.5),
    ]

    report = oracle_v1_plateau_report(windows)

    assert report["final_verdict"] == "FAIL"
    assert report["fail_global_reason"] == "NO_PLATEAU"
    assert len([w for w in report["windows_summary"] if w["verdict"] == "PASS"]) == 4
    assert report["candidates"] == []


def test_pass_plateau_at_i2_and_golden_json_is_stable() -> None:
    windows = [
        _wm(t0=0, f=0, tau=0, cond=200, bic=1, p=0.001),
        _wm(t0=1, f=100, tau=100, cond=200, bic=1, p=0.001),
        _wm(t0=2, f=10.0, tau=10.0, cond=10, bic=20, p=0.5),
        _wm(t0=3, f=11.0, tau=11.0, cond=10, bic=20, p=0.5),
        _wm(t0=4, f=12.0, tau=12.0, cond=10, bic=20, p=0.5),
    ]

    report = oracle_v1_plateau_report(windows)

    assert report["final_verdict"] == "PASS"
    assert report["chosen_t0"] == 2
    assert report["chosen_T"] == 0.1
    assert report["candidates"][0]["plateau_indices"] == [2, 3, 4]

    golden = _stable_json(report)
    assert golden == _stable_json(json.loads(golden))
