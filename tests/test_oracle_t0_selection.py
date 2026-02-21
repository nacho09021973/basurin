from __future__ import annotations

import json

from mvp.oracle_t0_selection import select_t0


def _stable_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_monotonicidad_con_k_y_thresholds() -> None:
    windows = [{"t0_ms": t0} for t0 in (0, 2, 4, 6, 8, 10)]
    metrics = [
        {"z_grad": 0.20, "cv_max": 0.03},
        {"z_grad": 0.40, "cv_max": 0.04},
        {"z_grad": 0.30, "cv_max": 0.06},
        {"z_grad": 0.35, "cv_max": 0.07},
        {"z_grad": 0.50, "cv_max": 0.08},
        {"z_grad": 0.60, "cv_max": 0.09},
    ]
    gates = ["PASS"] * 6

    base = select_t0(windows, metrics, gates, {"K_consec": 3, "z_grad_max": 0.5, "cv_thr": 0.10, "cv_gold": 0.05})
    stricter_k = select_t0(windows, metrics, gates, {"K_consec": 4, "z_grad_max": 0.5, "cv_thr": 0.10, "cv_gold": 0.05})
    stricter_thr = select_t0(windows, metrics, gates, {"K_consec": 3, "z_grad_max": 0.25, "cv_thr": 0.05, "cv_gold": 0.04})

    assert base["final_verdict"] == "PASS"
    assert stricter_k["final_verdict"] in {"PASS", "FAIL"}
    if stricter_k["final_verdict"] == "PASS":
        assert stricter_k["chosen_index"] >= base["chosen_index"]

    assert stricter_thr["final_verdict"] == "FAIL"
    assert stricter_thr["fail_global_reason"] in {"NO_CONSECUTIVE", "NO_PLATEAU"}


def test_determinismo_snapshot_json_normalizado() -> None:
    windows = [{"t0": 1.0}, {"t0": 2.0}, {"t0": 3.0}, {"t0": 4.0}, {"t0": 5.0}]
    metrics = [
        {"cv_max": 0.08, "z_grad": None},
        {"z_grad": 0.40, "cv_max": 0.04},
        {"cv_max": 0.04, "z_grad": 0.20},
        {"z_grad": 0.30, "cv_max": 0.04},
        {"z_grad": 0.20, "cv_max": 0.04},
    ]
    gates = [True, True, True, True, True]
    config = {"z_grad_max": 2.45, "cv_thr": 0.10, "cv_gold": 0.05, "K_consec": 3}

    report_a = select_t0(windows, metrics, gates, config)
    report_b = select_t0(windows, metrics, gates, config)

    snapshot = _stable_json(report_a)
    assert snapshot == _stable_json(report_b)
    assert report_a["final_verdict"] == "PASS"
    assert report_a["chosen_t0"] == 2.0
    assert report_a["quality"] == "GOLD"
