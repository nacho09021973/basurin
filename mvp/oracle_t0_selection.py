from __future__ import annotations

from typing import Any

DEFAULT_Z_GRAD_MAX = 2.45
DEFAULT_CV_THR = 0.10
DEFAULT_CV_GOLD = 0.05
DEFAULT_K_CONSEC = 3



def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    return {
        "z_grad_max": float(cfg.get("z_grad_max", DEFAULT_Z_GRAD_MAX)),
        "cv_thr": float(cfg.get("cv_thr", DEFAULT_CV_THR)),
        "cv_gold": float(cfg.get("cv_gold", DEFAULT_CV_GOLD)),
        "K_consec": int(cfg.get("K_consec", DEFAULT_K_CONSEC)),
    }



def _is_pass_gate(value: Any) -> bool:
    if isinstance(value, str):
        return value.upper() == "PASS"
    return bool(value)



def _to_number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out



def select_t0(
    windows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    gates: list[bool | str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Selección final del Oráculo t0 v1.2 basada en bloque consecutivo más temprano."""

    cfg = _normalize_config(config)
    n_windows = len(windows)

    if len(metrics) != n_windows or len(gates) != n_windows:
        raise ValueError("windows, metrics y gates deben tener la misma longitud")

    windows_summary: list[dict[str, Any]] = []
    has_gate_pass = False
    has_candidate = False

    for idx, win in enumerate(windows):
        gate_pass = _is_pass_gate(gates[idx])
        has_gate_pass = has_gate_pass or gate_pass

        metric = metrics[idx]
        z_grad = _to_number_or_none(metric.get("z_grad"))
        cv_max = _to_number_or_none(metric.get("cv_max"))

        z_ok = z_grad is not None and z_grad <= cfg["z_grad_max"]
        cv_ok = cv_max is not None and cv_max <= cfg["cv_thr"]
        is_candidate = gate_pass and z_ok and cv_ok
        has_candidate = has_candidate or is_candidate

        windows_summary.append(
            {
                "index": idx,
                "t0": win.get("t0") if "t0" in win else win.get("t0_ms"),
                "gate_pass": gate_pass,
                "z_grad": z_grad,
                "cv_max": cv_max,
                "candidate": is_candidate,
            }
        )

    fail_global_reason: str | None = None
    final_verdict = "FAIL"
    chosen_t0: float | None = None
    chosen_index: int | None = None
    quality: str | None = None
    chosen_block_indices: list[int] = []

    k_consec = cfg["K_consec"]
    if n_windows < (k_consec + 2):
        fail_global_reason = "INSUFFICIENT_WINDOWS"
    elif not has_gate_pass:
        fail_global_reason = "NO_VALID_WINDOW"
    else:
        first_block: list[int] | None = None
        for start in range(0, n_windows - k_consec + 1):
            block = list(range(start, start + k_consec))
            if all(windows_summary[i]["candidate"] for i in block):
                first_block = block
                break

        if first_block is not None:
            chosen_block_indices = first_block
            chosen_index = first_block[0]
            chosen_t0 = windows_summary[chosen_index]["t0"]
            all_gold = all((windows_summary[i]["cv_max"] is not None) and (windows_summary[i]["cv_max"] <= cfg["cv_gold"]) for i in first_block)
            quality = "GOLD" if all_gold else "STANDARD"
            final_verdict = "PASS"
            for i in first_block:
                windows_summary[i]["in_chosen_block"] = True
        elif has_candidate:
            fail_global_reason = "NO_CONSECUTIVE"
        else:
            fail_global_reason = "NO_PLATEAU"

    return {
        "thresholds": cfg,
        "n_windows": n_windows,
        "windows_summary": windows_summary,
        "chosen_t0": chosen_t0,
        "chosen_index": chosen_index,
        "chosen_block_indices": chosen_block_indices,
        "quality": quality,
        "final_verdict": final_verdict,
        "fail_global_reason": fail_global_reason,
    }
