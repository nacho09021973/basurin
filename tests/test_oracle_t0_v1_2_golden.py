from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mvp.oracle_t0_math import compute_oracle_t0_math
from mvp.oracle_t0_selection import select_t0

FIXTURE_PATH = Path("tests/fixtures/t0_sweep_full_results.json")
GOLDEN_PATH = Path("tests/golden/oracle_t0_v1_2_report.json")


def _round_float(value: float, ndigits: int = 8) -> float:
    return round(value, ndigits)


def _normalize(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _normalize(payload[key]) for key in sorted(payload)}
    if isinstance(payload, list):
        return [_normalize(item) for item in payload]
    if isinstance(payload, float):
        return _round_float(payload)
    return payload


def _load_synthetic_windows(path: Path) -> tuple[list[dict[str, Any]], list[bool]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows: list[dict[str, Any]] = []
    gates: list[bool] = []

    for point in payload["points"]:
        s3b = point.get("s3b") or {}
        windows.append(
            {
                "t0_ms": point.get("t0_ms"),
                "ln_f_220": s3b["ln_f_220"],
                "ln_Q_220": s3b["ln_Q_220"],
                "sigma_ln_f_220": s3b["sigma_ln_f_220"],
                "sigma_ln_Q_220": s3b["sigma_ln_Q_220"],
            }
        )
        gates.append((point.get("status") or "").upper() == "OK")

    return windows, gates


def test_oracle_t0_v1_2_synthetic_golden_report() -> None:
    windows, gates = _load_synthetic_windows(FIXTURE_PATH)

    metrics = compute_oracle_t0_math(windows, w=2)
    report = select_t0(windows, metrics, gates)

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert _normalize(report) == _normalize(golden)
