from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class WindowSummaryV1Error(ValueError):
    """Actionable mapper error with missing field + source path."""

    def __init__(self, *, field: str, path: Path | None, detail: str) -> None:
        self.field = field
        self.path = path
        super().__init__(f"WindowSummaryV1 missing {field} at {path}: {detail}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigma_from_iqr(iqr: float) -> float:
    return float(iqr) / 1.349


def _mode_220_from_payload(multimode_payload: dict[str, Any]) -> dict[str, Any]:
    for mode in multimode_payload.get("modes", []):
        if isinstance(mode, dict) and mode.get("label") == "220":
            return mode
    return {}


def _theta_from_point(point: dict[str, Any], mode_220: dict[str, Any]) -> tuple[float, float]:
    ln_f = point.get("s3b", {}).get("ln_f_220")
    ln_q = point.get("s3b", {}).get("ln_Q_220")
    if ln_f is None:
        ln_f = mode_220.get("ln_f")
    if ln_q is None:
        ln_q = mode_220.get("ln_Q")
    if ln_f is None or ln_q is None:
        raise WindowSummaryV1Error(field="theta", path=None, detail="missing ln_f_220 and/or ln_Q_220")
    return float(ln_f), float(ln_q)


def _sigma_theta(mode_220: dict[str, Any]) -> list[float]:
    sigma = mode_220.get("Sigma")
    if isinstance(sigma, list) and len(sigma) == 2 and isinstance(sigma[0], list) and isinstance(sigma[1], list):
        try:
            s_lnf = math.sqrt(float(sigma[0][0]))
            s_lnq = math.sqrt(float(sigma[1][1]))
            if math.isfinite(s_lnf) and math.isfinite(s_lnq) and s_lnf > 0 and s_lnq > 0:
                return [s_lnf, s_lnq]
        except (TypeError, ValueError):
            pass

    stability = mode_220.get("fit", {}).get("stability", {})
    p10_f = stability.get("lnf_p10")
    p90_f = stability.get("lnf_p90")
    p10_q = stability.get("lnQ_p10")
    p90_q = stability.get("lnQ_p90")
    if None not in (p10_f, p90_f, p10_q, p90_q):
        return [
            _sigma_from_iqr(float(p90_f) - float(p10_f)),
            _sigma_from_iqr(float(p90_q) - float(p10_q)),
        ]

    raise WindowSummaryV1Error(field="sigma_theta", path=None, detail="missing Sigma and IQR proxies for mode 220")


def map_sweep_point_to_window_summary_v1(sweep_payload: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    """Canonical mapper for t0_sweep_full point -> WindowSummaryV1.

    Canonical sweep artifact lives at:
    ``experiment/t0_sweep_full/**/outputs/t0_sweep_full_results.json``.

    Real schema anchors used by this mapper:
      - global summary: ``payload["summary"]`` (e.g. ``payload["summary"]["best_point"]``)
      - per-point summary: ``payload["points"][i]``
      - per-point multimode core: ``point["s3b"]`` (``ln_f_220``/``ln_Q_220``)
      - full uncertainties/source are resolved via subrun files under
        ``payload["subruns_root"]/point["subrun_id"]``.
    """

    subruns_root = Path(str(sweep_payload.get("subruns_root", "")))
    subrun_id = point.get("subrun_id")
    if not subrun_id:
        raise WindowSummaryV1Error(field="subrun_id", path=None, detail="point missing subrun_id")

    subrun_root = subruns_root / str(subrun_id)
    window_meta_path = subrun_root / "s2_ringdown_window" / "outputs" / "window_meta.json"
    multimode_path = subrun_root / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    s3_path = subrun_root / "s3_ringdown_estimates" / "outputs" / "estimates.json"

    window_meta = _read_json(window_meta_path) if window_meta_path.exists() else {}
    if not multimode_path.exists():
        raise WindowSummaryV1Error(field="modes.220", path=multimode_path, detail="multimode_estimates.json not found")
    multimode = _read_json(multimode_path)
    mode_220 = _mode_220_from_payload(multimode)
    if not mode_220:
        raise WindowSummaryV1Error(field="modes.220", path=multimode_path, detail="label 220 missing")

    theta = _theta_from_point(point, mode_220)
    sigma_theta = _sigma_theta(mode_220)

    s3 = _read_json(s3_path) if s3_path.exists() else {}
    combined = s3.get("combined", {}) if isinstance(s3, dict) else {}
    unc = s3.get("combined_uncertainty", {}) if isinstance(s3, dict) else {}

    return {
        "t0_ms": point.get("t0_ms"),
        "t0_s": (float(point["t0_ms"]) / 1000.0) if point.get("t0_ms") is not None else None,
        "T_s": window_meta.get("duration_s"),
        "theta": {"ln_f_220": theta[0], "ln_Q_220": theta[1]},
        "sigma_theta": {"sigma_ln_f_220": sigma_theta[0], "sigma_ln_Q_220": sigma_theta[1]},
        "snr": combined.get("snr_peak"),
        "cond": mode_220.get("fit", {}).get("stability", {}).get("sigma_cond"),
        "cov": {
            "Sigma": mode_220.get("Sigma"),
            "cov_logf_logQ": unc.get("cov_logf_logQ"),
            "r": unc.get("r"),
        },
        "whiteness_p": None,
        "coh": None,
        "source": {
            "subrun_id": subrun_id,
            "subrun_root": str(subrun_root),
            "window_meta_path": str(window_meta_path) if window_meta_path.exists() else None,
            "multimode_path": str(multimode_path),
            "s3_estimates_path": str(s3_path) if s3_path.exists() else None,
        },
    }
