from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SIGMA_FLOOR = 1e-12
FAIL_MISSING_FIELD = "FAIL_MISSING_FIELD"
FAIL_SIGMA_MISSING = "FAIL_SIGMA_MISSING"
FAIL_COV_MISSING = "FAIL_COV_MISSING"
FAIL_COND_MAX = "FAIL_COND_MAX"
FAIL_LOW_INFO_MISSING = "FAIL_LOW_INFO_MISSING"
FAIL_LOW_INFO = "FAIL_LOW_INFO"
FAIL_WHITE_MISSING = "FAIL_WHITE_MISSING"
FAIL_WHITE = "FAIL_WHITE"
FAIL_COH_MISSING = "FAIL_COH_MISSING"
FAIL_COH = "FAIL_COH"

WARN_WHITE_NOT_EVALUATED_LOW_POWER = "WARN_WHITE_NOT_EVALUATED_LOW_POWER"

Z_MAX = 2.0
K_PLATEAU = 3
COND_MAX = 100.0
DELTA_BIC_MIN = 10.0
P_WHITE_MIN = 0.01
N_LOW = 100


@dataclass(frozen=True)
class WindowMetrics:
    t0: float
    T: float
    f_median: float
    f_sigma: float
    tau_median: float
    tau_sigma: float
    cond_number: float
    delta_bic: float
    p_ljungbox: float
    n_samples: int
    chi2_coh: float | None = None
    valid_fraction: float | None = None


class WindowMetricsParseError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def sigma_from_iqr(iqr: float) -> float:
    return float(iqr) / 1.349


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _pick_value(payloads: list[dict[str, Any]], candidates: list[str]) -> Any:
    for payload in payloads:
        for cand in candidates:
            value = _get_nested(payload, cand)
            if value is not None:
                return value
    return None


def _require_float(payloads: list[dict[str, Any]], field: str, candidates: list[str]) -> float:
    value = _pick_value(payloads, candidates)
    if value is None:
        raise WindowMetricsParseError(FAIL_MISSING_FIELD, f"missing {field}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise WindowMetricsParseError(FAIL_MISSING_FIELD, f"invalid {field}: {value!r}") from exc
    if not math.isfinite(out):
        raise WindowMetricsParseError(FAIL_MISSING_FIELD, f"non-finite {field}: {out!r}")
    return out


def _require_int(payloads: list[dict[str, Any]], field: str, candidates: list[str]) -> int:
    value = _pick_value(payloads, candidates)
    if value is None:
        raise WindowMetricsParseError(FAIL_MISSING_FIELD, f"missing {field}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WindowMetricsParseError(FAIL_MISSING_FIELD, f"invalid {field}: {value!r}") from exc


def _optional_float(payloads: list[dict[str, Any]], candidates: list[str]) -> float | None:
    value = _pick_value(payloads, candidates)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _sigma_value(payloads: list[dict[str, Any]], *, sigma_candidates: list[str], iqr_candidates: list[str], field: str) -> float:
    sigma = _pick_value(payloads, sigma_candidates)
    if sigma is None:
        iqr = _pick_value(payloads, iqr_candidates)
        if iqr is not None:
            sigma = sigma_from_iqr(float(iqr))

    if sigma is None:
        raise WindowMetricsParseError(FAIL_SIGMA_MISSING, f"missing {field}")

    try:
        sigma_f = float(sigma)
    except (TypeError, ValueError) as exc:
        raise WindowMetricsParseError(FAIL_SIGMA_MISSING, f"invalid {field}: {sigma!r}") from exc

    if not math.isfinite(sigma_f):
        raise WindowMetricsParseError(FAIL_SIGMA_MISSING, f"non-finite {field}: {sigma_f!r}")
    return max(sigma_f, SIGMA_FLOOR)


def _canonical_payloads(subrun_root: Path) -> list[dict[str, Any]]:
    candidates = [
        subrun_root / "s2_ringdown_window" / "outputs" / "window_meta.json",
        subrun_root / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        subrun_root / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json",
        subrun_root / "s2_ringdown_window" / "stage_summary.json",
        subrun_root / "s3_ringdown_estimates" / "stage_summary.json",
        subrun_root / "s4c_kerr_consistency" / "stage_summary.json",
    ]
    payloads: list[dict[str, Any]] = []
    for path in candidates:
        if path.exists():
            payload = _read_json(path)
            payloads.append(payload)
            results = payload.get("results")
            if isinstance(results, dict):
                payloads.append(results)
    return payloads


def load_window_metrics_from_subrun(subrun_root: Path) -> WindowMetrics:
    payloads = _canonical_payloads(subrun_root)
    if not payloads:
        raise WindowMetricsParseError(FAIL_MISSING_FIELD, f"no canonical JSON artifacts found in {subrun_root}")

    t0 = _require_float(payloads, "t0", ["t0", "t0_ms", "t0_offset_ms", "window.t0", "window_meta.t0_offset_ms"])
    T = _require_float(payloads, "T", ["T", "duration_s", "window.duration_s", "window_meta.duration_s"])
    f_median = _require_float(payloads, "f_median", ["f_median", "combined.f_hz", "f_hz_median"])
    tau_median = _require_float(payloads, "tau_median", ["tau_median", "combined.tau_s", "tau_s_median"])
    cond_number = _require_float(payloads, "cond_number", ["cond_number", "condition_number", "diagnostics.cond_number"])
    delta_bic = _require_float(payloads, "delta_bic", ["delta_bic", "diagnostics.delta_bic"])
    p_ljungbox = _require_float(payloads, "p_ljungbox", ["p_ljungbox", "diagnostics.p_ljungbox"])
    n_samples = _require_int(payloads, "n_samples", ["n_samples", "window.n_samples", "window_meta.n_samples"])

    f_sigma = _sigma_value(
        payloads,
        sigma_candidates=["f_sigma", "combined_uncertainty.sigma_f_hz", "sigma_f_hz"],
        iqr_candidates=["f_iqr", "iqr_f_hz"],
        field="f_sigma",
    )
    tau_sigma = _sigma_value(
        payloads,
        sigma_candidates=["tau_sigma", "combined_uncertainty.sigma_tau_s", "sigma_tau_s"],
        iqr_candidates=["tau_iqr", "iqr_tau_s"],
        field="tau_sigma",
    )

    chi2_coh = _optional_float(payloads, ["chi2_coh", "diagnostics.chi2_coh"])
    valid_fraction = _optional_float(payloads, ["valid_fraction", "diagnostics.valid_fraction"])

    return WindowMetrics(
        t0=t0,
        T=T,
        f_median=f_median,
        f_sigma=f_sigma,
        tau_median=tau_median,
        tau_sigma=tau_sigma,
        cond_number=cond_number,
        delta_bic=delta_bic,
        p_ljungbox=p_ljungbox,
        n_samples=n_samples,
        chi2_coh=chi2_coh,
        valid_fraction=valid_fraction,
    )


def load_window_metrics_from_subruns(subrun_roots: list[Path]) -> list[WindowMetrics]:
    metrics = [load_window_metrics_from_subrun(root) for root in subrun_roots]
    return sorted(metrics, key=lambda item: item.t0)


def _sorted_unique(items: list[str]) -> list[str]:
    return sorted(set(items))


def _compute_z(a: float, b: float, sa: float, sb: float) -> float:
    denom = math.sqrt((sa * sa) + (sb * sb))
    return abs(b - a) / denom


def oracle_v1_plateau_report(windows: list[WindowMetrics], chi2_coh_max: float | None = None) -> dict[str, Any]:
    ordered = sorted(windows, key=lambda item: item.t0)
    n_windows = len(ordered)

    scores: list[float | None] = []
    for idx in range(n_windows):
        if idx == n_windows - 1:
            scores.append(None)
            continue
        left = ordered[idx]
        right = ordered[idx + 1]
        z_f = _compute_z(left.f_median, right.f_median, left.f_sigma, right.f_sigma)
        z_tau = _compute_z(left.tau_median, right.tau_median, left.tau_sigma, right.tau_sigma)
        scores.append(max(z_f, z_tau))

    windows_summary: list[dict[str, Any]] = []
    fail_counts: dict[str, int] = {}
    any_pass = False

    for idx, item in enumerate(ordered):
        fail_reasons: list[str] = []
        warnings: list[str] = []

        if item.cond_number is None:
            fail_reasons.append(FAIL_COV_MISSING)
        elif item.cond_number > COND_MAX:
            fail_reasons.append(FAIL_COND_MAX)

        if item.delta_bic is None:
            fail_reasons.append(FAIL_LOW_INFO_MISSING)
        elif item.delta_bic < DELTA_BIC_MIN:
            fail_reasons.append(FAIL_LOW_INFO)

        if item.n_samples > N_LOW:
            if item.p_ljungbox is None:
                fail_reasons.append(FAIL_WHITE_MISSING)
            elif item.p_ljungbox < P_WHITE_MIN:
                fail_reasons.append(FAIL_WHITE)
        else:
            warnings.append(WARN_WHITE_NOT_EVALUATED_LOW_POWER)

        if chi2_coh_max is not None:
            if item.chi2_coh is None:
                fail_reasons.append(FAIL_COH_MISSING)
            elif item.chi2_coh > chi2_coh_max:
                fail_reasons.append(FAIL_COH)

        fail_reasons = _sorted_unique(fail_reasons)
        warnings = _sorted_unique(warnings)
        verdict = "PASS" if not fail_reasons else "FAIL"
        any_pass = any_pass or verdict == "PASS"
        for code in fail_reasons:
            fail_counts[code] = fail_counts.get(code, 0) + 1

        windows_summary.append(
            {
                "index": idx,
                "t0": item.t0,
                "T": item.T,
                "verdict": verdict,
                "fail_reasons": fail_reasons,
                "warnings": warnings,
                "score_stab": scores[idx],
                "in_plateau": False,
                "plateau_start_index": None,
            }
        )

    candidates: list[dict[str, Any]] = []
    if n_windows >= K_PLATEAU:
        for start in range(0, n_windows - K_PLATEAU + 1):
            all_pass = all(windows_summary[pos]["verdict"] == "PASS" for pos in range(start, start + K_PLATEAU))
            if not all_pass:
                continue
            stable = all((scores[j] is not None and scores[j] <= Z_MAX) for j in range(start, start + K_PLATEAU - 1))
            if not stable:
                continue
            candidates.append({"start_index": start, "plateau_indices": list(range(start, start + K_PLATEAU))})

    chosen_t0: float | None = None
    chosen_t: float | None = None
    fail_global_reason: str | None = None
    final_verdict = "FAIL"

    if n_windows < K_PLATEAU:
        fail_global_reason = "NO_PLATEAU_INSUFFICIENT_WINDOWS"
    elif candidates:
        winner = candidates[0]
        start = winner["start_index"]
        chosen_t0 = ordered[start].t0
        chosen_t = ordered[start].T
        final_verdict = "PASS"
        for pos in winner["plateau_indices"]:
            windows_summary[pos]["in_plateau"] = True
            windows_summary[pos]["plateau_start_index"] = start
    elif not any_pass:
        fail_global_reason = "NO_VALID_WINDOW"
    else:
        fail_global_reason = "NO_PLATEAU"

    fail_counts_sorted = {key: fail_counts[key] for key in sorted(fail_counts)}
    warnings_global = _sorted_unique([w for row in windows_summary for w in row["warnings"]])

    return {
        "thresholds": {
            "z_max": Z_MAX,
            "k_plateau": K_PLATEAU,
            "cond_max": COND_MAX,
            "delta_bic_min": DELTA_BIC_MIN,
            "p_white_min": P_WHITE_MIN,
            "n_low": N_LOW,
            "chi2_coh_max": chi2_coh_max,
        },
        "windows_summary": windows_summary,
        "fail_counts": fail_counts_sorted,
        "candidates": candidates,
        "chosen_t0": chosen_t0,
        "chosen_T": chosen_t,
        "final_verdict": final_verdict,
        "fail_global_reason": fail_global_reason,
        "warnings_global": warnings_global,
    }
