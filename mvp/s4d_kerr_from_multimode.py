"""
s4d_kerr_from_multimode

Canonical stage (Phase B):
- Inputs: s3b multimode outputs (multimode_estimates.json; optional model_comparison.json)
- Outputs: kerr_from_multimode.json, kerr_from_multimode_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mvp.contracts import StageContext, abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "s4d_kerr_from_multimode"
M_MIN = 5.0
M_MAX = 500.0
A_MIN = 0.0
A_MAX = 0.99999
GRID_M_SIZE = 200
GRID_A_SIZE = 200
EPS_M = 1e-9 * (M_MAX - M_MIN)
EPS_A = 1e-9
BOUNDARY_FRACTION_THRESHOLD = 0.20
SPIN_PHYSICAL_FLOOR_WARNING_CODE = "SPIN_AT_PHYSICAL_FLOOR"
SPIN_PHYSICAL_FLOOR_WARNING_MSG = "Spin posterior saturated at A_MIN=0 (physical floor); continuing with warning."
MULTIMODE_OK = "MULTIMODE_OK"
SINGLEMODE_ONLY = "SINGLEMODE_ONLY"
RINGDOWN_NONINFORMATIVE = "RINGDOWN_NONINFORMATIVE"


def _base_params() -> dict[str, Any]:
    return {
        "M_MIN": M_MIN,
        "M_MAX": M_MAX,
        "A_MIN": A_MIN,
        "A_MAX": A_MAX,
        "GRID_M_SIZE": GRID_M_SIZE,
        "GRID_A_SIZE": GRID_A_SIZE,
        "EPS_M": EPS_M,
        "EPS_A": EPS_A,
        "gate": {
            "name": "KERR_GRID_SATURATION",
            "boundary_fraction_threshold": BOUNDARY_FRACTION_THRESHOLD,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=STAGE)
    p.add_argument("--run-id", required=True)
    return p


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_strictify(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_strictify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_strictify(v) for v in value]
    return value


def _write_json_strict_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    strict = _json_strictify(payload)
    text = json.dumps(strict, allow_nan=False, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, text.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_quantile_block(mode_obj: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    direct = mode_obj.get("estimates")
    if isinstance(direct, dict):
        f_hz = direct.get("f_hz")
        tau_s = direct.get("tau_s")
        if isinstance(f_hz, dict) and isinstance(tau_s, dict):
            vals = {
                "f_hz": {q: _to_float(f_hz.get(q)) for q in ("p10", "p50", "p90")},
                "tau_s": {q: _to_float(tau_s.get(q)) for q in ("p10", "p50", "p90")},
            }
            if all(vals["f_hz"][q] is not None and vals["tau_s"][q] is not None for q in ("p10", "p50", "p90")):
                return vals  # type: ignore[return-value]

    stability = (((mode_obj.get("fit") or {}).get("stability") or {}))
    if isinstance(stability, dict):
        lnf = {q: _to_float(stability.get(f"lnf_{q}")) for q in ("p10", "p50", "p90")}
        lnq = {q: _to_float(stability.get(f"lnQ_{q}")) for q in ("p10", "p50", "p90")}
        if all(lnf[q] is not None and lnq[q] is not None for q in ("p10", "p50", "p90")):
            f_vals = {q: float(math.exp(float(lnf[q]))) for q in ("p10", "p50", "p90")}
            tau_vals = {
                q: float(math.exp(float(lnq[q]) - float(lnf[q])) / math.pi)
                for q in ("p10", "p50", "p90")
            }
            return {"f_hz": f_vals, "tau_s": tau_vals}

    ln_f = _to_float(mode_obj.get("ln_f"))
    ln_q = _to_float(mode_obj.get("ln_Q"))
    if ln_f is not None and ln_q is not None:
        f = float(math.exp(ln_f))
        tau = float(math.exp(ln_q - ln_f) / math.pi)
        one = {"p10": f, "p50": f, "p90": f}
        two = {"p10": tau, "p50": tau, "p90": tau}
        return {"f_hz": one, "tau_s": two}

    return None


def _extract_mode_quantiles(multimode: dict[str, Any], label: str) -> dict[str, dict[str, float]]:
    estimates = multimode.get("estimates")
    if isinstance(estimates, dict):
        per_mode = estimates.get("per_mode")
        if isinstance(per_mode, dict):
            mode_node = per_mode.get(label)
            if isinstance(mode_node, dict):
                f_hz = mode_node.get("f_hz")
                tau_s = mode_node.get("tau_s")
                if isinstance(f_hz, dict) and isinstance(tau_s, dict):
                    vals = {
                        "f_hz": {q: _to_float(f_hz.get(q)) for q in ("p10", "p50", "p90")},
                        "tau_s": {q: _to_float(tau_s.get(q)) for q in ("p10", "p50", "p90")},
                    }
                    if all(vals["f_hz"][q] is not None and vals["tau_s"][q] is not None for q in ("p10", "p50", "p90")):
                        return vals  # type: ignore[return-value]

    modes = multimode.get("modes")
    if isinstance(modes, list):
        for node in modes:
            if not isinstance(node, dict):
                continue
            if str(node.get("label")) != label:
                continue
            vals = _extract_quantile_block(node)
            if vals is not None:
                return vals

    raise ValueError(f"Missing required fields in multimode_estimates for mode {label}: f_hz/tau_s p10/p50/p90")


def _triangular_sample(rng: random.Random, p10: float, p50: float, p90: float) -> float:
    lo = min(float(p10), float(p90))
    hi = max(float(p10), float(p90))
    mode = float(p50)
    mode = max(lo, min(hi, mode))
    if hi <= lo:
        return lo
    return float(rng.triangular(lo, hi, mode))


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}  # type: ignore[return-value]
    seq = sorted(float(v) for v in values)
    n = len(seq)

    def _pick(q: float) -> float:
        idx = int(round((n - 1) * q))
        idx = min(max(idx, 0), n - 1)
        return float(seq[idx])

    return {"p10": _pick(0.10), "p50": _pick(0.50), "p90": _pick(0.90)}


def _build_grid() -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    try:
        from mvp.kerr_qnm_fits import kerr_qnm
    except Exception as exc:
        raise RuntimeError(
            "Missing Kerr QNM forward model in repo; cannot invert f/tau to (M,a) without canonical model"
        ) from exc

    a_vals = [A_MIN + ((A_MAX - A_MIN) * i / float(GRID_A_SIZE - 1)) for i in range(GRID_A_SIZE)]
    m_vals = [M_MIN + ((M_MAX - M_MIN) * i / float(GRID_M_SIZE - 1)) for i in range(GRID_M_SIZE)]

    grid_m: list[float] = []
    grid_a: list[float] = []
    lnf_220: list[float] = []
    lntau_220: list[float] = []
    lnf_221: list[float] = []
    lntau_221: list[float] = []

    for m in m_vals:
        for a in a_vals:
            q220 = kerr_qnm(m, a, (2, 2, 0))
            q221 = kerr_qnm(m, a, (2, 2, 1))
            grid_m.append(float(m))
            grid_a.append(float(a))
            lnf_220.append(float(math.log(q220.f_hz)))
            lntau_220.append(float(math.log(q220.tau_s)))
            lnf_221.append(float(math.log(q221.f_hz)))
            lntau_221.append(float(math.log(q221.tau_s)))

    return grid_m, grid_a, lnf_220, lntau_220, lnf_221, lntau_221


def _invert_kerr_from_freqs_grid(
    f_220_hz: float,
    f_221_hz: float,
    grid_m: list[float],
    grid_a: list[float],
    lnf_220: list[float],
    lnf_221: list[float],
) -> tuple[float, float]:
    """Find (M, a) that minimises squared log-frequency residuals for modes 220 and 221.

    Uses only frequencies (not damping times) for a fast, deterministic nearest-grid lookup.
    Returns (M_solar, a_dimensionless).
    """
    t220 = math.log(f_220_hz)
    t221 = math.log(f_221_hz)
    best_i = 0
    best_e = float("inf")
    for i in range(len(grid_m)):
        r220 = lnf_220[i] - t220
        r221 = lnf_221[i] - t221
        e = r220 * r220 + r221 * r221
        if e < best_e:
            best_e = e
            best_i = i
    return float(grid_m[best_i]), float(grid_a[best_i])


def _extract_kerr_with_covariance_core(
    f_220_hz: float,
    f_221_hz: float,
    sigma_f220: float,
    sigma_f221: float,
    grid_m: list[float],
    grid_a: list[float],
    lnf_220: list[float],
    lnf_221: list[float],
) -> tuple[float, float, float, float, float]:
    """Propagate frequency uncertainties to (M, a) uncertainties via finite-difference Jacobian.

    Returns (M, a, sigma_M, sigma_a, cov_M_a).
    """
    M0, a0 = _invert_kerr_from_freqs_grid(f_220_hz, f_221_hz, grid_m, grid_a, lnf_220, lnf_221)

    df220 = max(sigma_f220 * 1e-3, f_220_hz * 1e-4)
    df221 = max(sigma_f221 * 1e-3, f_221_hz * 1e-4)

    M_p220, a_p220 = _invert_kerr_from_freqs_grid(f_220_hz + df220, f_221_hz, grid_m, grid_a, lnf_220, lnf_221)
    M_m220, a_m220 = _invert_kerr_from_freqs_grid(f_220_hz - df220, f_221_hz, grid_m, grid_a, lnf_220, lnf_221)
    dM_df220 = (M_p220 - M_m220) / (2.0 * df220)
    da_df220 = (a_p220 - a_m220) / (2.0 * df220)

    M_p221, a_p221 = _invert_kerr_from_freqs_grid(f_220_hz, f_221_hz + df221, grid_m, grid_a, lnf_220, lnf_221)
    M_m221, a_m221 = _invert_kerr_from_freqs_grid(f_220_hz, f_221_hz - df221, grid_m, grid_a, lnf_220, lnf_221)
    dM_df221 = (M_p221 - M_m221) / (2.0 * df221)
    da_df221 = (a_p221 - a_m221) / (2.0 * df221)

    var_M = (dM_df220 * sigma_f220) ** 2 + (dM_df221 * sigma_f221) ** 2
    var_a = (da_df220 * sigma_f220) ** 2 + (da_df221 * sigma_f221) ** 2
    cov_M_a = (dM_df220 * da_df220 * sigma_f220 ** 2
               + dM_df221 * da_df221 * sigma_f221 ** 2)

    sigma_M = math.sqrt(max(var_M, 0.0))
    sigma_a = math.sqrt(max(var_a, 0.0))
    return M0, a0, sigma_M, sigma_a, cov_M_a


def _best_idx_joint(
    obs: dict[str, dict[str, float]],
    lnf_220: list[float],
    lnq_220: list[float],
    lnf_221: list[float],
    lnq_221: list[float],
    inv_sigma_220: tuple[tuple[float, float], tuple[float, float]],
    inv_sigma_221: tuple[tuple[float, float], tuple[float, float]],
) -> int:
    def _obs_lnf_lnq(mode_obs: dict[str, float]) -> tuple[float, float]:
        t_f = math.log(float(mode_obs["f_hz"]))
        ln_q_obs = _to_float(mode_obs.get("ln_Q"))
        if ln_q_obs is None:
            ln_q_obs = math.log(float(mode_obs["tau_s"])) + t_f + math.log(math.pi)
        return t_f, float(ln_q_obs)

    t220_f, t220_q = _obs_lnf_lnq(obs["220"])
    t221_f, t221_q = _obs_lnf_lnq(obs["221"])

    best_i = 0
    best_e = float("inf")
    for i in range(len(lnf_220)):
        r220_f = lnf_220[i] - t220_f
        r220_q = lnq_220[i] - t220_q
        r221_f = lnf_221[i] - t221_f
        r221_q = lnq_221[i] - t221_q
        e = (
            (inv_sigma_220[0][0] * r220_f * r220_f)
            + (2.0 * inv_sigma_220[0][1] * r220_f * r220_q)
            + (inv_sigma_220[1][1] * r220_q * r220_q)
            + (inv_sigma_221[0][0] * r221_f * r221_f)
            + (2.0 * inv_sigma_221[0][1] * r221_f * r221_q)
            + (inv_sigma_221[1][1] * r221_q * r221_q)
        )
        if e < best_e:
            best_e = e
            best_i = i
    return best_i


def _best_idx_single(
    mode_obs: dict[str, float],
    lnf: list[float],
    lnq: list[float],
    inv_sigma: tuple[tuple[float, float], tuple[float, float]],
) -> int:
    tf = math.log(float(mode_obs["f_hz"]))
    tq = _to_float(mode_obs.get("ln_Q"))
    if tq is None:
        tq = math.log(float(mode_obs["tau_s"])) + tf + math.log(math.pi)
    best_i = 0
    best_e = float("inf")
    for i in range(len(lnf)):
        rf = lnf[i] - tf
        rq = lnq[i] - tq
        e = (inv_sigma[0][0] * rf * rf) + (2.0 * inv_sigma[0][1] * rf * rq) + (inv_sigma[1][1] * rq * rq)
        if e < best_e:
            best_e = e
            best_i = i
    return best_i


def _as_2x2_sigma(raw: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    row0, row1 = raw
    if not (isinstance(row0, list) and isinstance(row1, list) and len(row0) == 2 and len(row1) == 2):
        return None
    vals = [[_to_float(row0[0]), _to_float(row0[1])], [_to_float(row1[0]), _to_float(row1[1])]]
    if any(v is None for r in vals for v in r):
        return None
    return ((float(vals[0][0]), float(vals[0][1])), (float(vals[1][0]), float(vals[1][1])))


def _invert_2x2_sigma(sigma: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    inv, _ = _regularize_and_invert_2x2_sigma(sigma)
    return inv


def _should_abort_for_boundary(
    a_p50: float,
    m_p50: float,
    boundary_fraction: float,
    a_min: float,
    a_max: float,
    m_min: float,
    m_max: float,
    threshold: float,
) -> tuple[bool, str | None, bool]:
    on_m_min = abs(m_p50 - m_min) <= EPS_M
    on_m_max = abs(m_p50 - m_max) <= EPS_M
    on_a_min = abs(a_p50 - a_min) <= EPS_A
    on_a_max = abs(a_p50 - a_max) <= EPS_A

    if on_m_min or on_m_max:
        return True, "median_mass_on_grid_edge", False
    if on_a_max:
        return True, "median_spin_on_grid_edge", False
    if on_a_min and a_min == 0.0 and boundary_fraction >= threshold:
        return False, None, True
    if on_a_min:
        return True, "median_spin_on_grid_edge", False
    if boundary_fraction >= threshold:
        return True, "boundary_fraction_high", False
    return False, None, False


def _regularize_and_invert_2x2_sigma(
    sigma: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], dict[str, float]]:
    s00, s01 = sigma[0]
    s10, s11 = sigma[1]
    sym_off = 0.5 * (s01 + s10)
    s01 = float(sym_off)
    s10 = float(sym_off)
    s00 = float(s00)
    s11 = float(s11)

    det_before = (s00 * s11) - (s01 * s10)
    trace_half = 0.5 * (s00 + s11)
    base_jitter = max(1e-10, 1e-6 * trace_half)
    jitter = base_jitter
    det_after = float("nan")

    for _ in range(6):
        r00 = s00 + jitter
        r11 = s11 + jitter
        det_after = (r00 * r11) - (s01 * s10)
        if math.isfinite(det_after) and det_after > 0.0 and r00 > 0.0 and r11 > 0.0:
            inv_det = 1.0 / det_after
            disc = (r00 - r11) * (r00 - r11) + (4.0 * s01 * s01)
            disc = max(0.0, disc)
            root = math.sqrt(disc)
            eig_hi = 0.5 * (r00 + r11 + root)
            eig_lo = 0.5 * (r00 + r11 - root)
            cond_proxy = float("inf") if eig_lo <= 0.0 else float(eig_hi / eig_lo)
            diag = {
                "jitter_used": float(jitter),
                "det_before": float(det_before),
                "det_after": float(det_after),
                "cond_proxy": cond_proxy,
            }
            return ((r11 * inv_det, -s01 * inv_det), (-s10 * inv_det, r00 * inv_det)), diag
        jitter *= 10.0

    raise ValueError(
        "SIGMA_SINGULAR: Sigma regularization failed after jitter escalation "
        f"(det_before={det_before}, det_after={det_after})"
    )


def _extract_mode_inverse_sigma(multimode: dict[str, Any], label: str) -> tuple[tuple[tuple[float, float], tuple[float, float]], dict[str, float]]:
    modes = multimode.get("modes")
    if not isinstance(modes, list):
        raise ValueError(f"Missing multimode_estimates.modes for mode {label}; cannot build Mahalanobis metric")
    for node in modes:
        if not isinstance(node, dict) or str(node.get("label")) != label:
            continue
        sigma = _as_2x2_sigma(node.get("Sigma"))
        if sigma is None:
            raise ValueError(f"Missing/invalid Sigma for mode {label}; expected 2x2 finite matrix")
        return _regularize_and_invert_2x2_sigma(sigma)
    raise ValueError(f"Mode {label} not found in multimode_estimates.modes")




def _sigma_from_quantiles(node: dict[str, Any]) -> float:
    p10 = float(node["p10"])
    p90 = float(node["p90"])
    return max((p90 - p10) / 2.5631031311, 1e-12)


def _invert_kerr_from_freqs(f220_hz: float, f221_hz: float) -> tuple[float, float]:
    from mvp.kerr_qnm_fits import kerr_qnm

    if f220_hz <= 0 or f221_hz <= 0:
        raise ValueError("f220/f221 must be positive")
    ratio_target = f221_hz / f220_hz

    lo = A_MIN
    hi = A_MAX
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        q220 = kerr_qnm(50.0, mid, (2, 2, 0)).f_hz
        q221 = kerr_qnm(50.0, mid, (2, 2, 1)).f_hz
        ratio_mid = q221 / q220
        if ratio_mid < ratio_target:
            lo = mid
        else:
            hi = mid
    chi = max(A_MIN, min(A_MAX, 0.5 * (lo + hi)))
    f220_unit = kerr_qnm(1.0, chi, (2, 2, 0)).f_hz
    m_final = f220_unit / f220_hz
    return float(m_final), float(chi)


def _extract_kerr_with_covariance(multimode: dict[str, Any]) -> dict[str, Any]:
    from mvp.kerr_qnm_fits import kerr_qnm

    q220 = _extract_mode_quantiles(multimode, "220")
    q221 = _extract_mode_quantiles(multimode, "221")

    f220 = float(q220["f_hz"]["p50"])
    tau220 = float(q220["tau_s"]["p50"])
    f221 = float(q221["f_hz"]["p50"])
    tau221 = float(q221["tau_s"]["p50"])

    sf220 = _sigma_from_quantiles(q220["f_hz"])
    st220 = _sigma_from_quantiles(q220["tau_s"])
    sf221 = _sigma_from_quantiles(q221["f_hz"])
    st221 = _sigma_from_quantiles(q221["tau_s"])

    cov_220_221 = 0.0
    cov_node = multimode.get("cov_220_221")
    if isinstance(cov_node, (int, float)) and math.isfinite(float(cov_node)):
        cov_220_221 = float(cov_node)

    m0, chi0 = _invert_kerr_from_freqs(f220, f221)

    def state_from_obs(x_f220: float, x_f221: float) -> tuple[float, float]:
        return _invert_kerr_from_freqs(x_f220, x_f221)

    drel = 1e-4
    df220 = max(abs(f220) * drel, 1e-9)
    df221 = max(abs(f221) * drel, 1e-9)
    mp, cp = state_from_obs(f220 + df220, f221)
    mm, cm = state_from_obs(f220 - df220, f221)
    dmdf220 = (mp - mm) / (2.0 * df220)
    dcdf220 = (cp - cm) / (2.0 * df220)

    mp, cp = state_from_obs(f220, f221 + df221)
    mm, cm = state_from_obs(f220, f221 - df221)
    dmdf221 = (mp - mm) / (2.0 * df221)
    dcdf221 = (cp - cm) / (2.0 * df221)

    var_m = (dmdf220**2) * (sf220**2) + (dmdf221**2) * (sf221**2) + 2.0 * dmdf220 * dmdf221 * cov_220_221
    var_chi = (dcdf220**2) * (sf220**2) + (dcdf221**2) * (sf221**2) + 2.0 * dcdf220 * dcdf221 * cov_220_221
    cov_m_chi = dmdf220 * dcdf220 * (sf220**2) + dmdf221 * dcdf221 * (sf221**2) + (dmdf220 * dcdf221 + dmdf221 * dcdf220) * cov_220_221

    q221_pred = kerr_qnm(m0, chi0, (2, 2, 1))
    delta_f221 = f221 - q221_pred.f_hz
    delta_tau221_ms = 1e3 * (tau221 - q221_pred.tau_s)

    consistency_score = float(math.sqrt((delta_f221 / max(sf221, 1e-12)) ** 2 + ((tau221 - q221_pred.tau_s) / max(st221, 1e-12)) ** 2))

    return {
        "M_final_Msun": float(m0),
        "chi_final": float(chi0),
        "sigma_M": float(math.sqrt(max(var_m, 0.0))),
        "sigma_chi": float(math.sqrt(max(var_chi, 0.0))),
        "cov_M_chi": float(cov_m_chi),
        "delta_f221_Hz": float(delta_f221),
        "delta_tau221_ms": float(delta_tau221_ms),
        "consistency_score": consistency_score,
        "aux_input": {
            "f_220": f220, "sigma_f220": sf220, "tau_220": tau220, "sigma_tau220": st220,
            "f_221": f221, "sigma_f221": sf221, "tau_221": tau221, "sigma_tau221": st221,
            "cov_220_221": cov_220_221,
        },
    }

def _execute(ctx: StageContext) -> dict[str, Path]:
    if not isinstance(ctx.params, dict) or not ctx.params:
        ctx.params = _base_params()

    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    model_comparison_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"
    s3b_summary_path = ctx.run_dir / "s3b_multimode_estimates" / "stage_summary.json"

    inputs = check_inputs(
        ctx,
        paths={"multimode_estimates": multimode_path, "s3b_stage_summary": s3b_summary_path},
        optional={"model_comparison": model_comparison_path},
    )
    input_by_label = {row.get("label", ""): row for row in inputs}

    multimode = json.loads(multimode_path.read_text(encoding="utf-8"))
    s3b_summary = json.loads(s3b_summary_path.read_text(encoding="utf-8"))
    viability = s3b_summary.get("multimode_viability")
    if not isinstance(viability, dict) or viability.get("class") not in {
        MULTIMODE_OK,
        SINGLEMODE_ONLY,
        RINGDOWN_NONINFORMATIVE,
    }:
        abort(ctx, reason="Invalid or missing s3b multimode_viability contract in stage_summary.json")

    viability_class = str(viability.get("class"))
    viability_reasons = viability.get("reasons") if isinstance(viability.get("reasons"), list) else []
    if viability_class != MULTIMODE_OK:
        skip_payload = {
            "schema_name": "kerr_from_multimode",
            "schema_version": "mvp_kerr_from_multimode_v1",
            "json_strict": True,
            "created_utc": _utc_now_z(),
            "run_id": ctx.run_id,
            "stage": STAGE,
            "status": "SKIPPED_MULTIMODE_GATE",
            "multimode_viability": viability,
        }
        diag_payload = {
            "schema_name": "kerr_from_multimode_diagnostics",
            "schema_version": "mvp_kerr_from_multimode_diagnostics_v1",
            "json_strict": True,
            "created_utc": _utc_now_z(),
            "run_id": ctx.run_id,
            "stage": STAGE,
            "diagnostics": {
                "multimode_evaluated": False,
                "skips": ["MULTIMODE_DUE_TO_GATE"],
                "multimode_viability": viability,
            },
        }
        kerr_extract = {
            "schema_name": "kerr_extraction",
            "schema_version": "mvp_kerr_extraction_v1",
            "verdict": viability_class,
            "M_final_Msun": None,
            "chi_final": None,
            "sigma_M": None,
            "sigma_chi": None,
            "cov_M_chi": None,
            "delta_f221_Hz": None,
            "delta_tau221_ms": None,
            "consistency_score": None,
        }
        kerr_extract_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_extraction.json", kerr_extract)
        kerr_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_from_multimode.json", skip_payload)
        diag_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_from_multimode_diagnostics.json", diag_payload)
        extraction_skip_payload = {
            "schema_name": "kerr_extraction",
            "schema_version": "mvp_kerr_extraction_v1",
            "json_strict": True,
            "created_utc": _utc_now_z(),
            "run_id": ctx.run_id,
            "stage": STAGE,
            "verdict": "SKIPPED_MULTIMODE_GATE",
            "multimode_viability": viability,
            "M_final_Msun": None,
            "chi_final": None,
            "sigma_M": None,
            "sigma_chi": None,
            "cov_M_chi": None,
            "delta_f221_Hz": None,
            "delta_tau221_ms": None,
            "consistency_score": None,
        }
        extraction_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_extraction.json", extraction_skip_payload)
        ctx.params.update(
            {
                "multimode_viability_class": viability_class,
                "multimode_viability_reasons": viability_reasons,
                "multimode_evaluated": False,
                "skips": ["MULTIMODE_DUE_TO_GATE"],
            }
        )
        return {
            "kerr_from_multimode": kerr_path,
            "kerr_from_multimode_diagnostics": diag_path,
            "kerr_extraction": kerr_extract_path,
        }

    model_comparison = (
        json.loads(model_comparison_path.read_text(encoding="utf-8"))
        if model_comparison_path.exists()
        else None
    )

    try:
        q220 = _extract_mode_quantiles(multimode, "220")
        q221 = _extract_mode_quantiles(multimode, "221")
        inv_sigma_220, sigma_diag_220 = _extract_mode_inverse_sigma(multimode, "220")
        inv_sigma_221, sigma_diag_221 = _extract_mode_inverse_sigma(multimode, "221")
    except ValueError as exc:
        abort(ctx, reason=str(exc))

    ctx.params.update(
        {
            "matching_metric": "mahalanobis_sigma",
            "sigma_space_input": "ln_f_ln_Q",
            "sigma_space_matching": "ln_f_ln_Q",
            "sigma_transform_applied": False,
            "sigma_jitter_used_220": sigma_diag_220["jitter_used"],
            "sigma_jitter_used_221": sigma_diag_221["jitter_used"],
            "sigma_det_before_220": sigma_diag_220["det_before"],
            "sigma_det_before_221": sigma_diag_221["det_before"],
            "sigma_det_after_220": sigma_diag_220["det_after"],
            "sigma_det_after_221": sigma_diag_221["det_after"],
            "sigma_cond_proxy_220": sigma_diag_220["cond_proxy"],
            "sigma_cond_proxy_221": sigma_diag_221["cond_proxy"],
        }
    )

    for label, q in (("220", q220), ("221", q221)):
        for metric in ("f_hz", "tau_s"):
            vals = [q[metric]["p10"], q[metric]["p50"], q[metric]["p90"]]
            if not all(v is not None and float(v) > 0.0 and math.isfinite(float(v)) for v in vals):
                abort(ctx, reason=f"Missing required fields in multimode_estimates: mode {label} {metric} must define positive finite p10/p50/p90")

    try:
        grid_m, grid_a, lnf_220, lntau_220, lnf_221, lntau_221 = _build_grid()
    except RuntimeError as exc:
        abort(ctx, reason=str(exc))

    ln_pi = math.log(math.pi)
    lnq_220 = [float(lnf + lntau + ln_pi) for lnf, lntau in zip(lnf_220, lntau_220)]
    lnq_221 = [float(lnf + lntau + ln_pi) for lnf, lntau in zip(lnf_221, lntau_221)]

    seed = 0
    n_samples = 256
    rng = random.Random(seed)

    m_joint_samples: list[float] = []
    a_joint_samples: list[float] = []
    m220_samples: list[float] = []
    a220_samples: list[float] = []
    m221_samples: list[float] = []
    a221_samples: list[float] = []
    rejected = 0

    for _ in range(n_samples):
        obs220 = {
            "f_hz": _triangular_sample(rng, q220["f_hz"]["p10"], q220["f_hz"]["p50"], q220["f_hz"]["p90"]),
            "tau_s": _triangular_sample(rng, q220["tau_s"]["p10"], q220["tau_s"]["p50"], q220["tau_s"]["p90"]),
        }
        obs221 = {
            "f_hz": _triangular_sample(rng, q221["f_hz"]["p10"], q221["f_hz"]["p50"], q221["f_hz"]["p90"]),
            "tau_s": _triangular_sample(rng, q221["tau_s"]["p10"], q221["tau_s"]["p50"], q221["tau_s"]["p90"]),
        }
        try:
            idx_joint = _best_idx_joint(
                {"220": obs220, "221": obs221},
                lnf_220,
                lnq_220,
                lnf_221,
                lnq_221,
                inv_sigma_220,
                inv_sigma_221,
            )
            idx_220 = _best_idx_single(obs220, lnf_220, lnq_220, inv_sigma_220)
            idx_221 = _best_idx_single(obs221, lnf_221, lnq_221, inv_sigma_221)
        except Exception:
            rejected += 1
            continue

        m_joint_samples.append(grid_m[idx_joint])
        a_joint_samples.append(grid_a[idx_joint])
        m220_samples.append(grid_m[idx_220])
        a220_samples.append(grid_a[idx_220])
        m221_samples.append(grid_m[idx_221])
        a221_samples.append(grid_a[idx_221])

    if not m_joint_samples:
        abort(ctx, reason="No invertible samples in Kerr inversion from multimode estimates")

    m_quantiles = _quantiles(m_joint_samples)
    a_quantiles = _quantiles(a_joint_samples)
    m_p50 = float(m_quantiles["p50"])
    a_p50 = float(a_quantiles["p50"])

    count_m_min = sum(1 for m in m_joint_samples if abs(float(m) - M_MIN) <= EPS_M)
    count_m_max = sum(1 for m in m_joint_samples if abs(float(m) - M_MAX) <= EPS_M)
    count_a_min = sum(1 for a in a_joint_samples if abs(float(a) - A_MIN) <= EPS_A)
    count_a_max = sum(1 for a in a_joint_samples if abs(float(a) - A_MAX) <= EPS_A)
    boundary_hit_count = sum(
        1
        for m, a in zip(m_joint_samples, a_joint_samples)
        if (abs(float(m) - M_MIN) <= EPS_M)
        or (abs(float(m) - M_MAX) <= EPS_M)
        or (abs(float(a) - A_MIN) <= EPS_A)
        or (abs(float(a) - A_MAX) <= EPS_A)
    )
    boundary_fraction = float(boundary_hit_count / len(m_joint_samples))
    diagnostics = {
        "n_samples": n_samples,
        "n_accepted": len(m_joint_samples),
        "boundary_hits": boundary_hit_count,
        "boundary_fraction": boundary_fraction,
        "M_p50": m_p50,
        "a_p50": a_p50,
    }
    ctx.params.update(diagnostics)

    def _abort_boundary(reason_key: str) -> None:
        ctx.params.update(diagnostics)
        abort(
            ctx,
            reason=(
                f"{STAGE} failed: KERR_GRID_SATURATION: {reason_key}; "
                f"n_accepted={len(m_joint_samples)}; boundary_hits={boundary_hit_count}; "
                f"boundary_fraction={boundary_fraction:.6f}; M_p50={m_p50:.9g}; a_p50={a_p50:.9g}"
            ),
        )

    should_abort, abort_reason, warning_spin_physical_floor = _should_abort_for_boundary(
        a_p50=a_p50,
        m_p50=m_p50,
        boundary_fraction=boundary_fraction,
        a_min=A_MIN,
        a_max=A_MAX,
        m_min=M_MIN,
        m_max=M_MAX,
        threshold=BOUNDARY_FRACTION_THRESHOLD,
    )
    warnings: list[dict[str, str]] = []
    if warning_spin_physical_floor:
        warnings.append(
            {
                "warning_code": SPIN_PHYSICAL_FLOOR_WARNING_CODE,
                "warning_msg": SPIN_PHYSICAL_FLOOR_WARNING_MSG,
            }
        )
    if warnings:
        existing_warnings = ctx.params.get("warnings")
        if not isinstance(existing_warnings, list):
            existing_warnings = []
        existing_warnings.extend(warnings)
        ctx.params["warnings"] = existing_warnings
    ctx.params.update(
        {
            "multimode_viability_class": viability_class,
            "multimode_viability_reasons": viability_reasons,
            "multimode_evaluated": True,
            "skips": [],
        }
    )
    if should_abort and abort_reason is not None:
        _abort_boundary(abort_reason)

    per_mode = {
        "220": {"f_hz": q220["f_hz"], "tau_s": q220["tau_s"]},
        "221": {"f_hz": q221["f_hz"], "tau_s": q221["tau_s"]},
    }

    kerr_payload = {
        "schema_name": "kerr_from_multimode",
        "schema_version": "mvp_kerr_from_multimode_v1",
        "json_strict": True,
        "created_utc": _utc_now_z(),
        "run_id": ctx.run_id,
        "stage": STAGE,
        "source": {
            "multimode_estimates": {
                "relpath": "s3b_multimode_estimates/outputs/multimode_estimates.json",
                "sha256": input_by_label["multimode_estimates"]["sha256"],
            },
        },
        "conventions": {
            "units": {
                "f_hz": "Hz",
                "tau_s": "s",
                "mass_solar": "M_sun",
                "spin_dimensionless": "a",
            },
            "mode_labels": ["220", "221"],
            "mapping": (
                "Kerr inversion via deterministic dense grid search in (M_f_solar,a_f); "
                "forward model from mvp.kerr_qnm_fits.kerr_qnm for modes (2,2,0) and (2,2,1); "
                "objective in log-space over f_hz and tau_s."
            ),
        },
        "estimates": {
            "per_mode": per_mode,
            "kerr": {
                "M_f_solar": m_quantiles,
                "a_f": a_quantiles,
                "covariance": None,
            },
        },
        "consistency": {
            "metric_name": "delta_kerr",
            "value": None,
            "threshold": 0.1,
            "pass": False,
        },
        "trace": {
            "inversion": {
                "method": "deterministic_grid_search",
                "grid_or_solver": "grid_MxA_200x200_log_error_Mmax500_amax0p999",
                "seed": seed,
                "tie_break": "first_minimum_in_stable_grid_order",
            },
        },
    }

    if "model_comparison" in input_by_label:
        kerr_payload["source"]["model_comparison"] = {
            "relpath": "s3b_multimode_estimates/outputs/model_comparison.json",
            "sha256": input_by_label["model_comparison"]["sha256"],
        }

    m220_med = _quantiles(m220_samples)["p50"]
    m221_med = _quantiles(m221_samples)["p50"]
    a220_med = _quantiles(a220_samples)["p50"]
    a221_med = _quantiles(a221_samples)["p50"]
    m_med = _quantiles(m_joint_samples)["p50"]

    delta = None
    if m220_med and m221_med and a220_med is not None and a221_med is not None and m_med and float(m_med) > 0:
        delta = float(math.sqrt(((float(m220_med) - float(m221_med)) / float(m_med)) ** 2 + (float(a220_med) - float(a221_med)) ** 2))

    kerr_payload["consistency"]["value"] = delta
    kerr_payload["consistency"]["pass"] = bool(delta is not None and delta <= 0.1)

    extraction_core = _extract_kerr_with_covariance(multimode)
    kerr_extraction_payload = {
        "schema_name": "kerr_extraction",
        "schema_version": "mvp_kerr_extraction_v1",
        **{k: v for k, v in extraction_core.items() if k != "aux_input"},
        "verdict": "PASS",
    }

    diagnostics_payload = {
        "schema_name": "kerr_from_multimode_diagnostics",
        "schema_version": "mvp_kerr_from_multimode_diagnostics_v1",
        "json_strict": True,
        "created_utc": _utc_now_z(),
        "run_id": ctx.run_id,
        "stage": STAGE,
        "diagnostics": {
            "solver_status": {
                "status": "ok",
                "n_grid_points": len(grid_m),
                "n_samples": n_samples,
                "n_accepted": len(m_joint_samples),
                "n_rejected": rejected,
                "edge_hits": {
                    "M_min": count_m_min,
                    "M_max": count_m_max,
                    "a_min": count_a_min,
                    "a_max": count_a_max,
                    "joint_boundary_total": boundary_hit_count,
                },
                "saturation_fraction": boundary_fraction,
            },
            "conditioning": {
                "grid_mass_range_msun": [M_MIN, M_MAX],
                "grid_spin_range": [A_MIN, A_MAX],
                "grid_shape": [GRID_M_SIZE, GRID_A_SIZE],
                "objective": "mahalanobis_log_residuals_f_tau",
                "grid_limits": {
                    "M_min": M_MIN,
                    "M_max": M_MAX,
                    "a_min": A_MIN,
                    "a_max": A_MAX,
                },
                "boundary_counts": {
                    "M_min": count_m_min,
                    "M_max": count_m_max,
                    "a_min": count_a_min,
                    "a_max": count_a_max,
                },
                "boundary_fraction": boundary_fraction,
            },
            "rejected_fraction": float(rejected / n_samples) if n_samples > 0 else 0.0,
            "warnings": warnings,
            "notes": [
                "Uncertainty propagation uses deterministic triangular sampling from per-mode p10/p50/p90.",
                "Tie-break for equal objective values uses first minimum in stable nested-loop grid order.",
                "Covariance omitted in this phase; set to null by schema design.",
            ],
        },
    }

    kerr_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_from_multimode.json", kerr_payload)
    diag_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_from_multimode_diagnostics.json", diagnostics_payload)
    kerr_extraction_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_extraction.json", kerr_extraction_payload)

    # Compute kerr_extraction.json: point-estimate Kerr inversion with covariance propagation.
    f220_med = float(q220["f_hz"]["p50"])
    f221_med = float(q221["f_hz"]["p50"])
    tau220_med = float(q220["tau_s"]["p50"])
    tau221_med = float(q221["tau_s"]["p50"])
    sigma_f220 = max(float(q220["f_hz"]["p90"]) - float(q220["f_hz"]["p10"]), f220_med * 1e-6) / 2.0
    sigma_f221 = max(float(q221["f_hz"]["p90"]) - float(q221["f_hz"]["p10"]), f221_med * 1e-6) / 2.0

    M_ext, a_ext, sigma_M, sigma_a, cov_M_a = _extract_kerr_with_covariance_core(
        f220_med, f221_med, sigma_f220, sigma_f221, grid_m, grid_a, lnf_220, lnf_221
    )

    # Compute residuals: predicted f_221 and tau_221 from extracted (M, a) vs observed.
    delta_f221_Hz: float | None = None
    delta_tau221_ms: float | None = None
    consistency_score: float | None = None
    try:
        from mvp.kerr_qnm_fits import kerr_qnm
        predicted_221 = kerr_qnm(M_ext, a_ext, (2, 2, 1))
        delta_f221_Hz = float(predicted_221.f_hz - f221_med)
        delta_tau221_ms = float((predicted_221.tau_s - tau221_med) * 1e3)
        consistency_score = float(
            math.sqrt((delta_f221_Hz / max(f221_med, 1.0)) ** 2
                      + (delta_tau221_ms / max(abs(tau221_med * 1e3), 1e-6)) ** 2)
        )
    except Exception:
        pass

    extraction_payload = {
        "schema_name": "kerr_extraction",
        "schema_version": "mvp_kerr_extraction_v1",
        "json_strict": True,
        "created_utc": _utc_now_z(),
        "run_id": ctx.run_id,
        "stage": STAGE,
        "verdict": "PASS" if (consistency_score is not None and math.isfinite(consistency_score)) else "NONINFORMATIVE",
        "M_final_Msun": M_ext,
        "chi_final": a_ext,
        "sigma_M": sigma_M,
        "sigma_chi": sigma_a,
        "cov_M_chi": cov_M_a,
        "delta_f221_Hz": delta_f221_Hz,
        "delta_tau221_ms": delta_tau221_ms,
        "consistency_score": consistency_score,
        "method": "finite_diff_jacobian_from_grid_inversion",
    }
    extraction_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_extraction.json", extraction_payload)

    log_stage_paths(ctx)

    return {
        "kerr_from_multimode": kerr_path,
        "kerr_from_multimode_diagnostics": diag_path,
        "kerr_extraction": extraction_path,
        "kerr_extraction": kerr_extraction_path,
    }


def _exit_code(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _abort_with_reason(ctx: StageContext, reason: str) -> int:
    try:
        abort(ctx, reason=reason)
    except SystemExit as exc:
        return _exit_code(exc.code, 2)
    return 2


def main() -> int:
    args = build_argparser().parse_args()
    ctx = init_stage(args.run_id, STAGE)
    ctx.params = _base_params()
    try:
        artifacts = _execute(ctx)
    except NotImplementedError:
        return _abort_with_reason(ctx, f"{STAGE} failed: NOT_IMPLEMENTED")
    except SystemExit as exc:
        return _exit_code(exc.code, 1)
    except Exception as exc:  # deterministic abort reason; no traceback in reason field
        reason = f"{STAGE} failed: {type(exc).__name__}: {exc}"
        return _abort_with_reason(ctx, reason)

    if not artifacts:
        return _abort_with_reason(ctx, f"{STAGE} failed: NO_OUTPUTS")

    finalize(ctx, artifacts=artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
