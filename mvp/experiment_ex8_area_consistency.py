#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import sha256_file, utc_now_iso, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.kerr_qnm_fits import CHI_MAX, MSUN_S, kerr_Q, kerr_omega_dimless

STAGE = "experiment_ex8_area_consistency"
G_SI = 6.67430e-11
C_SI = 299792458.0
MSUN_SI = 1.98892e30


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _bisect(func, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> float | None:
    fa = func(a)
    fb = func(b)
    if not (math.isfinite(fa) and math.isfinite(fb)) or fa * fb > 0:
        return None
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = func(mid)
        if not math.isfinite(fm):
            return None
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return mid
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return (a + b) / 2.0


def _as_float_or_none(v: Any) -> float | None:
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def classify_consistency(tension_sigma: float | None, thresholds: tuple[float, float]) -> str:
    if tension_sigma is None or not math.isfinite(tension_sigma):
        return "UNDETERMINED"
    t_tension, t_inconsistent = thresholds
    if tension_sigma < t_tension:
        return "CONSISTENT"
    if tension_sigma < t_inconsistent:
        return "TENSION"
    return "INCONSISTENT"


def invert_kerr_qnm(f_hz: float, Q: float, l: int, m: int, n: int) -> dict[str, Any]:
    mode = (l, m, n)
    if f_hz <= 0 or Q <= 0:
        return {"M_solar": None, "chi": None, "converged": False}

    q0 = kerr_Q(0.0, mode)
    qmax = kerr_Q(CHI_MAX, mode)
    qmin, qhigh = min(q0, qmax), max(q0, qmax)
    if Q < qmin or Q > qhigh:
        return {"M_solar": None, "chi": None, "converged": False}

    f = lambda chi: kerr_Q(chi, mode) - Q
    f0 = f(0.0)
    f1 = f(CHI_MAX)
    if abs(f0) / Q < 1e-8:
        chi = 0.0
    elif abs(f1) / Q < 1e-8:
        chi = CHI_MAX
    else:
        chi = _bisect(f, 0.0, CHI_MAX, tol=1e-12, max_iter=200)
    if chi is None:
        return {"M_solar": None, "chi": None, "converged": False}

    q_at = kerr_Q(chi, mode)
    if abs(q_at - Q) / Q >= 1e-8:
        return {"M_solar": None, "chi": None, "converged": False}

    omega_dimless = kerr_omega_dimless(chi, mode)
    m_solar = omega_dimless / (2.0 * math.pi * f_hz * MSUN_S)
    if not math.isfinite(m_solar) or m_solar <= 0:
        return {"M_solar": None, "chi": None, "converged": False}
    return {"M_solar": float(m_solar), "chi": float(chi), "converged": True}


def compute_horizon_area(M_solar: float, chi: float) -> float:
    if M_solar <= 0 or abs(chi) >= 1:
        return float("nan")
    return 8.0 * math.pi * (M_solar ** 2) * (1.0 + math.sqrt(1.0 - chi ** 2))


def _extract_mode_220(estimates: dict[str, Any]) -> dict[str, float | None]:
    combined = estimates.get("combined") if isinstance(estimates.get("combined"), dict) else {}
    unc = estimates.get("combined_uncertainty") if isinstance(estimates.get("combined_uncertainty"), dict) else {}

    f_hz = _as_float_or_none(combined.get("f_hz"))
    q = _as_float_or_none(combined.get("Q"))
    sigma_f = _as_float_or_none(unc.get("sigma_f_hz"))
    sigma_q = _as_float_or_none(unc.get("sigma_Q"))

    if f_hz is None:
        f_hz = _as_float_or_none(estimates.get("f_hz"))
    if q is None:
        q = _as_float_or_none(estimates.get("Q"))
    if sigma_f is None:
        sigma_f = _as_float_or_none(estimates.get("sigma_f_hz"))
    if sigma_q is None:
        sigma_q = _as_float_or_none(estimates.get("sigma_Q"))

    return {"f_hz": f_hz, "Q": q, "sigma_f_hz": sigma_f, "sigma_Q": sigma_q}


def _extract_mode_221(mc: dict[str, Any]) -> dict[str, Any]:
    decision = mc.get("decision") if isinstance(mc.get("decision"), dict) else {}
    two_mode_preferred = decision.get("two_mode_preferred")
    if two_mode_preferred is None:
        two_mode_preferred = mc.get("two_mode_preferred")

    trace = mc.get("trace") if isinstance(mc.get("trace"), dict) else {}

    f_hz = _as_float_or_none(mc.get("f_hz_221"))
    q = _as_float_or_none(mc.get("Q_221"))
    if f_hz is None:
        ln_f = _as_float_or_none(trace.get("ln_f221"))
        f_hz = math.exp(ln_f) if ln_f is not None else None
    if q is None:
        ln_q = _as_float_or_none(trace.get("ln_Q221"))
        q = math.exp(ln_q) if ln_q is not None else None

    sigma_f = _as_float_or_none(mc.get("sigma_f_hz_221"))
    sigma_q = _as_float_or_none(mc.get("sigma_Q_221"))
    if sigma_f is None:
        sigma_lnf = _as_float_or_none(trace.get("sigma_lnf221"))
        if sigma_lnf is not None and f_hz is not None:
            sigma_f = sigma_lnf * f_hz
    if sigma_q is None:
        sigma_lnq = _as_float_or_none(trace.get("sigma_lnQ221"))
        if sigma_lnq is not None and q is not None:
            sigma_q = sigma_lnq * q

    return {
        "two_mode_preferred": bool(two_mode_preferred) if two_mode_preferred is not None else False,
        "f_hz": f_hz,
        "Q": q,
        "sigma_f_hz": sigma_f,
        "sigma_Q": sigma_q,
    }


def propagate_uncertainties(f_hz: float, sigma_f: float, Q: float, sigma_Q: float, l: int, m: int, n: int) -> dict[str, Any]:
    inv = invert_kerr_qnm(f_hz, Q, l, m, n)
    if not inv["converged"]:
        return {
            "M": None,
            "sigma_M": None,
            "chi": None,
            "sigma_chi": None,
            "A": None,
            "sigma_A": None,
            "area_p10": None,
            "area_p50": None,
            "area_p90": None,
            "inversion_converged": False,
        }

    M = float(inv["M_solar"])
    chi = float(inv["chi"])
    A = compute_horizon_area(M, chi)
    mode = (l, m, n)

    dQ = max(1e-9, abs(Q) * 0.01)
    plus = invert_kerr_qnm(f_hz, Q + dQ, l, m, n)
    minus = invert_kerr_qnm(f_hz, max(1e-12, Q - dQ), l, m, n)
    if plus["converged"] and minus["converged"]:
        dchi_dQ = (float(plus["chi"]) - float(minus["chi"])) / (2.0 * dQ)
    else:
        dchi_dQ = 0.0

    sigma_chi = abs(dchi_dQ) * max(0.0, sigma_Q)
    dM_df = -M / f_hz

    dchi = 1e-5
    op = kerr_omega_dimless(min(CHI_MAX, chi + dchi), mode)
    om = kerr_omega_dimless(max(0.0, chi - dchi), mode)
    oo = kerr_omega_dimless(chi, mode)
    domega_dchi = (op - om) / (2.0 * dchi)
    dM_dchi = M * (domega_dchi / oo) if oo != 0 else 0.0

    sigma_M = math.sqrt((dM_df ** 2) * (max(0.0, sigma_f) ** 2) + (dM_dchi ** 2) * (sigma_chi ** 2))

    root = math.sqrt(max(1e-15, 1.0 - chi ** 2))
    dA_dM = 2.0 * A / M
    dA_dchi = -8.0 * math.pi * (M ** 2) * (chi / root)
    sigma_A = math.sqrt((dA_dM ** 2) * (sigma_M ** 2) + (dA_dchi ** 2) * (sigma_chi ** 2))

    k = 1.282
    return {
        "M": M,
        "sigma_M": sigma_M,
        "chi": chi,
        "sigma_chi": sigma_chi,
        "A": A,
        "sigma_A": sigma_A,
        "area_p10": A - k * sigma_A,
        "area_p50": A,
        "area_p90": A + k * sigma_A,
        "inversion_converged": True,
    }


def _tension(delta: float, sigma: float) -> float | None:
    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        return None
    return abs(delta) / sigma


def _build_mode_payload(obs: dict[str, Any], prop: dict[str, Any]) -> dict[str, Any]:
    return {
        "f_hz": obs.get("f_hz"),
        "Q": obs.get("Q"),
        "sigma_f_hz": obs.get("sigma_f_hz"),
        "sigma_Q": obs.get("sigma_Q"),
        "M_solar": prop.get("M"),
        "sigma_M_solar": prop.get("sigma_M"),
        "chi": prop.get("chi"),
        "sigma_chi": prop.get("sigma_chi"),
        "area_GM2": prop.get("A"),
        "sigma_area_GM2": prop.get("sigma_A"),
        "area_p10": prop.get("area_p10"),
        "area_p50": prop.get("area_p50"),
        "area_p90": prop.get("area_p90"),
        "inversion_converged": prop.get("inversion_converged", False),
    }


def run_experiment(run_id: str, tension_sigma: float = 2.0, inconsistency_sigma: float = 3.0) -> dict[str, Any]:
    ctx = init_stage(run_id, STAGE, params={"tension_sigma": tension_sigma, "inconsistency_sigma": inconsistency_sigma})
    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    mc_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"

    check_inputs(ctx, {"estimates": estimates_path, "model_comparison": mc_path})

    try:
        estimates = _read_json(estimates_path)
        mc = _read_json(mc_path)

        mode220 = _extract_mode_220(estimates)
        if mode220["f_hz"] is None or mode220["Q"] is None:
            abort(
                ctx,
                "estimates.json missing f_hz/Q fields for mode 220. "
                f"Expected path: {estimates_path} "
                "Regenerate: python -m mvp.pipeline single --event-id <EVENT> --atlas-default",
            )

        mode221 = _extract_mode_221(mc)
        viable = True
        reason = None
        if not mode221["two_mode_preferred"]:
            viable = False
            reason = "two_mode_preferred_false"
        elif any(mode221[k] is None for k in ("f_hz", "Q", "sigma_f_hz", "sigma_Q")):
            viable = False
            reason = "mode_221_data_missing"

        p220 = propagate_uncertainties(
            float(mode220["f_hz"]), float(mode220["sigma_f_hz"] or 0.0), float(mode220["Q"]), float(mode220["sigma_Q"] or 0.0), 2, 2, 0
        )

        if viable:
            p221 = propagate_uncertainties(
                float(mode221["f_hz"]), float(mode221["sigma_f_hz"] or 0.0), float(mode221["Q"]), float(mode221["sigma_Q"] or 0.0), 2, 2, 1
            )
        else:
            p221 = {
                "M": None, "sigma_M": None, "chi": None, "sigma_chi": None,
                "A": None, "sigma_A": None, "area_p10": None, "area_p50": None,
                "area_p90": None, "inversion_converged": False,
            }

        status = "UNDETERMINED"
        consistency: dict[str, Any]
        if not viable:
            status = "MODE_221_NOT_VIABLE"
            consistency = {
                "status": status,
                "delta_M_solar": None,
                "sigma_delta_M": None,
                "tension_M_sigma": None,
                "delta_chi": None,
                "sigma_delta_chi": None,
                "tension_chi_sigma": None,
                "delta_area_GM2": None,
                "sigma_delta_area": None,
                "tension_area_sigma": None,
            }
        elif not p220["inversion_converged"] or not p221["inversion_converged"]:
            status = "INVERSION_FAILED"
            consistency = {
                "status": status,
                "delta_M_solar": None,
                "sigma_delta_M": None,
                "tension_M_sigma": None,
                "delta_chi": None,
                "sigma_delta_chi": None,
                "tension_chi_sigma": None,
                "delta_area_GM2": None,
                "sigma_delta_area": None,
                "tension_area_sigma": None,
            }
        else:
            dM = p220["M"] - p221["M"]
            sM = math.sqrt((p220["sigma_M"] ** 2) + (p221["sigma_M"] ** 2))
            dchi = p220["chi"] - p221["chi"]
            schi = math.sqrt((p220["sigma_chi"] ** 2) + (p221["sigma_chi"] ** 2))
            dA = p220["A"] - p221["A"]
            sA = math.sqrt((p220["sigma_A"] ** 2) + (p221["sigma_A"] ** 2))

            tM = _tension(dM, sM)
            tchi = _tension(dchi, schi)
            tA = _tension(dA, sA)
            status = classify_consistency(tA, (tension_sigma, inconsistency_sigma))

            consistency = {
                "status": status,
                "delta_M_solar": dM,
                "sigma_delta_M": sM,
                "tension_M_sigma": tM,
                "delta_chi": dchi,
                "sigma_delta_chi": schi,
                "tension_chi_sigma": tchi,
                "delta_area_GM2": dA,
                "sigma_delta_area": sA,
                "tension_area_sigma": tA,
            }

        output = {
            "schema_version": "ex8_area_consistency_v1",
            "run_id": run_id,
            "event_id": str(estimates.get("event_id") or run_id),
            "inputs_sha256": {
                "estimates_json": sha256_file(estimates_path),
                "model_comparison_json": sha256_file(mc_path),
            },
            "parameters": {
                "tension_sigma": tension_sigma,
                "inconsistency_sigma": inconsistency_sigma,
                "propagation_method": "delta_method",
                "inversion_tolerance": 1e-8,
            },
            "mode_220": _build_mode_payload(mode220, p220),
            "mode_221": _build_mode_payload(mode221, p221),
            "consistency": consistency,
            "hawking_area": {
                "enabled": False,
                "A_initial_source": None,
                "A_initial_GM2": None,
                "violation_margin_GM2": None,
                "policy": None,
            },
            "viability": {
                "mode_221_viable": viable,
                "two_mode_preferred": mode221["two_mode_preferred"],
                "reason": reason,
            },
            "created": utc_now_iso(),
        }

        out_path = ctx.outputs_dir / "area_consistency.json"
        write_json_atomic(out_path, output)

        finalize(
            ctx,
            artifacts={"area_consistency": out_path},
            results={"status": status, "mode_221_viable": viable},
        )
        log_stage_paths(ctx)
        return output
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EX8 area consistency for modes 220/221")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tension-sigma", type=float, default=2.0)
    ap.add_argument("--inconsistency-sigma", type=float, default=3.0)
    args = ap.parse_args(argv)

    run_experiment(args.run, args.tension_sigma, args.inconsistency_sigma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
