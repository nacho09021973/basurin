#!/usr/bin/env python3
"""Canonical stage s7: beyond-Kerr deviation score from multimode + Kerr extraction."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import write_json_atomic
from mvp.contracts import StageContext, abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "s7_beyond_kerr_deviation_score"


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _chi2_cdf_2dof(x: float) -> float:
    if x < 0.0:
        return 0.0
    return 1.0 - math.exp(-x / 2.0)


def _beyond_kerr_verdict(chi2: float) -> str:
    if chi2 < 4.605:
        return "GR_CONSISTENT"
    if chi2 < 9.210:
        return "GR_TENSION"
    return "GR_INCONSISTENT"


def _compute_score(
    M_final: float,
    chi_final: float,
    f221_obs: float,
    tau221_obs: float,
    sigma_f221: float,
    sigma_tau221: float,
) -> dict[str, Any]:
    from mvp.kerr_qnm_fits import kerr_qnm

    pred = kerr_qnm(M_final, chi_final, (2, 2, 1))
    delta_f_norm = (f221_obs - pred.f_hz) / max(sigma_f221, pred.f_hz * 1e-6)
    delta_tau_norm = (tau221_obs - pred.tau_s) / max(sigma_tau221, pred.tau_s * 1e-6)
    chi2 = delta_f_norm**2 + delta_tau_norm**2
    epsilon_f = (f221_obs - pred.f_hz) / pred.f_hz
    epsilon_tau = (tau221_obs - pred.tau_s) / pred.tau_s
    return {
        "chi2_kerr_2dof": chi2,
        "cdf_proxy": _chi2_cdf_2dof(chi2),
        "verdict": _beyond_kerr_verdict(chi2),
        "epsilon_f": epsilon_f,
        "epsilon_tau": epsilon_tau,
        "delta_f_norm": delta_f_norm,
        "delta_tau_norm": delta_tau_norm,
        "predicted_f221_hz": pred.f_hz,
        "predicted_tau221_s": pred.tau_s,
        "gr_threshold_90pct": 4.605,
        "gr_threshold_99pct": 9.210,
    }


def _require(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"Missing required JSON path: {path}")
        cur = cur[part]
    return cur


def _as_float(obj: Any, path: str) -> float:
    val = _require(obj, path)
    try:
        out = float(val)
    except Exception as exc:
        raise ValueError(f"Invalid numeric value at JSON path: {path}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite numeric value at JSON path: {path}")
    return out


def _execute(ctx: StageContext) -> dict[str, Path]:
    kerr_path = ctx.run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    check_inputs(ctx, {"kerr_extraction": kerr_path, "multimode_estimates": multimode_path})

    kerr = json.loads(kerr_path.read_text(encoding="utf-8"))
    if kerr.get("verdict") == "SKIPPED_MULTIMODE_GATE" or kerr.get("M_final_Msun") is None:
        payload = {
            "schema_name": "beyond_kerr_score",
            "schema_version": "mvp_beyond_kerr_score_v1",
            "created_utc": _utc_now_z(),
            "run_id": ctx.run_id,
            "stage": STAGE,
            "verdict": "SKIPPED_S4D_GATE",
            "chi2_kerr_2dof": None,
            "cdf_proxy": None,
            "epsilon_f": None,
            "epsilon_tau": None,
            "delta_f_norm": None,
            "delta_tau_norm": None,
            "predicted_f221_hz": None,
            "predicted_tau221_s": None,
            "gr_threshold_90pct": 4.605,
            "gr_threshold_99pct": 9.210,
            "source_s4d_verdict": kerr.get("verdict"),
        }
        out_path = ctx.outputs_dir / "beyond_kerr_score.json"
        write_json_atomic(out_path, payload)
        return {"beyond_kerr_score": out_path}

    multimode = json.loads(multimode_path.read_text(encoding="utf-8"))
    try:
        f_221 = _as_float(multimode, "estimates.per_mode.221.f_hz.p50")
        tau_221 = _as_float(multimode, "estimates.per_mode.221.tau_s.p50")
        f_221_p10 = _as_float(multimode, "estimates.per_mode.221.f_hz.p10")
        f_221_p90 = _as_float(multimode, "estimates.per_mode.221.f_hz.p90")
        tau_221_p10 = _as_float(multimode, "estimates.per_mode.221.tau_s.p10")
        tau_221_p90 = _as_float(multimode, "estimates.per_mode.221.tau_s.p90")
    except ValueError as exc:
        abort(
            ctx,
            (
                f"{exc}. Expected input: {multimode_path}. "
                f"Regenerate upstream with: python -m mvp.s3b_multimode_estimates --run-id {ctx.run_id}"
            ),
        )

    sigma_f221 = max((f_221_p90 - f_221_p10) / 2.0, 0.0)
    sigma_tau221 = max((tau_221_p90 - tau_221_p10) / 2.0, 0.0)

    try:
        score = _compute_score(
            M_final=float(kerr["M_final_Msun"]),
            chi_final=float(kerr["chi_final"]),
            f221_obs=f_221,
            tau221_obs=tau_221,
            sigma_f221=sigma_f221,
            sigma_tau221=sigma_tau221,
        )
    except Exception as exc:
        abort(ctx, f"Failed computing beyond-Kerr score: {exc}")

    payload = {
        "schema_name": "beyond_kerr_score",
        "schema_version": "mvp_beyond_kerr_score_v1",
        "created_utc": _utc_now_z(),
        "run_id": ctx.run_id,
        "stage": STAGE,
        **score,
    }
    out_path = ctx.outputs_dir / "beyond_kerr_score.json"
    write_json_atomic(out_path, payload)
    return {"beyond_kerr_score": out_path}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=STAGE)
    p.add_argument("--run-id", required=True)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    ctx = init_stage(args.run_id, STAGE, params={})
    try:
        artifacts = _execute(ctx)
        out_path = artifacts["beyond_kerr_score"]
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        finalize(
            ctx,
            artifacts=artifacts,
            verdict="PASS",
            results={
                "verdict": payload.get("verdict"),
                "chi2_kerr_2dof": payload.get("chi2_kerr_2dof"),
            },
        )
        log_stage_paths(ctx)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
