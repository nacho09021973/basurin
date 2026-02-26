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

from mvp.contracts import StageContext, abort, check_inputs, finalize, init_stage

STAGE = "s4d_kerr_from_multimode"
MASS_MIN = 5.0
MASS_MAX = 200.0
SPIN_MIN = 0.0
SPIN_MAX = 0.99
BOUNDARY_FRACTION_THRESHOLD = 0.20


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


def _at_boundary(value: float, lo: float, hi: float, eps: float) -> bool:
    return abs(float(value) - float(lo)) <= float(eps) or abs(float(value) - float(hi)) <= float(eps)


def _build_grid() -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    try:
        from mvp.kerr_qnm_fits import kerr_qnm
    except Exception as exc:
        raise RuntimeError(
            "Missing Kerr QNM forward model in repo; cannot invert f/tau to (M,a) without canonical model"
        ) from exc

    a_vals = [SPIN_MIN + ((SPIN_MAX - SPIN_MIN) * i / 199.0) for i in range(200)]
    m_vals = [MASS_MIN + ((MASS_MAX - MASS_MIN) * i / 199.0) for i in range(200)]

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


def _best_idx_joint(
    obs: dict[str, dict[str, float]],
    lnf_220: list[float],
    lntau_220: list[float],
    lnf_221: list[float],
    lntau_221: list[float],
) -> int:
    t220_f = math.log(float(obs["220"]["f_hz"]))
    t220_tau = math.log(float(obs["220"]["tau_s"]))
    t221_f = math.log(float(obs["221"]["f_hz"]))
    t221_tau = math.log(float(obs["221"]["tau_s"]))

    best_i = 0
    best_e = float("inf")
    for i in range(len(lnf_220)):
        e = (
            (lnf_220[i] - t220_f) ** 2
            + (lntau_220[i] - t220_tau) ** 2
            + (lnf_221[i] - t221_f) ** 2
            + (lntau_221[i] - t221_tau) ** 2
        )
        if e < best_e:
            best_e = e
            best_i = i
    return best_i


def _best_idx_single(
    mode_obs: dict[str, float],
    lnf: list[float],
    lntau: list[float],
) -> int:
    tf = math.log(float(mode_obs["f_hz"]))
    tt = math.log(float(mode_obs["tau_s"]))
    best_i = 0
    best_e = float("inf")
    for i in range(len(lnf)):
        e = (lnf[i] - tf) ** 2 + (lntau[i] - tt) ** 2
        if e < best_e:
            best_e = e
            best_i = i
    return best_i


def _execute(ctx: StageContext) -> dict[str, Path]:
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    model_comparison_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"

    inputs = check_inputs(
        ctx,
        paths={"multimode_estimates": multimode_path},
        optional={"model_comparison": model_comparison_path},
    )
    input_by_label = {row.get("label", ""): row for row in inputs}

    multimode = json.loads(multimode_path.read_text(encoding="utf-8"))
    model_comparison = (
        json.loads(model_comparison_path.read_text(encoding="utf-8"))
        if model_comparison_path.exists()
        else None
    )

    try:
        q220 = _extract_mode_quantiles(multimode, "220")
        q221 = _extract_mode_quantiles(multimode, "221")
    except ValueError as exc:
        abort(ctx, reason=str(exc))

    for label, q in (("220", q220), ("221", q221)):
        for metric in ("f_hz", "tau_s"):
            vals = [q[metric]["p10"], q[metric]["p50"], q[metric]["p90"]]
            if not all(v is not None and float(v) > 0.0 and math.isfinite(float(v)) for v in vals):
                abort(ctx, reason=f"Missing required fields in multimode_estimates: mode {label} {metric} must define positive finite p10/p50/p90")

    try:
        grid_m, grid_a, lnf_220, lntau_220, lnf_221, lntau_221 = _build_grid()
    except RuntimeError as exc:
        abort(ctx, reason=str(exc))

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
            idx_joint = _best_idx_joint({"220": obs220, "221": obs221}, lnf_220, lntau_220, lnf_221, lntau_221)
            idx_220 = _best_idx_single(obs220, lnf_220, lntau_220)
            idx_221 = _best_idx_single(obs221, lnf_221, lntau_221)
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

    per_mode = {
        "220": {"f_hz": q220["f_hz"], "tau_s": q220["tau_s"]},
        "221": {"f_hz": q221["f_hz"], "tau_s": q221["tau_s"]},
    }

    kerr_payload = {
        "schema_name": "kerr_from_multimode",
        "schema_version": 1,
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
                "M_f_solar": _quantiles(m_joint_samples),
                "a_f": _quantiles(a_joint_samples),
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
                "grid_or_solver": "grid_MxA_200x200_log_error",
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
    m_q = _quantiles(m_joint_samples)
    a_q = _quantiles(a_joint_samples)
    m_med = m_q["p50"]
    a_med = a_q["p50"]

    eps_m = 1e-6 * (MASS_MAX - MASS_MIN)
    eps_a = 1e-6
    m_min_hits = sum(1 for m in m_joint_samples if abs(float(m) - MASS_MIN) <= eps_m)
    m_max_hits = sum(1 for m in m_joint_samples if abs(float(m) - MASS_MAX) <= eps_m)
    a_min_hits = sum(1 for a in a_joint_samples if abs(float(a) - SPIN_MIN) <= eps_a)
    a_max_hits = sum(1 for a in a_joint_samples if abs(float(a) - SPIN_MAX) <= eps_a)
    boundary_hits_any = 0
    for m, a in zip(m_joint_samples, a_joint_samples):
        if (
            abs(float(m) - MASS_MIN) <= eps_m
            or abs(float(m) - MASS_MAX) <= eps_m
            or abs(float(a) - SPIN_MIN) <= eps_a
            or abs(float(a) - SPIN_MAX) <= eps_a
        ):
            boundary_hits_any += 1

    boundary_fraction = float(boundary_hits_any / len(m_joint_samples)) if m_joint_samples else 0.0

    if _at_boundary(float(m_med), MASS_MIN, MASS_MAX, eps_m):
        abort(ctx, reason=f"{STAGE} failed: KERR_BOUNDARY_HIT_M")
    if _at_boundary(float(a_med), SPIN_MIN, SPIN_MAX, eps_a):
        abort(ctx, reason=f"{STAGE} failed: KERR_BOUNDARY_HIT_A")
    if boundary_fraction >= BOUNDARY_FRACTION_THRESHOLD:
        abort(ctx, reason=f"{STAGE} failed: KERR_BOUNDARY_FRACTION_HIGH")

    delta = None
    if m220_med and m221_med and a220_med is not None and a221_med is not None and m_med and float(m_med) > 0:
        delta = float(math.sqrt(((float(m220_med) - float(m221_med)) / float(m_med)) ** 2 + (float(a220_med) - float(a221_med)) ** 2))

    kerr_payload["consistency"]["value"] = delta
    has_boundary_contact = bool(
        boundary_fraction > 0.0
        or _at_boundary(float(m_med), MASS_MIN, MASS_MAX, eps_m)
        or _at_boundary(float(a_med), SPIN_MIN, SPIN_MAX, eps_a)
    )
    kerr_payload["estimates"]["kerr"]["M_f_solar"] = m_q
    kerr_payload["estimates"]["kerr"]["a_f"] = a_q
    kerr_payload["consistency"]["pass"] = bool(delta is not None and delta <= 0.1 and not has_boundary_contact)

    diagnostics_payload = {
        "schema_name": "kerr_from_multimode_diagnostics",
        "schema_version": 1,
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
                "boundary_fraction": boundary_fraction,
                "boundary_counts": {
                    "M_min": m_min_hits,
                    "M_max": m_max_hits,
                    "a_min": a_min_hits,
                    "a_max": a_max_hits,
                },
            },
            "conditioning": {
                "grid_mass_range_msun": [MASS_MIN, MASS_MAX],
                "grid_spin_range": [SPIN_MIN, SPIN_MAX],
                "grid_shape": [200, 200],
                "objective": "sum_squared_log_residuals_f_tau",
                "grid_limits": {
                    "mass_min": MASS_MIN,
                    "mass_max": MASS_MAX,
                    "spin_min": SPIN_MIN,
                    "spin_max": SPIN_MAX,
                },
            },
            "rejected_fraction": float(rejected / n_samples) if n_samples > 0 else 0.0,
            "notes": [
                "Uncertainty propagation uses deterministic triangular sampling from per-mode p10/p50/p90.",
                "Tie-break for equal objective values uses first minimum in stable nested-loop grid order.",
                "Covariance omitted in this phase; set to null by schema design.",
            ],
        },
    }

    kerr_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_from_multimode.json", kerr_payload)
    diag_path = _write_json_strict_atomic(ctx.outputs_dir / "kerr_from_multimode_diagnostics.json", diagnostics_payload)

    print(f"OUT_ROOT={ctx.out_root}")
    print(f"STAGE_DIR={ctx.stage_dir}")
    print(f"OUTPUTS_DIR={ctx.outputs_dir}")
    print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
    print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")

    return {
        "kerr_from_multimode": kerr_path,
        "kerr_from_multimode_diagnostics": diag_path,
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
