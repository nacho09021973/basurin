#!/usr/bin/env python3
"""
stages/ringdown_featuremap_v0_stage.py
--------------------------------------
THE BRIDGE: Maps ringdown parameters (f, tau, Q) into holographic ratio space.

This is the stage that was missing. Without it, the holographic pipeline
(geometry -> spectrum -> dictionary) and the ringdown pipeline
(synth -> inference -> f,tau,Q) operate as two disconnected projects.

Physics (conjectured, falsifiable):
  f   = sqrt(lambda_0) / (2*pi*L)
  tau = L / (alpha * sqrt(lambda_0) * (r_1 - 1))
  Q   = 1 / (2 * alpha * (r_1 - 1))

Inverse (used here):
  r_1_pred = 1 + 1 / (2 * alpha * Q_obs)

Inputs:
  --run <run_id>
  --ringdown-params <path>  JSON with {"f_220": ..., "tau_220": ..., "Q_220": ...}
                            OR JSONL from recovery_cases (uses estimate fields)
  --alpha <float>           Calibration parameter (default: 1.0)
  --k-ratios <int>          Number of ratios to predict (default: 1, only r_1)

Outputs:
  runs/<run>/ringdown_featuremap_v0/
    manifest.json
    stage_summary.json
    outputs/mapped_features.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    get_runs_root,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "ringdown_featuremap_v0"
EXIT_CONTRACT_FAIL = 2


def inverse_phi(
    f_hz: float,
    tau_s: float,
    alpha: float,
    k_ratios: int,
    f_221_hz: float | None = None,
    tau_221_s: float | None = None,
) -> dict:
    """Map observed ringdown (f, tau) -> predicted holographic ratios.

    Core formula:
      Q = pi * f * tau
      r_1 = 1 + 1 / (2 * alpha * Q)

    V1-A overtone:
      r_1 = 1 + 1 / (2 * alpha * Q_220)
      r_2 = 1 + 3 / (2 * alpha * Q_221)  (if overtone is available)

    Legacy fallback (for backward compatibility):
      r_n = 1 + n^2 / (2 * alpha * Q_220)
    """
    if f_hz <= 0 or tau_s <= 0:
        return {"error": "f_hz and tau_s must be positive", "ratios": None}

    Q = math.pi * f_hz * tau_s

    if Q <= 0 or alpha <= 0:
        return {"error": f"invalid Q={Q} or alpha={alpha}", "ratios": None}

    ratios = []
    overtone_used = False
    Q_221 = None

    if f_221_hz is not None and tau_221_s is not None and f_221_hz > 0 and tau_221_s > 0:
        Q_221 = math.pi * f_221_hz * tau_221_s
        if Q_221 > 0:
            overtone_used = True

    for n in range(1, k_ratios + 1):
        if n == 1:
            r_n = 1.0 + 1.0 / (2.0 * alpha * Q)
        elif n == 2 and overtone_used:
            r_n = 1.0 + 3.0 / (2.0 * alpha * Q_221)  # V1-A overtone
        else:
            r_n = 1.0 + (n ** 2) / (2.0 * alpha * Q)
        ratios.append(r_n)

    # Also predict M2_0 from f (the scale observable).
    # omega_0 = 2*pi*f, and omega_0 = sqrt(M2_0)/L.
    # Since L is shared across the atlas, we use omega_0^2 as a
    # scale-independent proxy for M2_0 (i.e., M2_0 * L^2 = omega_0^2).
    omega_0 = 2.0 * math.pi * f_hz
    M2_0_proxy = omega_0 ** 2  # = M2_0 * L^2

    return {
        "f_hz": f_hz,
        "tau_s": tau_s,
        "Q": Q,
        "Q_221": Q_221,
        "alpha": alpha,
        "ratios": ratios,
        "log_ratios": [math.log(r) for r in ratios],
        "M2_0_proxy": M2_0_proxy,
        "overtone_used": overtone_used,
    }


def forward_phi(M2_0: float, r_1: float, L: float, alpha: float, r_2: float | None = None) -> dict:
    """Map holographic theory (M2_0, r_1, L) -> predicted ringdown params.

    Core formulas:
      omega_0 = sqrt(|M2_0|) / L
      f = omega_0 / (2*pi)
      gamma = alpha * omega_0 * (r_1 - 1)
      tau = 1 / gamma
      Q = pi * f * tau
    """
    if M2_0 <= 0 or L <= 0 or r_1 <= 1.0:
        return {"error": "invalid inputs", "f_hz": None, "tau_s": None}

    omega_0 = math.sqrt(abs(M2_0)) / L
    f_pred = omega_0 / (2.0 * math.pi)
    gamma = alpha * omega_0 * (r_1 - 1.0)

    if gamma <= 0:
        return {"error": "gamma <= 0", "f_hz": f_pred, "tau_s": None}

    tau_pred = 1.0 / gamma
    Q_pred = math.pi * f_pred * tau_pred

    out = {
        "f_hz": f_pred,
        "tau_s": tau_pred,
        "Q": Q_pred,
        "omega_0": omega_0,
        "gamma": gamma,
    }
    if r_2 is not None:
        if r_2 <= 1.0:
            return {"error": "invalid r_2 <= 1", "f_hz": f_pred, "tau_s": tau_pred}
        # V1-A overtone: explicit r2 from overtone quality factor
        f_221 = f_pred * math.sqrt(r_1)
        Q_221 = 3.0 / (2.0 * alpha * (r_2 - 1.0))
        tau_221 = Q_221 / (math.pi * f_221)
        out.update({"f_221": f_221, "tau_221": tau_221, "Q_221": Q_221})
    return out


def load_ringdown_params(path: Path) -> list[dict]:
    """Load ringdown parameters from JSON or JSONL.

    Supports:
    - Single event: {"f_220": ..., "tau_220": ..., "Q_220": ...}
    - Single event with truth: {"truth": {"f_220": ..., ...}}
    - JSONL from recovery_cases: one JSON per line with estimate fields
    """
    text = path.read_text(encoding="utf-8").strip()

    # Try JSONL first (multiple lines)
    lines = text.split("\n")
    if len(lines) > 1:
        cases = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case = _extract_params(obj)
            if case:
                cases.append(case)
        return cases

    # Single JSON
    obj = json.loads(text)
    if isinstance(obj, list):
        return [c for c in (_extract_params(o) for o in obj) if c]
    case = _extract_params(obj)
    return [case] if case else []


def _extract_params(obj: dict) -> dict | None:
    """Extract (f, tau, Q) from various JSON formats."""
    case_id = obj.get("case_id", obj.get("id", "unknown"))
    status = obj.get("status", "OK")

    if status != "OK":
        return None

    # From recovery_cases.jsonl (estimate fields)
    est = obj.get("estimate", {})
    if est.get("f_220_hat") is not None:
        f = est["f_220_hat"]
        tau = est["tau_220_hat"]
        Q = est.get("Q_220_hat", math.pi * f * tau)
        f_221 = est.get("f_221_hat")
        tau_221 = est.get("tau_221_hat")
        truth = obj.get("truth", {})
        return {
            "case_id": case_id,
            "f_220": f,
            "tau_220": tau,
            "Q_220": Q,
            "f_221": f_221,
            "tau_221": tau_221,
            "truth": truth if truth else None,
        }

    # From synthetic_event.json (truth fields)
    truth = obj.get("truth", {})
    if truth.get("f_220") is not None:
        f = truth["f_220"]
        tau = truth["tau_220"]
        Q = truth.get("Q_220", math.pi * f * tau)
        f_221 = truth.get("f_221")
        tau_221 = truth.get("tau_221")
        return {
            "case_id": case_id,
            "f_220": f,
            "tau_220": tau,
            "Q_220": Q,
            "f_221": f_221,
            "tau_221": tau_221,
            "truth": truth,
        }

    # Direct fields
    if obj.get("f_220") is not None:
        f = obj["f_220"]
        tau = obj["tau_220"]
        Q = obj.get("Q_220", math.pi * f * tau)
        f_221 = obj.get("f_221")
        tau_221 = obj.get("tau_221")
        return {
            "case_id": case_id,
            "f_220": f,
            "tau_220": tau,
            "Q_220": Q,
            "f_221": f_221,
            "tau_221": tau_221,
            "truth": None,
        }

    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ringdown featuremap v0: map (f, tau, Q) -> holographic ratio space"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument("--ringdown-params", required=True,
                     help="JSON/JSONL with ringdown parameters")
    ap.add_argument("--alpha", type=float, default=1.0,
                     help="Calibration parameter (default: 1.0)")
    ap.add_argument("--k-ratios", type=int, default=1, dest="k_ratios",
                     help="Number of ratios to predict (default: 1)")
    ap.add_argument("--out-root", default="runs",
                     help="Output root (default: runs)")
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)
    require_run_valid(out_root, args.run)

    params_path = Path(args.ringdown_params)
    if not params_path.is_absolute():
        params_path = Path.cwd() / params_path
    if not params_path.exists():
        print(f"ERROR: ringdown params not found: {params_path}", file=sys.stderr)
        return EXIT_CONTRACT_FAIL

    cases = load_ringdown_params(params_path)
    if not cases:
        print("ERROR: no valid ringdown cases found", file=sys.stderr)
        return EXIT_CONTRACT_FAIL

    # Map each case to ratio space
    mapped = []
    for case in cases:
        result = inverse_phi(
            case["f_220"],
            case["tau_220"],
            args.alpha,
            args.k_ratios,
            f_221_hz=case.get("f_221"),
            tau_221_s=case.get("tau_221"),
        )
        mapped.append({
            "case_id": case["case_id"],
            "input": {
                "f_220": case["f_220"],
                "tau_220": case["tau_220"],
                "Q_220": case["Q_220"],
                "f_221": case.get("f_221"),
                "tau_221": case.get("tau_221"),
            },
            "mapped": result,
            "truth": case.get("truth"),
        })

    # Write outputs
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)

    output_path = outputs_dir / "mapped_features.json"
    payload = {
        "schema_version": "ringdown_featuremap_v0",
        "created": utc_now_iso(),
        "config": {
            "alpha": args.alpha,
            "k_ratios": args.k_ratios,
        },
        "n_cases": len(mapped),
        "feature_key": "ratios",
        "feature_dim": args.k_ratios,
        "cases": mapped,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write_manifest(stage_dir, {"mapped_features": output_path})
    write_stage_summary(stage_dir, {
        "stage": STAGE_NAME,
        "run": args.run,
        "config": {
            "alpha": args.alpha,
            "k_ratios": args.k_ratios,
            "ringdown_params": str(params_path),
        },
        "inputs": {
            "ringdown_params": {
                "path": str(params_path),
                "sha256": sha256_file(params_path),
            },
        },
        "n_cases": len(mapped),
        "verdict": "PASS",
    })

    print(f"[ringdown_featuremap_v0] Mapped {len(mapped)} cases to ratio space")
    print(f"  alpha={args.alpha}, k_ratios={args.k_ratios}")
    print(f"  output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
