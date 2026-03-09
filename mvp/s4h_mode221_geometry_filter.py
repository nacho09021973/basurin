"""s4h_mode221_geometry_filter — Canonical stage: filter atlas geometries compatible
with the observed mode-221 ringdown via chi-squared test in (f, tau) space.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.golden_geometry_spec import (
    DEFAULT_MODE_CHI2_THRESHOLD_90,
    MODE_221,
    VERDICT_NO_COMMON_GEOMETRIES,
    VERDICT_PASS,
    VERDICT_SKIPPED_221_UNAVAILABLE,
    _utc_now_iso,
    chi2_mode,
    passes_mode_threshold,
)
from mvp.s4g_mode220_geometry_filter import load_atlas_entries

STAGE = "s4h_mode221_geometry_filter"
OBS_FILE_REL = "s4h_mode221_geometry_filter/inputs/mode221_obs.json"
OUTPUT_FILE = "mode221_filter.json"


def extract_mode221_predictions(entry: dict[str, Any]) -> "tuple[float, float] | None":
    m221 = entry.get("mode_221")
    if isinstance(m221, dict):
        f = m221.get("f_hz")
        tau = m221.get("tau_s")
        if f is not None and tau is not None:
            return float(f), float(tau)

    meta = entry.get("metadata") or {}
    mode_str = str(meta.get("mode", "")).replace(" ", "")
    if mode_str in {"(2,2,1)", "221"}:
        f = entry.get("f_hz")
        tau = entry.get("tau_s")
        if f is not None and tau is not None:
            return float(f), float(tau)

    return None


def filter_mode221(
    *,
    obs_f_hz: float,
    obs_tau_s: float,
    sigma_f_hz: float,
    sigma_tau_s: float,
    atlas_entries: list[dict[str, Any]],
    chi2_threshold: float,
) -> list[str]:
    passed: list[str] = []
    for entry in atlas_entries:
        gid = entry.get("geometry_id")
        if not isinstance(gid, str):
            continue
        pred = extract_mode221_predictions(entry)
        if pred is None:
            continue
        pred_f, pred_tau = pred
        chi2 = chi2_mode(obs_f_hz, obs_tau_s, pred_f, pred_tau, sigma_f_hz, sigma_tau_s)
        if passes_mode_threshold(chi2, chi2_threshold):
            passed.append(gid)
    return sorted(passed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: mode-221 geometry filter")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--threshold-221", type=float, default=DEFAULT_MODE_CHI2_THRESHOLD_90)
    args = ap.parse_args(argv)

    ctx = init_stage(
        args.run_id,
        STAGE,
        params={"atlas_path": args.atlas_path, "threshold_221": float(args.threshold_221)},
    )

    obs_path = ctx.run_dir / OBS_FILE_REL
    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()
    if not atlas_path.exists():
        abort(ctx, f"Atlas not found. expected: {atlas_path}. Command to regenerate upstream: provide --atlas-path <ATLAS_PATH>.")

    inputs = check_inputs(ctx, {
        "mode221_obs": obs_path,
        "atlas": atlas_path,
    })

    try:
        check_inputs(ctx, {"atlas": atlas_path}, optional={"mode221_obs": obs_path})

        if not obs_path.exists():
            payload: dict[str, Any] = {
                "schema_name": "golden_geometry_mode_filter",
                "schema_version": "v1",
                "created_utc": _utc_now_iso(),
                "run_id": args.run_id,
                "stage": STAGE,
                "mode": MODE_221,
                "obs_f_hz": None,
                "obs_tau_s": None,
                "sigma_f_hz": None,
                "sigma_tau_s": None,
                "chi2_threshold": args.threshold_221,
                "atlas_path": str(atlas_path),
                "geometry_ids": [],
                "n_passed": 0,
                "verdict": VERDICT_SKIPPED_221_UNAVAILABLE,
            }
            out_path = ctx.outputs_dir / OUTPUT_FILE
            write_json_atomic(out_path, payload)
            finalize(
                ctx,
                artifacts={"mode221_filter": out_path},
                verdict="PASS",
                results={
                    "mode": MODE_221,
                    "verdict": VERDICT_SKIPPED_221_UNAVAILABLE,
                    "reason": f"mode221 obs file not found: {obs_path}",
                },
            )
            log_stage_paths(ctx)
            print(f"[{STAGE}] verdict={VERDICT_SKIPPED_221_UNAVAILABLE}")
            return 0

        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        obs_f_hz = float(obs["obs_f_hz"])
        obs_tau_s = float(obs["obs_tau_s"])
        sigma_f_hz = float(obs["sigma_f_hz"])
        sigma_tau_s = float(obs["sigma_tau_s"])

        atlas_entries = load_atlas_entries(atlas_path)
        geometry_ids = filter_mode221(
            obs_f_hz=obs_f_hz,
            obs_tau_s=obs_tau_s,
            sigma_f_hz=sigma_f_hz,
            sigma_tau_s=sigma_tau_s,
            atlas_entries=atlas_entries,
            chi2_threshold=args.threshold_221,
        )

        verdict = VERDICT_PASS if geometry_ids else VERDICT_NO_COMMON_GEOMETRIES
        payload = {
            "schema_name": "golden_geometry_mode_filter",
            "schema_version": "v1",
            "created_utc": _utc_now_iso(),
            "run_id": args.run_id,
            "stage": STAGE,
            "mode": MODE_221,
            "obs_f_hz": obs_f_hz,
            "obs_tau_s": obs_tau_s,
            "sigma_f_hz": sigma_f_hz,
            "sigma_tau_s": sigma_tau_s,
            "chi2_threshold": args.threshold_221,
            "atlas_path": str(atlas_path),
            "geometry_ids": geometry_ids,
            "n_passed": len(geometry_ids),
            "verdict": verdict,
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={"mode221_filter": out_path},
            verdict="PASS",
            results={
                "mode": MODE_221,
                "chi2_threshold": float(args.threshold_221),
                "n_atlas_mode221": sum(1 for e in atlas_entries if extract_mode221_predictions(e) is not None),
                "n_passed": len(geometry_ids),
                "verdict": verdict,
            },
        )
        log_stage_paths(ctx)
        print(f"[{STAGE}] n_passed={len(geometry_ids)} verdict={verdict}")
        return 0
    except SystemExit:
        raise
    except KeyError as exc:
        abort(ctx, f"missing key in observations file: {exc}. expected: {obs_path}")
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
