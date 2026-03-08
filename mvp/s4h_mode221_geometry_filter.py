"""s4h_mode221_geometry_filter — Canonical stage: filter atlas geometries compatible
with the observed mode-221 ringdown via chi-squared test in (f, tau) space.

Input (per-run observation file):
    <run_dir>/s4h_mode221_geometry_filter/inputs/mode221_obs.json
    {
        "obs_f_hz":   <float>,
        "obs_tau_s":  <float>,
        "sigma_f_hz": <float>,
        "sigma_tau_s":<float>
    }

Atlas (passed via --atlas-path):
    Either a list or {"entries": [...]} of geometry dicts.
    Entries that provide mode-221 predictions must have one of:
      - "mode_221": {"f_hz": ..., "tau_s": ...}           (unified golden atlas)
      - metadata.mode == "(2,2,1)" with top-level f_hz + tau_s  (existing atlas format)

Output:
    <run_dir>/s4h_mode221_geometry_filter/outputs/mode221_filter.json
    <run_dir>/s4h_mode221_geometry_filter/stage_summary.json
    <run_dir>/s4h_mode221_geometry_filter/manifest.json
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

from basurin_io import (
    resolve_out_root,
    require_run_valid,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)
from mvp.golden_geometry_spec import (
    DEFAULT_MODE_CHI2_THRESHOLD_90,
    MODE_221,
    VERDICT_PASS,
    VERDICT_NO_COMMON_GEOMETRIES,
    VERDICT_SKIPPED_221_UNAVAILABLE,
    chi2_mode,
    passes_mode_threshold,
    _utc_now_iso,
)
from mvp.s4g_mode220_geometry_filter import load_atlas_entries

STAGE = "s4h_mode221_geometry_filter"
OBS_FILE_REL = "s4h_mode221_geometry_filter/inputs/mode221_obs.json"
OUTPUT_FILE = "mode221_filter.json"

# ---------------------------------------------------------------------------
# Pure helpers (reusable by experiment)
# ---------------------------------------------------------------------------


def extract_mode221_predictions(entry: dict[str, Any]) -> "tuple[float, float] | None":
    """Return (pred_f_hz, pred_tau_s) for mode 221 from an atlas entry, or None.

    Supports two atlas formats:
    1. Unified golden atlas: entry has a "mode_221" sub-dict with "f_hz" and "tau_s".
    2. Existing atlas format: entry has metadata.mode == "(2,2,1)" with top-level
       "f_hz" and "tau_s".
    """
    # Unified format
    m221 = entry.get("mode_221")
    if isinstance(m221, dict):
        f = m221.get("f_hz")
        tau = m221.get("tau_s")
        if f is not None and tau is not None:
            return float(f), float(tau)

    # Existing atlas format (metadata.mode)
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
    """Return sorted list of geometry_ids whose mode-221 prediction passes chi2 < threshold.

    This is a pure function; callers may scale sigmas before passing them in.
    """
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


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: mode-221 geometry filter")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument(
        "--threshold-221",
        type=float,
        default=DEFAULT_MODE_CHI2_THRESHOLD_90,
        help=f"chi² threshold for mode-221 compatibility (default: {DEFAULT_MODE_CHI2_THRESHOLD_90})",
    )
    args = ap.parse_args(argv)

    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    run_dir = out_root / args.run_id
    stage_dir = run_dir / STAGE
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    obs_path = run_dir / OBS_FILE_REL
    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    if not atlas_path.exists():
        print(f"ERROR: atlas not found: {atlas_path}", file=sys.stderr)
        return 2

    # Mode 221 is optional: if the obs file is absent, record SKIPPED verdict.
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
        out_path = outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        summary = {
            "stage": STAGE,
            "run_id": args.run_id,
            "mode": MODE_221,
            "verdict": VERDICT_SKIPPED_221_UNAVAILABLE,
            "reason": f"mode221 obs file not found: {obs_path}",
        }
        stage_summary = write_stage_summary(stage_dir, summary)
        manifest = write_manifest(
            stage_dir,
            {"mode221_filter": out_path, "stage_summary": stage_summary},
        )
        print(f"OUT_ROOT={out_root}")
        print(f"STAGE_DIR={stage_dir}")
        print(f"OUTPUTS_DIR={outputs_dir}")
        print(f"STAGE_SUMMARY={stage_summary}")
        print(f"MANIFEST={manifest}")
        print(f"[{STAGE}] verdict={VERDICT_SKIPPED_221_UNAVAILABLE}")
        return 0

    try:
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

        out_path = outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)

        summary = {
            "stage": STAGE,
            "run_id": args.run_id,
            "mode": MODE_221,
            "chi2_threshold": args.threshold_221,
            "n_atlas_mode221": sum(1 for e in atlas_entries if extract_mode221_predictions(e) is not None),
            "n_passed": len(geometry_ids),
            "verdict": verdict,
        }
        stage_summary = write_stage_summary(stage_dir, summary)
        manifest = write_manifest(
            stage_dir,
            {"mode221_filter": out_path, "stage_summary": stage_summary},
        )

        print(f"OUT_ROOT={out_root}")
        print(f"STAGE_DIR={stage_dir}")
        print(f"OUTPUTS_DIR={outputs_dir}")
        print(f"STAGE_SUMMARY={stage_summary}")
        print(f"MANIFEST={manifest}")
        print(f"[{STAGE}] n_passed={len(geometry_ids)} verdict={verdict}")
        return 0

    except KeyError as exc:
        print(f"ERROR: missing key in observations file: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
