"""s4g_mode220_geometry_filter — Canonical stage: filter atlas geometries compatible
with the observed mode-220 ringdown via chi-squared test in (f, tau) space.

Input (per-run observation file):
    <run_dir>/s4g_mode220_geometry_filter/inputs/mode220_obs.json
    {
        "obs_f_hz":   <float>,   # observed ringdown frequency (Hz)
        "obs_tau_s":  <float>,   # observed damping time (s)
        "sigma_f_hz": <float>,   # 1-sigma uncertainty on frequency (Hz)
        "sigma_tau_s":<float>    # 1-sigma uncertainty on damping time (s)
    }

Atlas (passed via --atlas-path):
    Either a list or {"entries": [...]} of geometry dicts.
    Entries that provide mode-220 predictions must have one of:
      - "mode_220": {"f_hz": ..., "tau_s": ...}           (unified golden atlas)
      - metadata.mode == "(2,2,0)" with top-level f_hz + tau_s  (existing atlas format)

Output:
    <run_dir>/s4g_mode220_geometry_filter/outputs/mode220_filter.json
    <run_dir>/s4g_mode220_geometry_filter/stage_summary.json
    <run_dir>/s4g_mode220_geometry_filter/manifest.json
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
    write_json_atomic,
)
from mvp.contracts import (
    init_stage,
    check_inputs,
    finalize,
    abort,
    log_stage_paths,
)
from mvp.golden_geometry_spec import (
    DEFAULT_MODE_CHI2_THRESHOLD_90,
    MODE_220,
    VERDICT_PASS,
    VERDICT_NO_COMMON_GEOMETRIES,
    chi2_mode,
    passes_mode_threshold,
    _utc_now_iso,
)

STAGE = "s4g_mode220_geometry_filter"
OBS_FILE_REL = "s4g_mode220_geometry_filter/inputs/mode220_obs.json"
OUTPUT_FILE = "geometries_220.json"

# ---------------------------------------------------------------------------
# Pure helpers (reusable by experiment)
# ---------------------------------------------------------------------------


def load_atlas_entries(atlas_path: Path) -> list[dict[str, Any]]:
    """Load atlas from a file; return a flat list of entry dicts."""
    with open(atlas_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("entries", "atlas"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError(f"Atlas must be a JSON list or dict with 'entries' key; got {type(raw).__name__}")


def extract_mode220_predictions(entry: dict[str, Any]) -> "tuple[float, float] | None":
    """Return (pred_f_hz, pred_tau_s) for mode 220 from an atlas entry, or None.

    Supports two atlas formats:
    1. Unified golden atlas: entry has a "mode_220" sub-dict with "f_hz" and "tau_s".
    2. Existing atlas format: entry has metadata.mode == "(2,2,0)" with top-level
       "f_hz" and "tau_s".
    """
    # Unified format
    m220 = entry.get("mode_220")
    if isinstance(m220, dict):
        f = m220.get("f_hz")
        tau = m220.get("tau_s")
        if f is not None and tau is not None:
            return float(f), float(tau)

    # Existing atlas format (metadata.mode)
    meta = entry.get("metadata") or {}
    mode_str = str(meta.get("mode", "")).replace(" ", "")
    if mode_str in {"(2,2,0)", "220"}:
        f = entry.get("f_hz")
        tau = entry.get("tau_s")
        if f is not None and tau is not None:
            return float(f), float(tau)

    return None


def filter_mode220(
    *,
    obs_f_hz: float,
    obs_tau_s: float,
    sigma_f_hz: float,
    sigma_tau_s: float,
    atlas_entries: list[dict[str, Any]],
    chi2_threshold: float,
) -> "tuple[list[str], list[dict[str, Any]]]":
    """Return (accepted_geometry_ids, accepted_geometries) for mode-220 chi2 < threshold.

    Parameters
    ----------
    obs_f_hz, obs_tau_s   : observed frequency and damping time.
    sigma_f_hz, sigma_tau_s : 1-sigma uncertainties (must be > 0).
    atlas_entries          : list of atlas entry dicts.
    chi2_threshold         : chi² cut-off (strict: chi2 < threshold).

    Returns
    -------
    accepted_geometry_ids : sorted list of geometry_id strings that passed.
    accepted_geometries   : list of full atlas entry dicts, ordered by geometry_id.

    This is a pure function; callers may scale sigmas before passing them in.
    """
    passed: list[tuple[str, dict[str, Any]]] = []
    for entry in atlas_entries:
        gid = entry.get("geometry_id")
        if not isinstance(gid, str):
            continue
        pred = extract_mode220_predictions(entry)
        if pred is None:
            continue
        pred_f, pred_tau = pred
        chi2 = chi2_mode(obs_f_hz, obs_tau_s, pred_f, pred_tau, sigma_f_hz, sigma_tau_s)
        if passes_mode_threshold(chi2, chi2_threshold):
            passed.append((gid, entry))
    passed.sort(key=lambda t: t[0])
    accepted_ids = [t[0] for t in passed]
    accepted_entries = [t[1] for t in passed]
    return accepted_ids, accepted_entries


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: mode-220 geometry filter")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument(
        "--threshold-220",
        type=float,
        default=DEFAULT_MODE_CHI2_THRESHOLD_90,
        help=f"chi² threshold for mode-220 compatibility (default: {DEFAULT_MODE_CHI2_THRESHOLD_90})",
    )
    args = ap.parse_args(argv)

    ctx = init_stage(args.run_id, STAGE, params={
        "threshold_220": args.threshold_220,
    })

    obs_path = ctx.run_dir / OBS_FILE_REL
    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    if not obs_path.exists():
        abort(ctx, f"mode-220 observations file not found: {obs_path}")

    if not atlas_path.exists():
        abort(ctx, f"atlas not found: {atlas_path}")

    inputs = check_inputs(ctx, {
        "mode220_obs": obs_path,
        "atlas": atlas_path,
    })

    try:
        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        obs_f_hz = float(obs["obs_f_hz"])
        obs_tau_s = float(obs["obs_tau_s"])
        sigma_f_hz = float(obs["sigma_f_hz"])
        sigma_tau_s = float(obs["sigma_tau_s"])

        atlas_entries = load_atlas_entries(atlas_path)
        n_geometries_scanned = sum(
            1 for e in atlas_entries if extract_mode220_predictions(e) is not None
        )
        accepted_geometry_ids, accepted_geometries = filter_mode220(
            obs_f_hz=obs_f_hz,
            obs_tau_s=obs_tau_s,
            sigma_f_hz=sigma_f_hz,
            sigma_tau_s=sigma_tau_s,
            atlas_entries=atlas_entries,
            chi2_threshold=args.threshold_220,
        )
        n_geometries_accepted = len(accepted_geometry_ids)

        verdict = VERDICT_PASS if accepted_geometry_ids else VERDICT_NO_COMMON_GEOMETRIES

        payload: dict[str, Any] = {
            "schema_name": "golden_geometry_mode_filter",
            "schema_version": "v1",
            "created_utc": _utc_now_iso(),
            "run_id": args.run_id,
            "stage": STAGE,
            "mode": MODE_220,
            "obs_f_hz": obs_f_hz,
            "obs_tau_s": obs_tau_s,
            "sigma_f_hz": sigma_f_hz,
            "sigma_tau_s": sigma_tau_s,
            "chi2_threshold": args.threshold_220,
            "atlas_path": str(atlas_path),
            "n_geometries_scanned": n_geometries_scanned,
            "n_geometries_accepted": n_geometries_accepted,
            "accepted_geometry_ids": accepted_geometry_ids,
            "accepted_geometries": accepted_geometries,
            "verdict": verdict,
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)

        finalize(ctx, artifacts={"geometries_220": out_path}, verdict=verdict, results={
            "mode": MODE_220,
            "chi2_threshold": args.threshold_220,
            "n_geometries_scanned": n_geometries_scanned,
            "n_geometries_accepted": n_geometries_accepted,
        })
        log_stage_paths(ctx)
        print(f"[{STAGE}] n_geometries_scanned={n_geometries_scanned} n_geometries_accepted={n_geometries_accepted} verdict={verdict}")
        return 0

    except SystemExit:
        raise
    except KeyError as exc:
        abort(ctx, f"missing key in observations file: {exc}")
    except Exception as exc:
        abort(ctx, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
