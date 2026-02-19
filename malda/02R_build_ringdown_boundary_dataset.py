#!/usr/bin/env python3
"""
02R_build_ringdown_boundary_dataset.py — CUERDAS-MALDACENA (Stage 02R bridge, v1.0)

Purpose
-------
Convert Stage-01 ringdown artifacts (poles_*.json, coincident_pairs.json, null_test.json)
into a boundary-only HDF5 dataset + manifest.json consumable by:

  run_pipeline.py --experiment <exp> --stage 02_emergent_geometry_engine --mode inference --data-dir <OUT_DIR>

Contract / Epistemic honesty
----------------------------
- This script DOES NOT inject theoretical targets (no GR templates, no bulk truth).
- It builds deterministic *surrogate embeddings* (G_R_real/imag, G2_ringdown) purely from
  extracted poles as feature-engineering, and stores full provenance in the output HDF5.

Path rules (aligned with readme_rutas.md spirit)
------------------------------------------------
- Relative paths are interpreted root-relative (PROJECT_ROOT / path).
- Relative paths containing '..' are rejected.
- Resolved paths must not escape PROJECT_ROOT.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except Exception as e:
    raise SystemExit(f"[ERROR] h5py not available: {e}")

SCRIPT_VERSION = "02R_build_ringdown_boundary_dataset.py v1.0 (2026-01-12)"


# ----------------------------
# Path resolution (root-relative, no '..', no escape)
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


def _reject_dotdot(p: Path) -> None:
    # Reject any '..' segment in a *relative* path
    if any(part == ".." for part in p.parts):
        raise ValueError(f"Relative path contains '..' (forbidden): {p}")


def resolve_root_relative(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        resolved = p.resolve(strict=False)
    else:
        _reject_dotdot(p)
        resolved = (PROJECT_ROOT / p).resolve(strict=False)

    # Reject paths escaping project root (for auditability)
    try:
        resolved.relative_to(PROJECT_ROOT.resolve(strict=False))
    except Exception:
        raise ValueError(f"Path escapes PROJECT_ROOT ({PROJECT_ROOT}): {path_str} -> {resolved}")
    return resolved


# ----------------------------
# JSON helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------
# Ringdown parsing
# ----------------------------

@dataclass
class Pole:
    freq_hz: float
    damping_1_over_s: float
    amp_abs: float


def parse_poles_json(poles_payload: Dict[str, Any]) -> List[Pole]:
    poles_list = poles_payload.get("poles", [])
    out: List[Pole] = []
    for p in poles_list:
        try:
            f = float(p.get("freq_hz"))
            g = float(p.get("damping_1_over_s"))
            a = float(p.get("amp_abs", 1.0))
            if not np.isfinite(f) or not np.isfinite(g) or not np.isfinite(a):
                continue
            # Keep only positive-frequency poles by default (consistent with Stage 01 filters)
            if f <= 0:
                continue
            out.append(Pole(freq_hz=f, damping_1_over_s=g, amp_abs=max(a, 0.0)))
        except Exception:
            continue
    return out


def pick_best_pair(cp_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pairs = cp_payload.get("pairs", [])
    if not isinstance(pairs, list) or not pairs:
        return None
    # Best = minimal score_2d
    best = None
    best_score = float("inf")
    for p in pairs:
        try:
            s = float(p.get("score_2d"))
            if np.isfinite(s) and s < best_score:
                best_score = s
                best = p
        except Exception:
            continue
    return best


def compute_p_values_from_null(best_score: float, null_scores: List[float]) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Return:
      p_unc  = (n_better+1)/(N+1) where N includes invalid (non-finite) scores, but only finite count as "better"
      p_cond = (n_better+1)/(N_valid+1) where N_valid excludes invalid
    This mirrors Stage-01 smoothing style and provides an explicit conditional alternative.
    """
    if not np.isfinite(best_score) or not isinstance(null_scores, list) or len(null_scores) == 0:
        return None, None, 0, 0

    scores = np.array([float(s) for s in null_scores], dtype=float)
    finite = np.isfinite(scores)
    N = int(scores.size)
    N_valid = int(np.sum(finite))
    n_better = int(np.sum((scores[finite] <= best_score))) if N_valid > 0 else 0

    p_unc = float(n_better + 1) / float(N + 1)
    p_cond = float(n_better + 1) / float(N_valid + 1) if N_valid > 0 else None
    return p_unc, p_cond, N, N_valid


# ----------------------------
# Surrogate embeddings (data-driven, deterministic)
# ----------------------------

def build_omega_grid_hz(poles: List[Pole], n_omega: int, fmin_hz: Optional[float], fmax_hz: Optional[float]) -> np.ndarray:
    if fmin_hz is not None and fmax_hz is not None and fmax_hz > fmin_hz > 0:
        lo, hi = float(fmin_hz), float(fmax_hz)
    elif poles:
        freqs = np.array([p.freq_hz for p in poles], dtype=float)
        fmin = float(np.min(freqs))
        fmax = float(np.max(freqs))
        lo = max(1e-3, 0.5 * fmin)
        hi = max(lo + 1e-3, 1.5 * fmax)
    else:
        lo, hi = 1.0, 1024.0

    return np.linspace(lo, hi, int(n_omega), dtype=np.float64)


def poles_to_gr(omega_grid_hz: np.ndarray, poles: List[Pole], normalization: str = "unit_peak") -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a complex-valued surrogate response:
      GR(ω) = Σ a_n / (ω - ω_n), with ω real and ω_n = 2π f_n - i γ_n.

    Returns real/imag arrays shaped (Nw, Nk) with Nk=1 (k=0 only).
    """
    Nw = int(omega_grid_hz.size)
    if not poles or Nw <= 0:
        return np.zeros((Nw, 1), dtype=np.float64), np.zeros((Nw, 1), dtype=np.float64)

    # residues: use amp_abs, stabilized
    amps = np.array([max(p.amp_abs, 0.0) for p in poles], dtype=np.float64)
    if not np.any(amps > 0):
        amps = np.ones_like(amps)

    # Normalize residues to avoid extreme scaling
    amps = amps / (float(np.max(amps)) + 1e-12)

    omega = (2.0 * np.pi * omega_grid_hz).astype(np.float64)  # rad/s, real
    omega = omega.reshape(-1, 1)  # (Nw,1)

    w_poles = []
    for p, a in zip(poles, amps):
        w = (2.0 * np.pi * float(p.freq_hz)) - 1j * float(p.damping_1_over_s)
        w_poles.append((w, float(a)))

    GR = np.zeros((Nw, 1), dtype=np.complex128)
    for w, a in w_poles:
        GR[:, 0] += a / (omega[:, 0] - w)

    # Optional normalization
    if normalization == "unit_peak":
        mag = np.abs(GR[:, 0])
        mmax = float(np.max(mag)) if mag.size else 0.0
        if mmax > 0:
            GR[:, 0] /= mmax

    return np.real(GR).astype(np.float64), np.imag(GR).astype(np.float64)


def build_x_grid_s(n_x: int, x_min_s: float, x_max_s: float) -> np.ndarray:
    n_x = int(n_x)
    x_min_s = float(x_min_s)
    x_max_s = float(x_max_s)
    if not (x_max_s > x_min_s >= 0.0):
        raise ValueError(f"Invalid x range: x_min_s={x_min_s}, x_max_s={x_max_s}")
    # Avoid x=0 exactly for log features downstream; keep strictly positive lower bound if possible
    x0 = max(x_min_s, 1e-6)
    return np.linspace(x0, x_max_s, n_x, dtype=np.float64)


def poles_to_g2(x_grid_s: np.ndarray, poles: List[Pole], normalization: str = "unit_peak") -> np.ndarray:
    """
    Build a positive surrogate 1D observable from poles:
      s(x) = Σ a_n * exp((-γ_n + i ω_r_n) x)
      G2(x) = |s(x)|^2  (>=0)

    This is an embedding, not claimed to be a physical CFT correlator.
    """
    Nx = int(x_grid_s.size)
    if not poles or Nx <= 0:
        return np.zeros((Nx,), dtype=np.float64)

    amps = np.array([max(p.amp_abs, 0.0) for p in poles], dtype=np.float64)
    if not np.any(amps > 0):
        amps = np.ones_like(amps)
    amps = amps / (float(np.max(amps)) + 1e-12)

    x = x_grid_s.astype(np.float64).reshape(-1, 1)  # (Nx,1)
    s = np.zeros((Nx, 1), dtype=np.complex128)

    for p, a in zip(poles, amps):
        w_r = 2.0 * np.pi * float(p.freq_hz)        # rad/s
        g = float(p.damping_1_over_s)               # 1/s
        s[:, 0] += float(a) * np.exp((-g + 1j * w_r) * x[:, 0])

    G2 = (np.abs(s[:, 0]) ** 2).astype(np.float64)

    if normalization == "unit_peak":
        mmax = float(np.max(G2)) if G2.size else 0.0
        if mmax > 0:
            G2 = G2 / mmax

    return G2


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 02R: build boundary dataset from ringdown_* artifacts (poles -> surrogate boundary embeddings)."
    )
    ap.add_argument("--run-dir", required=True, type=str, help="Run directory containing ringdown_* and data_boundary/")
    ap.add_argument("--ringdown-dirs", required=True, nargs="+", help="One or more ringdown_* subdirectories inside --run-dir")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory to create (will contain manifest.json and *.h5)")

    ap.add_argument("--d", type=int, default=4, help="Boundary dimension d to store (default: 4)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature metadata to store (default: 0.0)")

    ap.add_argument("--n-omega", type=int, default=256, help="Number of omega grid points (Hz)")
    ap.add_argument("--fmin-hz", type=float, default=None, help="Override omega grid min frequency (Hz)")
    ap.add_argument("--fmax-hz", type=float, default=None, help="Override omega grid max frequency (Hz)")
    ap.add_argument("--gr-normalization", type=str, default="unit_peak", choices=["unit_peak", "none"],
                    help="Normalize GR by max |GR| (unit_peak) or leave raw (none)")

    ap.add_argument("--n-x", type=int, default=256, help="Number of x grid points (seconds)")
    ap.add_argument("--x-min-s", type=float, default=1e-4, help="Minimum x (seconds), strictly positive recommended")
    ap.add_argument("--x-max-s", type=float, default=0.2, help="Maximum x (seconds)")
    ap.add_argument("--g2-normalization", type=str, default="unit_peak", choices=["unit_peak", "none"],
                    help="Normalize G2 by max (unit_peak) or leave raw (none)")

    args = ap.parse_args()

    run_dir = resolve_root_relative(args.run_dir)
    out_dir = resolve_root_relative(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to infer event_id from data_boundary/<EVENT>_boundary.h5
    event_id = None
    db_dir = run_dir / "data_boundary"
    if db_dir.exists() and db_dir.is_dir():
        candidates = sorted(db_dir.glob("*_boundary.h5"))
        if candidates:
            event_id = candidates[0].name.replace("_boundary.h5", "")
    if event_id is None:
        event_id = run_dir.name  # fallback

    print("=" * 70)
    print("PHASE 2 — Stage 02R (Ringdown → Boundary Dataset)")
    print(f"Script:    {SCRIPT_VERSION}")
    print(f"Run dir:   {run_dir}")
    print(f"Out dir:   {out_dir}")
    print(f"Event id:  {event_id}")
    print("=" * 70)

    manifest: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "script": SCRIPT_VERSION,
        "version": "02R-v1",
        "source_run_dir": str(run_dir),
        "event_id": event_id,
        "config": {
            "d": int(args.d),
            "temperature": float(args.temperature),
            "n_omega": int(args.n_omega),
            "fmin_hz": None if args.fmin_hz is None else float(args.fmin_hz),
            "fmax_hz": None if args.fmax_hz is None else float(args.fmax_hz),
            "gr_normalization": args.gr_normalization,
            "n_x": int(args.n_x),
            "x_min_s": float(args.x_min_s),
            "x_max_s": float(args.x_max_s),
            "g2_normalization": args.g2_normalization,
            "k_grid": [0.0],
        },
        "geometries": [],
    }

    omega_grid_hz = None  # build per-system if auto bounds depend on poles

    for rd in args.ringdown_dirs:
        rd_rel = Path(rd)
        _reject_dotdot(rd_rel)
        rd_dir = (run_dir / rd_rel).resolve(strict=False)
        try:
            rd_dir.relative_to(run_dir)
        except Exception:
            raise ValueError(f"ringdown-dir escapes run_dir: {rd} -> {rd_dir}")

        if not rd_dir.exists() or not rd_dir.is_dir():
            raise FileNotFoundError(f"Ringdown dir not found: {rd_dir}")

        # Input files
        poles_path = rd_dir / "poles_joint.json"
        if not poles_path.exists():
            # fallback: first poles_*.json if present
            alt = sorted(rd_dir.glob("poles_*.json"))
            poles_path = alt[0] if alt else poles_path

        cp_path = rd_dir / "coincident_pairs.json"
        null_path = rd_dir / "null_test.json"

        poles_payload = read_json(poles_path) if poles_path.exists() else {}
        cp_payload = read_json(cp_path) if cp_path.exists() else {}
        null_payload = read_json(null_path) if null_path.exists() else {}

        poles = parse_poles_json(poles_payload)

        # best coincident pair / score
        best_pair = pick_best_pair(cp_payload)
        best_score = None
        best_p_value_stage01 = None
        if best_pair is not None:
            try:
                best_score = float(best_pair.get("score_2d"))
            except Exception:
                best_score = None
            try:
                pv = best_pair.get("p_value", None)
                best_p_value_stage01 = None if pv is None else float(pv)
            except Exception:
                best_p_value_stage01 = None

        # null test stats
        null_scores = _safe_get(null_payload, ["null_test", "scores_per_trial"], default=None)
        null_stats = _safe_get(null_payload, ["null_test", "statistics"], default={}) or {}
        n_invalid = None
        n_trials = None
        if isinstance(null_stats, dict):
            try:
                n_invalid = int(null_stats.get("n_invalid_trials")) if "n_invalid_trials" in null_stats else None
            except Exception:
                n_invalid = None
        if isinstance(null_scores, list):
            n_trials = int(len(null_scores))

        # compute p-values (explicit unconditional + conditional)
        p_unc = p_cond = None
        N = N_valid = 0
        if best_score is not None and isinstance(null_scores, list):
            p_unc, p_cond, N, N_valid = compute_p_values_from_null(best_score, null_scores)

        # Build grids and surrogate boundary embeddings
        omega_grid_hz = build_omega_grid_hz(poles, args.n_omega, args.fmin_hz, args.fmax_hz)
        k_grid = np.array([0.0], dtype=np.float64)

        GR_real, GR_imag = poles_to_gr(
            omega_grid_hz,
            poles,
            normalization=args.gr_normalization,
        )

        x_grid = build_x_grid_s(args.n_x, args.x_min_s, args.x_max_s)
        G2_ringdown = poles_to_g2(
            x_grid,
            poles,
            normalization=args.g2_normalization,
        )

        # Output HDF5
        system_name = f"{event_id}__{rd_rel.name}"
        out_h5 = out_dir / f"{system_name}.h5"

        with h5py.File(out_h5, "w") as f:
            # File-level attrs
            f.attrs["created_at"] = utc_now_iso()
            f.attrs["script"] = SCRIPT_VERSION
            f.attrs["name"] = system_name
            f.attrs["system_name"] = system_name
            f.attrs["category"] = "ringdown"
            f.attrs["family"] = "unknown"
            f.attrs["operators"] = "[]"  # boundary-only: no operator spectrum provided

            # boundary group (what Stage 02 consumes)
            b = f.create_group("boundary")
            b.create_dataset("omega_grid", data=omega_grid_hz)
            b.create_dataset("k_grid", data=k_grid)
            b.create_dataset("G_R_real", data=GR_real)
            b.create_dataset("G_R_imag", data=GR_imag)
            b.create_dataset("x_grid", data=x_grid)
            b.create_dataset("G2_ringdown", data=G2_ringdown)

            b.attrs["d"] = int(args.d)
            b.attrs["temperature"] = float(args.temperature)
            b.attrs["T"] = float(args.temperature)

            # provenance / quality metadata (attrs for quick audit)
            b.attrs["source_run_dir"] = str(run_dir)
            b.attrs["source_ringdown_dir"] = str(rd_dir)
            b.attrs["source_poles_file"] = str(poles_path)
            b.attrs["source_coincident_pairs_file"] = str(cp_path) if cp_path.exists() else ""
            b.attrs["source_null_test_file"] = str(null_path) if null_path.exists() else ""

            if best_score is not None and np.isfinite(best_score):
                b.attrs["best_score_2d"] = float(best_score)
            if best_p_value_stage01 is not None and np.isfinite(best_p_value_stage01):
                b.attrs["best_p_value_stage01"] = float(best_p_value_stage01)

            if p_unc is not None and np.isfinite(p_unc):
                b.attrs["p_value_unconditional_including_invalid"] = float(p_unc)
            if p_cond is not None and np.isfinite(p_cond):
                b.attrs["p_value_conditional_valid_only"] = float(p_cond)

            if n_trials is not None:
                b.attrs["null_n_trials"] = int(n_trials)
            if n_invalid is not None:
                b.attrs["null_n_invalid_trials"] = int(n_invalid)
            if N:
                b.attrs["null_N_scores_total"] = int(N)
            if N_valid:
                b.attrs["null_N_scores_valid"] = int(N_valid)

            # raw ringdown JSON snapshots (for full traceability)
            raw = f.create_group("ringdown_raw")
            raw.create_dataset("poles_json", data=np.string_(json.dumps(poles_payload)))
            raw.create_dataset("coincident_pairs_json", data=np.string_(json.dumps(cp_payload)))
            raw.create_dataset("null_test_json", data=np.string_(json.dumps(null_payload)))

        # manifest entry
        manifest["geometries"].append(
            {
                "name": system_name,
                "family": "unknown",
                "category": "ringdown",
                "d": int(args.d),
                "file": str(out_h5.name),
                "source_ringdown_dir": str(rd_rel.as_posix()),
                "poles_file": poles_path.name if poles_path.exists() else "",
                "has_null_test": bool(null_path.exists()),
                "has_coincident_pairs": bool(cp_path.exists()),
            }
        )

        print(f"[OK] wrote: {out_h5}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {manifest_path}")

    print("=" * 70)
    print("[OK] Stage 02R completed")
    print("Next step: run_pipeline.py --experiment <exp> --stage 02_emergent_geometry_engine --mode inference --data-dir <OUT_DIR> --checkpoint <MODEL>")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
