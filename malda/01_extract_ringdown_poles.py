#!/usr/bin/env python3
"""
01_extract_ringdown_poles.py — CUERDAS-MALDACENA (LIGO boundary adapter, stage 01)

Purpose
-------
Operational extraction of a small set of damped complex exponentials ("poles")
from boundary strain data stored by 00_load_ligo_data.py in:

  <RUN_DIR>/data_boundary/<EVENT>_boundary.h5

This script is intentionally self-contained:
- No SciPy, no GWpy required.
- No injection of Kerr / GR ringdown theory.
- Purely signal-processing + deterministic IO.

Method (operational)
--------------------
ESPRIT / matrix-pencil style subspace method:

Given a 1D real signal y[k] sampled at dt, we model (in a selected window):

  y[k] ≈ Re( Σ_{m=1..r} a_m z_m^k )

where z_m are discrete-time poles. We estimate z_m from Hankel matrices
built from y, then map to continuous-time exponents:

  q_m = log(z_m) / dt   (q = sigma + i*omega)

For QNM-style frequency notation compatible with exp(-i ω t), we also provide:

  ω_m = i * q_m

so that Im(ω_m) < 0 indicates a decaying mode (operational stability).

Outputs (inside RUN_DIR)
------------------------
ringdown/
  ringdown_spec.json
  poles_H1.json
  poles_L1.json              (if present)
  poles_joint.json           (if both present; simple merge)
  poles_H1.csv
  poles_L1.csv
  poles_joint.csv
  summary.json

Also updates <RUN_DIR>/run_manifest.json (adds ringdown artifact keys).

Routing rules (anti-path-hell)
------------------------------
- Script runnable from any CWD.
- All relative input paths are interpreted relative to PROJECT_ROOT
  (= directory containing this script), and relative paths must not contain '..'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SCRIPT_VERSION = "01_extract_ringdown_poles.py/v1.1"


# ----------------------------
# Small utilities (determinism)
# ----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _safe_relpath_no_up(path_str: str) -> None:
    parts = Path(path_str).parts
    if ".." in parts:
        raise ValueError(f"Relative path must not contain '..': {path_str}")


def _resolve_root_relative(project_root: Path, p: str) -> Path:
    if os.path.isabs(p):
        return Path(p)
    _safe_relpath_no_up(p)
    return (project_root / p).resolve()


def _mkdirp(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_symlink(target_abs: Path, link_path: Path) -> None:
    if not target_abs.is_absolute():
        raise ValueError("target_abs must be absolute")
    if link_path.exists() and link_path.is_dir() and not link_path.is_symlink():
        raise IsADirectoryError(
            f"Refusing to replace directory with symlink: {link_path}. "
            "Pass a symlink path (e.g. outputs/latest)."
        )
    _mkdirp(link_path.parent)
    tmp = link_path.with_name(link_path.name + ".tmp")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        tmp.symlink_to(str(target_abs))
        os.replace(tmp, link_path)
    finally:
        if tmp.exists() or tmp.is_symlink():
            try:
                tmp.unlink()
            except OSError:
                pass


# ----------------------------
# Complex JSON helpers
# ----------------------------

def c2(z: complex) -> List[float]:
    return [float(np.real(z)), float(np.imag(z))]


def c_from2(v: Any) -> complex:
    v = list(v)
    return complex(float(v[0]), float(v[1]))


# ----------------------------
# Signal helpers
# ----------------------------

def detrend(y: np.ndarray, mode: str) -> np.ndarray:
    y = y.astype(np.float64)
    if mode == "none":
        return y
    if mode == "mean":
        return y - float(np.mean(y))
    # linear
    x = np.arange(y.size, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - (m * x + b)


def apply_rfft_bandpass(y: np.ndarray, fs: float, hp: Optional[float], lp: Optional[float]) -> np.ndarray:
    """Simple FFT-domain bandpass without SciPy. hp/lp in Hz."""
    if hp is None and lp is None:
        return y
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(y.size, d=1.0 / fs)
    mask = np.ones_like(f, dtype=bool)
    if hp is not None:
        mask &= (f >= float(hp))
    if lp is not None:
        mask &= (f <= float(lp))
    Yf = np.where(mask, Y, 0.0 + 0.0j)
    return np.fft.irfft(Yf, n=y.size).astype(np.float64)


# ----------------------------
# ESPRIT / matrix-pencil core
# ----------------------------

@dataclass
class PoleFit:
    z: np.ndarray          # (r,) complex
    q: np.ndarray          # (r,) complex  (continuous exponent)
    omega_qnm: np.ndarray  # (r,) complex  (omega = i*q)
    a: np.ndarray          # (r,) complex amplitudes (least squares)
    residual_rms: float
    relative_rms: float
    singular_values: np.ndarray  # of Y0
    rank: int
    L: int


def esprit_poles(y: np.ndarray, dt: float, L: int, rank: int, sv_thresh: float) -> PoleFit:
    """
    Estimate discrete-time poles z_m using a subspace method:
      - Build Hankel Y0, Y1
      - SVD(Y0) and choose rank r
      - A = U_r^H Y1 V_r S_r^{-1}
      - eig(A) -> z

    Parameters
    ----------
    y : 1D float array
    dt : sampling interval (seconds)
    L : Hankel rows (must satisfy 2 <= L <= N-2)
    rank : if >0, fixed rank; if 0, choose by singular values threshold
    sv_thresh : relative threshold (e.g. 1e-3) for auto rank selection
    """
    y = y.astype(np.float64)
    N = y.size
    if N < 16:
        raise ValueError(f"Need at least 16 samples for ESPRIT; got N={N}")
    if L < 2 or L > N - 2:
        raise ValueError(f"Invalid L={L} for N={N}; need 2 <= L <= N-2")

    K = N - L
    # Build Hankel matrices without heavy allocations (still OK at these sizes)
    Y0 = np.empty((L, K), dtype=np.float64)
    Y1 = np.empty((L, K), dtype=np.float64)
    for i in range(K):
        Y0[:, i] = y[i : i + L]
        Y1[:, i] = y[i + 1 : i + L + 1]

    # SVD
    U, s, Vh = np.linalg.svd(Y0, full_matrices=False)
    if rank and rank > 0:
        r = int(rank)
    else:
        # auto rank based on relative singular values
        if s.size == 0:
            raise ValueError("SVD returned no singular values")
        cutoff = float(sv_thresh) * float(s[0])
        r = int(np.sum(s >= cutoff))
        r = max(1, r)

    r = min(r, min(U.shape[1], Vh.shape[0]))
    Ur = U[:, :r]
    Sr = s[:r]
    Vr = Vh.conj().T[:, :r]

    # A = Ur^T Y1 Vr diag(1/Sr)
    # Note: Ur is real here but keep conj.T for clarity
    A = (Ur.conj().T @ Y1 @ Vr) @ np.diag(1.0 / Sr)

    # eigenvalues
    z = np.linalg.eigvals(A).astype(np.complex128)

    # continuous exponent
    q = np.log(z) / float(dt)  # principal branch
    omega_qnm = 1j * q

    # amplitudes via least squares on complex exponentials
    k = np.arange(N, dtype=np.float64)
    V = np.power(z[None, :], k[:, None])  # (N, r)
    # Solve in complex least squares; target is real y, but allow complex fit then take Re later
    a, *_ = np.linalg.lstsq(V, y.astype(np.complex128), rcond=None)

    y_hat = (V @ a)
    resid = y.astype(np.complex128) - y_hat
    residual_rms = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
    denom = float(np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-30)
    relative_rms = float(residual_rms / denom)

    return PoleFit(
        z=z,
        q=q,
        omega_qnm=omega_qnm,
        a=a.astype(np.complex128),
        residual_rms=residual_rms,
        relative_rms=relative_rms,
        singular_values=s.astype(np.float64),
        rank=r,
        L=int(L),
    )


def _sort_and_filter(pf: PoleFit, require_decay: bool, max_modes: int) -> PoleFit:
    z, q, w, a = pf.z, pf.q, pf.omega_qnm, pf.a

    finite = np.isfinite(np.real(w)) & np.isfinite(np.imag(w)) & np.isfinite(np.real(z)) & np.isfinite(np.imag(z))
    z, q, w, a = z[finite], q[finite], w[finite], a[finite]

    if require_decay:
        # Decay for exp(-i ω t) corresponds to Im(ω) < 0
        keep = np.imag(w) < 0.0
        z, q, w, a = z[keep], q[keep], w[keep], a[keep]

    if z.size == 0:
        return PoleFit(
            z=z, q=q, omega_qnm=w, a=a,
            residual_rms=pf.residual_rms, relative_rms=pf.relative_rms,
            singular_values=pf.singular_values, rank=0, L=pf.L
        )

    amp = np.abs(a)
    # Sorting: least damped first (Im(ω) closest to 0 from below), then larger amplitude
    order = np.lexsort((-amp, -np.imag(w)))  # lexsort uses last key as primary; we want (-Im) primary
    z, q, w, a = z[order], q[order], w[order], a[order]

    if max_modes > 0 and z.size > max_modes:
        z, q, w, a = z[:max_modes], q[:max_modes], w[:max_modes], a[:max_modes]

    return PoleFit(
        z=z, q=q, omega_qnm=w, a=a,
        residual_rms=pf.residual_rms, relative_rms=pf.relative_rms,
        singular_values=pf.singular_values, rank=int(z.size), L=pf.L
    )


# ----------------------------
# IO (HDF5 boundary)
# ----------------------------

def load_boundary_h5(path: Path) -> Dict[str, Any]:
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("h5py is required to read boundary .h5. Install with: pip install h5py") from e

    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        # normalize numpy scalars
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in attrs.items()}

        t_rel = f["time/t_rel"][:].astype(np.float64)
        h1 = f["strain/H1"][:].astype(np.float64)
        l1 = f["strain/L1"][:] .astype(np.float64) if "strain/L1" in f else None
        fs = float(attrs.get("fs_hz", float(f["time"].attrs.get("fs_hz", 0.0))))
        dt = float(attrs.get("dt", float(f["time"].attrs.get("dt", 1.0 / fs if fs else 0.0))))
        event = str(attrs.get("event", "UNKNOWN_EVENT"))
        gps = float(attrs.get("gps", 0.0))

        return {
            "attrs": attrs,
            "event": event,
            "gps": gps,
            "fs": fs,
            "dt": dt,
            "t_rel": t_rel,
            "H1": h1,
            "L1": l1,
        }


def _find_boundary_from_manifest(run_dir: Path) -> Path:
    """
    Resolve boundary HDF5 path from run_manifest.json, with a conservative fallback:
    if manifest keys are absent, look for exactly one file matching data_boundary/*_boundary.h5.
    """
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
        rel = artifacts.get("ligo_boundary_h5") or artifacts.get("boundary_h5_rel")
        if rel:
            return (run_dir / rel).resolve()

    # Fallback: search deterministically
    candidates = sorted((run_dir / "data_boundary").glob("*_boundary.h5"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) == 0:
        raise FileNotFoundError(
            "Could not locate boundary HDF5. Expected either run_manifest.json with artifacts.ligo_boundary_h5 "
            "or a file matching data_boundary/*_boundary.h5 inside run_dir."
        )
    raise RuntimeError(
        "Multiple boundary HDF5 candidates found; please pass --boundary-h5 explicitly:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )



# ----------------------------
# Window selection (operational)
# ----------------------------

def pick_window(t_rel: np.ndarray,
                y_ref: np.ndarray,
                t0_rel: Optional[float],
                duration: float,
                peak_search_start: float,
                peak_search_end: float,
                start_offset: float) -> Tuple[int, int, float]:
    """
    Returns (i0, i1, t0_used).
    If t0_rel is None, choose t_peak within [peak_search_start, peak_search_end]
    (by max |y_ref|) then set t0 = t_peak + start_offset.
    """
    if duration <= 0:
        raise ValueError("duration must be > 0")

    if t0_rel is None:
        mask = (t_rel >= float(peak_search_start)) & (t_rel <= float(peak_search_end))
        if not np.any(mask):
            raise ValueError("peak search window does not overlap available t_rel")
        idxs = np.where(mask)[0]
        local = idxs[np.argmax(np.abs(y_ref[idxs]))]
        t_peak = float(t_rel[local])
        t0_used = float(t_peak + start_offset)
    else:
        t0_used = float(t0_rel)

    t1_used = t0_used + float(duration)

    # indices (inclusive/exclusive)
    i0 = int(np.searchsorted(t_rel, t0_used, side="left"))
    i1 = int(np.searchsorted(t_rel, t1_used, side="right"))
    i0 = max(0, min(i0, t_rel.size))
    i1 = max(0, min(i1, t_rel.size))

    if i1 - i0 < 16:
        raise ValueError(f"Selected window too short: Nw={i1-i0}. Adjust --duration or --t0-rel.")
    return i0, i1, t0_used


# ----------------------------
# Output writers
# ----------------------------

def write_poles_json(out_path: Path, event: str, ifo: str, t0: float, duration: float, fs: float, dt: float,
                     preprocess: Dict[str, Any], window_info: Dict[str, Any], pf: PoleFit) -> None:
    poles = []
    for i in range(pf.z.size):
        w = pf.omega_qnm[i]
        poles.append({
            "i": int(i),
            "z": c2(pf.z[i]),
            "q": c2(pf.q[i]),
            "omega_qnm": c2(w),
            "freq_hz": float(np.real(w) / (2.0 * np.pi)),
            "damping_1_over_s": float(-np.imag(w)),  # >0 for decay if Im(w)<0
            "a": c2(pf.a[i]),
            "amp_abs": float(np.abs(pf.a[i])),
        })

    payload = {
        "created_at": _utc_now_iso(),
        "script": "01_extract_ringdown_poles.py",
        "script_version": SCRIPT_VERSION,
        "event": event,
        "ifo": ifo,
        "fs_hz": float(fs),
        "dt": float(dt),
        "t0_rel": float(t0),
        "duration_s": float(duration),
        "preprocess": preprocess,
        "window": window_info,
        "fit": {
            "method": "esprit_matrix_pencil",
            "L": int(pf.L),
            "rank_used": int(pf.rank),
            "residual_rms": float(pf.residual_rms),
            "relative_rms": float(pf.relative_rms),
            "singular_values": [float(x) for x in pf.singular_values[: min(64, pf.singular_values.size)]],
            "singular_values_truncated": bool(pf.singular_values.size > 64),
        },
        "poles": poles,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_poles_csv(out_path: Path, pf: PoleFit) -> None:
    # columns: idx, omega_re, omega_im, freq_hz, damping, amp_abs, a_re, a_im
    lines = ["i,omega_re,omega_im,freq_hz,damping_1_over_s,amp_abs,a_re,a_im,z_re,z_im,q_re,q_im"]
    for i in range(pf.z.size):
        w = pf.omega_qnm[i]
        lines.append(
            ",".join([
                str(i),
                f"{np.real(w):.18e}",
                f"{np.imag(w):.18e}",
                f"{(np.real(w)/(2*np.pi)):.18e}",
                f"{(-np.imag(w)):.18e}",
                f"{(np.abs(pf.a[i])):.18e}",
                f"{np.real(pf.a[i]):.18e}",
                f"{np.imag(pf.a[i]):.18e}",
                f"{np.real(pf.z[i]):.18e}",
                f"{np.imag(pf.z[i]):.18e}",
                f"{np.real(pf.q[i]):.18e}",
                f"{np.imag(pf.q[i]):.18e}",
            ])
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifest(run_dir: Path, new_artifacts: Dict[str, str], new_metadata: Dict[str, Any]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "manifest_version": "2.0",
        "created_at": _utc_now_iso(),
        "run_dir": str(run_dir),
        "artifacts": {},
        "metadata": {},
    }
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.update(new_artifacts)
    manifest["artifacts"] = artifacts

    meta = dict(manifest.get("metadata", {}))
    meta.update(new_metadata)
    manifest["metadata"] = meta

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract operational ringdown poles (ESPRIT/matrix pencil) from boundary HDF5.")
    p.add_argument("--run-dir", required=True, help="Existing run directory created by 00_load_ligo_data.py (root-relative allowed).")
    p.add_argument("--boundary-h5", default=None, help="Override boundary HDF5 path (root-relative allowed). If omitted, use run_manifest.json.")
    p.add_argument("--out-dir-name", default="ringdown", help="Output directory name inside run_dir (default: ringdown).")

    # window selection
    p.add_argument("--t0-rel", type=float, default=None, help="Start time (seconds) relative to event gps (t_rel). If omitted, auto from peak.")
    p.add_argument("--duration", type=float, default=0.25, help="Window duration in seconds (default: 0.25).")
    p.add_argument("--peak-search-start", type=float, default=-0.5, help="Auto-peak search start (t_rel) if --t0-rel omitted.")
    p.add_argument("--peak-search-end", type=float, default=0.5, help="Auto-peak search end (t_rel) if --t0-rel omitted.")
    p.add_argument("--start-offset", type=float, default=0.0, help="If auto peak used, t0 = t_peak + start_offset (seconds).")

    # preprocessing
    p.add_argument("--detrend", choices=["none", "mean", "linear"], default="mean", help="Detrend mode applied to window before fit.")
    p.add_argument("--hp-hz", type=float, default=None, help="Optional high-pass cutoff in Hz (FFT-domain).")
    p.add_argument("--lp-hz", type=float, default=None, help="Optional low-pass cutoff in Hz (FFT-domain).")

    # ESPRIT params
    p.add_argument("--L", type=int, default=0, help="Hankel rows. 0 => auto (min(Nw//2, 4096)).")
    p.add_argument("--rank", type=int, default=0, help="Fixed rank. 0 => auto from singular values.")
    p.add_argument("--sv-thresh", type=float, default=1e-3, help="Relative singular value threshold for auto rank (default: 1e-3).")
    p.add_argument("--require-decay", action="store_true", help="Filter to modes with Im(omega_qnm) < 0 (decaying).")
    p.add_argument("--max-modes", type=int, default=16, help="Keep at most this many modes after sorting (default: 16).")

    # routing convenience
    p.add_argument("--set-latest", default=None, help="Optional stable symlink (root-relative allowed) to point to run_dir.")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    project_root = Path(__file__).resolve().parent.parent  # repo root (basurin/)

    run_dir = _resolve_root_relative(project_root, args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    if args.boundary_h5:
        boundary_h5 = _resolve_root_relative(project_root, args.boundary_h5)
    else:
        boundary_h5 = _find_boundary_from_manifest(run_dir)

    if not boundary_h5.exists():
        raise FileNotFoundError(f"Boundary HDF5 not found: {boundary_h5}")

    boundary_sha = _sha256_file(boundary_h5)

    data = load_boundary_h5(boundary_h5)
    event = data["event"]
    fs = float(data["fs"])
    dt = float(data["dt"])
    t_rel = data["t_rel"]
    h1 = data["H1"]
    l1 = data["L1"]

    # reference signal for peak picking
    if l1 is not None:
        y_ref = np.abs(h1) + np.abs(l1)
    else:
        y_ref = np.abs(h1)

    i0, i1, t0_used = pick_window(
        t_rel=t_rel,
        y_ref=y_ref,
        t0_rel=args.t0_rel,
        duration=args.duration,
        peak_search_start=args.peak_search_start,
        peak_search_end=args.peak_search_end,
        start_offset=args.start_offset,
    )

    # extract windows
    tw = t_rel[i0:i1].astype(np.float64)
    h1w = h1[i0:i1].astype(np.float64)
    l1w = l1[i0:i1].astype(np.float64) if l1 is not None else None

    # preprocess per channel
    h1p = detrend(h1w, args.detrend)
    if args.hp_hz is not None or args.lp_hz is not None:
        h1p = apply_rfft_bandpass(h1p, fs, args.hp_hz, args.lp_hz)

    l1p = None
    if l1w is not None:
        l1p = detrend(l1w, args.detrend)
        if args.hp_hz is not None or args.lp_hz is not None:
            l1p = apply_rfft_bandpass(l1p, fs, args.hp_hz, args.lp_hz)

    Nw = h1p.size
    L = int(args.L) if args.L and args.L > 0 else int(min(max(2, Nw // 2), 4096))

    # fit per channel
    pf_h1 = esprit_poles(h1p, dt=dt, L=L, rank=args.rank, sv_thresh=args.sv_thresh)
    pf_h1 = _sort_and_filter(pf_h1, require_decay=bool(args.require_decay), max_modes=int(args.max_modes))

    pf_l1 = None
    if l1p is not None:
        pf_l1 = esprit_poles(l1p, dt=dt, L=L, rank=args.rank, sv_thresh=args.sv_thresh)
        pf_l1 = _sort_and_filter(pf_l1, require_decay=bool(args.require_decay), max_modes=int(args.max_modes))

    # joint: simple merge by concatenation then unique-ish by rounding omega_qnm
    pf_joint = None
    if pf_l1 is not None and pf_h1.rank > 0 and pf_l1.rank > 0:
        w_all = np.concatenate([pf_h1.omega_qnm, pf_l1.omega_qnm])
        a_all = np.concatenate([pf_h1.a, pf_l1.a])
        z_all = np.concatenate([pf_h1.z, pf_l1.z])
        q_all = np.concatenate([pf_h1.q, pf_l1.q])

        # bucket by rounded (Re,Im) to reduce duplicates
        key = np.round(np.vstack([np.real(w_all), np.imag(w_all)]).T, decimals=3)
        _, idx = np.unique(key, axis=0, return_index=True)
        idx = np.sort(idx)

        pf_joint = PoleFit(
            z=z_all[idx], q=q_all[idx], omega_qnm=w_all[idx], a=a_all[idx],
            residual_rms=float("nan"), relative_rms=float("nan"),
            singular_values=np.array([], dtype=np.float64), rank=int(idx.size), L=L
        )
        pf_joint = _sort_and_filter(pf_joint, require_decay=bool(args.require_decay), max_modes=int(args.max_modes))

    # output paths
    out_dir = run_dir / args.out_dir_name
    _mkdirp(out_dir)

    created_at = _utc_now_iso()

    preprocess = {"detrend": args.detrend, "hp_hz": args.hp_hz, "lp_hz": args.lp_hz}
    window_info = {
        "t0_rel": float(t0_used),
        "duration_s": float(args.duration),
        "i0": int(i0),
        "i1": int(i1),
        "n_samples": int(i1 - i0),
        "t_rel_start": float(t_rel[i0]),
        "t_rel_end": float(t_rel[i1 - 1]),
    }

    spec = {
        "script": "01_extract_ringdown_poles.py",
        "script_version": SCRIPT_VERSION,
        "created_at": created_at,
        "params": {
            "run_dir": str(args.run_dir),
            "boundary_h5": str(args.boundary_h5) if args.boundary_h5 else None,
            "out_dir_name": args.out_dir_name,
            "t0_rel": args.t0_rel,
            "duration": args.duration,
            "peak_search_start": args.peak_search_start,
            "peak_search_end": args.peak_search_end,
            "start_offset": args.start_offset,
            "detrend": args.detrend,
            "hp_hz": args.hp_hz,
            "lp_hz": args.lp_hz,
            "L": L,
            "rank": args.rank,
            "sv_thresh": args.sv_thresh,
            "require_decay": bool(args.require_decay),
            "max_modes": args.max_modes,
            "set_latest": args.set_latest,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "boundary_h5_abs": str(boundary_h5),
            "boundary_h5_sha256": boundary_sha,
        },
        "window": window_info,
    }
    spec_canonical = _json_canonical_dumps(spec)
    spec_sha_canonical = _sha256_bytes(spec_canonical.encode("utf-8"))

    spec_path = out_dir / "ringdown_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    spec_sha_file = _sha256_file(spec_path)

    # write per-IFO outputs
    poles_h1_json = out_dir / "poles_H1.json"
    poles_h1_csv = out_dir / "poles_H1.csv"
    write_poles_json(poles_h1_json, event, "H1", t0_used, args.duration, fs, dt, preprocess, window_info, pf_h1)
    write_poles_csv(poles_h1_csv, pf_h1)

    poles_l1_json = poles_l1_csv = None
    if pf_l1 is not None:
        poles_l1_json = out_dir / "poles_L1.json"
        poles_l1_csv = out_dir / "poles_L1.csv"
        write_poles_json(poles_l1_json, event, "L1", t0_used, args.duration, fs, dt, preprocess, window_info, pf_l1)
        write_poles_csv(poles_l1_csv, pf_l1)

    poles_joint_json = poles_joint_csv = None
    if pf_joint is not None:
        poles_joint_json = out_dir / "poles_joint.json"
        poles_joint_csv = out_dir / "poles_joint.csv"
        write_poles_json(poles_joint_json, event, "H1+L1", t0_used, args.duration, fs, dt, preprocess, window_info, pf_joint)
        write_poles_csv(poles_joint_csv, pf_joint)

    # write stage summary
    summary = {
        "created_at": created_at,
        "script": "01_extract_ringdown_poles.py",
        "script_version": SCRIPT_VERSION,
        "event": event,
        "closure_criterion": "ringdown_poles_extracted",
        "inputs": {
            "run_dir": str(run_dir),
            "boundary_h5_rel": boundary_h5.relative_to(run_dir).as_posix() if boundary_h5.is_relative_to(run_dir) else None,
            "boundary_h5_sha256": boundary_sha,
        },
        "spec_sha256_canonical": spec_sha_canonical,
        "spec_sha256_file": spec_sha_file,
        "outputs": {
            "ringdown_dir_rel": out_dir.relative_to(run_dir).as_posix(),
            "ringdown_spec_rel": spec_path.relative_to(run_dir).as_posix(),
            "poles_h1_json_rel": poles_h1_json.relative_to(run_dir).as_posix(),
            "poles_h1_csv_rel": poles_h1_csv.relative_to(run_dir).as_posix(),
            "poles_l1_json_rel": poles_l1_json.relative_to(run_dir).as_posix() if poles_l1_json else None,
            "poles_l1_csv_rel": poles_l1_csv.relative_to(run_dir).as_posix() if poles_l1_csv else None,
            "poles_joint_json_rel": poles_joint_json.relative_to(run_dir).as_posix() if poles_joint_json else None,
            "poles_joint_csv_rel": poles_joint_csv.relative_to(run_dir).as_posix() if poles_joint_csv else None,
        },
        "fit_quality": {
            "H1_relative_rms": float(pf_h1.relative_rms),
            "L1_relative_rms": float(pf_l1.relative_rms) if pf_l1 is not None else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # update manifest (do not delete existing keys)
    new_artifacts = {
        "ringdown_dir": out_dir.relative_to(run_dir).as_posix(),
        "ringdown_spec": spec_path.relative_to(run_dir).as_posix(),
        "ringdown_poles_h1_json": poles_h1_json.relative_to(run_dir).as_posix(),
        "ringdown_poles_h1_csv": poles_h1_csv.relative_to(run_dir).as_posix(),
        "ringdown_summary": (out_dir / "summary.json").relative_to(run_dir).as_posix(),
    }
    if poles_l1_json:
        new_artifacts["ringdown_poles_l1_json"] = poles_l1_json.relative_to(run_dir).as_posix()
    if poles_l1_csv:
        new_artifacts["ringdown_poles_l1_csv"] = poles_l1_csv.relative_to(run_dir).as_posix()
    if poles_joint_json:
        new_artifacts["ringdown_poles_joint_json"] = poles_joint_json.relative_to(run_dir).as_posix()
    if poles_joint_csv:
        new_artifacts["ringdown_poles_joint_csv"] = poles_joint_csv.relative_to(run_dir).as_posix()

    update_manifest(
        run_dir,
        new_artifacts=new_artifacts,
        new_metadata={
            "ringdown_stage_created_at": created_at,
            "ringdown_spec_sha256_canonical": spec_sha_canonical,
            "ringdown_spec_sha256_file": spec_sha_file,
        },
    )

    # optional latest symlink
    if args.set_latest:
        link = _resolve_root_relative(project_root, args.set_latest)
        _atomic_symlink(run_dir.resolve(), link)

    print(f"RUN_DIR: {run_dir}")
    print(f"Boundary: {boundary_h5}")
    print(f"Ringdown window: t0_rel={t0_used:.6f}, duration={args.duration:.6f}, Nw={i1-i0}")
    print(f"Wrote: {out_dir.relative_to(run_dir).as_posix()}/poles_H1.json")
    if poles_l1_json:
        print(f"Wrote: {out_dir.relative_to(run_dir).as_posix()}/poles_L1.json")
    if poles_joint_json:
        print(f"Wrote: {out_dir.relative_to(run_dir).as_posix()}/poles_joint.json")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
