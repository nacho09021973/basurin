#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BASURIN — F3-closure (External Contrast Closure via Feature-Space Compatibility)

Implements:
  - C6a: Structural compatibility gate (feature_key + dim + schema)
  - C6b: OOD/manifold test via kNN distance calibrated on atlas self-distances

Cases:
  A) REAL: ringdown_features vs atlas_points (expected INCOMPATIBLE_FEATURE_SPACE)
  B) POS: synthetic external inliers vs synthetic atlas in common feature space (expected PASS)
  C) NEG-OOD: synthetic external outliers vs synthetic atlas in common feature space (expected FAIL_OOD)

Outputs (BASURIN-style):
  runs/<run_id>/f3_closure/
    manifest.json
    stage_summary.json
    outputs/
      decision.json
      calibration.json
      cases/A_real.json
      cases/B_positive.json
      cases/C_negative_ood.json
      inputs_hashes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np


# -----------------------------
# Utilities: IO / hashing
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)

def relpath_under(root: Path, p: Path) -> str:
    return str(p.resolve().relative_to(root.resolve()))

def ensure_under(root: Path, p: Path) -> None:
    p = p.resolve()
    r = root.resolve()
    if r not in p.parents and p != r:
        raise RuntimeError(f"Refusing to write outside run root: {p} not under {r}")

def load_points_any(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Flexible loader:
      - .npy: expects (N,D) array
      - .json: supports several schemas:
          a) {"points": [[...]], "feature_key": "...", "dim": D}
          b) {"X": [[...]], "meta": {"feature_key": "...", "dim": D}}
          c) list of dict rows: [{"log_f220":..., ...}, ...] -> will attempt column inference
    Returns (points_array, meta_dict)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    meta: Dict[str, Any] = {}

    if suffix == ".npy":
        X = np.load(path)
        if X.ndim != 2:
            raise ValueError(f"{path}: expected 2D array, got shape {X.shape}")
        meta["schema"] = "npy_2d"
        meta["dim"] = int(X.shape[1])
        return X.astype(float), meta

    if suffix == ".json":
        obj = read_json(path)

        # Case: dict with points
        if isinstance(obj, dict):
            # common keys that may hold the point matrix
            for k in ("points", "X", "data", "features"):
                if k in obj and isinstance(obj[k], list):
                    raw = obj[k]
                    # If points are dict-rows, infer numeric columns and build matrix
                    if raw and isinstance(raw[0], dict):
                        meta_obj = (obj.get("meta", {}) or {})
                        cols = meta_obj.get("columns") or obj.get("columns")
                        if not cols:
                            cols = sorted([c for c,v in raw[0].items() if isinstance(v, (int, float))])
                        X = np.array([[r.get(c) for c in cols] for r in raw], dtype=float)
                        if X.ndim != 2:
                            raise ValueError(f"{path}: '{k}' rows not 2D after conversion, got {X.shape}")
                        meta["schema"] = f"json_dict_{k}_rows"
                        meta["columns"] = cols
                        meta["feature_key"] = obj.get("feature_key") or obj.get("key") or meta_obj.get("feature_key")
                        meta["dim"] = int(obj.get("dim") or meta_obj.get("dim") or X.shape[1])
                        return X, meta

                    # Otherwise assume raw is a numeric matrix (list of lists)
                    X = np.array(raw, dtype=float)
                    if X.ndim != 2:
                        raise ValueError(f"{path}: '{k}' not 2D after conversion, got {X.shape}")
                    meta_obj = (obj.get("meta", {}) or {})
                    meta["schema"] = f"json_dict_{k}"
                    meta["feature_key"] = obj.get("feature_key") or obj.get("key") or meta_obj.get("feature_key")
                    meta["dim"] = int(obj.get("dim") or meta_obj.get("dim") or X.shape[1])
                    return X, meta

            # Case: ROOT dict-of-columns (ringdown_features style)
            # Example: {"log_f220":[...], "log_tau220":[...], "log_Q220":[...], "spin_median":[...], ...}
            # We build (N,4) using preferred columns if present.
            if all(isinstance(v, list) for v in obj.values()):
                preferred_sets = [
                    ["log_f220", "log_tau220", "log_Q220", "spin_median"],
                    ["log_f220", "log_tau220", "log_Q220", "spin"],
                ]
                cols = None
                for cand in preferred_sets:
                    if all(c in obj for c in cand):
                        cols = cand
                        break
                if cols is not None:
                    n = len(obj[cols[0]])
                    # Basic consistency check
                    for c in cols[1:]:
                        if len(obj[c]) != n:
                            raise ValueError(f"{path}: column length mismatch for '{c}'")
                    X = np.array([[obj[c][i] for c in cols] for i in range(n)], dtype=float)
                    meta["schema"] = "json_root_dictofcols"
                    meta["columns"] = cols
                    meta["feature_key"] = "qnm"
                    meta["dim"] = int(X.shape[1])
                    return X, meta

            # Case: dict containing rows
            if "rows" in obj and isinstance(obj["rows"], list) and obj["rows"] and isinstance(obj["rows"][0], dict):
                rows = obj["rows"]
                # numeric columns
                cols = sorted([c for c in rows[0].keys() if isinstance(rows[0][c], (int, float))])
                X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
                meta["schema"] = "json_dict_rows"
                meta["columns"] = cols
                meta["feature_key"] = obj.get("feature_key") or (obj.get("meta", {}) or {}).get("feature_key")
                meta["dim"] = int(X.shape[1])
                return X, meta

        # Case: list of dict rows (ringdown_features often looks like this)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            rows = obj
            # Prefer canonical ringdown columns if present
            preferred = ["log_f220", "log_tau220", "log_Q220", "spin"]
            if all(p in rows[0] for p in preferred):
                cols = preferred
            else:
                cols = sorted([c for c in rows[0].keys() if isinstance(rows[0][c], (int, float))])
            X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
            meta["schema"] = "json_list_rows"
            meta["columns"] = cols
            meta["dim"] = int(X.shape[1])
            return X, meta

        # Case: ROOT dict-of-columns (ringdown_features style)
        # Example (your file): {"log_f220":[...], "log_tau220":[...], "log_Q220":[...], "spin_median":[...], ...}
        # Build (N,4) deterministically.
        preferred_sets = [
            ["log_f220", "log_tau220", "log_Q220", "spin_median"],
            ["log_f220", "log_tau220", "log_Q220", "spin"],
        ]
        for cols in preferred_sets:
            if all(c in obj for c in cols) and all(isinstance(obj[c], list) for c in cols):
                n = len(obj[cols[0]])
                for c in cols[1:]:
                    if len(obj[c]) != n:
                        raise ValueError(f"{path}: column length mismatch for '{c}'")
                X = np.array([[obj[c][i] for c in cols] for i in range(n)], dtype=float)
                meta["schema"] = "json_root_dictofcols"
                meta["columns"] = cols
                meta["feature_key"] = "qnm"
                meta["dim"] = int(X.shape[1])
                return X, meta

        # Case: ROOT dict of scalar features (single-point ringdown summary)
        # Example: {"log_f220": 5.43, "log_tau220": -4.72, "log_Q220": 1.85, "spin_median": 0.93, ...}
        preferred_sets_scalar = [
            ["log_f220", "log_tau220", "log_Q220", "spin_median"],
            ["log_f220", "log_tau220", "log_Q220", "spin"],
        ]
        for cols in preferred_sets_scalar:
            if all(c in obj for c in cols) and all(isinstance(obj[c], (int, float)) for c in cols):
                X = np.array([[float(obj[c]) for c in cols]], dtype=float)  # shape (1,4)
                meta["schema"] = "json_root_scalars_single_point"
                meta["columns"] = cols
                meta["feature_key"] = "qnm"
                meta["dim"] = int(X.shape[1])
                return X, meta

        raise ValueError(f"{path}: unsupported json schema for points")

    raise ValueError(f"{path}: unsupported file type {suffix}")


# -----------------------------
# C6 contracts
# -----------------------------

@dataclass
class C6aResult:
    status: str  # PASS / FAIL
    failure_mode: Optional[str]
    details: Dict[str, Any]

def c6a_structural(atlas_meta: Dict[str, Any], ext_meta: Dict[str, Any],
                   atlas_dim: int, ext_dim: int,
                   atlas_feature_key: Optional[str], ext_feature_key: Optional[str]) -> C6aResult:
    # Explicit: both feature_key and dim must match
    if atlas_feature_key is None or ext_feature_key is None:
        return C6aResult(
            status="FAIL",
            failure_mode="SCHEMA_MISMATCH",
            details={
                "reason": "Missing feature_key in one or both inputs",
                "atlas_feature_key": atlas_feature_key,
                "ext_feature_key": ext_feature_key,
            },
        )
    if atlas_feature_key != ext_feature_key or atlas_dim != ext_dim:
        return C6aResult(
            status="FAIL",
            failure_mode="INCOMPATIBLE_FEATURE_SPACE",
            details={
                "atlas_feature_key": atlas_feature_key,
                "ext_feature_key": ext_feature_key,
                "atlas_dim": atlas_dim,
                "ext_dim": ext_dim,
            },
        )
    return C6aResult(
        status="PASS",
        failure_mode=None,
        details={
            "feature_key": atlas_feature_key,
            "dim": atlas_dim,
        },
    )

@dataclass
class C6bResult:
    status: str  # PASS / FAIL_OOD
    details: Dict[str, Any]

def knn_min_distances(X_ref: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """
    Computes min Euclidean distance from each query point to any ref point.
    O(N*M) but acceptable for modest sizes; deterministic.
    """
    # (M,1,D) - (1,N,D) -> (M,N,D)
    diffs = X_query[:, None, :] - X_ref[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)  # (M,N)
    dmin = np.sqrt(np.min(d2, axis=1))  # (M,)
    return dmin

def atlas_self_dmins(X: np.ndarray) -> np.ndarray:
    """
    Leave-one-out min distance within atlas: for each i, min distance to j!=i.
    """
    N = X.shape[0]
    dmins = np.empty(N, dtype=float)
    for i in range(N):
        diffs = X[i][None, :] - np.delete(X, i, axis=0)
        d2 = np.sum(diffs * diffs, axis=1)
        dmins[i] = float(np.sqrt(np.min(d2)))
    return dmins

def c6b_ood(atlas_points: np.ndarray, ext_points: np.ndarray,
           p_tau: float, frac_inlier_pass: float) -> Tuple[C6bResult, Dict[str, Any]]:
    d_self = atlas_self_dmins(atlas_points)
    tau = float(np.percentile(d_self, p_tau))

    d_ext = knn_min_distances(atlas_points, ext_points)
    frac_inlier = float(np.mean(d_ext <= tau))
    d_med = float(np.median(d_ext))

    if d_ext.size == 1:
        status = "PASS" if float(d_ext[0]) <= tau else "FAIL_OOD"
    else:
        status = "PASS" if (frac_inlier >= frac_inlier_pass and d_med <= tau) else "FAIL_OOD"

    calib = {
        "p_tau": p_tau,
        "tau": tau,
        "self_dmin_stats": {
            "p50": float(np.percentile(d_self, 50)),
            "p90": float(np.percentile(d_self, 90)),
            "p95": float(np.percentile(d_self, 95)),
            "p99": float(np.percentile(d_self, 99)),
            "min": float(np.min(d_self)),
            "max": float(np.max(d_self)),
            "mean": float(np.mean(d_self)),
        },
    }

    res = C6bResult(
        status=status,
        details={
            "frac_inlier": frac_inlier,
            "frac_inlier_pass": frac_inlier_pass,
            "d_med": d_med,
            "tau": tau,
            "n_atlas": int(atlas_points.shape[0]),
            "n_external": int(ext_points.shape[0]),
        },
    )
    return res, calib


# -----------------------------
# Synthetic controls
# -----------------------------

def make_synth_atlas_and_external(seed: int, n_atlas: int, n_ext: int, dim: int,
                                 ood_shift_sigma: float = 5.0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # atlas: mixture of 2 Gaussians
    means = np.stack([rng.normal(0, 1, size=dim), rng.normal(0, 1, size=dim)], axis=0)
    cov = np.diag(rng.uniform(0.5, 1.5, size=dim))
    labels = rng.integers(0, 2, size=n_atlas)
    atlas = np.array([rng.multivariate_normal(means[l], cov) for l in labels], dtype=float)

    # inlier external: sample from same mixture
    labels2 = rng.integers(0, 2, size=n_ext)
    ext_in = np.array([rng.multivariate_normal(means[l], cov) for l in labels2], dtype=float)

    # ood external: shift along first principal-ish direction (use unit vector)
    u = rng.normal(0, 1, size=dim)
    u = u / (np.linalg.norm(u) + 1e-12)
    # Estimate typical scale from atlas std projected on u
    proj = atlas @ u
    sigma_proj = float(np.std(proj) + 1e-12)
    shift = ood_shift_sigma * sigma_proj * u
    ext_ood = ext_in + shift[None, :]

    return {"atlas": atlas, "ext_in": ext_in, "ext_ood": ext_ood}


# -----------------------------
# Main stage
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id (e.g., 2026-01-20__F3_closure)")
    ap.add_argument("--atlas-points", required=True, help="Path to atlas_points.json/.npy (ratios dim=9)")
    ap.add_argument("--ringdown-features", required=True, help="Path to ringdown features .json/.npy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-tau", type=float, default=95.0)
    ap.add_argument("--frac-inlier-pass", type=float, default=0.90)
    ap.add_argument("--n-atlas-synth", type=int, default=400)
    ap.add_argument("--n-external-synth", type=int, default=60)
    ap.add_argument("--common-dim", type=int, default=4)
    args = ap.parse_args()

    run_root = Path("runs") / args.run
    stage_root = run_root / "f3_closure"
    out_root = stage_root / "outputs"
    cases_root = out_root / "cases"

    # Enforce BASURIN rule: outputs only under runs/<run_id>/
    stage_root.mkdir(parents=True, exist_ok=True)
    ensure_under(run_root, stage_root)

    # Load inputs
    atlas_path = Path(args.atlas_points)
    ring_path = Path(args.ringdown_features)

    atlas_X, atlas_meta = load_points_any(atlas_path)
    ring_X, ring_meta = load_points_any(ring_path)

    # Try to infer feature_key for atlas (should be 'ratios'); ringdown should be 'qnm' or similar.
    # If not present in file, we set a conservative key that makes mismatch explicit.
    atlas_feature_key = atlas_meta.get("feature_key") or "ratios"
    ring_feature_key = ring_meta.get("feature_key") or "qnm"

    atlas_dim = int(atlas_X.shape[1])
    ring_dim = int(ring_X.shape[1])

    # CASE A: REAL structural gate
    c6a_A = c6a_structural(
        atlas_meta=atlas_meta,
        ext_meta=ring_meta,
        atlas_dim=atlas_dim,
        ext_dim=ring_dim,
        atlas_feature_key=atlas_feature_key,
        ext_feature_key=ring_feature_key,
    )
    caseA = {
        "case": "A_real",
        "inputs": {
            "atlas_points": str(atlas_path),
            "ringdown_features": str(ring_path),
        },
        "atlas": {"feature_key": atlas_feature_key, "dim": atlas_dim, "schema": atlas_meta.get("schema")},
        "external": {"feature_key": ring_feature_key, "dim": ring_dim, "schema": ring_meta.get("schema"), "columns": ring_meta.get("columns")},
        "C6a": {
            "status": c6a_A.status,
            "failure_mode": c6a_A.failure_mode,
            "details": c6a_A.details,
        },
        "C6b": {"status": "SKIP", "details": {"reason": "C6a not PASS"}},
        "expected": {"C6a": "FAIL(INCOMPATIBLE_FEATURE_SPACE)"},
    }

    # Synthetic controls in common feature space
    synth = make_synth_atlas_and_external(
        seed=args.seed,
        n_atlas=args.n_atlas_synth,
        n_ext=args.n_external_synth,
        dim=args.common_dim,
        ood_shift_sigma=5.0,
    )
    common_feature_key = "common_latent"
    common_dim = int(args.common_dim)

    # CASE B: POS
    atlas_s = synth["atlas"]
    ext_in = synth["ext_in"]
    c6a_B = c6a_structural({}, {}, common_dim, common_dim, common_feature_key, common_feature_key)
    if c6a_B.status == "PASS":
        c6b_B, calib_B = c6b_ood(atlas_s, ext_in, args.p_tau, args.frac_inlier_pass)
    else:
        c6b_B, calib_B = C6bResult("SKIP", {"reason": "C6a not PASS"}), {}

    caseB = {
        "case": "B_positive",
        "atlas": {"feature_key": common_feature_key, "dim": common_dim, "n": int(atlas_s.shape[0])},
        "external": {"feature_key": common_feature_key, "dim": common_dim, "n": int(ext_in.shape[0])},
        "C6a": {"status": c6a_B.status, "failure_mode": c6a_B.failure_mode, "details": c6a_B.details},
        "C6b": {"status": c6b_B.status, "details": c6b_B.details},
        "expected": {"C6a": "PASS", "C6b": "PASS"},
    }

    # CASE C: NEG-OOD
    ext_ood = synth["ext_ood"]
    c6a_C = c6a_structural({}, {}, common_dim, common_dim, common_feature_key, common_feature_key)
    if c6a_C.status == "PASS":
        c6b_C, calib_C = c6b_ood(atlas_s, ext_ood, args.p_tau, args.frac_inlier_pass)
    else:
        c6b_C, calib_C = C6bResult("SKIP", {"reason": "C6a not PASS"}), {}

    caseC = {
        "case": "C_negative_ood",
        "atlas": {"feature_key": common_feature_key, "dim": common_dim, "n": int(atlas_s.shape[0])},
        "external": {"feature_key": common_feature_key, "dim": common_dim, "n": int(ext_ood.shape[0])},
        "C6a": {"status": c6a_C.status, "failure_mode": c6a_C.failure_mode, "details": c6a_C.details},
        "C6b": {"status": c6b_C.status, "details": c6b_C.details},
        "expected": {"C6a": "PASS", "C6b": "FAIL_OOD"},
    }

    # Decide overall PASS/FAIL exactly as specified
    ok_A = (caseA["C6a"]["status"] == "FAIL" and caseA["C6a"]["failure_mode"] == "INCOMPATIBLE_FEATURE_SPACE")
    ok_B = (caseB["C6a"]["status"] == "PASS" and caseB["C6b"]["status"] == "PASS")
    ok_C = (caseC["C6a"]["status"] == "PASS" and caseC["C6b"]["status"] == "FAIL_OOD")

    overall = "PASS" if (ok_A and ok_B and ok_C) else "FAIL"

    decision = {
        "stage": "f3_closure",
        "timestamp_utc": utc_now_iso(),
        "overall_status": overall,
        "criteria": {
            "A_real_expected": "C6a FAIL(INCOMPATIBLE_FEATURE_SPACE)",
            "B_positive_expected": "C6a PASS and C6b PASS",
            "C_negative_ood_expected": "C6a PASS and C6b FAIL_OOD",
        },
        "checks": {"ok_A": ok_A, "ok_B": ok_B, "ok_C": ok_C},
        "cases": {
            "A_real": {"status": "PASS" if ok_A else "FAIL", "C6a": caseA["C6a"], "C6b": caseA["C6b"]},
            "B_positive": {"status": "PASS" if ok_B else "FAIL", "C6a": caseB["C6a"], "C6b": caseB["C6b"]},
            "C_negative_ood": {"status": "PASS" if ok_C else "FAIL", "C6a": caseC["C6a"], "C6b": caseC["C6b"]},
        },
        "notes": {
            "honesty": "No physical QNM<->ratios mapping is assumed; compatibility is only enforced in synthetic controls (common_latent).",
            "real_case": "If feature_key or dim mismatch, pipeline must stop with INCOMPATIBLE_FEATURE_SPACE.",
        },
    }

    # Write outputs
    ensure_under(run_root, out_root)
    (cases_root).mkdir(parents=True, exist_ok=True)

    write_json(out_root / "decision.json", decision)
    write_json(out_root / "calibration.json", {"B_positive": calib_B, "C_negative_ood": calib_C})
    write_json(cases_root / "A_real.json", caseA)
    write_json(cases_root / "B_positive.json", caseB)
    write_json(cases_root / "C_negative_ood.json", caseC)

    # Input hashes & script hash
    script_path = Path(__file__).resolve()
    inputs_hashes = {
        "atlas_points": {"path": str(atlas_path), "sha256": sha256_file(atlas_path)},
        "ringdown_features": {"path": str(ring_path), "sha256": sha256_file(ring_path)},
        "stage_script": {"path": str(script_path), "sha256": sha256_file(script_path)},
    }
    write_json(out_root / "inputs_hashes.json", inputs_hashes)

    # Stage summary
    stage_summary = {
        "stage": "f3_closure",
        "run_id": args.run,
        "timestamp_utc": utc_now_iso(),
        "params": {
            "seed": args.seed,
            "p_tau": args.p_tau,
            "frac_inlier_pass": args.frac_inlier_pass,
            "n_atlas_synth": args.n_atlas_synth,
            "n_external_synth": args.n_external_synth,
            "common_dim": args.common_dim,
        },
        "inputs": inputs_hashes,
        "outputs": {
            "decision": "outputs/decision.json",
            "calibration": "outputs/calibration.json",
            "cases": {
                "A_real": "outputs/cases/A_real.json",
                "B_positive": "outputs/cases/B_positive.json",
                "C_negative_ood": "outputs/cases/C_negative_ood.json",
            },
            "inputs_hashes": "outputs/inputs_hashes.json",
        },
        "status": overall,
    }
    write_json(stage_root / "stage_summary.json", stage_summary)

    # Manifest
    # Keep it simple: list all files we created with sha256 and relative path
    created_files = [
        stage_root / "stage_summary.json",
        out_root / "decision.json",
        out_root / "calibration.json",
        out_root / "inputs_hashes.json",
        cases_root / "A_real.json",
        cases_root / "B_positive.json",
        cases_root / "C_negative_ood.json",
    ]
    # Build manifest with correct relative paths
    manifest_entries = []
    for p in created_files:
        manifest_entries.append({
            "path": relpath_under(run_root, p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })
    manifest = {
        "run_id": args.run,
        "stage": "f3_closure",
        "timestamp_utc": utc_now_iso(),
        "artifacts": manifest_entries,
    }
    write_json(stage_root / "manifest.json", manifest)

    print("=== F3-closure ===")
    print(f"run_id: {args.run}")
    print(f"overall: {overall}")
    print(f"A_real: ok={ok_A} -> C6a={caseA['C6a']}")
    print(f"B_positive: ok={ok_B} -> C6b={caseB['C6b']}")
    print(f"C_negative_ood: ok={ok_C} -> C6b={caseC['C6b']}")
    print(f"Outputs: {stage_root}")

if __name__ == "__main__":
    main()
