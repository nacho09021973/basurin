#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, hashlib, time, platform, sys
from pathlib import Path
import numpy as np

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    in_summary = repo / "runs" / args.run_id / "ringdown" / "outputs" / "summary.json"
    if not in_summary.exists():
        raise FileNotFoundError(f"No existe {in_summary}")

    out_dir = repo / "runs" / args.run_id / "ringdown_features"
    outputs = out_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    s = json.loads(in_summary.read_text(encoding="utf-8"))

    f220 = float(s["f_qnm_hz"])
    tau_s = float(s["tau_qnm_ms"]) / 1000.0
    Q220 = float(np.pi * f220 * tau_s)

    mass_src = None
    if s.get("mass_source") and s["mass_source"] is not None:
        mass_src = float(s["mass_source"]["median"])

    spin = float(s["spin"]["median"])
    category = str(s["category"])

    feats = {
        "category": category,
        "f220_hz": f220,
        "tau220_s": tau_s,
        "Q220": Q220,
        "log_f220": float(np.log(f220)),
        "log_tau220": float(np.log(tau_s)),
        "log_Q220": float(np.log(Q220)),
        "spin_median": spin,
        "mass_source_median": mass_src,
    }

    # Guardar features
    write_json(outputs / "features.json", feats)

    # Vector numérico estable (orden fijo)
    names = [
        "log_f220", "log_tau220", "log_Q220",
        "spin_median",
    ]
    vec = np.array([feats[k] for k in names], dtype=np.float64)
    np.save(outputs / "features.npy", vec)

    # Stage summary + manifest
    stage_summary = {
        "stage": "ringdown_features",
        "run_id": args.run_id,
        "created_at": now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "inputs": {"summary_json": str(in_summary.relative_to(repo)), "sha256": sha256_file(in_summary)},
        "outputs": {
            "features_json": str((outputs / "features.json").relative_to(repo)),
            "features_npy": str((outputs / "features.npy").relative_to(repo)),
        },
        "feature_order": names,
    }
    write_json(out_dir / "stage_summary.json", stage_summary)

    manifest = {
        "run_id": args.run_id,
        "stage": "ringdown_features",
        "inputs": [{"path": str(in_summary), "sha256": sha256_file(in_summary)}],
        "outputs": [
            {"path": str((outputs / "features.json").relative_to(repo)), "sha256": sha256_file(outputs / "features.json")},
            {"path": str((outputs / "features.npy").relative_to(repo)), "sha256": sha256_file(outputs / "features.npy")},
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"[ringdown_features] OK -> {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
