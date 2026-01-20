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
    ap = argparse.ArgumentParser(description="C6_ringdown_meta: enlace ringdown<->atlas sin distancias")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas-points", required=True, help="atlas_points.json (feature_key=ratios)")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    run_id = args.run_id

    feat_json = repo/"runs"/run_id/"ringdown_features"/"outputs"/"features.json"
    feat_npy  = repo/"runs"/run_id/"ringdown_features"/"outputs"/"features.npy"
    if not feat_json.exists() or not feat_npy.exists():
        raise FileNotFoundError("Faltan ringdown_features outputs (features.json/features.npy)")

    atlas_p = Path(args.atlas_points)
    if not atlas_p.is_absolute():
        atlas_p = repo/atlas_p
    if not atlas_p.exists():
        raise FileNotFoundError(f"No existe {atlas_p}")

    rd = json.loads(feat_json.read_text(encoding="utf-8"))
    x = np.load(feat_npy).reshape(-1)
    atlas = json.loads(atlas_p.read_text(encoding="utf-8"))

    # Resumen del atlas
    points = atlas.get("points", [])
    deltas = [p.get("delta") for p in points if p.get("delta") is not None]
    regimes = [p.get("regime") for p in points if p.get("regime") is not None]
    feat_dim = None
    if points and "features" in points[0]:
        feat_dim = len(points[0]["features"])

    # Resultado profesional: declaramos incompatibilidad de espacios
    out = {
        "contract": "C6_ringdown_meta",
        "run_id": run_id,
        "status": "INCOMPATIBLE_FEATURE_SPACE",
        "message": "El atlas está en espacio 'ratios' (dim=9) y ringdown_features está en espacio QNM (dim=4). No se calculan distancias sin un feature-map explícito.",
        "ringdown_features": {
            "category": rd.get("category"),
            "mass_source_median": rd.get("mass_source_median"),
            "spin_median": rd.get("spin_median"),
            "vector_dim": int(x.shape[0]),
            "vector_order": ["log_f220","log_tau220","log_Q220","spin_median"],
            "vector": [float(v) for v in x.tolist()],
        },
        "atlas_points": {
            "path": str(atlas_p.relative_to(repo)),
            "feature_key": atlas.get("feature_key"),
            "n_points": int(atlas.get("n_points", len(points))),
            "feature_dim": feat_dim,
            "delta_min": float(min(deltas)) if deltas else None,
            "delta_max": float(max(deltas)) if deltas else None,
            "regime_counts": {r: regimes.count(r) for r in sorted(set(regimes))},
        },
        "next_step": {
            "recommended": "Definir un feature-map explícito para comparar ringdown en el espacio ratios (o exportar features del atlas en el espacio QNM) y entonces usar C6_ringdown (distancias).",
            "options": [
                "phi: ringdown -> ratios (requiere modelado explícito)",
                "psi: ratios -> QNM-features (requiere entrenamiento/regresión externa)",
                "exportar del pipeline un embedding común (p.ej. tangentes locales/latente) para ambos"
            ]
        }
    }

    out_dir = repo/"runs"/run_id/"C6_ringdown_meta"
    outputs = out_dir/"outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    write_json(outputs/"validation.json", out)

    stage_summary = {
        "stage": "C6_ringdown_meta",
        "run_id": run_id,
        "created_at": now_iso(),
        "python": sys.version.replace("\n"," "),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "inputs": {
            "ringdown_features_json": str(feat_json.relative_to(repo)),
            "ringdown_features_npy": str(feat_npy.relative_to(repo)),
            "atlas_points": str(atlas_p.relative_to(repo)),
        },
        "sha256": {
            "ringdown_features_json": sha256_file(feat_json),
            "ringdown_features_npy": sha256_file(feat_npy),
            "atlas_points": sha256_file(atlas_p),
        },
        "outputs": {"validation_json": str((outputs/"validation.json").relative_to(repo))}
    }
    write_json(out_dir/"stage_summary.json", stage_summary)

    manifest = {
        "run_id": run_id,
        "stage": "C6_ringdown_meta",
        "inputs": [
            {"path": str(feat_json), "sha256": sha256_file(feat_json)},
            {"path": str(feat_npy), "sha256": sha256_file(feat_npy)},
            {"path": str(atlas_p), "sha256": sha256_file(atlas_p)},
        ],
        "outputs": [
            {"path": str((outputs/"validation.json").relative_to(repo)), "sha256": sha256_file(outputs/"validation.json")}
        ],
    }
    write_json(out_dir/"manifest.json", manifest)

    print(f"[C6_ringdown_meta] OK -> {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
