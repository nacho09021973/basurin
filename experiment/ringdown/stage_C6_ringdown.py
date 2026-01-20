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

def robust_threshold(d: np.ndarray) -> float:
    # Umbral conservador: mediana + 6*MAD
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-12
    return med + 6.0 * mad

def main() -> int:
    ap = argparse.ArgumentParser(description="C6_ringdown: OOD/distancia contra atlas")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas", required=True, help="Ruta a atlas.json (en runs/<id>/dictionary/atlas.json)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    run_id = args.run_id

    feat_npy = repo / "runs" / run_id / "ringdown_features" / "outputs" / "features.npy"
    feat_json = repo / "runs" / run_id / "ringdown_features" / "outputs" / "features.json"
    if not feat_npy.exists() or not feat_json.exists():
        raise FileNotFoundError("No encuentro features de ringdown_features (features.npy/features.json)")

    atlas_path = (repo / args.atlas).resolve() if not str(args.atlas).startswith(str(repo)) else Path(args.atlas).resolve()
    if not atlas_path.exists():
        raise FileNotFoundError(f"No existe atlas: {atlas_path}")

    # Cargar
    x = np.load(feat_npy).astype(np.float64).reshape(-1)
    meta = json.loads(feat_json.read_text(encoding="utf-8"))
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))

    # Profesional: NO inferir estructura si no existe
    # Esperamos que el atlas contenga un array de puntos en el mismo espacio de features
    # Convención: atlas["points"] = [{"id":..., "features":[...]}...]
    points = atlas.get("points", None)
    if points is None:
        raise KeyError("El atlas.json no contiene clave 'points'. No puedo calcular distancias sin features explícitas.")

    ids = []
    X = []
    for p in points:
        if "features" not in p:
            continue
        v = np.array(p["features"], dtype=np.float64).reshape(-1)
        X.append(v)
        ids.append(p.get("id", None))

    if len(X) == 0:
        raise ValueError("atlas['points'] existe pero no hay 'features' utilizables.")

    X = np.vstack(X)
    if X.shape[1] != x.shape[0]:
        raise ValueError(f"Dim mismatch: ringdown features dim={x.shape[0]} vs atlas dim={X.shape[1]}")

    # Distancias
    d = np.linalg.norm(X - x[None, :], axis=1)

    # kNN
    k = min(args.k, len(d))
    idx = np.argsort(d)[:k]
    nn = [{"rank": i+1, "id": ids[j], "dist": float(d[j])} for i, j in enumerate(idx)]

    # Umbral OOD (robusto)
    thr = robust_threshold(d)
    ood = bool(float(np.min(d)) > thr)

    out = {
        "contract": "C6_ringdown",
        "run_id": run_id,
        "atlas": str(atlas_path.relative_to(repo)),
        "k": k,
        "ringdown_meta": {
            "category": meta.get("category"),
            "mass_source_median": meta.get("mass_source_median"),
            "spin_median": meta.get("spin_median"),
        },
        "dist": {
            "min": float(np.min(d)),
            "p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99)),
            "threshold": float(thr),
            "ood": ood,
        },
        "nearest_neighbors": nn,
    }

    # Escribir artefactos BASURIN-style
    out_dir = repo / "runs" / run_id / "C6_ringdown"
    outputs = out_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    write_json(outputs / "validation.json", out)

    stage_summary = {
        "stage": "C6_ringdown",
        "run_id": run_id,
        "created_at": now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "inputs": {
            "features_npy": str(feat_npy.relative_to(repo)),
            "features_json": str(feat_json.relative_to(repo)),
            "atlas": str(atlas_path.relative_to(repo)),
        },
        "sha256": {
            "features_npy": sha256_file(feat_npy),
            "features_json": sha256_file(feat_json),
            "atlas": sha256_file(atlas_path),
        },
        "params": {"k": k},
        "outputs": {"validation_json": str((outputs / "validation.json").relative_to(repo))},
    }
    write_json(out_dir / "stage_summary.json", stage_summary)

    manifest = {
        "run_id": run_id,
        "stage": "C6_ringdown",
        "inputs": [
            {"path": str(feat_npy), "sha256": sha256_file(feat_npy)},
            {"path": str(feat_json), "sha256": sha256_file(feat_json)},
            {"path": str(atlas_path), "sha256": sha256_file(atlas_path)},
        ],
        "outputs": [
            {"path": str((outputs / "validation.json").relative_to(repo)), "sha256": sha256_file(outputs / "validation.json")}
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"[C6_ringdown] OK -> {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
