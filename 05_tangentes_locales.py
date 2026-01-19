#!/usr/bin/env python3
"""
05_tangentes_locales.py
BASURIN – Test de tangentes locales y dimensión intrínseca (Fase 1)

Diagnóstico post-hoc de geometría emergente en el espacio de ratios:
- PCA local en vecindarios kNN
- Dimensión efectiva (participation ratio)
- Test tangente vs ortogonal mediante proxy local de C3

Autor: BASURIN
"""

import argparse
import json
import os
import hashlib
import time
import numpy as np
import h5py


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_ratio_features(masses, k_features):
    m0 = masses[:, 0]
    X = []
    for n in range(1, k_features + 1):
        X.append((masses[:, n] ** 2) / (m0 ** 2))
    return np.stack(X, axis=1)


def pca_local(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.cov(Xc, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def effective_dimension(evals):
    s1 = np.sum(evals)
    s2 = np.sum(evals ** 2)
    if s2 == 0:
        return 0.0
    return (s1 ** 2) / s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k-features", type=int, default=4)
    ap.add_argument("--k-neighbors", default="10,20,30")
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--n-perturb", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--standardize", choices=["on", "off"], default="on")
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run)
    spec_path = os.path.join(run_dir, "spectrum", "spectrum.h5")
    out_dir = os.path.join(run_dir, "tangentes", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(spec_path, "r") as h5:
        masses = h5["masses"][:]
        delta = h5["delta_uv"][:]

    X = compute_ratio_features(masses, args.k_features)

    if args.standardize == "on":
        mu = X.mean(axis=0)
        sig = X.std(axis=0) + 1e-12
        X = (X - mu) / sig

    N = X.shape[0]
    rng = np.random.default_rng(123)
    idx_points = rng.choice(N, size=min(args.n_points, N), replace=False)

    k_list = [int(x) for x in args.k_neighbors.split(",")]

    results = {}

    for k in k_list:
        deff_list = []
        rho_list = []

        for i in idx_points:
            xi = X[i]
            di = delta[i]

            dists = np.linalg.norm(X - xi, axis=1)
            nn_idx = np.argsort(dists)[1:k+1]

            Xn = X[nn_idx]
            dn = delta[nn_idx]

            evals, evecs = pca_local(Xn)
            deff = effective_dimension(evals)
            m = max(1, int(round(deff)))

            # regresión local lineal
            A = np.c_[np.ones(len(Xn)), Xn - xi]
            coef, *_ = np.linalg.lstsq(A, dn - di, rcond=None)

            def delta_hat(x):
                v = x - xi
                return di + coef[0] + np.dot(coef[1:], v)

            # perturbaciones
            errs_par = []
            errs_ort = []

            for _ in range(args.n_perturb):
                z = rng.normal(size=X.shape[1])

                v_par = (evecs[:, :m] @ (evecs[:, :m].T @ z))
                v_ort = z - v_par

                if np.linalg.norm(v_par) > 0:
                    v_par = args.eps * v_par / np.linalg.norm(v_par)
                if np.linalg.norm(v_ort) > 0:
                    v_ort = args.eps * v_ort / np.linalg.norm(v_ort)

                errs_par.append(abs(delta_hat(xi + v_par) - di))
                errs_ort.append(abs(delta_hat(xi + v_ort) - di))

            rho = (np.mean(errs_ort) + 1e-12) / (np.mean(errs_par) + 1e-12)

            deff_list.append(deff)
            rho_list.append(rho)

        results[str(k)] = {
            "d_eff_mean": float(np.mean(deff_list)),
            "d_eff_std": float(np.std(deff_list)),
            "rho_median": float(np.median(rho_list)),
            "rho_mean": float(np.mean(rho_list))
        }

    # guardar resultados
    res_path = os.path.join(out_dir, "results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # stage summary
    summary = {
        "stage": "tangentes_locales",
        "timestamp": time.time(),
        "config": vars(args),
        "inputs": {
            "spectrum": spec_path,
            "spectrum_sha256": sha256_file(spec_path)
        },
        "outputs": {
            "results": "outputs/results.json"
        }
    }

    with open(os.path.join(run_dir, "tangentes", "stage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "files": {
            "stage_summary": "stage_summary.json",
            "results": "outputs/results.json"
        }
    }

    with open(os.path.join(run_dir, "tangentes", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
