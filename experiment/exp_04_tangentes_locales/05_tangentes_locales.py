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
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import h5py


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def compute_ratio_features(masses, k_features, mass_kind):
    m0 = masses[:, 0]
    X = []
    for n in range(1, k_features + 1):
        if mass_kind == "masses":
            X.append((masses[:, n] ** 2) / (m0 ** 2))
        elif mass_kind == "M2":
            X.append(masses[:, n] / m0)
        else:
            raise ValueError(f"mass_kind inválido: {mass_kind}")
    return np.stack(X, axis=1)


def resolve_spectrum_path(run_dir: Path) -> Path:
    stage_dir = run_dir / "spectrum"
    candidate = stage_dir / "outputs" / "spectrum.h5"
    if candidate.exists():
        return candidate
    return stage_dir / "spectrum.h5"


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
    ap.add_argument("--eps-floor", type=float, default=1e-12)
    ap.add_argument("--standardize", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rho-threshold", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run
    spec_path = resolve_spectrum_path(run_dir)
    stage_dir = run_dir / "tangentes_locales"
    out_dir = stage_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(spec_path, "r") as h5:
        if "masses" in h5:
            mass_dataset = "masses"
            mass_kind = "masses"
        elif "M2" in h5:
            mass_dataset = "M2"
            mass_kind = "M2"
        else:
            available = ", ".join(sorted(h5.keys()))
            raise ValueError(
                "spectrum.h5 debe incluir dataset 'masses' o 'M2'. "
                f"Disponibles: {available}"
            )
        masses = h5[mass_dataset][:]
        delta = h5["delta_uv"][:]

    X = compute_ratio_features(masses, args.k_features, mass_kind)

    if masses.ndim != 2:
        raise ValueError("masses debe ser 2D (N, n_features).")
    if delta.ndim != 1:
        raise ValueError("delta_uv debe ser 1D (N,).")
    if masses.shape[0] != delta.shape[0]:
        raise ValueError("masses y delta_uv deben tener el mismo número de filas.")
    if X.shape[0] != delta.shape[0]:
        raise ValueError("X y delta_uv deben tener el mismo número de filas.")
    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o Inf.")
    if args.k_neighbors is None:
        raise ValueError("k_neighbors no puede ser None.")
    if args.eps <= 0:
        raise ValueError("eps debe ser > 0.")
    if args.eps_floor <= 0:
        raise ValueError("eps_floor debe ser > 0.")
    if args.n_perturb <= 0:
        raise ValueError("n_perturb debe ser > 0.")

    if args.standardize == "on":
        mu = X.mean(axis=0)
        sig = X.std(axis=0) + 1e-12
        X = (X - mu) / sig

    N = X.shape[0]
    rng = np.random.default_rng(args.seed)
    idx_points = rng.choice(N, size=min(args.n_points, N), replace=False)

    k_list = [int(x) for x in args.k_neighbors.split(",")]
    if any(k < 2 for k in k_list):
        raise ValueError("k_neighbors debe ser >= 2.")
    if any(k >= N for k in k_list):
        raise ValueError("k_neighbors debe ser < N.")

    feature_columns = ["d_eff", "m", "parallel", "perp", "rho_clipped", "log10_rho"]
    feature_key = "tangentes_locales_v1"

    results = {}
    per_point_outputs = {}
    ids = [f"idx_{i}" for i in idx_points]

    for k in k_list:
        deff_list = []
        rho_list = []
        rho_clipped_list = []
        log10_rho_list = []
        m_list = []
        parallel_means = []
        per_point_rows = []

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
            m_list.append(m)

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

            perp = np.mean(errs_ort)
            parallel = np.mean(errs_par)
            rho_clipped = perp / max(parallel, args.eps_floor)
            rho = rho_clipped
            parallel_means.append(parallel)

            deff_list.append(deff)
            rho_list.append(rho)
            rho_clipped_list.append(rho_clipped)
            log10_rho_list.append(np.log10(rho_clipped))
            per_point_rows.append(
                [
                    float(deff),
                    float(m),
                    float(parallel),
                    float(perp),
                    float(rho_clipped),
                    float(np.log10(rho_clipped)),
                ]
            )

        frac_parallel_below_floor = float(
            np.mean(np.array(parallel_means) < args.eps_floor)
        )

        results[str(k)] = {
            "d_eff_mean": float(np.mean(deff_list)),
            "d_eff_std": float(np.std(deff_list)),
            "d_eff_p10": float(np.percentile(deff_list, 10)),
            "d_eff_p50": float(np.percentile(deff_list, 50)),
            "d_eff_p90": float(np.percentile(deff_list, 90)),
            "m_mean": float(np.mean(m_list)),
            "m_std": float(np.std(m_list)),
            "rho_median": float(np.median(rho_list)),
            "rho_mean": float(np.mean(rho_list)),
            "rho_p10": float(np.percentile(rho_list, 10)),
            "rho_p50": float(np.percentile(rho_list, 50)),
            "rho_p90": float(np.percentile(rho_list, 90)),
            "eps_floor": float(args.eps_floor),
            "rho_clipped_mean": float(np.mean(rho_clipped_list)),
            "rho_clipped_p10": float(np.percentile(rho_clipped_list, 10)),
            "rho_clipped_p50": float(np.percentile(rho_clipped_list, 50)),
            "rho_clipped_p90": float(np.percentile(rho_clipped_list, 90)),
            "log10_rho_p10": float(np.percentile(log10_rho_list, 10)),
            "log10_rho_p50": float(np.percentile(log10_rho_list, 50)),
            "log10_rho_p90": float(np.percentile(log10_rho_list, 90)),
            "frac_parallel_below_floor": frac_parallel_below_floor,
            "n_samples": int(len(rho_list))
        }
        per_point_outputs[str(k)] = {
            "ids": ids,
            "Y": per_point_rows,
            "meta": {
                "feature_key": feature_key,
                "columns": feature_columns,
                "k_neighbors": int(k),
                "n_points": int(len(ids)),
                "schema_version": "1",
                "created": utc_now_iso(),
                "source_stage": "tangentes_locales",
            },
        }

    # guardar resultados
    res_path = out_dir / "results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    features_paths = {}
    for k, payload in per_point_outputs.items():
        features_path = out_dir / f"features_points_k{k}.json"
        with open(features_path, "w") as f:
            json.dump(payload, f, indent=2)
        features_paths[k] = f"outputs/features_points_k{k}.json"

    # stage summary
    summary = {
        "stage": "tangentes_locales",
        "timestamp": utc_now_iso(),
        "config": vars(args),
        "numpy_version": np.__version__,
        "notes": {
            "effective_dimension_criterion": (
                "d_eff = (sum(evals)**2) / sum(evals**2); m = max(1, round(d_eff))"
            ),
            "eps_floor": float(args.eps_floor),
            "rho_clipped_definition": (
                "rho_clipped = perp / max(parallel, eps_floor); log10_rho = log10(rho_clipped)"
            )
        },
        "interpretation": {
            "rho >> 1": "predominantly orthogonal variation to local tangent",
            "rho ~ 1": "no clear tangent structure"
        },
        "inputs": {
            "spectrum": str(spec_path),
            "mass_dataset": mass_dataset,
            "mass_kind": mass_kind,
            "mass_dataset_path": f"/{mass_dataset}",
            "spectrum_sha256": sha256_file(spec_path) if spec_path.exists() else None,
            "script_sha256": sha256_file(Path(__file__))
        },
        "outputs": {
            "results": "outputs/results.json",
            "features_points": features_paths,
        }
    }

    stage_summary_path = stage_dir / "stage_summary.json"
    with open(stage_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    manifest_path = stage_dir / "manifest.json"
    manifest_files = {
        "outputs/results.json": sha256_file(res_path),
        "stage_summary.json": sha256_file(stage_summary_path)
    }
    for k in k_list:
        features_path = out_dir / f"features_points_k{k}.json"
        manifest_files[f"outputs/features_points_k{k}.json"] = sha256_file(features_path)
    manifest = {"files": manifest_files}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    for k in k_list:
        entry = results[str(k)]
        print(
            "k={k} d_eff_p50={d_eff_p50:.4f} rho_p50={rho_p50:.4f} "
            "log10_rho_p50={log10_rho_p50:.4f} frac_parallel_below_floor={frac:.4f}"
            .format(
                k=k,
                d_eff_p50=entry["d_eff_p50"],
                rho_p50=entry["rho_p50"],
                log10_rho_p50=entry["log10_rho_p50"],
                frac=entry["frac_parallel_below_floor"],
            )
        )
    print(f"Stage written to: {stage_dir}")


if __name__ == "__main__":
    main()
