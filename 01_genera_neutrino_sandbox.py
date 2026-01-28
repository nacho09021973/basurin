#!/usr/bin/env python3
"""BASURIN — Generador 01 (sandbox): observables tipo "neutrino" como proxy espectral.

Objetivo
--------
Genera un dataset sintético *etiquetado* (Δ como parámetro latente) y lo escribe
con el mismo esquema mínimo que espera `04_diccionario.py`:

  runs/<run>/spectrum/outputs/spectrum.h5
    - delta_uv : (n_delta,)
    - m2L2     : (n_delta,)          (placeholder; no usado por Bloque C)
    - M2       : (n_delta, n_modes)  ("masas"/observables por modo)
    - z_grid   : (N,)                (grid auxiliar; no usado por Bloque C)
    - attrs: d, L, n_delta, n_modes, ...

Interpretación
--------------
- Aquí `M2[:, j]` NO proviene de Sturm–Liouville. Es un observable positivo tipo
  integral de línea D_eff = ∫ A(ρ)^2 ds sobre distintos perfiles de densidad.
- Los features que usará `04_diccionario.py` serán ratios r_n = M2_n / M2_0.

Familias disponibles
--------------------
1) `eft_power` (suave, controlable):
   A(ρ) = 1 + alpha * ((ρ/ρ0)^n - 1)

2) `symmetron` (piecewise, adversarial):
   A(ρ) = 1 + alpha_s0*(1 - ρ/ρcrit)   si ρ <= ρcrit
          1                           si ρ >  ρcrit

Mapeo Δ -> parámetro
--------------------
- `--map-mode linear`: alpha = alpha_min + (alpha_max-alpha_min)*t
- `--map-mode quad`  : alpha = alpha_min + (alpha_max-alpha_min)*t^2
  con t = (Δ-Δmin)/(Δmax-Δmin)

Uso
---
  python 01_genera_neutrino_sandbox.py --run ir_neutrino \
    --family eft_power --n-delta 80 --n-modes 6 \
    --profiles vacuum,crust,mantle,core,mixed \
    --noise-rel 0.0

Luego:
  python 04_diccionario.py --run ir_neutrino --enable-c3 --k-features 3 --n-bootstrap 0

Notas BASURIN
-------------
- Este generador vive en "01" por convención: *generadores/sandbox*.
- Es determinista dado `--seed`.
- Es auditable: produce manifest.json y stage_summary.json con hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Config:
    run: str
    family: str
    profiles: List[str]

    # Sweep Δ
    delta_min: float
    delta_max: float
    n_delta: int
    n_alpha: int
    grid_mode: str

    # "Modos" (observables por perfil)
    n_modes: int

    # Grid
    n_grid: int
    s_max: float

    # EFT power
    rho0: float
    power_n: float

    # Symmetron
    rho_crit: float

    # Map Δ -> alpha
    map_mode: str
    alpha_min: float
    alpha_max: float

    # Ruido
    noise_rel: float

    # Meta
    d: int
    L: float
    seed: int
    run_kind: str = "spectrum_only"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="01: Genera sandbox neutrino (proxy espectral) en outputs/spectrum.h5")
    p.add_argument("--run", required=True, type=str, help="Nombre del run (runs/<run>/...) ")
    p.add_argument("--family", type=str, default="eft_power", choices=["eft_power", "symmetron"],
                   help="Familia de A(rho) a usar")
    p.add_argument("--profiles", type=str, default="vacuum,crust,mantle,core,mixed",
                   help="Lista separada por comas de perfiles (vacuum,crust,mantle,core,mixed)")

    # Δ sweep
    p.add_argument("--delta-min", type=float, default=1.55, dest="delta_min")
    p.add_argument("--delta-max", type=float, default=5.50, dest="delta_max")
    p.add_argument("--n-delta", type=int, default=80, dest="n_delta")
    p.add_argument("--n-alpha", type=int, default=None, dest="n_alpha",
                   help="Número de puntos en alpha (default: = n_delta)")
    p.add_argument("--grid-mode", type=str, default="paired", dest="grid_mode",
                   choices=["paired", "cartesian"],
                   help="paired: alpha ~ delta indexado; cartesian: producto cartesiano")

    # modos
    p.add_argument("--n-modes", type=int, default=None, dest="n_modes",
                   help="Número de modos/observables (default: = len(profiles))")

    # grid
    p.add_argument("--n-grid", type=int, default=1024, dest="n_grid")
    p.add_argument("--s-max", type=float, default=1.0, dest="s_max")

    # EFT power
    p.add_argument("--rho0", type=float, default=2.6, help="Referencia rho0 (g/cm^3) (paper: superficie terrestre)")
    p.add_argument("--power-n", type=float, default=1.0, dest="power_n", help="Exponente n en eft_power")

    # symmetron
    p.add_argument("--rho-crit", type=float, default=4.0, dest="rho_crit", help="Densidad crítica symmetron")

    # map
    p.add_argument("--map-mode", type=str, default="linear", choices=["linear", "quad"], dest="map_mode")
    p.add_argument("--alpha-min", type=float, default=-0.05, dest="alpha_min")
    p.add_argument("--alpha-max", type=float, default=0.05, dest="alpha_max")

    # noise
    p.add_argument("--noise-rel", type=float, default=0.0, dest="noise_rel",
                   help="Ruido relativo multiplicativo sobre M2 (std dev). 0 = determinista")

    # meta
    p.add_argument("--d", type=int, default=3, help="Dimensión frontera (solo para compatibilidad attrs)")
    p.add_argument("--L", type=float, default=1.0, help="Escala L (solo para compatibilidad attrs)")
    p.add_argument("--seed", type=int, default=42, help="Semilla RNG")

    a = p.parse_args()
    profiles = [x.strip() for x in a.profiles.split(",") if x.strip()]
    if not profiles:
        p.error("--profiles no puede estar vacío")

    n_modes = a.n_modes if a.n_modes is not None else len(profiles)
    n_alpha = a.n_alpha if a.n_alpha is not None else a.n_delta

    return Config(
        run=a.run,
        family=a.family,
        profiles=profiles,
        delta_min=a.delta_min,
        delta_max=a.delta_max,
        n_delta=a.n_delta,
        n_alpha=n_alpha,
        grid_mode=a.grid_mode,
        n_modes=n_modes,
        n_grid=a.n_grid,
        s_max=a.s_max,
        rho0=a.rho0,
        power_n=a.power_n,
        rho_crit=a.rho_crit,
        map_mode=a.map_mode,
        alpha_min=a.alpha_min,
        alpha_max=a.alpha_max,
        noise_rel=a.noise_rel,
        d=a.d,
        L=a.L,
        seed=a.seed,
    )


def validate(cfg: Config) -> None:
    if not cfg.run:
        raise ValueError("--run es obligatorio")
    if cfg.delta_max <= cfg.delta_min:
        raise ValueError("delta_max debe ser > delta_min")
    if cfg.n_delta < 2:
        raise ValueError("n_delta debe ser >= 2")
    if cfg.n_alpha < 1:
        raise ValueError("n_alpha debe ser >= 1")
    if cfg.n_grid < 16:
        raise ValueError("n_grid debe ser >= 16")
    if cfg.s_max <= 0:
        raise ValueError("s_max debe ser > 0")
    if cfg.n_modes < 2:
        raise ValueError("n_modes debe ser >= 2 (para ratios r_n)")
    if len(cfg.profiles) < cfg.n_modes:
        raise ValueError("len(profiles) debe ser >= n_modes")
    if cfg.L <= 0:
        raise ValueError("L debe ser > 0")


# -----------------------------
# Physics-ish forward
# -----------------------------

def profile_rho(s: np.ndarray, name: str) -> np.ndarray:
    """Perfiles de densidad ρ(s) (g/cm^3) en un parámetro de trayectoria s∈[0,s_max].

    Son proxies controlados (no geofísica). La intención es inducir regímenes:
    - vacuum: ~0
    - crust/mantle/core: constantes
    - mixed: piecewise (para degeneracias)
    """
    name = name.lower()
    if name == "vacuum":
        return np.full_like(s, 1e-6)
    if name == "crust":
        return np.full_like(s, 2.6)
    if name == "mantle":
        return np.full_like(s, 4.5)
    if name == "core":
        return np.full_like(s, 11.0)
    if name == "mixed":
        # 1/3 crust, 1/3 mantle, 1/3 core
        rho = np.empty_like(s)
        n = len(s)
        a = n // 3
        b = 2 * n // 3
        rho[:a] = 2.6
        rho[a:b] = 4.5
        rho[b:] = 11.0
        return rho
    raise ValueError(f"Perfil desconocido: {name}")


def A_eft_power(rho: np.ndarray, *, alpha: float, n: float, rho0: float) -> np.ndarray:
    return 1.0 + alpha * ((rho / rho0) ** n - 1.0)


def A_symmetron_raw(rho: np.ndarray, *, alpha_s0: float, rho_crit: float) -> np.ndarray:
    A = np.ones_like(rho, dtype=np.float64)
    m = rho <= rho_crit
    A[m] = 1.0 + alpha_s0 * (1.0 - rho[m] / rho_crit)
    A[~m] = 1.0
    return A


def A_symmetron_normalized(
    rho: np.ndarray, *, alpha_s0: float, rho_crit: float, rho0: float
) -> tuple[np.ndarray, float]:
    A_raw = A_symmetron_raw(rho, alpha_s0=alpha_s0, rho_crit=rho_crit)
    A0_raw = float(A_symmetron_raw(np.array([rho0], dtype=np.float64), alpha_s0=alpha_s0, rho_crit=rho_crit)[0])
    if not np.isfinite(A0_raw) or abs(A0_raw) <= 1e-12:
        raise ValueError("symmetron A0_raw no finito o demasiado cercano a 0 para normalizar")
    A_norm = A_raw / A0_raw
    return A_norm, A0_raw


def line_integral_A2(s: np.ndarray, A: np.ndarray) -> float:
    # D_eff = ∫ ds A(s)^2
    return float(np.trapezoid(A * A, s))


def delta_to_alpha(delta: float, cfg: Config) -> float:
    t = (delta - cfg.delta_min) / (cfg.delta_max - cfg.delta_min)
    t = float(np.clip(t, 0.0, 1.0))
    if cfg.map_mode == "linear":
        u = t
    else:
        u = t * t
    return cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * u


def A_raw_family(rho: np.ndarray, *, alpha: float, cfg: Config) -> np.ndarray:
    if cfg.family == "eft_power":
        return A_eft_power(rho, alpha=alpha, n=cfg.power_n, rho0=cfg.rho0)
    return A_symmetron_raw(rho, alpha_s0=alpha, rho_crit=cfg.rho_crit)


def normalize_A(rho: np.ndarray, *, alpha: float, cfg: Config) -> tuple[np.ndarray, float]:
    A_raw = A_raw_family(rho, alpha=alpha, cfg=cfg)
    A0_raw = float(A_raw_family(np.array([cfg.rho0], dtype=np.float64), alpha=alpha, cfg=cfg)[0])
    if not np.isfinite(A0_raw) or abs(A0_raw) <= 1e-12:
        raise ValueError("A0_raw no finito o demasiado cercano a 0 para normalizar")
    return A_raw / A0_raw, A0_raw


def write_abort_summary(
    *,
    stage_dir: Path,
    cfg: Config,
    reason: str,
    max_abs_A_minus_1: float | None,
    threshold: float,
    detail: str | None = None,
) -> None:
    payload = {
        "stage": "spectrum",
        "status": "FAIL",
        "reason": reason,
        "contracts": {
            "EFT_DOMAIN": {
                "status": "FAIL",
                "max_abs_A_minus_1": max_abs_A_minus_1,
                "threshold": threshold,
            }
        },
        "normalization": "A := A_raw/A_raw(rho0)",
        "rho0_ref": float(cfg.rho0),
        "out_root": str(stage_dir.parent),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        payload["detail"] = detail
    summary_path = stage_dir / "stage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------
# IO helpers
# -----------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = parse_args()
    try:
        validate(cfg)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    np.random.seed(cfg.seed)

    runs_root = Path(os.environ.get("BASURIN_RUNS_ROOT", "runs"))
    stage_dir = runs_root / cfg.run / "spectrum"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Grid s
    s = np.linspace(0.0, cfg.s_max, cfg.n_grid, dtype=np.float64)

    # Sweep Δ y alpha
    delta_values = np.linspace(cfg.delta_min, cfg.delta_max, cfg.n_delta, dtype=np.float64)
    if cfg.grid_mode == "paired":
        alpha_values = np.array([delta_to_alpha(float(dlt), cfg) for dlt in delta_values], dtype=np.float64)
        delta_per_point = delta_values
        alpha_per_point = alpha_values
        grid_order = "paired_index"
    else:
        alpha_values = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha, dtype=np.float64)
        delta_per_point = np.repeat(delta_values, cfg.n_alpha)
        alpha_per_point = np.tile(alpha_values, cfg.n_delta)
        grid_order = "delta_outer_alpha_inner"

    n_total = int(len(delta_per_point))
    delta_uv = delta_per_point

    # Compute M2 (n_delta, n_modes)
    profiles = cfg.profiles[: cfg.n_modes]
    M2 = np.zeros((n_total, cfg.n_modes), dtype=np.float64)

    # Precompute rho profiles
    rho_profiles = {name: profile_rho(s, name) for name in profiles}

    max_abs_A_minus_1 = 0.0
    max_abs_where: dict | None = None
    clamp_applied = False
    A0_raw_values: list[float] = []
    A0_raw_abort: dict | None = None

    for i, (dlt, alpha) in enumerate(zip(delta_per_point, alpha_per_point)):
        symmetron_A0_raw = None

        for j, pname in enumerate(profiles):
            rho = rho_profiles[pname]
            try:
                A, symmetron_A0_raw = normalize_A(rho, alpha=alpha, cfg=cfg)
            except ValueError:
                A0_raw_abort = {
                    "delta_index": int(i),
                    "delta_uv": float(dlt),
                    "alpha": float(alpha),
                    "profile": pname,
                    "rho0": float(cfg.rho0),
                    "rho_crit": float(cfg.rho_crit),
                    "family": cfg.family,
                }
                break

            abs_delta = np.abs(A - 1.0)
            local_max = float(np.max(abs_delta))
            if local_max > max_abs_A_minus_1:
                idx = int(np.argmax(abs_delta))
                max_abs_A_minus_1 = local_max
                max_abs_where = {
                    "delta_index": int(i),
                    "delta_uv": float(dlt),
                    "alpha": float(alpha),
                    "profile": pname,
                    "s_index": idx,
                    "s_value": float(s[idx]),
                    "rho_value": float(rho[idx]),
                    "A_value": float(A[idx]),
                }

            # Positividad: evitar A<=0 (clamp suave, para no romper log/rel)
            if np.any(A < 1e-6):
                clamp_applied = True
            A = np.maximum(A, 1e-6)

            D_eff = line_integral_A2(s, A)

            # Convertimos D_eff en una cantidad tipo "M^2" positiva y con escala.
            # Escala base para evitar M0 ~ 0.
            base = 1.0
            M2[i, j] = base + D_eff

        A0_raw_values.append(float(symmetron_A0_raw))

        if A0_raw_abort:
            break

    if A0_raw_abort:
        write_abort_summary(
            stage_dir=stage_dir,
            cfg=cfg,
            reason="EFT_DOMAIN_VIOLATION",
            max_abs_A_minus_1=None,
            threshold=1.0,
            detail="A0_raw_nonfinite_or_near_zero",
        )
        print("ABORT: EFT_DOMAIN_VIOLATION (A0_raw non-finite/near-zero)", file=sys.stderr)
        return 1

    # Ruido relativo multiplicativo (opcional)
    if cfg.noise_rel > 0:
        noise = np.random.normal(loc=0.0, scale=cfg.noise_rel, size=M2.shape)
        M2 = M2 * (1.0 + noise)
        M2 = np.maximum(M2, 1e-8)

    # Placeholder m2L2 (no usado por 04, pero requerido por el loader)
    m2L2 = np.zeros(n_total, dtype=np.float64)

    # Validación básica
    validation = {
        "finite_ok": bool(np.isfinite(M2).all() and np.isfinite(delta_uv).all()),
        "positive_ok": bool((M2 > 0).all()),
        "M0_min": float(np.min(M2[:, 0])),
        "M0_max": float(np.max(M2[:, 0])),
        "ratio_min_max": None,
        "notes": "M2 es un observable proxy (integral de A^2). No es SL.",
        "eft_domain_max_abs_delta": float(max_abs_A_minus_1),
        "eft_domain_threshold": 1.0,
        "eft_domain_clamp_applied": bool(clamp_applied),
        "symmetron_normalize_by_A0": bool(cfg.family == "symmetron"),
        "symmetron_A0_raw": A0_raw_values if cfg.family == "symmetron" else None,
    }
    ratios = M2[:, 1:] / np.maximum(M2[:, [0]], 1e-12)
    validation["ratio_min_max"] = [float(np.min(ratios)), float(np.max(ratios))]

    if not (validation["finite_ok"] and validation["positive_ok"]):
        print("ERROR: dataset inválido (NaN/Inf o no-positivo)", file=sys.stderr)
        return 1

    if max_abs_A_minus_1 >= 1.0:
        write_abort_summary(
            stage_dir=stage_dir,
            cfg=cfg,
            reason="EFT_DOMAIN_VIOLATION",
            max_abs_A_minus_1=float(max_abs_A_minus_1),
            threshold=1.0,
        )
        print("ABORT: EFT_DOMAIN_VIOLATION (|A-1| >= 1)", file=sys.stderr)
        return 1

    # Write spectrum.h5
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        return 1

    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    h5_path = outputs_dir / "spectrum.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("z_grid", data=s)  # compat: 04 espera z_grid
        h5.create_dataset("delta_uv", data=delta_uv)
        h5.create_dataset("m2L2", data=m2L2)
        h5.create_dataset("M2", data=M2)

        # attrs esperados por 04_diccionario.load_spectrum
        h5.attrs["d"] = int(cfg.d)
        h5.attrs["L"] = float(cfg.L)
        h5.attrs["n_delta"] = int(n_total)
        h5.attrs["n_modes"] = int(cfg.n_modes)

        # extras de auditoría
        h5.attrs["generator"] = "01_genera_neutrino_sandbox.py"
        h5.attrs["family"] = cfg.family
        h5.attrs["profiles"] = ",".join(profiles)
        h5.attrs["map_mode"] = cfg.map_mode
        h5.attrs["alpha_min"] = float(cfg.alpha_min)
        h5.attrs["alpha_max"] = float(cfg.alpha_max)
        h5.attrs["rho0"] = float(cfg.rho0)
        h5.attrs["power_n"] = float(cfg.power_n)
        h5.attrs["rho_crit"] = float(cfg.rho_crit)
        h5.attrs["noise_rel"] = float(cfg.noise_rel)
        h5.attrs["seed"] = int(cfg.seed)
        h5.attrs["symmetron_normalize_by_A0"] = bool(cfg.family == "symmetron")
        if A0_raw_values:
            h5.attrs["symmetron_A0_raw_min"] = float(np.min(A0_raw_values))
            h5.attrs["symmetron_A0_raw_max"] = float(np.max(A0_raw_values))
        h5.attrs["created"] = datetime.now(timezone.utc).isoformat()

    # Write validation.json
    val_path = outputs_dir / "validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    # stage_summary.json
    summary = {
        "stage": "spectrum",
        "script": "01_genera_neutrino_sandbox.py",
        "version": "0.1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "run": cfg.run,
        "config": {**asdict(cfg), "n_total": n_total},
        "inputs": {
            "generator": "neutrino_sandbox",
            "generator_script": "01_genera_neutrino_sandbox.py",
            "generator_seed": cfg.seed,
            "generator_config": asdict(cfg),
        },
        "outputs": {
            "spectrum_h5": "outputs/spectrum.h5",
            "validation": "outputs/validation.json",
        },
        "status": "PASS",
        "contracts": {
            "EFT_DOMAIN": {
                "status": "PASS",
                "max_abs_A_minus_1": float(max_abs_A_minus_1),
                "threshold": 1.0,
            }
        },
        "normalization": "A := A_raw/A_raw(rho0)",
        "rho0_ref": float(cfg.rho0),
        "symmetron_normalize_by_A0": bool(cfg.family == "symmetron"),
        "symmetron_A0_raw": A0_raw_values if cfg.family == "symmetron" else None,
        "hashes": {
            "outputs/spectrum.h5": sha256_file(h5_path),
            "outputs/validation.json": sha256_file(val_path),
        },
        "compatibility": {
            "consumer": "04_diccionario.load_spectrum",
            "note": "Esquema mínimo compatible: delta_uv, m2L2, M2, z_grid y attrs d,L,n_delta,n_modes.",
        },
        "grid": {
            "mode": cfg.grid_mode,
            "n_alpha": int(cfg.n_alpha),
            "n_delta": int(cfg.n_delta),
            "n_total": int(n_total),
            "alpha_values": [float(x) for x in alpha_values],
            "delta_values": [float(x) for x in delta_values],
            "alpha_per_point": [float(x) for x in alpha_per_point],
            "delta_per_point": [float(x) for x in delta_per_point],
            "order": grid_order,
        },
    }
    summary_path = stage_dir / "stage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # manifest.json
    manifest = {
        "stage": "spectrum",
        "run": cfg.run,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {
            "spectrum": "outputs/spectrum.h5",
            "validation": "outputs/validation.json",
            "summary": "stage_summary.json",
        },
    }
    manifest_path = stage_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Console summary
    print("OK: dataset generado")
    print(f"  run: {cfg.run}")
    print(f"  out: {h5_path}")
    print(f"  family: {cfg.family}")
    print(f"  profiles (modes): {profiles}")
    print(f"  Δ in [{cfg.delta_min:.3f}, {cfg.delta_max:.3f}], n_delta={cfg.n_delta}")
    print(f"  grid_mode={cfg.grid_mode}, n_alpha={cfg.n_alpha}, n_total={n_total}")
    print(f"  n_modes={cfg.n_modes}, noise_rel={cfg.noise_rel}")
    print(f"  ratio range: [{validation['ratio_min_max'][0]:.6f}, {validation['ratio_min_max'][1]:.6f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
