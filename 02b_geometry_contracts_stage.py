#!/usr/bin/env python3
"""BASURIN — Stage: Contratos geométricos/espectrales post-hoc (Fase 2)

Este stage evalúa validadores *post-hoc* sobre la geometría reconstruida y,
si existe, sobre el espectro SL (Bloque B).

IO (determinista, contrato BASURIN):
  runs/<run>/geometry_contracts/
    - manifest.json
    - stage_summary.json
    - outputs/
        - contracts.json

Inputs esperados:
  - Geometría: runs/<run>/geometry/<geometry-file>  (H5)
    datasets: z_grid, A_of_z, f_of_z; attrs: d, L
  - (Opcional) Espectro: runs/<run>/spectrum/outputs/spectrum.h5
    datasets: M2 (o M2_D), attrs: d, L

Gauge (confirmado):
  ds^2 = e^{2A(z)}[-f(z)dt^2 + d\vec{x}^2 + dz^2/f(z)]

Notas teóricas importantes:
  - NEC y c-theorem se evalúan de forma invariante en coordenada radial propia r,
    implementado sin reparametrizar manualmente toda la malla.

Uso:
  python 02b_geometry_contracts_stage.py --run <run_id> [--geometry-file ads_puro.h5]

"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Contratos (módulo puro, sin IO)
# Nota: el nombre del fichero empieza por dígitos; lo cargamos por ruta.
import importlib.util

_CONTRACTS_PATH = Path(__file__).with_name('02b_contratos_geometricos.py')
_spec = importlib.util.spec_from_file_location('basurin_geom_contracts', _CONTRACTS_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise RuntimeError(f'No se pudo cargar {_CONTRACTS_PATH}')
_geom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geom)  # type: ignore

check_uv_ads = _geom.check_uv_ads
check_nec_einstein = _geom.check_nec_einstein
check_c_theorem = _geom.check_c_theorem
check_horizon_regularity = _geom.check_horizon_regularity
check_regge_trajectory = _geom.check_regge_trajectory


# -----------------------------
# Utilidades IO BASURIN
# -----------------------------

def compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -----------------------------
# Lectura de inputs
# -----------------------------

def load_geometry(h5_path: Path) -> Dict[str, Any]:
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        sys.exit(1)

    with h5py.File(h5_path, "r") as h5:
        z = h5["z_grid"][:]
        A = h5["A_of_z"][:]
        f = h5["f_of_z"][:]
        d = int(h5.attrs["d"])
        L = float(h5.attrs["L"])
        family = h5.attrs.get("family", "unknown")

    return {
        "z": z,
        "A": A,
        "f": f,
        "d": d,
        "L": L,
        "N": int(len(z)),
        "z_min": float(z[0]),
        "z_max": float(z[-1]),
        "family": family,
    }


def load_spectrum_M2(h5_path: Path) -> Optional[np.ndarray]:
    """Carga M2 para contrato Regge. Devuelve None si no hay archivo."""
    if not h5_path.exists():
        return None

    try:
        import h5py
    except ImportError:
        return None

    with h5py.File(h5_path, "r") as h5:
        if "M2" in h5:
            return h5["M2"][:]
        if "M2_D" in h5:
            return h5["M2_D"][:]

    return None


# -----------------------------
# Geometría diferencial mínima
# -----------------------------

def compute_ricci_scalar(z: np.ndarray, A: np.ndarray, f: np.ndarray, d: int) -> np.ndarray:
    """Escalar de Ricci para el gauge conforme.

    R = e^{-2A}[-2 d A'' - d(d-1)(A')^2 - d (f'/f) A']
    """
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    # Permite z no uniforme
    A1 = np.gradient(A, z)
    A2 = np.gradient(A1, z)
    f1 = np.gradient(f, z)
    f_safe = np.where(np.abs(f) > 1e-12, f, 1e-12)

    R = np.exp(-2.0 * A) * (
        -2.0 * d * A2
        - d * (d - 1) * (A1 ** 2)
        - d * (f1 / f_safe) * A1
    )
    return R


def ricci_uv_stats(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    d: int,
    L: float,
    n_uv_points: int = 20,
    *,
    gauge_assumed: str = "domain_wall_conformal_z",
    gauge_definition: str = "ds^2=e^{2A}[-f dt^2+dx^2+dz^2/f] (B=A)",
    dynamic_class: str = "none",
) -> Dict[str, Any]:
    n = min(int(n_uv_points), len(z))
    if n < 5:
        return {
            "status": "SKIP",
            "reason": "insuficientes puntos UV",
            "contract_class": "geometric_pure",
            "gauge_assumed": gauge_assumed,
            "gauge_definition": gauge_definition,
            "dynamic_class": dynamic_class,
            "requires": [],
        }

    R = compute_ricci_scalar(z[:n], A[:n], f[:n], d)
    R_expected = -d * (d + 1) / (L * L)

    rel = np.abs(R - R_expected) / max(abs(R_expected), 1e-30)

    status = "PASS" if float(np.median(rel)) < 0.01 else "FAIL"
    return {
        "status": status,
        "R_expected": float(R_expected),
        "R_median": float(np.median(R)),
        "R_p10": float(np.percentile(R, 10)),
        "R_p90": float(np.percentile(R, 90)),
        "rel_error_R_median": float(np.median(rel)),
        "rel_error_R_p90": float(np.percentile(rel, 90)),
        "n_uv_points": int(n),
        "notes": "Check UV Ricci ~ -d(d+1)/L^2 (sensibles a discretización; usar como auditor complementario)",
        "contract_class": "geometric_pure",
        "gauge_assumed": gauge_assumed,
        "gauge_definition": gauge_definition,
        "dynamic_class": dynamic_class,
        "requires": [],
    }


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Config:
    run: str
    geometry_file: str = "ads_puro.h5"
    # Contracts params
    n_uv_points: int = 20
    margin_frac: float = 0.10
    smooth_window: int = 11
    poly_order: int = 3
    method: str = "auto"  # auto|savgol|spline (en el módulo)
    spline_s: Optional[float] = None
    f_floor: float = 1e-6
    nec_tol: float = 0.0
    c_tol: float = 0.0
    regge_skip_low: int = 3
    regge_r2_threshold: float = 0.95

    # Declarativos
    gauge: str = "domain_wall_conformal_z"
    dynamic_class_assumed: str = "Einstein_2deriv_NEC"  # para B,C


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="BASURIN: stage de contratos geométricos post-hoc")
    p.add_argument("--run", required=True, type=str, help="Nombre del run")
    p.add_argument("--geometry-file", default="ads_puro.h5", type=str, dest="geometry_file")

    # Contratos
    p.add_argument("--n-uv-points", default=20, type=int, dest="n_uv_points")
    p.add_argument("--margin-frac", default=0.10, type=float, dest="margin_frac")
    p.add_argument("--smooth-window", default=11, type=int, dest="smooth_window")
    p.add_argument("--poly-order", default=3, type=int, dest="poly_order")
    p.add_argument("--method", default="auto", choices=["auto", "savgol", "spline"], dest="method")
    p.add_argument("--spline-s", default=None, type=float, dest="spline_s")
    p.add_argument("--f-floor", default=1e-6, type=float, dest="f_floor")
    p.add_argument("--nec-tol", default=0.0, type=float, dest="nec_tol")
    p.add_argument("--c-tol", default=0.0, type=float, dest="c_tol")
    p.add_argument("--regge-skip-low", default=3, type=int, dest="regge_skip_low")
    p.add_argument("--regge-r2-threshold", default=0.95, type=float, dest="regge_r2_threshold")

    args = p.parse_args()
    return Config(**{k: v for k, v in vars(args).items()})


# -----------------------------
# Stage
# -----------------------------

def run_stage(cfg: Config) -> Dict[str, Path]:
    run_dir = Path("runs") / cfg.run
    geo_path = run_dir / "geometry" / cfg.geometry_file

    if not geo_path.exists():
        raise FileNotFoundError(f"No existe geometría: {geo_path}")

    geo = load_geometry(geo_path)
    z, A, f = geo["z"], geo["A"], geo["f"]
    d, L = geo["d"], geo["L"]

    # Optional spectrum
    spec_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    M2 = load_spectrum_M2(spec_path)

    # Contratos
    contracts: Dict[str, Any] = {}

    contracts["A_uv_ads"] = check_uv_ads(
        z, A, f, d=d, L=L, n_uv_points=cfg.n_uv_points
    )
    # Auditor complementario: Ricci UV
    contracts["A_uv_ricci"] = ricci_uv_stats(
        z,
        A,
        f,
        d=d,
        L=L,
        n_uv_points=cfg.n_uv_points,
        gauge_assumed=cfg.gauge,
        gauge_definition="ds^2=e^{2A}[-f dt^2+d\vec{x}^2+dz^2/f] (B=A)",
        dynamic_class="none",
    )

    contracts["B_nec"] = check_nec_einstein(
        z,
        A,
        f,
        smooth_window=cfg.smooth_window,
        poly_order=cfg.poly_order,
        method=cfg.method,
        spline_s=cfg.spline_s,
        margin_frac=cfg.margin_frac,
        f_floor=cfg.f_floor,
        tol=cfg.nec_tol,
    )

    contracts["C_c_theorem"] = check_c_theorem(
        z,
        A,
        f,
        d=d,
        smooth_window=cfg.smooth_window,
        poly_order=cfg.poly_order,
        method=cfg.method,
        spline_s=cfg.spline_s,
        margin_frac=cfg.margin_frac,
        f_floor=cfg.f_floor,
        tol=cfg.c_tol,
    )

    contracts["E_horizon"] = check_horizon_regularity(
        z,
        A,
        f,
        smooth_window=cfg.smooth_window,
        poly_order=cfg.poly_order,
        method=cfg.method,
        spline_s=cfg.spline_s,
    )

    if M2 is None:
        contracts["F_regge"] = {"status": "SKIP", "reason": f"No existe {spec_path}"}
    else:
        regge = check_regge_trajectory(M2, n_skip_low=cfg.regge_skip_low, r2_threshold=cfg.regge_r2_threshold)
        regge["spectrum_source"] = "../spectrum/outputs/spectrum.h5"
        contracts["F_regge"] = regge

    # IO
    stage_dir = run_dir / "geometry_contracts"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    contracts_path = outputs_dir / "contracts.json"
    write_json(contracts_path, contracts)

    # stage_summary
    summary = {
        "stage": "02b_geometry_contracts",
        "version": "0.1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "geometry": {
            "d": d,
            "L": L,
            "N": geo["N"],
            "z_min": geo["z_min"],
            "z_max": geo["z_max"],
            "family": geo["family"],
            "source": f"../geometry/{cfg.geometry_file}",
        },
        "inputs": {
            "geometry_h5": f"../geometry/{cfg.geometry_file}",
            "spectrum_h5": "../spectrum/outputs/spectrum.h5" if spec_path.exists() else None,
        },
        "contracts": {k: v.get("status", "UNKNOWN") if isinstance(v, dict) else "UNKNOWN" for k, v in contracts.items()},
        "contract_groups": {
            "geometric_pure": ["A_uv_ads", "A_uv_ricci", "E_horizon"],
            "einstein_nec": ["B_nec", "C_c_theorem"],
            "spectral_ir": ["F_regge"],
        },
        "interpretation_notes": {
            "geometric_pure": "No asume ecuaciones de Einstein; checks de consistencia geométrica y gauge.",
            "einstein_nec": "Asume Einstein (2 derivadas) + materia que satisface NEC; valida realizabilidad dentro de esa clase.",
            "spectral_ir": "Depende de espectro SL (Bloque B) y diagnostica asintótica IR (Regge/soft-wall).",
        },
        "hashes": {
            "outputs/contracts.json": compute_file_hash(contracts_path),
        },
    }

    summary_path = stage_dir / "stage_summary.json"
    write_json(summary_path, summary)

    # manifest
    manifest = {
        "stage": "02b_geometry_contracts",
        "run": cfg.run,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {
            "contracts": "outputs/contracts.json",
            "summary": "stage_summary.json",
        },
        "input_geometry": f"../geometry/{cfg.geometry_file}",
        "input_spectrum": "../spectrum/outputs/spectrum.h5" if spec_path.exists() else None,
    }

    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest)

    return {
        "contracts": contracts_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> int:
    cfg = parse_args()
    try:
        out = run_stage(cfg)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("=== Contratos geométricos post-hoc (Fase 2) ===")
    print(f"run: {cfg.run}")
    print(f"outputs: {out['contracts']}")
    print(f"summary:  {out['summary']}")
    print(f"manifest: {out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
