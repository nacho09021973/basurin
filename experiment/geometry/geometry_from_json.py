"""Geometry compiler for explicit geometry.json inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from basurin_io import sha256_file

__version__ = "0.1.0"

DEFAULT_Z_MIN = 0.01
DEFAULT_Z_MAX = 1.0


def _expect_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} debe ser numérico")
    return float(value)


def _resolve_dimension(geom: dict[str, Any]) -> int:
    for key in ("d", "boundary_dimension", "dimension"):
        if key in geom:
            value = geom[key]
            if not isinstance(value, int):
                raise ValueError(f"{key} debe ser entero")
            if value < 2:
                raise ValueError(f"{key} debe ser >= 2")
            return value
    raise ValueError("Falta dimensión: espera d, boundary_dimension o dimension")


def load_geometry_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"geometry.json no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("geometry.json debe ser un objeto JSON")

    geometry_type = payload.get("geometry_type") or payload.get("type")
    if not isinstance(geometry_type, str) or not geometry_type:
        raise ValueError("geometry_type es obligatorio (string)")

    if geometry_type != "ads_like_minimal":
        raise ValueError(
            "geometry_type no soportado: "
            f"{geometry_type}. Solo 'ads_like_minimal' en este compiler."
        )

    d = _resolve_dimension(payload)

    # Prefer schema: payload["parameters"]
    params = payload.get("parameters", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("parameters debe ser un objeto")

    # L: prefer parameters.L, fallback legacy top-level L
    L_val = params.get("L", payload.get("L"))
    L = _expect_number(L_val, "L")
    if L <= 0:
        raise ValueError("L debe ser > 0")

    # z_min / z_max: prefer parameters.{z_min,z_max}, fallback legacy top-level
    z_min = params.get("z_min", payload.get("z_min"))
    if z_min is not None:
        z_min = _expect_number(z_min, "z_min")
        if z_min <= 0:
            raise ValueError("z_min debe ser > 0")

    z_max = params.get("z_max", payload.get("z_max"))
    if z_max is not None:
        z_max = _expect_number(z_max, "z_max")

    if z_min is not None and z_max is not None and z_max <= z_min:
        raise ValueError("z_max debe ser > z_min")

    return {
        "geometry_type": geometry_type,
        "d": d,
        "L": L,
        "z_min": z_min,
        "z_max": z_max,
        "parameters": params,   # <-- añade esto
        "raw": payload,
    }


def compile_geometry_numeric(
    geom: dict[str, Any],
    n_z: int,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    d = _resolve_dimension(geom)
    if n_z < 5:
        raise ValueError("n_z debe ser >= 5")
    if z_min <= 0:
        raise ValueError("z_min debe ser > 0")
    if z_max <= z_min:
        raise ValueError("z_max debe ser > z_min")

    if geom.get("geometry_type") != "ads_like_minimal":
        raise ValueError("Solo se soporta geometry_type=ads_like_minimal")
    params = geom.get("parameters", {})
    if "L" in params:
        L_raw = params.get("L")
        schema_used = "parameters"
    elif "L" in geom:
        L_raw = geom.get("L")
        schema_used = "legacy_top_level"
    else:
        raise ValueError("L faltante: se esperaba parameters.L (o legacy L)")

    try:
        L = float(L_raw)
    except Exception as e:
        raise ValueError("L debe ser numérico") from e

    z = np.linspace(z_min, z_max, n_z, dtype=np.float64)
    A = np.log(L / z)
    f = np.ones_like(z)

    return {
        "z": [float(v) for v in z],
        "A": [float(v) for v in A],
        "f": [float(v) for v in f],
        "d": d,
        "L": L,
        "N": int(n_z),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "family": "ads_like_minimal",
        "geometry_type": geom["geometry_type"],
    }


def write_geometry_numeric(out_path: Path, payload: dict[str, Any]) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return sha256_file(out_path)
