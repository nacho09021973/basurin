#!/usr/bin/env python3
"""
Stage BASURIN (experimental): OPE coefficients proxy via bulk overlap.

Contrato:
  runs/<run_id>/experiment/ope_coefficients_bulk_overlap/

Inputs esperados:
  - runs/<run>/RUN_VALID/outputs/run_valid.json (gate obligatorio)
  - runs/<run>/spectrum/outputs/spectrum.h5 con datasets:
      - z_grid
      - phi (autofunciones)
      - delta_uv
  - (opcional) runs/<run>/geometry/outputs/geometry.h5 para medida

Outputs obligatorios:
  - manifest.json
  - stage_summary.json
  - outputs/ope_coefficients.json
  - outputs/verdict.json
  - outputs/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "experiment/ope_coefficients_bulk_overlap"
SCHEMA_VERSION = "ope_coefficients_bulk_overlap_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _list_datasets(h5: h5py.File) -> list[str]:
    datasets: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5.visititems(visitor)
    return datasets


def _select_dataset(dataset_paths: Iterable[str], candidates: list[str]) -> str | None:
    ordered_paths = sorted(dataset_paths)
    for candidate in candidates:
        for path in ordered_paths:
            if Path(path).name == candidate:
                return path
    return None


def _load_spectrum_inputs(spectrum_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    missing: list[str] = []
    with h5py.File(spectrum_path, "r") as h5:
        dataset_paths = _list_datasets(h5)
        delta_path = _select_dataset(dataset_paths, ["delta_uv", "Delta", "delta", "delta_grid"])
        z_path = _select_dataset(dataset_paths, ["z_grid", "z"])
        phi_path = _select_dataset(dataset_paths, ["phi"])

        if delta_path is None:
            missing.append("delta_uv")
            delta = np.array([], dtype=float)
        else:
            delta = np.asarray(h5[delta_path][...], dtype=float).ravel()

        if z_path is None:
            missing.append("z_grid")
            z_grid = np.array([], dtype=float)
        else:
            z_grid = np.asarray(h5[z_path][...], dtype=float).ravel()

        if phi_path is None:
            missing.append("phi")
            phi = np.array([], dtype=float)
        else:
            phi_raw = np.asarray(h5[phi_path][...], dtype=float)
            if phi_raw.ndim == 2:
                phi = phi_raw
            elif phi_raw.ndim == 3:
                phi = phi_raw[:, :, 0]
            else:
                missing.append("phi")
                phi = np.array([], dtype=float)

    return z_grid, phi, delta, missing


def _load_geometry_weight(
    geometry_path: Path,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    assumptions: dict[str, Any] = {}
    if not geometry_path.exists():
        assumptions["measure_fallback"] = "geometry_missing"
        return np.ones_like(z_grid), assumptions

    try:
        with h5py.File(geometry_path, "r") as h5:
            if "A_of_z" not in h5 or "z_grid" not in h5:
                assumptions["measure_fallback"] = "geometry_missing_keys"
                return np.ones_like(z_grid), assumptions
            geo_z = np.asarray(h5["z_grid"][...], dtype=float).ravel()
            A_of_z = np.asarray(h5["A_of_z"][...], dtype=float).ravel()
            d_attr = h5.attrs.get("d")
            if d_attr is None:
                assumptions["measure_fallback"] = "geometry_missing_d"
                return np.ones_like(z_grid), assumptions
            d = float(d_attr)
    except OSError:
        assumptions["measure_fallback"] = "geometry_unreadable"
        return np.ones_like(z_grid), assumptions

    if geo_z.size != A_of_z.size or geo_z.size == 0:
        assumptions["measure_fallback"] = "geometry_invalid_grid"
        return np.ones_like(z_grid), assumptions

    A_interp = np.interp(z_grid, geo_z, A_of_z)
    weight = np.exp((d - 1.0) * A_interp)
    assumptions["geometry_weight"] = "w(z)=exp((d-1)*A(z))"
    assumptions["geometry_d"] = d
    return weight, assumptions


def _integrate(z_grid: np.ndarray, values: np.ndarray, method: str) -> tuple[float, str]:
    if z_grid.size == 0:
        return 0.0, "empty"
    def trapz() -> float:
        diffs = np.diff(z_grid)
        if diffs.size == 0:
            return 0.0
        return float(np.sum((values[:-1] + values[1:]) * diffs / 2.0))

    if method == "simpson":
        n = z_grid.size
        if n < 3 or n % 2 == 0:
            return trapz(), "trapz_fallback"
        h = (z_grid[-1] - z_grid[0]) / (n - 1)
        coeffs = np.ones(n)
        coeffs[1:-1:2] = 4
        coeffs[2:-1:2] = 2
        return float(h / 3.0 * np.sum(coeffs * values)), "simpson"
    return trapz(), "trapz"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPE coefficients proxy via bulk overlap")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--runs-root", default="runs", help="Runs root path")
    parser.add_argument("--n-light", type=int, default=3, help="Número de operadores light")
    parser.add_argument("--n-tower", type=int, default=5, help="Niveles en torre sintética")
    parser.add_argument(
        "--integration",
        choices=["trapz", "simpson"],
        default="trapz",
        help="Método de integración",
    )
    parser.add_argument(
        "--measure",
        choices=["flat", "geometry"],
        default="flat",
        help="Medida de integración",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runs_root = resolve_out_root(args.runs_root)
        validate_run_id(args.run, runs_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_dir = runs_root / args.run
    try:
        run_valid_payload = require_run_valid(runs_root, args.run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    if not spectrum_path.exists():
        print(f"ERROR: spectrum.h5 missing at {spectrum_path}", file=sys.stderr)
        return 2

    try:
        assert_within_runs(run_dir, spectrum_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=runs_root)

    created_utc = datetime.now(timezone.utc).isoformat()
    z_grid, phi, delta_uv, missing = _load_spectrum_inputs(spectrum_path)

    geometry_path = run_dir / "geometry" / "outputs" / "geometry.h5"
    measure_assumptions: dict[str, Any] = {}
    if args.measure == "geometry" and z_grid.size:
        weight, assumptions = _load_geometry_weight(geometry_path, z_grid)
        measure_assumptions.update(assumptions)
        measure_used = "geometry" if "geometry_weight" in assumptions else "flat"
    else:
        weight = np.ones_like(z_grid)
        measure_used = "flat"
        if args.measure == "geometry":
            measure_assumptions["measure_fallback"] = "no_z_grid"

    ope_coefficients: dict[str, float] = {}
    light_ops: list[str] = []
    tower_prefixes: list[str] = []
    integration_used = args.integration

    if not missing:
        finite_mask = np.isfinite(delta_uv)
        deltas = delta_uv[finite_mask]
        if deltas.size == 0:
            missing.append("delta_uv")
        else:
            order = np.argsort(deltas)
            n_light = min(max(args.n_light, 1), deltas.size)
            selected = order[:n_light]
            width = max(3, len(str(n_light - 1)))
            light_ops = [f"op_{idx:0{width}d}" for idx in range(n_light)]

            if phi.size == 0:
                missing.append("phi")
            else:
                if phi.shape[0] < deltas.size:
                    missing.append("phi")
                elif z_grid.size != phi.shape[1]:
                    missing.append("z_grid")
                else:
                    phi_light = phi[selected, :]
                    tower_prefixes = [f"[{op} {op}]_" for op in light_ops]
                    for i, op_i in enumerate(light_ops):
                        for j, op_j in enumerate(light_ops):
                            for k, op_k in enumerate(light_ops):
                                integrand = weight * phi_light[i] * phi_light[j] * phi_light[k]
                                value, method_used = _integrate(z_grid, integrand, args.integration)
                                integration_used = method_used
                                key = f"{op_i}_{op_j}_{op_k}"
                                ope_coefficients[key] = value
                        for n in range(max(args.n_tower, 1)):
                            op3 = f"[{op_i} {op_i}]_{n}"
                            integrand = weight * phi_light[i] ** 3
                            value, method_used = _integrate(z_grid, integrand, args.integration)
                            integration_used = method_used
                            scaled = value / float(n + 1)
                            key = f"{op_i}_{op_i}_{op3}"
                            ope_coefficients[key] = scaled

    overall_verdict = "PASS" if not missing else "UNDERDETERMINED"
    missing_inputs = sorted(set(missing)) if missing else None

    coefficients_values = list(ope_coefficients.values())
    statistics = {
        "n_coefficients": len(ope_coefficients),
        "median_abs_coefficient": _median([abs(v) for v in coefficients_values]),
        "median_coefficient": _median(coefficients_values),
    }

    ope_payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "run_id": args.run,
            "provenance": {
                "stage": STAGE_NAME,
                "spectrum": str(spectrum_path.relative_to(run_dir)),
                "run_valid": str((run_dir / "RUN_VALID" / "outputs" / "run_valid.json").relative_to(run_dir)),
            },
            "model_assumptions": {
                "g0": 1.0,
                "measure_requested": args.measure,
                "measure_used": measure_used,
                "integration": integration_used,
                "n_light": args.n_light,
                "n_tower": args.n_tower,
                "tower_scaling": "1/(n+1)" if args.n_tower else None,
                **measure_assumptions,
            },
        },
        "conventions": {
            "light_ops": light_ops,
            "tower_prefixes": tower_prefixes,
        },
        "ope_coefficients": ope_coefficients,
        "statistics": statistics,
    }

    verdict_payload = {
        "overall_verdict": overall_verdict,
        "reason": "missing_inputs" if missing_inputs else None,
        "missing_inputs": missing_inputs,
        "n_coefficients": statistics["n_coefficients"],
        "median_abs_coefficient": statistics["median_abs_coefficient"],
        "median_coefficient": statistics["median_coefficient"],
    }

    report_payload = {
        "summary": "OPE coefficients proxy via bulk overlap",
        "overall_verdict": overall_verdict,
        "missing_inputs": missing_inputs,
        "n_light": args.n_light,
        "n_tower": args.n_tower,
        "measure": measure_used,
        "integration": integration_used,
        "statistics": statistics,
        "assumptions": ope_payload["metadata"]["model_assumptions"],
    }

    ope_path = outputs_dir / "ope_coefficients.json"
    verdict_path = outputs_dir / "verdict.json"
    report_path = outputs_dir / "report.json"

    _write_json(ope_path, ope_payload)
    _write_json(verdict_path, verdict_payload)
    _write_json(report_path, report_payload)

    stage_summary = {
        "stage": STAGE_NAME,
        "run": args.run,
        "created_utc": created_utc,
        "inputs": {
            "run_valid": {
                "path": str((run_dir / "RUN_VALID" / "outputs" / "run_valid.json").relative_to(run_dir)),
                "sha256": sha256_file(run_dir / "RUN_VALID" / "outputs" / "run_valid.json"),
                "verdict": run_valid_payload.get("verdict") or run_valid_payload.get("overall_verdict"),
            },
            "spectrum": {
                "path": str(spectrum_path.relative_to(run_dir)),
                "sha256": sha256_file(spectrum_path),
            },
            "geometry": {
                "path": str(geometry_path.relative_to(run_dir)),
                "sha256": sha256_file(geometry_path) if geometry_path.exists() else None,
            },
        },
        "outputs": {
            "ope_coefficients": "outputs/ope_coefficients.json",
            "verdict": "outputs/verdict.json",
            "report": "outputs/report.json",
        },
        "hashes": {
            "outputs/ope_coefficients.json": sha256_file(ope_path),
            "outputs/verdict.json": sha256_file(verdict_path),
            "outputs/report.json": sha256_file(report_path),
        },
        "missing_inputs": missing_inputs,
        "verdict": overall_verdict,
    }

    summary_path = write_stage_summary(stage_dir, stage_summary)
    write_manifest(
        stage_dir,
        {
            "ope_coefficients": ope_path,
            "verdict": verdict_path,
            "report": report_path,
            "summary": summary_path,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
