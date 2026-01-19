#!/usr/bin/env python3
"""
01_mix_spectra.py

Etapa 01 (dataset): mezcla/concatena dos espectros spectrum.h5 (mismo formato)
para crear un run de salida con un spectrum.h5 combinado.

Contrato IO BASURIN (canónico):
  runs/<run_out>/spectrum/
    - manifest.json
    - stage_summary.json
    - outputs/
        - spectrum.h5

Compatibilidad (legado): lee spectrum/spectrum.h5 si no existe outputs/spectrum.h5.
"""

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import h5py


STAGE = "spectrum"
SCRIPT_NAME = "01_mix_spectra.py"
VERSION = "v0.2.0"

def resolve_in_spectrum(run: str) -> Path:
    root = Path("runs") / run / STAGE
    cand1 = root / "outputs" / "spectrum.h5"
    if cand1.exists():
        return cand1
    return root / "spectrum.h5"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mezcla dos runs (spectrum.h5) concatenando muestras para crear un run mezclado."
    )
    ap.add_argument("--run-out", required=True, help="Run de salida (p.ej. ir_mix)")
    ap.add_argument("--run-a", required=True, help="Run A (p.ej. ir_A)")
    ap.add_argument("--run-b", required=True, help="Run B (p.ej. ir_B)")
    ap.add_argument("--tolerance-delta", type=float, default=1e-12,
                    help="Tolerancia para comparar delta_uv entre A y B.")
    args = ap.parse_args()

    a_path = resolve_in_spectrum(args.run_a)
    b_path = resolve_in_spectrum(args.run_b)

    stage_dir = Path("runs") / args.run_out / STAGE
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_spectrum = outputs_dir / "spectrum.h5"
    out_manifest = stage_dir / "manifest.json"
    out_summary = stage_dir / "stage_summary.json"

    if not a_path.exists():
        raise FileNotFoundError(f"No existe: {a_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"No existe: {b_path}")

    # Carga y validación estricta
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        da = np.array(fa["delta_uv"])
        db = np.array(fb["delta_uv"])

        if da.shape != db.shape or np.max(np.abs(da - db)) > args.tolerance_delta:
            raise ValueError(
                "delta_uv no coincide entre run-a y run-b. "
                "Ajusta sweep_delta para que sea idéntico (o incrementa --tolerance-delta si procede)."
            )

        M2a = np.array(fa["M2"])
        M2b = np.array(fb["M2"])

        if M2a.ndim != 2 or M2b.ndim != 2:
            raise ValueError("M2 debe ser 2D: (n_samples, n_modes).")

        if M2a.shape[1] != M2b.shape[1]:
            raise ValueError(f"n_modes distinto: {M2a.shape[1]} vs {M2b.shape[1]}")

        # Concatenación: duplicamos muestreo de Delta (dos familias para los mismos Delta)
        M2 = np.vstack([M2a, M2b])
        delta_uv = np.concatenate([da, db])

        # Opcionales: m2L2, z_grid
        m2a = np.array(fa["m2L2"]) if "m2L2" in fa else None
        m2b = np.array(fb["m2L2"]) if "m2L2" in fb else None
        m2 = np.concatenate([m2a, m2b]) if (m2a is not None and m2b is not None) else None

        zga = np.array(fa["z_grid"]) if "z_grid" in fa else None
        zgb = np.array(fb["z_grid"]) if "z_grid" in fb else None
        z_grid = zga if zga is not None else zgb  # informativo

    # Escritura del spectrum combinado
    with h5py.File(out_spectrum, "w") as fo:
        fo.create_dataset("delta_uv", data=delta_uv)
        fo.create_dataset("M2", data=M2)
        if m2 is not None:
            fo.create_dataset("m2L2", data=m2)
        if z_grid is not None:
            fo.create_dataset("z_grid", data=z_grid)

        fo.attrs["created_by"] = SCRIPT_NAME
        fo.attrs["version"] = VERSION
        fo.attrs["run_a"] = args.run_a
        fo.attrs["run_b"] = args.run_b
        fo.attrs["note"] = (
            "Concatenacion vertical de dos familias espectrales (mismo delta_uv). "
            "delta_uv se duplica para etiquetar cada muestra."
        )

    # Hashes (auditoría)
    in_hash_a = sha256_file(a_path)
    in_hash_b = sha256_file(b_path)
    out_hash = sha256_file(out_spectrum)

    # Manifest
    manifest = {
        "stage": STAGE,
        "run_out": args.run_out,
        "created": utc_now_iso(),
        "files": {
            "spectrum": "outputs/spectrum.h5",
            "stage_summary": "stage_summary.json",
            "manifest": "manifest.json",
        },
        "hashes": {
            "outputs/spectrum.h5": out_hash,
        },
        "inputs": {
            "run_a": args.run_a,
            "run_b": args.run_b,
            "run_a_spectrum": str(a_path.as_posix()),
            "run_b_spectrum": str(b_path.as_posix()),
        },
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Stage summary
    summary = {
        "stage": STAGE,
        "script": SCRIPT_NAME,
        "version": VERSION,
        "timestamp_utc": utc_now_iso(),
        "params": {
            "run_out": args.run_out,
            "run_a": args.run_a,
            "run_b": args.run_b,
            "tolerance_delta": args.tolerance_delta,
        },
        "inputs": {
            "run_a_spectrum": str(a_path.as_posix()),
            "run_b_spectrum": str(b_path.as_posix()),
            "sha256": {
                str(a_path.as_posix()): in_hash_a,
                str(b_path.as_posix()): in_hash_b,
            },
        },
        "outputs": {
            "spectrum": str(out_spectrum.as_posix()),
            "sha256": {
                str(out_spectrum.as_posix()): out_hash,
            },
        },
        "shape": {
            "n_samples": int(M2.shape[0]),
            "n_modes": int(M2.shape[1]),
            "delta_uv_len": int(delta_uv.shape[0]),
        },
        "notes": [
            "Este script produce un dataset mezclado (duplicando delta_uv) para Bloque C.",
            "No ejecuta Sturm–Liouville; opera sobre spectrum.h5 existentes.",
        ],
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"OK: creado {out_spectrum}")
    print(f"  n_samples={M2.shape[0]} (={M2a.shape[0]}+{M2b.shape[0]}), n_modes={M2.shape[1]}")
    print(f"  manifest: {out_manifest}")
    print(f"  summary:  {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
