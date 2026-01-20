#!/usr/bin/env python3
"""
stage_ringdown.py
=================
Wrapper BASURIN-style para ejecutar ringdown_bayesian_v3.py y escribir artefactos
en runs/<run_id>/ringdown/ siguiendo el protocolo determinista de IO.

Uso ejemplo:
  python experiment/ringdown/stage_ringdown.py \
    --run-id 2026-01-20__ringdown_v3_smoke \
    --data experiment/ringdown/data/l1.npy \
    --detectors L1 \
    --redshift 0.09 \
    --category stellar \
    --multiweight equal \
    --overtones auto \
    --auto-fmin 20 --auto-fmax 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    # ISO8601 naive; suficiente para auditoría local
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def main() -> int:
    p = argparse.ArgumentParser(description="BASURIN stage wrapper: ringdown v3")
    p.add_argument("--run-id", required=True, help="run_id (e.g., 2026-01-20__ringdown_v3_smoke)")
    p.add_argument("--repo-root", default=".", help="Root del repo (default: .)")

    # Inputs del ringdown
    p.add_argument("--data", nargs="+", required=True, help="Archivos .npy (uno o varios detectores)")
    p.add_argument("--detectors", nargs="+", default=None, help="Nombres detectores (H1 L1 etc.)")
    p.add_argument("--fs", type=float, default=4096.0)
    p.add_argument("--redshift", type=float, default=None)
    p.add_argument("--category", choices=["light", "stellar", "imbh", "heavy"], default=None)
    p.add_argument("--multiweight", choices=["equal", "snr", "sigma"], default="equal")
    p.add_argument("--overtones", choices=["on", "off", "auto"], default="auto")
    p.add_argument("--auto-fmin", type=float, default=20.0)
    p.add_argument("--auto-fmax", type=float, default=500.0)

    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    script_ringdown = repo_root / "experiment" / "ringdown" / "ringdown_bayesian_v3.py"
    if not script_ringdown.exists():
        raise FileNotFoundError(f"No encuentro {script_ringdown}")

    # Stage paths
    run_dir = repo_root / "runs" / args.run_id / "ringdown"
    out_dir = run_dir / "outputs"
    safe_mkdir(out_dir)

    # Validar inputs
    data_paths = [Path(x).resolve() for x in args.data]
    for dp in data_paths:
        if not dp.exists():
            raise FileNotFoundError(f"Input data no existe: {dp}")

    if args.detectors is not None and len(args.detectors) != len(data_paths):
        raise ValueError("Si pasas --detectors, debe tener la misma longitud que --data")

    # Construir comando de ringdown (lo ejecutamos tal cual, sin reinterpretar ciencia)
    cmd: List[str] = [
        sys.executable,
        str(script_ringdown),
        "--data",
        *[str(x) for x in data_paths],
        "--outdir",
        str(out_dir),
        "--fs",
        str(args.fs),
        "--multiweight",
        args.multiweight,
        "--overtones",
        args.overtones,
        "--auto-fmin",
        str(args.auto_fmin),
        "--auto-fmax",
        str(args.auto_fmax),
    ]
    if args.detectors:
        cmd += ["--detectors", *args.detectors]
    if args.redshift is not None:
        cmd += ["--redshift", str(args.redshift)]
    if args.category is not None:
        cmd += ["--category", args.category]

    # Metadata base
    stage_summary: Dict = {
        "stage": "ringdown",
        "run_id": args.run_id,
        "created_at": now_iso(),
        "repo_root": str(repo_root),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "code": {
            "entrypoint": str(script_ringdown.relative_to(repo_root)),
            "entrypoint_sha256": sha256_file(script_ringdown),
            "wrapper": str(Path(__file__).resolve().relative_to(repo_root)),
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        },
        "params": {
            "data": [str(p) for p in data_paths],
            "detectors": args.detectors,
            "fs": args.fs,
            "redshift": args.redshift,
            "category": args.category,
            "multiweight": args.multiweight,
            "overtones": args.overtones,
            "auto_fmin": args.auto_fmin,
            "auto_fmax": args.auto_fmax,
            "outdir": str(out_dir.relative_to(repo_root)),
        },
        "command": cmd,
    }

    # Ejecutar
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    dt = time.time() - t0

    # Guardar logs
    (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    stage_summary["runtime_sec"] = round(dt, 6)
    stage_summary["returncode"] = proc.returncode

    # Manifest (inputs + outputs)
    inputs_manifest = []
    for dp in data_paths:
        inputs_manifest.append({
            "path": str(dp),
            "sha256": sha256_file(dp),
        })

    # Outputs esperados
    expected = ["summary.json", "config.json", "samples.npy", "corner.png", "fit.png"]
    outputs_manifest = []
    for name in expected:
        op = out_dir / name
        if op.exists():
            outputs_manifest.append({
                "path": str(op.relative_to(repo_root)),
                "sha256": sha256_file(op),
            })
        else:
            outputs_manifest.append({
                "path": str(op.relative_to(repo_root)),
                "missing": True,
            })

    manifest = {
        "run_id": args.run_id,
        "stage": "ringdown",
        "inputs": inputs_manifest,
        "outputs": outputs_manifest,
        "logs": {
            "stdout": str((run_dir / "stdout.txt").relative_to(repo_root)),
            "stderr": str((run_dir / "stderr.txt").relative_to(repo_root)),
        }
    }

    write_json(run_dir / "stage_summary.json", stage_summary)
    write_json(run_dir / "manifest.json", manifest)

    if proc.returncode != 0:
        print(f"[ringdown stage] FAILED (returncode={proc.returncode})")
        print(f"See logs: {run_dir}/stderr.txt")
        return proc.returncode

    print(f"[ringdown stage] OK -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
