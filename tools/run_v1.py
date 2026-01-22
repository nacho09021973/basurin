#!/usr/bin/env python3
"""BASURIN — Runner v1: orquestación 01→02b→03→04→RUN_VALID.

Uso:
  python tools/run_v1.py --run <run_id> [--mode geometry|spectrum_only]
                          [--skip-02b] [--geometry-file ads_puro.h5]
                          [--generator neutrino] [--keep-runs]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import get_run_dir, validate_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner v1 BASURIN (01→02b→03→04→RUN_VALID)")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument(
        "--mode",
        choices=["geometry", "spectrum_only"],
        default="geometry",
        help="Modo de ejecución (default: geometry)",
    )
    parser.add_argument("--skip-02b", action="store_true", help="Omitir 02b_geometry_contracts_stage")
    parser.add_argument(
        "--geometry-file",
        default="ads_puro.h5",
        help="Archivo H5 de geometría (default: ads_puro.h5)",
    )
    parser.add_argument(
        "--generator",
        choices=["neutrino"],
        default="neutrino",
        help="Generador para spectrum_only (default: neutrino)",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="No borrar runs/<run> antes de ejecutar",
    )
    return parser.parse_args()


def run_cmd(args: list[str]) -> int:
    result = subprocess.run(args, check=False)
    return int(result.returncode)


def main() -> int:
    args = parse_args()
    validate_run_id(args.run, Path("runs"))
    run_dir = get_run_dir(args.run)

    if not args.keep_runs and run_dir.exists():
        shutil.rmtree(run_dir)

    commands: list[list[str]] = []
    if args.mode == "geometry":
        commands.append([sys.executable, "01_genera_ads_puro.py", "--run", args.run])
        if not args.skip_02b:
            commands.append([sys.executable, "02b_geometry_contracts_stage.py", "--run", args.run])
        commands.append(
            [
                sys.executable,
                "03_sturm_liouville.py",
                "--run",
                args.run,
                "--geometry-file",
                args.geometry_file,
            ]
        )
    else:
        if args.generator == "neutrino":
            commands.append([sys.executable, "01_genera_neutrino_sandbox.py", "--run", args.run])
    commands.extend(
        [
            [sys.executable, "04_diccionario.py", "--run", args.run],
            [sys.executable, "tools/contract_run_valid.py", "--run", args.run],
        ]
    )

    for cmd in commands:
        returncode = run_cmd(cmd)
        if returncode != 0:
            return returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
