#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import write_json_atomic
from mvp.experiment_t0_sweep_full import _new_subrun_trace, _write_subrun_trace


DEFAULT_LOSC_ROOT = REPO_ROOT / "data" / "losc"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_event_ids(args: argparse.Namespace) -> list[str]:
    events: list[str] = []
    for item in args.event_id:
        token = item.strip()
        if token:
            events.append(token)
    if args.events_file:
        path = Path(args.events_file).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"ERROR: events file not found: {path}")
        for raw in path.read_text(encoding="utf-8").splitlines():
            token = raw.strip()
            if token and not token.startswith("#"):
                events.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for event_id in events:
        if event_id not in seen:
            deduped.append(event_id)
            seen.add(event_id)
    if not deduped:
        raise SystemExit("ERROR: no EVENT_ID provided. Use --event-id and/or --events-file")
    return deduped


def _select_event_h5(event_id: str, detector: str, losc_root: Path) -> Path:
    event_dir = (losc_root / event_id).resolve()
    candidates = sorted(
        {
            *[p for p in event_dir.glob(f"*{detector}*.hdf5") if p.is_file()],
            *[p for p in event_dir.glob(f"*{detector}*.h5") if p.is_file()],
        },
        key=lambda p: p.name,
    )
    if len(candidates) == 1:
        return candidates[0]

    find_hint = f"find {shlex.quote(str(event_dir))} \\( -iname '*.hdf5' -o -iname '*.h5' \\) -type f"
    if not event_dir.exists():
        raise SystemExit(
            "ERROR: setup incompleto: falta directorio local LOSC para el evento.\n"
            f"  event_id={event_id}\n"
            f"  esperado={event_dir}\n"
            f"  ayuda: monta HDF5 locales en data/losc/<EVENT_ID>/ y reintenta.\n"
            f"  check: {find_hint}"
        )

    if len(candidates) == 0:
        raise SystemExit(
            "ERROR: setup incompleto: no hay HDF5 locales requeridos.\n"
            f"  event_id={event_id} detector={detector}\n"
            f"  esperado patrón=*{detector}*.hdf5|*.h5 en {event_dir}\n"
            f"  check: {find_hint}"
        )

    shown = ", ".join(p.name for p in candidates[:6])
    suffix = " ..." if len(candidates) > 6 else ""
    raise SystemExit(
        "ERROR: setup ambiguo: múltiples HDF5 candidatos para detector.\n"
        f"  event_id={event_id} detector={detector}\n"
        f"  candidatos={shown}{suffix}\n"
        "  deja exactamente 1 por detector o usa un wrapper dedicado para resolver ambigüedad."
    )


def _run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("[batch] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _event_run_id(batch_run_id: str, event_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in event_id)
    return f"{batch_run_id}__{clean}"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ejecuta batch local (s1->s2->t0_sweep_full(run)->s5) por EVENT_ID")
    ap.add_argument("--event-id", action="append", default=[], help="EVENT_ID (repeatable)")
    ap.add_argument("--events-file", default=None, help="Archivo con EVENT_ID por línea")
    ap.add_argument("--batch-run-id", default=None, help="ID del batch (default: batch_UTC)")
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--t0-ms", type=int, default=0, help="t0 usado por s5_event_row")
    ap.add_argument("--t0-grid-ms", default="0", help="grid para experiment_t0_sweep_full, ej: 0,2,4")
    ap.add_argument("--losc-root", default=str(DEFAULT_LOSC_ROOT))
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    event_ids = _read_event_ids(args)

    atlas_path = Path(args.atlas_path).expanduser().resolve()
    if not atlas_path.exists():
        raise SystemExit(f"ERROR: atlas no encontrado: {atlas_path}")

    runs_root = Path(args.runs_root).expanduser().resolve()
    batch_run_id = args.batch_run_id or f"batch_{_utc_stamp()}"
    batch_dir = runs_root / batch_run_id
    batch_outputs = batch_dir / "outputs"
    batch_outputs.mkdir(parents=True, exist_ok=True)

    losc_root = Path(args.losc_root).expanduser().resolve()
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(batch_dir / "runs")

    inventory: list[str] = []

    for event_id in event_ids:
        h1 = _select_event_h5(event_id, "H1", losc_root)
        l1 = _select_event_h5(event_id, "L1", losc_root)
        run_id = _event_run_id(batch_run_id, event_id)

        _run_cmd(
            [
                sys.executable,
                "mvp/s1_fetch_strain.py",
                "--run",
                run_id,
                "--event-id",
                event_id,
                "--local-hdf5",
                f"H1={h1}",
                "--local-hdf5",
                f"L1={l1}",
                "--reuse-if-present",
                "--offline",
            ],
            env,
        )

        strain_npz = batch_dir / "runs" / run_id / "s1_fetch_strain" / "outputs" / "strain.npz"
        _run_cmd(
            [
                sys.executable,
                "mvp/s2_ringdown_window.py",
                "--run-id",
                run_id,
                "--event-id",
                event_id,
                "--strain-npz",
                str(strain_npz),
            ],
            env,
        )

        _run_cmd(
            [
                sys.executable,
                "-m",
                "mvp.experiment_t0_sweep_full",
                "--run-id",
                run_id,
                "--phase",
                "run",
                "--atlas-path",
                str(atlas_path),
                "--seed",
                str(args.seed),
                "--t0-grid-ms",
                str(args.t0_grid_ms),
                "--resume-missing",
            ],
            env,
        )

        _run_cmd(
            [
                sys.executable,
                "mvp/s5_event_row.py",
                "--runs-root",
                str(batch_dir / "runs"),
                "--run-id",
                run_id,
                "--seed",
                str(args.seed),
                "--t0-ms",
                str(args.t0_ms),
            ],
            env,
        )

        subrun_id = f"{run_id}__t0ms{int(args.t0_ms):04d}"
        seed_dir = batch_dir / "runs" / run_id / "experiment" / f"t0_sweep_full_seed{int(args.seed)}"
        subrun_dir = seed_dir / "runs" / subrun_id
        trace_path = subrun_dir / "derived" / "subrun_trace.json"
        if not trace_path.exists():
            trace = _new_subrun_trace(subrun_id, int(args.seed), int(args.t0_ms), stages=[])
            trace["stages"].append({"stage": "batch_wrapper", "status": "injected_for_s5_target"})
            _write_subrun_trace(trace_path, trace)

        inventory.append(run_id)

    run_ids_path = batch_outputs / "run_ids.txt"
    run_ids_path.write_text("\n".join(inventory) + "\n", encoding="utf-8")
    write_json_atomic(
        batch_outputs / "batch_manifest.json",
        {
            "schema_version": "run_experiment_batch_v1",
            "batch_run_id": batch_run_id,
            "events": event_ids,
            "run_ids": inventory,
            "seed": int(args.seed),
            "t0_ms": int(args.t0_ms),
            "t0_grid_ms": str(args.t0_grid_ms),
            "atlas_path": str(atlas_path),
            "losc_root": str(losc_root),
        },
    )
    print(f"[batch] OK inventory: {run_ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
