#!/usr/bin/env python3
"""Experimento reproducible: barrido de t0 para auditar aparición del modo 221.

Escribe exclusivamente bajo:
  runs/<exp_run_id>/experiment/t0_sweep_221/

Para cada punto del grid ejecuta subruns aislados de s2 y s3b, sin modificar
stages canónicos.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
)

EXPERIMENT_NAME = "t0_sweep_221"
EXPERIMENT_STAGE = f"experiment/{EXPERIMENT_NAME}"
DEFAULT_S2_DT_START_S = 0.003
MTSUN_SECONDS = 4.92549095e-6


def _parse_grid(grid_text: str) -> list[float]:
    vals: list[float] = []
    for part in grid_text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("--t0-grid vacío: proporciona offsets separados por coma")
    return vals


def _find_event_mass_msun(event_id: str, base_run_dir: Path) -> tuple[float | None, list[str]]:
    keys = (
        "Mf_msun",
        "mf_msun",
        "M_final_msun",
        "m_final_msun",
        "final_mass_msun",
    )
    candidates = [
        base_run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        base_run_dir / "s1_fetch_strain" / "outputs" / "provenance.json",
        Path("docs/ringdown/event_metadata") / f"{event_id}_metadata.json",
    ]
    found_paths: list[str] = [str(p) for p in candidates if p.exists()]

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key in keys:
                    if key in node:
                        try:
                            val = float(node[key])
                            if val > 0:
                                return val, found_paths
                        except Exception:
                            pass
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return None, found_paths


def _m_to_seconds(value_m: float, mass_msun: float) -> float:
    return float(value_m) * float(mass_msun) * MTSUN_SECONDS


def _format_t0_token(t0: float) -> str:
    txt = f"{t0:.9g}"
    txt = txt.replace("+", "")
    return txt


def _extract_row(event_id: str, t0: float, units: str, payload: dict[str, Any], subrun_path: str) -> dict[str, Any]:
    flags = list(payload.get("results", {}).get("quality_flags") or [])
    mode_221 = None
    for mode in payload.get("modes", []):
        if str(mode.get("label")) == "221":
            mode_221 = mode
            break

    has_221 = False
    valid_fraction = None
    if isinstance(mode_221, dict):
        ln_f = mode_221.get("ln_f")
        sigma = mode_221.get("Sigma")
        stability = mode_221.get("fit", {}).get("stability", {}) if isinstance(mode_221.get("fit"), dict) else {}
        vf = stability.get("valid_fraction")
        if vf is not None:
            try:
                valid_fraction = float(vf)
            except Exception:
                valid_fraction = None
        has_221 = ln_f is not None and sigma is not None

    reason = None
    if not has_221:
        reason = next((f for f in flags if str(f).startswith("221_")), "no_221_block")

    return {
        "event_id": event_id,
        "t0": float(t0),
        "units": units,
        "has_221": bool(has_221),
        "valid_fraction_221": valid_fraction,
        "reason": reason,
        "flags": flags,
        "subrun_path": subrun_path,
    }


def _run_stage_cmd(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)


def _run_one_point(
    *,
    event_id: str,
    base_run_dir: Path,
    exp_stage_dir: Path,
    t0_value: float,
    units: str,
    mass_msun: float,
    python_exe: str,
    out_root: Path,
) -> dict[str, Any]:
    t0_token = _format_t0_token(t0_value)
    subrun_id = f"t0_{t0_token}"
    subruns_root = exp_stage_dir / "subruns"
    subrun_dir = subruns_root / subrun_id
    (subrun_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
    write_json_atomic(subrun_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    if units == "ms":
        offset_seconds = float(t0_value) / 1000.0
    else:
        offset_seconds = _m_to_seconds(float(t0_value), mass_msun)
    dt_start_s = DEFAULT_S2_DT_START_S + offset_seconds

    base_strain = base_run_dir / "s1_fetch_strain" / "outputs" / "strain.npz"
    if not base_strain.exists():
        raise FileNotFoundError(
            "Input faltante para experimento. "
            f"Ruta esperada exacta: {base_strain}. "
            f"Comando para regenerar upstream: python mvp/pipeline.py single --event-id {event_id} --atlas-path <ATLAS_PATH>."
        )
    base_s3 = base_run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if not base_s3.exists():
        raise FileNotFoundError(
            "Input faltante para experimento. "
            f"Ruta esperada exacta: {base_s3}. "
            f"Comando para regenerar upstream: python mvp/s3_ringdown_estimates.py --run-id {base_run_dir.name}."
        )

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(subruns_root.resolve())

    cmd_s2 = [
        python_exe,
        "mvp/s2_ringdown_window.py",
        "--run-id",
        subrun_id,
        "--event-id",
        event_id,
        "--dt-start-s",
        str(dt_start_s),
        "--strain-npz",
        str(base_strain.resolve()),
    ]
    cp2 = _run_stage_cmd(cmd_s2, env)
    if cp2.returncode != 0:
        raise RuntimeError(f"s2 falló en t0={t0_value} {units}: {cp2.stderr.strip() or cp2.stdout.strip()}")

    cmd_s3b = [
        python_exe,
        "mvp/s3b_multimode_estimates.py",
        "--run-id",
        subrun_id,
        "--s3-estimates",
        str(base_s3.resolve()),
    ]
    cp3 = _run_stage_cmd(cmd_s3b, env)
    if cp3.returncode != 0:
        raise RuntimeError(f"s3b falló en t0={t0_value} {units}: {cp3.stderr.strip() or cp3.stdout.strip()}")

    s3b_out = subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    if not s3b_out.exists():
        raise FileNotFoundError(f"Output esperado no encontrado: {s3b_out}")
    payload = json.loads(s3b_out.read_text(encoding="utf-8"))
    rel = str(subrun_dir.relative_to(out_root))
    return _extract_row(event_id, t0_value, units, payload, rel + "/")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["event_id", "t0", "units", "has_221", "valid_fraction_221", "reason", "flags", "subrun_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            out = dict(row)
            out["flags"] = json.dumps(row.get("flags", []), ensure_ascii=False)
            wr.writerow(out)


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("has_221")]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (float(r.get("valid_fraction_221") or 0.0), -abs(float(r.get("t0") or 0.0))))


def _write_experiment_contract(
    *,
    exp_stage_dir: Path,
    outputs_dir: Path,
    argv: list[str],
    script_path: Path,
    input_hashes: dict[str, str],
    status: str,
    error: str | None,
) -> tuple[Path, Path]:
    output_files = sorted(p for p in outputs_dir.glob("*") if p.is_file())
    output_hashes = {str(p.relative_to(exp_stage_dir)): sha256_file(p) for p in output_files}

    summary = {
        "stage": EXPERIMENT_STAGE,
        "run": exp_stage_dir.parts[-3],
        "created_utc": utc_now_iso(),
        "command": " ".join(shlex.quote(x) for x in argv),
        "script_sha256": sha256_file(script_path),
        "input_hashes": input_hashes,
        "outputs": [{"path": rel, "sha256": sha} for rel, sha in output_hashes.items()],
        "verdict": status,
    }
    if error:
        summary["error"] = error

    stage_summary_path = write_json_atomic(exp_stage_dir / "stage_summary.json", summary)
    manifest = {
        "schema_version": "mvp_manifest_v1",
        "created": utc_now_iso(),
        "artifacts": {
            "stage_summary": "stage_summary.json",
            "outputs": [str(p.relative_to(exp_stage_dir)) for p in output_files],
        },
        "hashes": {
            "stage_summary": sha256_file(stage_summary_path),
            **output_hashes,
        },
        "inputs": input_hashes,
        "verdict": status,
    }
    manifest_path = write_json_atomic(exp_stage_dir / "manifest.json", manifest)
    return stage_summary_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Experimento t0_sweep_221 (GW250114 / GW231028_153006)")
    ap.add_argument("--event-id", required=True, choices=["GW250114", "GW231028_153006"])
    ap.add_argument("--base-run-id", required=True)
    ap.add_argument("--exp-run-id", required=True)
    ap.add_argument("--t0-grid", required=True, help='Offsets separados por coma. En units=M son unidades geométricas M; en units=ms son milisegundos.')
    ap.add_argument("--units", required=True, choices=["M", "ms"])
    ap.add_argument("--max-workers", type=int, default=1)
    args = ap.parse_args(argv)

    out_root = resolve_out_root("runs")
    validate_run_id(args.base_run_id, out_root)
    validate_run_id(args.exp_run_id, out_root)
    require_run_valid(out_root, args.base_run_id)

    base_run_dir = out_root / args.base_run_id
    exp_stage_dir = out_root / args.exp_run_id / "experiment" / EXPERIMENT_NAME
    outputs_dir = exp_stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    grid = _parse_grid(args.t0_grid)
    mass_msun, candidates = _find_event_mass_msun(args.event_id, base_run_dir)
    if args.units == "ms" and mass_msun is None:
        expected = base_run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
        raise RuntimeError(
            "No se puede convertir ms->M sin Mf. "
            f"Ruta esperada exacta: {expected}. "
            f"Comando para regenerar upstream: python mvp/s3_ringdown_estimates.py --run-id {args.base_run_id}. "
            f"Candidatos detectados: {candidates or ['<ninguno>']}"
        )
    if args.units == "M" and mass_msun is None:
        raise RuntimeError("No se puede convertir M->segundos sin Mf (m_final_msun).")

    input_hashes: dict[str, str] = {}
    for p in (
        base_run_dir / "s1_fetch_strain" / "manifest.json",
        base_run_dir / "s2_ringdown_window" / "manifest.json",
        base_run_dir / "s3_ringdown_estimates" / "manifest.json",
        base_run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
    ):
        if p.exists():
            input_hashes[str(p.relative_to(out_root / args.base_run_id))] = sha256_file(p)

    rows: list[dict[str, Any]] = []
    err: str | None = None
    status = "PASS"
    try:
        if args.max_workers <= 1:
            for t0 in grid:
                rows.append(
                    _run_one_point(
                        event_id=args.event_id,
                        base_run_dir=base_run_dir,
                        exp_stage_dir=exp_stage_dir,
                        t0_value=t0,
                        units=args.units,
                        mass_msun=float(mass_msun),
                        python_exe=sys.executable,
                        out_root=out_root,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futs = {
                    ex.submit(
                        _run_one_point,
                        event_id=args.event_id,
                        base_run_dir=base_run_dir,
                        exp_stage_dir=exp_stage_dir,
                        t0_value=t0,
                        units=args.units,
                        mass_msun=float(mass_msun),
                        python_exe=sys.executable,
                        out_root=out_root,
                    ): t0
                    for t0 in grid
                }
                for fut in as_completed(futs):
                    rows.append(fut.result())
        rows.sort(key=lambda r: float(r["t0"]))

        table_json = outputs_dir / "t0_sweep_table.json"
        table_csv = outputs_dir / "t0_sweep_table.csv"
        best_path = outputs_dir / "best_t0.json"
        diag_path = outputs_dir / "diagnostics.json"

        write_json_atomic(table_json, rows)
        _write_csv(table_csv, rows)

        best = _select_best(rows)
        write_json_atomic(best_path, {"best_t0": best})

        counts = Counter((r.get("reason") if not r.get("has_221") else "has_221") for r in rows)
        write_json_atomic(diag_path, {"counts": dict(sorted(counts.items()))})
    except Exception as exc:
        status = "FAIL"
        err = f"{type(exc).__name__}: {exc}"

    stage_summary_path, manifest_path = _write_experiment_contract(
        exp_stage_dir=exp_stage_dir,
        outputs_dir=outputs_dir,
        argv=[sys.executable, str(Path(__file__).as_posix()), *(argv or sys.argv[1:])],
        script_path=Path(__file__).resolve(),
        input_hashes=input_hashes,
        status=status,
        error=err,
    )

    print(f"OUT_ROOT={out_root}")
    print(f"STAGE_DIR={exp_stage_dir}")
    print(f"OUTPUTS_DIR={outputs_dir}")
    print(f"STAGE_SUMMARY={stage_summary_path}")
    print(f"MANIFEST={manifest_path}")

    if status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
