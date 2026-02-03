#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------


import argparse
import json
import math

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)


def _compute_qnm_from_mass_spin(mass_msun: float, spin: float) -> tuple[float, float]:
    """
    Hook para calcular (f_220, tau_220) desde (M, a).
    NOTA: No se incluyen aquí coeficientes Berti para evitar hardcode no auditado.
    Integra tu implementación existente cuando la tengas canónica.

    Por defecto: exige que el usuario pase --f-220 y --tau-220.
    """
    raise ValueError("QNM from (mass,spin) no implementado en v1: usa --f-220 y --tau-220.")


def _load_batch(path: Path) -> tuple[list[float], list[int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit("ERROR: --batch-json debe ser un objeto JSON.")
    snr_grid = payload.get("snr_grid")
    seeds = payload.get("seeds")
    if not isinstance(snr_grid, list) or not isinstance(seeds, list):
        raise SystemExit("ERROR: --batch-json requiere keys 'snr_grid' y 'seeds' (listas).")
    if not snr_grid or not seeds:
        raise SystemExit("ERROR: --batch-json requiere snr_grid y seeds no vacíos.")
    return [float(x) for x in snr_grid], [int(x) for x in seeds]


def _case_id(idx: int, snr: float, seed: int) -> str:
    snr_tag = f"{snr:.6g}".replace(".", "p").replace("-", "m")
    return f"case_{idx:04d}_snr_{snr_tag}_seed_{seed}"


def _write_strain_npz(
    path: Path,
    seed: int,
    snr: float,
    f_220: float,
    tau_220: float,
    *,
    fs: float = 4096.0,
    n_samples: int = 1024,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    t = np.arange(n_samples, dtype=float) / float(fs)
    signal = np.exp(-t / float(tau_220)) * np.sin(2.0 * math.pi * float(f_220) * t)
    noise = rng.normal(0.0, 1.0, size=n_samples)
    signal_rms = math.sqrt(float(np.mean(signal**2))) if signal.size else 1.0
    noise_rms = math.sqrt(float(np.mean(noise**2))) if noise.size else 1.0
    scale = float(snr) * noise_rms / (signal_rms + 1e-12)
    h = scale * signal + noise
    np.savez(
        path,
        t=t,
        h=h,
        fs=np.asarray(fs, dtype=float),
        snr_target=np.asarray(snr, dtype=float),
        seed=np.asarray(seed, dtype=int),
        f_220=np.asarray(f_220, dtype=float),
        tau_220=np.asarray(tau_220, dtype=float),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical ringdown synthetic event generator (ringdown_synth).")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", default="runs", help="Runs root (default: runs)")
    ap.add_argument("--batch-json", default=None, help="Batch JSON schema: {'snr_grid':[...], 'seeds':[...]}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snr", type=float, default=12.0)
    ap.add_argument("--mass-msun", type=float, default=None)
    ap.add_argument("--spin", type=float, default=None)
    ap.add_argument("--f-220", type=float, default=None, help="Hz (required unless mass/spin implemented)")
    ap.add_argument("--tau-220", type=float, default=None, help="seconds (required unless mass/spin implemented)")
    args = ap.parse_args()

    try:
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "ringdown_synth", base_dir=out_root)

    out_json = outputs_dir / "synthetic_event.json"
    out_events_index = outputs_dir / "synthetic_events.json"

    if args.f_220 is not None and args.tau_220 is not None:
        f_220 = float(args.f_220)
        tau_220 = float(args.tau_220)
    elif args.mass_msun is not None and args.spin is not None:
        f_220, tau_220 = _compute_qnm_from_mass_spin(float(args.mass_msun), float(args.spin))
    else:
        raise SystemExit("ERROR: proporciona --f-220 y --tau-220 (o implementa mass/spin).")

    q_220 = math.pi * f_220 * tau_220

    base_event = {
        "schema_version": "ringdown_synth_event_v1",
        "truth": {"f_220": f_220, "tau_220": tau_220, "Q_220": q_220},
        "notes": {
            "source": "stages/ringdown_synth_stage.py",
            "mass_msun": args.mass_msun,
            "spin": args.spin,
        },
    }

    artifacts: dict[str, Path] = {}

    if args.batch_json:
        snr_grid, seeds = _load_batch(Path(args.batch_json))
        cases_dir = outputs_dir / "cases"
        events: list[dict[str, object]] = []
        idx = 0
        for snr in snr_grid:
            for seed in seeds:
                case_id = _case_id(idx, snr, seed)
                strain_path = cases_dir / case_id / "strain.npz"
                _write_strain_npz(strain_path, seed, snr, f_220, tau_220)
                events.append(
                    {
                        **base_event,
                        "case_id": case_id,
                        "seed": int(seed),
                        "snr": float(snr),
                        "snr_target": float(snr),
                        "strain_npz": f"cases/{case_id}/strain.npz",
                        "paths": {
                            "strain_npz": f"ringdown_synth/outputs/cases/{case_id}/strain.npz"
                        },
                        "outputs": {"strain": f"outputs/cases/{case_id}/strain.npz"},
                    }
                )
                artifacts[f"case_{case_id}_strain"] = strain_path
                idx += 1

        out_events = outputs_dir / "synthetic_events.json"
        with open(out_events, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        artifacts["synthetic_events"] = out_events

        # canonical list for EXP01/EXP02 consumption
        out_events_list = outputs_dir / "synthetic_events_list.json"
        with open(out_events_list, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        artifacts["synthetic_events_list"] = out_events_list

        summary = {
            "stage": "ringdown_synth",
            "params": {
                "batch_json": args.batch_json,
                "mass_msun": args.mass_msun,
                "spin": args.spin,
                "f_220": args.f_220,
                "tau_220": args.tau_220,
                "snr_grid": snr_grid,
                "seeds": seeds,
            },
            "inputs": {
                "batch_json": args.batch_json,
            },
            "outputs": {
                "synthetic_events": "outputs/synthetic_events.json",
                "cases_dir": "outputs/cases",
            },
        }
        write_stage_summary(stage_dir, summary)
        write_manifest(stage_dir, artifacts, extra={"verdict": "PASS"})
        _ = sha256_file(out_events)
    else:
        payload = {
            **base_event,
            "seed": int(args.seed),
            "snr_target": float(args.snr),
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Generate strain.npz for EXP01 consumption
        out_strain = outputs_dir / "strain.npz"
        _write_strain_npz(out_strain, int(args.seed), float(args.snr), f_220, tau_220)
        artifacts["strain_npz"] = out_strain

        summary = {
            "stage": "ringdown_synth",
            "params": {
                "seed": int(args.seed),
                "snr_target": float(args.snr),
                "mass_msun": args.mass_msun,
                "spin": args.spin,
                "f_220": args.f_220,
                "tau_220": args.tau_220,
            },
            "outputs": {
                "synthetic_event": "outputs/synthetic_event.json",
                "synthetic_events": "outputs/synthetic_events.json",
                "strain_npz": "outputs/strain.npz",
            },
        }
        write_stage_summary(stage_dir, summary)

        index_payload = {
            "schema_version": "ringdown_synth_events_index_v1",
            "n_events": 1,
            "events": [
                {
                    "path": "synthetic_event.json",
                    "strain_npz": "strain.npz",
                    "truth": base_event["truth"],
                    "snr_target": float(args.snr),
                }
            ],
        }
        with open(out_events_index, "w", encoding="utf-8") as f:
            json.dump(index_payload, f, indent=2)

        # canonical list for EXP01/EXP02 consumption
        out_events_list = outputs_dir / "synthetic_events_list.json"
        with open(out_events_list, "w", encoding="utf-8") as f:
            json.dump(index_payload["events"], f, indent=2)

        artifacts["synthetic_event"] = out_json
        artifacts["synthetic_events"] = out_events_index
        artifacts["synthetic_events_list"] = out_events_list
        write_manifest(stage_dir, artifacts, extra={"verdict": "PASS"})

        _ = sha256_file(out_json)
        _ = sha256_file(out_events_index)
        _ = sha256_file(out_events_list)
        _ = sha256_file(out_strain)


if __name__ == "__main__":
    main()
