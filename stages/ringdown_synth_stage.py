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
from pathlib import Path

from basurin_io import ensure_stage_dirs, get_run_dir, sha256_file, write_manifest, write_stage_summary


def _compute_qnm_from_mass_spin(mass_msun: float, spin: float) -> tuple[float, float]:
    """
    Hook para calcular (f_220, tau_220) desde (M, a).
    NOTA: No se incluyen aquí coeficientes Berti para evitar hardcode no auditado.
    Integra tu implementación existente cuando la tengas canónica.

    Por defecto: exige que el usuario pase --f-220 y --tau-220.
    """
    raise ValueError("QNM from (mass,spin) no implementado en v1: usa --f-220 y --tau-220.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical ringdown synthetic event generator (ringdown_synth).")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snr", type=float, default=12.0)
    ap.add_argument("--mass-msun", type=float, default=None)
    ap.add_argument("--spin", type=float, default=None)
    ap.add_argument("--f-220", type=float, default=None, help="Hz (required unless mass/spin implemented)")
    ap.add_argument("--tau-220", type=float, default=None, help="seconds (required unless mass/spin implemented)")
    args = ap.parse_args()

    run_dir = get_run_dir(args.run).resolve()
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "ringdown_synth")

    out_json = outputs_dir / "synthetic_event.json"

    if args.f_220 is not None and args.tau_220 is not None:
        f_220 = float(args.f_220)
        tau_220 = float(args.tau_220)
    elif args.mass_msun is not None and args.spin is not None:
        f_220, tau_220 = _compute_qnm_from_mass_spin(float(args.mass_msun), float(args.spin))
    else:
        raise SystemExit("ERROR: proporciona --f-220 y --tau-220 (o implementa mass/spin).")

    q_220 = math.pi * f_220 * tau_220

    payload = {
        "schema_version": "ringdown_synth_event_v1",
        "seed": int(args.seed),
        "snr_target": float(args.snr),
        "truth": {"f_220": f_220, "tau_220": tau_220, "Q_220": q_220},
        "notes": {
            "source": "stages/ringdown_synth_stage.py",
            "mass_msun": args.mass_msun,
            "spin": args.spin,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

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
        "outputs": {"synthetic_event": "outputs/synthetic_event.json"},
    }
    write_stage_summary(stage_dir, summary)

    artifacts = {"synthetic_event": out_json}
    write_manifest(stage_dir, artifacts, extra={"verdict": "PASS"})

    # Sanity: ensure file hash stable in summary (optional)
    _ = sha256_file(out_json)


if __name__ == "__main__":
    main()
