#!/usr/bin/env python3
"""
BASURIN — Canonical stage: ringdown_synth

Generates a synthetic ringdown event with known QNM parameters (Berti et al.)
for use as ground truth in downstream experiments.

Output:
    runs/<run_id>/ringdown_synth/
        ├── manifest.json
        ├── stage_summary.json
        └── outputs/
            └── synthetic_event.json

Usage:
    python stages/ringdown_synth_stage.py --run <run_id> --seed 42

Note: This stage does NOT require RUN_VALID (it can be the first stage).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import (
    ensure_stage_dirs,
    get_runs_root,
    sha256_file,
    utc_now_iso,
    write_manifest,
    write_stage_summary,
)

# =============================================================================
# Physical constants and Berti coefficients
# =============================================================================

G = 6.67430e-11  # m^3 kg^-1 s^-2
C = 299792458.0  # m/s
MSUN = 1.98892e30  # kg

# Berti coefficients for mode 220 (fundamental)
BERTI_F_220 = (1.5251, -1.1568, 0.1292)
BERTI_Q_220 = (0.7000, 1.4187, -0.4990)


@dataclass(frozen=True)
class SynthConfig:
    """Configuration for synthetic event generation."""

    mass_msun: float = 70.0
    spin: float = 0.7
    snr_nominal: float = 25.0
    t_ref: float = 0.0  # Reference time marker (s)
    duration_available: float = 2.0  # Total data duration (s)
    fs: int = 4096  # Sampling rate (Hz)
    seed: int = 42


def kerr_qnm_220(mass_msun: float, spin: float) -> tuple[float, float, float]:
    """Compute QNM parameters for mode 220.

    Returns:
        f_hz: Frequency in Hz
        tau_ms: Decay time in milliseconds
        Q: Quality factor
    """
    spin = float(np.clip(spin, 0.01, 0.99))

    w_bar = BERTI_F_220[0] + BERTI_F_220[1] * (1 - spin) ** BERTI_F_220[2]
    Q = BERTI_Q_220[0] + BERTI_Q_220[1] * (1 - spin) ** BERTI_Q_220[2]

    T_g = (G * mass_msun * MSUN) / (C**3)
    f_hz = w_bar / (2 * np.pi * T_g)
    tau_s = Q / (np.pi * f_hz)
    tau_ms = tau_s * 1000.0

    return float(f_hz), float(tau_ms), float(max(Q, 0.1))


def generate_synthetic_event(cfg: SynthConfig) -> dict:
    """Generate a synthetic ringdown event."""
    np.random.seed(cfg.seed)

    f_220, tau_220, Q_220 = kerr_qnm_220(cfg.mass_msun, cfg.spin)

    event = {
        "schema_version": "1.0.0",
        "generator": "ringdown_synth_stage.py",
        "seed": cfg.seed,
        "parameters": {
            "mass_msun": cfg.mass_msun,
            "spin": cfg.spin,
        },
        "qnm_truth": {
            "f_220_hz": f_220,
            "tau_220_ms": tau_220,
            "Q_220": Q_220,
        },
        "signal_properties": {
            "snr_nominal": cfg.snr_nominal,
            "t_ref": cfg.t_ref,
            "duration_available": cfg.duration_available,
            "fs": cfg.fs,
        },
        "metadata": {
            "model": "Kerr QNM (Berti et al.)",
            "mode": "220",
            "created": utc_now_iso(),
        },
    }

    return event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ringdown event")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--mass", type=float, default=70.0, help="BH mass (Msun)")
    parser.add_argument("--spin", type=float, default=0.7, help="BH spin (0-1)")
    parser.add_argument("--snr", type=float, default=25.0, help="Nominal SNR")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-root", default=None, help="Runs root directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runs_root = Path(args.runs_root) if args.runs_root else get_runs_root()

    cfg = SynthConfig(
        mass_msun=args.mass,
        spin=args.spin,
        snr_nominal=args.snr,
        seed=args.seed,
    )

    # Create stage directories
    stage_dir, outputs_dir = ensure_stage_dirs(
        args.run, "ringdown_synth", base_dir=runs_root
    )

    # Generate event
    event = generate_synthetic_event(cfg)

    # Write synthetic_event.json
    event_path = outputs_dir / "synthetic_event.json"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2, sort_keys=True)
        f.write("\n")

    # Write stage_summary.json
    summary = {
        "stage": "ringdown_synth",
        "run": args.run,
        "version": "1.0.0",
        "config": asdict(cfg),
        "outputs": {
            "synthetic_event": "outputs/synthetic_event.json",
        },
        "qnm_summary": event["qnm_truth"],
        "hashes": {
            "outputs/synthetic_event.json": sha256_file(event_path),
        },
    }
    write_stage_summary(stage_dir, summary)

    # Write manifest.json
    write_manifest(
        stage_dir,
        {"synthetic_event": event_path},
        extra={"version": "1.0.0"},
    )

    print(f"[ringdown_synth] OK")
    print(f"  run: {args.run}")
    print(f"  mass: {cfg.mass_msun} Msun")
    print(f"  spin: {cfg.spin}")
    print(f"  f_220: {event['qnm_truth']['f_220_hz']:.2f} Hz")
    print(f"  tau_220: {event['qnm_truth']['tau_220_ms']:.2f} ms")
    print(f"  Q_220: {event['qnm_truth']['Q_220']:.2f}")
    print(f"  output: {event_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
