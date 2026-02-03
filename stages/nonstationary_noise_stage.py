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

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)


def _variants_v1(line_f0: float) -> list[dict[str, object]]:
    line_f0 = float(line_f0)
    return [
        {
            "id": "drift_lowfreq",
            "type": "drift_lowfreq",
            "params": {"amplitude_scale": 0.20, "f_hz": 0.5},
        },
        {
            "id": "line_f0",
            "type": "line",
            "params": {"f_hz": line_f0, "amplitude_scale": 0.03},
        },
        {
            "id": "line_2f0",
            "type": "line",
            "params": {"f_hz": 2.0 * line_f0, "amplitude_scale": 0.02},
        },
        {
            "id": "glitch_gaussian_pulse",
            "type": "glitch_gaussian_pulse",
            "params": {"amplitude_scale": 0.50, "t0_rel": 0.35, "sigma_s": 0.008},
        },
        {
            "id": "glitch_chirp_short",
            "type": "glitch_chirp_short",
            "params": {
                "amplitude_scale": 0.35,
                "f0_hz": 30.0,
                "f1_hz": 180.0,
                "t0_rel": 0.15,
                "duration_s": 0.05,
            },
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical nonstationary noise variant catalog (v1)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--sweep", default="v1")
    ap.add_argument("--line-f0", type=float, default=50.0)
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)

    if args.sweep != "v1":
        print(f"ERROR: unsupported sweep {args.sweep}", file=sys.stderr)
        return 2

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "nonstationary_noise", base_dir=out_root)
    variants = _variants_v1(float(args.line_f0))

    output_path = outputs_dir / "nonstationary_variants.json"
    payload = {
        "schema_version": "nonstationary_noise_variants_v1",
        "sweep_id": args.sweep,
        "line_f0": float(args.line_f0),
        "variants": variants,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    summary = {
        "stage": "nonstationary_noise",
        "params": {"sweep": args.sweep, "line_f0": float(args.line_f0)},
        "outputs": {"nonstationary_variants": "outputs/nonstationary_variants.json"},
    }
    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "nonstationary_variants": output_path,
            "stage_summary": summary_path,
        },
        extra={"sha256": sha256_file(output_path), "version": "1"},
    )

    print(f"OK: nonstationary_noise variants written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
