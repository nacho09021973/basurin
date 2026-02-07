#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for cand in [HERE.parents[1], HERE.parents[2]]:
    if (cand / "basurin_io.py").exists():
        sys.path.insert(0, str(cand))
        break

import phi_core
from basurin_io import (
    compute_sha256,
    ensure_dir,
    load_run_valid_verdict,
    read_json,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE = "ringdown_synth"
MODEL_FAMILY = "phi_phenomenological_v0"
EPISTEMIC = "conjectural/phenomenological"


def _run_rel(run_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_root.resolve()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-json", required=True)
    ap.add_argument("--sigma-rel", required=True, type=float)
    ap.add_argument("--seed-base", required=True, type=int)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    run_root = Path(args.root) / "runs" / args.run
    run_pass, run_valid_path = load_run_valid_verdict(run_root)
    if not run_pass:
        print("[BASURIN ABORT] RUN_VALID != PASS", file=sys.stderr)
        return 2

    atlas_path = Path(args.atlas_json)
    if not atlas_path.is_absolute():
        atlas_path = Path(args.root) / atlas_path
    atlas = read_json(atlas_path)
    geoms = atlas.get("geometries", [])

    stage_dir = run_root / STAGE
    outputs_dir = stage_dir / "outputs"
    ensure_dir(outputs_dir)

    cases = []
    for geom in geoms:
        idx = int(geom["geometry_index"])
        out = phi_core.forward_model(float(geom["M2_0"]), float(geom["r1"]), float(geom["L"]), alpha=1.0)
        seed_f = args.seed_base + 2 * idx
        seed_tau = args.seed_base + 2 * idx + 1
        f_obs = phi_core.add_gaussian_noise(out["f"], args.sigma_rel, seed_f)
        tau_obs = phi_core.add_gaussian_noise(out["tau"], args.sigma_rel, seed_tau)
        q_obs = math.pi * f_obs * tau_obs
        cases.append(
            {
                "case_id": f"case_{idx}",
                "geometry_index": idx,
                "f_obs": float(f_obs),
                "tau_obs": float(tau_obs),
                "Q_obs": float(q_obs),
                "noise_sigma_rel": float(args.sigma_rel),
                "seed_base": int(args.seed_base),
            }
        )

    events = {
        "version": "synthetic_events.v0",
        "model": {"family": MODEL_FAMILY, "epistemic_status": EPISTEMIC},
        "n_cases": len(cases),
        "cases": cases,
    }
    out_path = outputs_dir / "synthetic_events.json"
    write_json_atomic(out_path, events)

    verdict = "PASS" if events["n_cases"] == len(geoms) and all(c["f_obs"] > 0 and c["tau_obs"] > 0 and c["Q_obs"] > 0 for c in cases) else "FAIL"

    inputs = {
        "run_valid": {"path": _run_rel(run_root, run_valid_path), "sha256": compute_sha256(run_valid_path)},
        "atlas_json": {"path": _run_rel(run_root, atlas_path), "sha256": compute_sha256(atlas_path)},
    }
    outputs = {
        "synthetic_events_json": {"path": _run_rel(run_root, out_path), "sha256": compute_sha256(out_path)},
    }

    write_stage_summary(
        stage_dir,
        stage=STAGE,
        impl_module="stages.stage_ringdown_synth",
        params={"sigma_rel": args.sigma_rel, "seed_base": args.seed_base, "atlas_json": str(args.atlas_json)},
        inputs=inputs,
        outputs=outputs,
        verdict=verdict,
        model_family=MODEL_FAMILY,
        epistemic_status=EPISTEMIC,
    )

    write_manifest(
        stage_dir,
        stage=STAGE,
        artifact_relpaths=[Path("manifest.json"), Path("stage_summary.json"), Path("outputs/synthetic_events.json")],
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
