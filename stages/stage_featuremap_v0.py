#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

STAGE = "ringdown_featuremap_v0"
MODEL_FAMILY = "phi_phenomenological_v0"
EPISTEMIC = "conjectural/phenomenological"


def _run_rel(run_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_root.resolve()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--synthetic-events-json", required=True)
    ap.add_argument("--stage-subdir", default=None)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    run_root = Path(args.root) / "runs" / args.run
    run_pass, run_valid_path = load_run_valid_verdict(run_root)
    if not run_pass:
        print("[BASURIN ABORT] RUN_VALID != PASS", file=sys.stderr)
        return 2

    synth_path = Path(args.synthetic_events_json)
    if not synth_path.is_absolute():
        synth_path = Path(args.root) / synth_path
    synth = read_json(synth_path)
    cases = synth.get("cases", [])

    stage_dir = run_root / args.stage_subdir if args.stage_subdir else run_root / "experiment" / "ringdown" / "featuremap_v0"
    outputs_dir = stage_dir / "outputs"
    ensure_dir(outputs_dir)

    features = []
    for case in cases:
        inv = phi_core.inverse_model(float(case["f_obs"]), float(case["tau_obs"]), alpha=1.0)
        features.append(
            {
                "case_id": case["case_id"],
                "geometry_index": int(case["geometry_index"]),
                "r1_pred": float(inv["r1_pred"]),
                "Q_obs": float(inv["Q_obs"]),
            }
        )

    out_obj = {
        "version": "mapped_features.v0",
        "model": {"family": MODEL_FAMILY, "epistemic_status": EPISTEMIC},
        "n_cases": len(features),
        "features": features,
    }
    out_path = outputs_dir / "mapped_features.json"
    write_json_atomic(out_path, out_obj)
    verdict = "PASS" if out_obj["n_cases"] == len(cases) and all(x["r1_pred"] > 1 and x["Q_obs"] > 0 for x in features) else "FAIL"

    write_stage_summary(
        stage_dir,
        stage=STAGE,
        impl_module="stages.stage_featuremap_v0",
        params={"synthetic_events_json": str(args.synthetic_events_json)},
        inputs={
            "run_valid": {"path": _run_rel(run_root, run_valid_path), "sha256": compute_sha256(run_valid_path)},
            "synthetic_events_json": {"path": _run_rel(run_root, synth_path), "sha256": compute_sha256(synth_path)},
        },
        outputs={
            "mapped_features_json": {"path": _run_rel(run_root, out_path), "sha256": compute_sha256(out_path)},
        },
        verdict=verdict,
        model_family=MODEL_FAMILY,
        epistemic_status=EPISTEMIC,
    )
    write_manifest(
        stage_dir,
        stage=STAGE,
        artifact_relpaths=[Path("manifest.json"), Path("stage_summary.json"), Path("outputs/mapped_features.json")],
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
