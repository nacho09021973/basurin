#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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

STAGE = "geometry_select_v0"
MODEL_FAMILY = "phi_phenomenological_v0"
EPISTEMIC = "conjectural/phenomenological"


def _run_rel(run_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_root.resolve()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-json", required=True)
    ap.add_argument("--mapped-features-json", required=True)
    ap.add_argument("--topk", required=True, type=int)
    ap.add_argument("--acc-top1-threshold", required=True, type=float)
    ap.add_argument("--acc-topk-threshold", required=True, type=float)
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
    mapped_path = Path(args.mapped_features_json)
    if not mapped_path.is_absolute():
        mapped_path = Path(args.root) / mapped_path

    atlas = read_json(atlas_path)
    mapped = read_json(mapped_path)

    atlas_ratios = [np.array([float(g["r1"])]) for g in atlas.get("geometries", [])]

    stage_dir = run_root / "experiment" / "ringdown" / STAGE
    outputs_dir = stage_dir / "outputs"
    ensure_dir(outputs_dir)

    case_results = []
    top1_hits = 0
    topk_hits = 0
    features = mapped.get("features", [])
    for feat in features:
        true_idx = int(feat["geometry_index"])
        ranking_pairs = phi_core.rank_atlas(np.array([float(feat["r1_pred"])]), atlas_ratios)
        ranking = [{"geometry_index": int(i), "distance": float(d)} for i, d in ranking_pairs]
        top1 = ranking[0]["geometry_index"]
        topk_set = {x["geometry_index"] for x in ranking[: args.topk]}
        top1_hits += int(top1 == true_idx)
        topk_hits += int(true_idx in topk_set)
        case_results.append(
            {
                "case_id": feat["case_id"],
                "true_geometry_index": true_idx,
                "ranking": ranking,
            }
        )

    n_cases = len(features)
    acc1 = (top1_hits / n_cases) if n_cases else 0.0
    acck = (topk_hits / n_cases) if n_cases else 0.0
    verdict = "PASS" if acc1 >= args.acc_top1_threshold and acck >= args.acc_topk_threshold else "FAIL"

    out_obj = {
        "version": "geometry_ranking.v0",
        "model": {"family": MODEL_FAMILY, "epistemic_status": EPISTEMIC},
        "n_cases": n_cases,
        "topk": args.topk,
        "accuracy_top1": acc1,
        "accuracy_topk": acck,
        "cases": case_results,
        "verdict": verdict,
    }
    out_path = outputs_dir / "geometry_ranking.json"
    write_json_atomic(out_path, out_obj)

    write_stage_summary(
        stage_dir,
        stage=STAGE,
        impl_module="stages.stage_geometry_select_v0",
        params={
            "topk": args.topk,
            "acc_top1_threshold": args.acc_top1_threshold,
            "acc_topk_threshold": args.acc_topk_threshold,
            "atlas_json": str(args.atlas_json),
            "mapped_features_json": str(args.mapped_features_json),
        },
        inputs={
            "run_valid": {"path": _run_rel(run_root, run_valid_path), "sha256": compute_sha256(run_valid_path)},
            "atlas_json": {"path": _run_rel(run_root, atlas_path), "sha256": compute_sha256(atlas_path)},
            "mapped_features_json": {"path": _run_rel(run_root, mapped_path), "sha256": compute_sha256(mapped_path)},
        },
        outputs={
            "geometry_ranking_json": {"path": _run_rel(run_root, out_path), "sha256": compute_sha256(out_path)},
        },
        verdict=verdict,
        model_family=MODEL_FAMILY,
        epistemic_status=EPISTEMIC,
    )
    write_manifest(
        stage_dir,
        stage=STAGE,
        artifact_relpaths=[Path("manifest.json"), Path("stage_summary.json"), Path("outputs/geometry_ranking.json")],
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
