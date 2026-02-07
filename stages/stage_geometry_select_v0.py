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


def _bootstrap_metrics(
    obs_vec: np.ndarray,
    atlas_ratios: list[np.ndarray],
    topk: int,
    true_idx: int | None,
    bootstrap_k: int,
    bootstrap_seed: int,
) -> dict[str, float | None]:
    """Estimate ranking stability via deterministic feature bootstrap.

    Resamples feature dimensions with replacement and reruns ranker.
    """
    if bootstrap_k <= 0:
        return {}

    base_ranking_pairs = phi_core.rank_atlas(obs_vec, atlas_ratios)
    base_topk = [int(i) for i, _ in base_ranking_pairs[:topk]]
    base_winner = base_topk[0] if base_topk else None

    winner_hits = 0
    overlap_sum = 0.0
    true_ranks: list[int] = []
    n_dim = int(obs_vec.shape[0])

    for rep in range(bootstrap_k):
        rng = np.random.default_rng(bootstrap_seed + rep)
        sample_idx = rng.integers(0, n_dim, size=n_dim)
        obs_rep = obs_vec[sample_idx]
        atlas_rep = [ar[sample_idx] for ar in atlas_ratios]
        rep_ranking_pairs = phi_core.rank_atlas(obs_rep, atlas_rep)
        rep_topk = [int(i) for i, _ in rep_ranking_pairs[:topk]]

        rep_winner = rep_topk[0] if rep_topk else None
        winner_hits += int(rep_winner == base_winner)

        denom = max(topk, 1)
        overlap = len(set(base_topk).intersection(rep_topk)) / denom
        overlap_sum += float(overlap)

        if true_idx is not None:
            true_rank = next((pos for pos, (idx, _) in enumerate(rep_ranking_pairs) if int(idx) == true_idx), None)
            if true_rank is not None:
                true_ranks.append(int(true_rank))

    rank_variance_true = float(np.var(np.asarray(true_ranks, dtype=float))) if true_ranks else None
    return {
        "winner_stability": winner_hits / float(bootstrap_k),
        "overlap_topk": overlap_sum / float(bootstrap_k),
        "rank_variance_true": rank_variance_true,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-json", required=True)
    ap.add_argument("--mapped-features-json", required=True)
    ap.add_argument("--topk", required=True, type=int)
    ap.add_argument("--acc-top1-threshold", required=True, type=float)
    ap.add_argument("--acc-topk-threshold", required=True, type=float)
    ap.add_argument("--bootstrap-k", required=False, type=int, default=0)
    ap.add_argument("--bootstrap-seed", required=False, type=int, default=123)
    ap.add_argument("--stage-subdir", default=None)
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

    stage_dir = run_root / args.stage_subdir if args.stage_subdir else run_root / "experiment" / "ringdown" / STAGE
    outputs_dir = stage_dir / "outputs"
    ensure_dir(outputs_dir)

    case_results = []
    top1_hits = 0
    topk_hits = 0
    features = mapped.get("features", [])
    bootstrap_case_metrics: list[dict[str, float | None]] = []
    for feat in features:
        true_idx = int(feat["geometry_index"])
        obs_vec = np.array([float(feat["r1_pred"])], dtype=float)
        ranking_pairs = phi_core.rank_atlas(obs_vec, atlas_ratios)
        ranking = [{"geometry_index": int(i), "distance": float(d)} for i, d in ranking_pairs]
        top1 = ranking[0]["geometry_index"]
        topk_set = {x["geometry_index"] for x in ranking[: args.topk]}
        top1_hits += int(top1 == true_idx)
        topk_hits += int(true_idx in topk_set)
        bootstrap = _bootstrap_metrics(
            obs_vec=obs_vec,
            atlas_ratios=atlas_ratios,
            topk=args.topk,
            true_idx=true_idx,
            bootstrap_k=args.bootstrap_k,
            bootstrap_seed=args.bootstrap_seed,
        )
        if bootstrap:
            bootstrap_case_metrics.append(bootstrap)
        case_results.append(
            {
                "case_id": feat["case_id"],
                "true_geometry_index": true_idx,
                "ranking": ranking,
                **({"bootstrap": bootstrap} if bootstrap else {}),
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
    if args.bootstrap_k > 0:
        out_obj["bootstrap"] = {
            "k": args.bootstrap_k,
            "seed": args.bootstrap_seed,
            "winner_stability": float(np.mean([m["winner_stability"] for m in bootstrap_case_metrics])) if bootstrap_case_metrics else None,
            "overlap_topk": float(np.mean([m["overlap_topk"] for m in bootstrap_case_metrics])) if bootstrap_case_metrics else None,
            "rank_variance_true": float(np.mean([m["rank_variance_true"] for m in bootstrap_case_metrics if m["rank_variance_true"] is not None])) if any(m["rank_variance_true"] is not None for m in bootstrap_case_metrics) else None,
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
            "bootstrap_k": args.bootstrap_k,
            "bootstrap_seed": args.bootstrap_seed,
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
