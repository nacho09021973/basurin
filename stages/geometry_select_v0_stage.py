#!/usr/bin/env python3
"""
stages/geometry_select_v0_stage.py
----------------------------------
THE SELECTOR: Given mapped ringdown features and an atlas of holographic
geometries, rank the atlas entries by proximity and select top-k matches.

This is the consumer that converts the bridge output into a scientific
conclusion: "for this ringdown signal, the most likely geometry is X".

Inputs:
  --run <run_id>
  --mapped-features <path>   Output of ringdown_featuremap_v0_stage.py
  --atlas <path>             atlas.json from 04_diccionario.py
  --top-k <int>              Number of top matches to report (default: 3)

Outputs:
  runs/<run>/geometry_select_v0/
    manifest.json
    stage_summary.json
    outputs/geometry_ranking.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    get_runs_root,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "geometry_select_v0"
EXIT_CONTRACT_FAIL = 2


def log_ratio_distance(r_obs: list[float], r_atlas: list[float]) -> float:
    """Euclidean distance in log-ratio space.

    Uses log because ratios are positive and relative scale matters
    more than absolute differences.
    """
    n = min(len(r_obs), len(r_atlas))
    d_sq = 0.0
    for i in range(n):
        if r_obs[i] <= 0 or r_atlas[i] <= 0:
            return float("inf")
        diff = math.log(r_obs[i]) - math.log(r_atlas[i])
        d_sq += diff * diff
    return math.sqrt(d_sq)


def joint_distance(
    obs_ratios: list[float],
    obs_M2_proxy: float | None,
    atlas_ratios: list[float],
    atlas_M2_0: float | None,
    L: float = 1.0,
) -> float:
    """Joint distance using both ratios AND eigenvalue scale.

    The ratio r_1 comes from Q (damping), while M2_0 comes from f (frequency).
    Using both gives two independent constraints, dramatically improving
    discrimination.

    Distance = sqrt( d_ratio^2 + d_scale^2 )
    where d_ratio is in log-ratio space and d_scale is in log-M2 space.
    """
    d_ratio = log_ratio_distance(obs_ratios, atlas_ratios)

    if obs_M2_proxy is not None and atlas_M2_0 is not None and atlas_M2_0 > 0:
        # obs_M2_proxy = omega_0^2 = M2_0 * L^2 (from featuremap)
        atlas_M2_proxy = atlas_M2_0 * L * L
        if obs_M2_proxy > 0 and atlas_M2_proxy > 0:
            d_scale = abs(math.log(obs_M2_proxy) - math.log(atlas_M2_proxy))
        else:
            d_scale = float("inf")
    else:
        d_scale = 0.0  # fallback to ratio-only

    return math.sqrt(d_ratio ** 2 + d_scale ** 2)


def rank_atlas(
    obs_ratios: list[float],
    atlas_theories: list[dict],
    top_k: int,
    obs_M2_proxy: float | None = None,
    L: float = 1.0,
) -> list[dict]:
    """Rank atlas theories by proximity to observed features.

    Uses joint distance on (ratios, M2_0) when M2_0_proxy is available.
    Falls back to ratio-only distance otherwise.
    """
    scored = []
    for theory in atlas_theories:
        atlas_ratios = theory.get("ratios", [])
        if not atlas_ratios:
            continue
        dist = joint_distance(
            obs_ratios, obs_M2_proxy,
            atlas_ratios, theory.get("M2_0"),
            L=L,
        )
        scored.append({
            "theory_id": theory["id"],
            "delta": theory["delta"],
            "M2_0": theory.get("M2_0"),
            "atlas_ratios": atlas_ratios,
            "distance": dist,
        })

    scored.sort(key=lambda x: x["distance"])
    return scored[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Geometry selector v0: rank atlas theories by ringdown proximity"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument("--mapped-features", required=True,
                     help="mapped_features.json from featuremap stage")
    ap.add_argument("--atlas", required=True,
                     help="atlas.json from 04_diccionario.py")
    ap.add_argument("--top-k", type=int, default=3, dest="top_k",
                     help="Number of top matches (default: 3)")
    ap.add_argument("--out-root", default="runs",
                     help="Output root (default: runs)")
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)
    require_run_valid(out_root, args.run)

    # Load mapped features
    mapped_path = Path(args.mapped_features)
    if not mapped_path.is_absolute():
        mapped_path = Path.cwd() / mapped_path
    if not mapped_path.exists():
        print(f"ERROR: mapped features not found: {mapped_path}", file=sys.stderr)
        return EXIT_CONTRACT_FAIL

    mapped = json.loads(mapped_path.read_text(encoding="utf-8"))
    cases = mapped.get("cases", [])
    if not cases:
        print("ERROR: no cases in mapped features", file=sys.stderr)
        return EXIT_CONTRACT_FAIL

    # Load atlas
    atlas_path = Path(args.atlas)
    if not atlas_path.is_absolute():
        atlas_path = Path.cwd() / atlas_path
    if not atlas_path.exists():
        print(f"ERROR: atlas not found: {atlas_path}", file=sys.stderr)
        return EXIT_CONTRACT_FAIL

    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    atlas_section = atlas.get("atlas", atlas)
    theories = atlas_section.get("theories", [])
    if not theories:
        print("ERROR: no theories in atlas", file=sys.stderr)
        return EXIT_CONTRACT_FAIL

    # Rank for each case
    rankings = []
    n_correct_top1 = 0
    n_correct_topk = 0
    n_evaluable = 0

    for case in cases:
        m = case.get("mapped", {})
        obs_ratios = m.get("ratios")
        if obs_ratios is None:
            rankings.append({
                "case_id": case.get("case_id"),
                "error": m.get("error", "no ratios"),
                "top_k": [],
            })
            continue

        obs_M2_proxy = m.get("M2_0_proxy")
        top_k_results = rank_atlas(obs_ratios, theories, args.top_k,
                                    obs_M2_proxy=obs_M2_proxy)

        # Check correctness if truth contains delta (for synthetic validation)
        truth = case.get("truth")
        correct_id = None
        hit_top1 = False
        hit_topk = False

        if truth and "source_theory_id" in truth:
            correct_id = truth["source_theory_id"]
            if top_k_results:
                hit_top1 = (top_k_results[0]["theory_id"] == correct_id)
                hit_topk = any(r["theory_id"] == correct_id for r in top_k_results)
            n_evaluable += 1
            if hit_top1:
                n_correct_top1 += 1
            if hit_topk:
                n_correct_topk += 1

        rankings.append({
            "case_id": case.get("case_id"),
            "observed_ratios": obs_ratios,
            "top_k": top_k_results,
            "truth_id": correct_id,
            "hit_top1": hit_top1 if correct_id is not None else None,
            "hit_topk": hit_topk if correct_id is not None else None,
        })

    # Compute accuracy
    accuracy_top1 = n_correct_top1 / n_evaluable if n_evaluable > 0 else None
    accuracy_topk = n_correct_topk / n_evaluable if n_evaluable > 0 else None

    # Determine verdict
    if n_evaluable > 0:
        verdict = "PASS" if (accuracy_top1 >= 0.70 and accuracy_topk >= 0.95) else "FAIL"
    else:
        verdict = "NO_TRUTH"  # Can't evaluate without ground truth

    # Write outputs
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)

    output_path = outputs_dir / "geometry_ranking.json"
    payload = {
        "schema_version": "geometry_select_v0",
        "created": utc_now_iso(),
        "config": {
            "top_k": args.top_k,
        },
        "n_cases": len(rankings),
        "n_evaluable": n_evaluable,
        "accuracy": {
            "top1": accuracy_top1,
            "topk": accuracy_topk,
            "k": args.top_k,
            "n_atlas": len(theories),
        },
        "verdict": verdict,
        "rankings": rankings,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write_manifest(stage_dir, {"geometry_ranking": output_path})
    write_stage_summary(stage_dir, {
        "stage": STAGE_NAME,
        "run": args.run,
        "config": {
            "top_k": args.top_k,
            "mapped_features": str(mapped_path),
            "atlas": str(atlas_path),
        },
        "inputs": {
            "mapped_features": {
                "path": str(mapped_path),
                "sha256": sha256_file(mapped_path),
            },
            "atlas": {
                "path": str(atlas_path),
                "sha256": sha256_file(atlas_path),
            },
        },
        "n_cases": len(rankings),
        "n_evaluable": n_evaluable,
        "accuracy_top1": accuracy_top1,
        "accuracy_topk": accuracy_topk,
        "verdict": verdict,
    })

    print(f"[geometry_select_v0] Ranked {len(rankings)} cases against {len(theories)} theories")
    if n_evaluable > 0:
        print(f"  accuracy_top1 = {accuracy_top1:.1%} (threshold: 70%)")
        print(f"  accuracy_top{args.top_k} = {accuracy_topk:.1%} (threshold: 95%)")
        print(f"  verdict: {verdict}")
    else:
        print("  verdict: NO_TRUTH (no ground truth to evaluate)")
    print(f"  output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
