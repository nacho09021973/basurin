#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

import phi_core
from basurin_io import (
    compute_sha256,
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "experiment/ringdown/EDC_01__degeneracy_audit"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_rel(run_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_dir.resolve()))


def _resolve_atlas_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "inputs" / "atlas.json",
        run_dir / "dictionary" / "outputs" / "atlas.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    expected = " | ".join(str(x) for x in candidates)
    raise SystemExit(f"[BASURIN ABORT] missing atlas artifact; expected one of: {expected}")


def _bootstrap_metrics(obs_vec: np.ndarray, atlas_ratios: list[np.ndarray], topk: int, true_idx: int | None, bootstrap_k: int, bootstrap_seed: int) -> dict[str, float | None]:
    if bootstrap_k <= 0:
        return {}

    base_ranking = phi_core.rank_atlas(obs_vec, atlas_ratios)
    base_topk = [int(i) for i, _ in base_ranking[:topk]]
    base_winner = base_topk[0] if base_topk else None

    winner_hits = 0
    overlap_sum = 0.0
    true_ranks: list[int] = []
    n_dim = int(obs_vec.shape[0])

    for rep in range(bootstrap_k):
        rng = np.random.default_rng(bootstrap_seed + rep)
        sample_idx = rng.integers(0, n_dim, size=n_dim)
        obs_rep = obs_vec[sample_idx]
        atlas_rep = [r[sample_idx] for r in atlas_ratios]
        ranking = phi_core.rank_atlas(obs_rep, atlas_rep)
        rep_topk = [int(i) for i, _ in ranking[:topk]]
        rep_winner = rep_topk[0] if rep_topk else None
        winner_hits += int(rep_winner == base_winner)

        overlap_sum += len(set(base_topk).intersection(rep_topk)) / max(topk, 1)

        if true_idx is not None:
            pos = next((p for p, (idx, _) in enumerate(ranking) if int(idx) == true_idx), None)
            if pos is not None:
                true_ranks.append(int(pos))

    rank_variance_true = float(np.var(np.asarray(true_ranks, dtype=float))) if true_ranks else None
    return {
        "winner_stability": winner_hits / float(bootstrap_k),
        "overlap_topk": overlap_sum / float(bootstrap_k),
        "rank_variance_true": rank_variance_true,
    }


def _group_metrics(rows: list[dict[str, Any]], topk: int, include_bootstrap: bool) -> dict[str, Any]:
    n = len(rows)
    top1 = sum(int(r["hit_top1"]) for r in rows)
    topk_hits = sum(int(r["hit_topk"]) for r in rows)
    margins = [float(r["margin"]) for r in rows if r.get("margin") is not None]

    payload: dict[str, Any] = {
        "n_cases": n,
        "acc_top1": (top1 / n) if n else None,
        "acc_topk": (topk_hits / n) if n else None,
        "k": topk,
    }
    if margins:
        payload["median_margin"] = float(np.median(np.asarray(margins, dtype=float)))

    if include_bootstrap and rows:
        bm = [r["bootstrap"] for r in rows if "bootstrap" in r]
        if bm:
            payload["bootstrap"] = {
                "winner_stability": float(np.mean([x["winner_stability"] for x in bm])),
                "overlap_topk": float(np.mean([x["overlap_topk"] for x in bm])),
                "rank_variance_true": float(np.mean([x["rank_variance_true"] for x in bm if x["rank_variance_true"] is not None])) if any(x["rank_variance_true"] is not None for x in bm) else None,
            }

    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="EDC_01 degeneracy audit")
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m-per-group", type=int, default=3)
    ap.add_argument("--n-cases-per-geom", type=int, default=2)
    ap.add_argument("--bootstrap-k", type=int, default=0)
    ap.add_argument("--bootstrap-seed", type=int, default=123)
    ap.add_argument("--q-low", type=float, default=0.33)
    ap.add_argument("--q-high", type=float, default=0.66)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--acc-top1-a-threshold", type=float, default=0.6)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    synth_path = (run_dir / "ringdown_synth" / "outputs" / "synthetic_events.json").resolve()
    if not synth_path.exists():
        print(f"[BASURIN ABORT] missing canonical synthetic events: {synth_path}", file=sys.stderr)
        return 2

    try:
        atlas_path = _resolve_atlas_path(run_dir)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    synth = _read_json(synth_path)
    atlas = _read_json(atlas_path)

    geometries = atlas.get("geometries", [])
    if not geometries:
        print(f"[BASURIN ABORT] atlas without geometries at {atlas_path}", file=sys.stderr)
        return 2

    atlas_ratios = [np.array([float(g["r1"])], dtype=float) for g in geometries]
    geom_ids = [int(g["geometry_index"]) for g in geometries]
    r1_map = {int(g["geometry_index"]): float(g["r1"]) for g in geometries}

    deltas: dict[int, float] = {}
    for gi in geom_ids:
        log_i = math.log(r1_map[gi])
        others = [abs(log_i - math.log(r1_map[gj])) for gj in geom_ids if gj != gi]
        deltas[gi] = min(others) if others else 0.0

    delta_values = np.asarray([deltas[g] for g in geom_ids], dtype=float)
    q_low = float(np.quantile(delta_values, args.q_low))
    q_high = float(np.quantile(delta_values, args.q_high))

    g_a = sorted([g for g in geom_ids if deltas[g] >= q_high])[: args.m_per_group]
    g_b = sorted([g for g in geom_ids if q_low <= deltas[g] < q_high])[: args.m_per_group]
    g_c = sorted([g for g in geom_ids if deltas[g] < q_low])[: args.m_per_group]

    cases = synth.get("cases", [])
    by_geom: dict[int, list[dict[str, Any]]] = {}
    for case in cases:
        gid = case.get("geometry_index", case.get("geometry_id"))
        if gid is None:
            continue
        by_geom.setdefault(int(gid), []).append(case)

    for gid in list(by_geom.keys()):
        by_geom[gid] = sorted(by_geom[gid], key=lambda c: str(c.get("case_id", "")))

    rows_by_group: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}

    for group_name, group_geoms in (("A", g_a), ("B", g_b), ("C", g_c)):
        for gid in group_geoms:
            selected_cases = by_geom.get(gid, [])[: args.n_cases_per_geom]
            for case in selected_cases:
                inv = phi_core.inverse_model(float(case["f_obs"]), float(case["tau_obs"]), alpha=1.0)
                obs_vec = np.array([float(inv["r1_pred"])], dtype=float)
                ranking_pairs = phi_core.rank_atlas(obs_vec, atlas_ratios)
                ranking = [{"geometry_index": int(i), "distance": float(d)} for i, d in ranking_pairs]
                topk = ranking[: args.k]
                top1_id = topk[0]["geometry_index"] if topk else None
                truth = int(case.get("geometry_index", case.get("geometry_id")))
                margin = None
                if len(topk) >= 2:
                    margin = float(topk[1]["distance"] - topk[0]["distance"])

                row: dict[str, Any] = {
                    "case_id": str(case.get("case_id")),
                    "geometry_id": gid,
                    "truth_geometry_id": truth,
                    "hit_top1": bool(top1_id == truth),
                    "hit_topk": any(x["geometry_index"] == truth for x in topk),
                    "margin": margin,
                }
                if args.bootstrap_k > 0:
                    row["bootstrap"] = _bootstrap_metrics(
                        obs_vec=obs_vec,
                        atlas_ratios=atlas_ratios,
                        topk=args.k,
                        true_idx=truth,
                        bootstrap_k=args.bootstrap_k,
                        bootstrap_seed=args.bootstrap_seed,
                    )
                rows_by_group[group_name].append(row)

    group_payload: dict[str, Any] = {}
    for gname, g_ids in (("A", g_a), ("B", g_b), ("C", g_c)):
        g_rows = rows_by_group[gname]
        group_payload[gname] = {
            "geom_ids": g_ids,
            "case_ids": [r["case_id"] for r in g_rows],
            "metrics": _group_metrics(g_rows, args.k, args.bootstrap_k > 0),
        }

    acc_a = group_payload["A"]["metrics"].get("acc_top1")
    acc_c = group_payload["C"]["metrics"].get("acc_top1")
    threshold = float(args.acc_top1_a_threshold)

    if acc_a is not None and acc_a < threshold:
        verdict = "MODEL_MISSPECIFIED"
    elif acc_a is not None and acc_c is not None and acc_c > acc_a:
        verdict = "MODEL_HALLUCINATING"
    elif acc_a is not None and acc_c is not None and acc_a >= threshold and acc_c < 0.4:
        verdict = "DEGENERACY_INEVITABLE"
    else:
        verdict = "INCONCLUSIVE"

    evidence = (
        f"threshold_acc_top1_A={threshold:.3f}; "
        f"acc_top1_A={acc_a}; acc_top1_C={acc_c}; "
        f"q_low={args.q_low}; q_high={args.q_high}"
    )

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    result_path = outputs_dir / "edc_results.json"

    params: dict[str, Any] = {
        "k": args.k,
        "m_per_group": args.m_per_group,
        "n_cases_per_geom": args.n_cases_per_geom,
        "bootstrap_k": args.bootstrap_k,
        "bootstrap_seed": args.bootstrap_seed,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "acc_top1_a_threshold": threshold,
    }
    if args.sigma is not None:
        params["sigma"] = args.sigma

    inputs = {
        "synthetic_events": {
            "path": _run_rel(run_dir, synth_path),
            "sha256": compute_sha256(synth_path),
        },
        "atlas": {
            "path": _run_rel(run_dir, atlas_path),
            "sha256": compute_sha256(atlas_path),
        },
    }

    result_payload = {
        "schema_version": "edc_01_degeneracy_audit.v1",
        "run_id": args.run,
        "params": params,
        "inputs": inputs,
        "groups": group_payload,
        "verdict": verdict,
        "evidence": evidence,
    }
    result_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")

    write_manifest(stage_dir, {"edc_results": result_path})
    write_stage_summary(
        stage_dir,
        {
            "stage": STAGE_NAME,
            "run": args.run,
            "params": params,
            "inputs": inputs,
            "outputs": {
                "edc_results": {
                    "path": _run_rel(run_dir, result_path),
                    "sha256": compute_sha256(result_path),
                }
            },
            "verdict": verdict,
            "evidence": evidence,
        },
    )

    print(f"[EDC_01] verdict={verdict}")
    print(f"[EDC_01] output={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
