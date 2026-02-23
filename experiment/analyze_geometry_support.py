#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import resolve_out_root, sha256_file, utc_now_iso, validate_run_id, write_json_atomic

EXPERIMENT_NAME = "geometry_support_v1"
DEFAULT_TARGET_GEOMETRY = "bK_141_a0.95_df-0.20_dQ+0.20"


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den == 0.0:
        return None
    return num / den


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_values[lo])
    w = k - lo
    return float(sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w)


def _find_rank(ranked_ids: list[str], geometry_id: str) -> int:
    for i, gid in enumerate(ranked_ids):
        if gid == geometry_id:
            return i
    return -1


def _load_runs_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            "Missing required runs list. "
            f"Expected: {path}\n"
            "Command to regenerate upstream (example): "
            "python mvp/pipeline.py multi --events <E1,E2,...> --atlas-path <atlas_path>\n"
            "Candidates detected: <none>"
        )
    run_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        x = line.strip()
        if x and not x.startswith("#"):
            run_ids.append(x)
    if not run_ids:
        raise ValueError(f"Runs list is empty: {path}")
    return run_ids


def _load_atlas_map(atlas_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(atlas_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    return {str(e.get("geometry_id")): e for e in entries if "geometry_id" in e}


def _spin_family(geometry_id: str | None) -> str:
    if not geometry_id:
        return "unknown"
    if "_a0.95_" in geometry_id:
        return "a0.95"
    if "_a0.80_" in geometry_id:
        return "a0.80"
    return "other"


def analyze_geometry_support(
    out_root: Path,
    run_id: str,
    runs_ids_path: Path,
    atlas_path: Path,
    target_geometry_id: str,
    k_values: list[int],
    worst_n: int,
) -> dict[str, Any]:
    validate_run_id(run_id, out_root)
    stage_dir = out_root / run_id / "experiment" / EXPERIMENT_NAME
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    source_run_ids = _load_runs_list(runs_ids_path)
    atlas_by_id = _load_atlas_map(atlas_path)

    per_event: list[dict[str, Any]] = []
    ranked_ids_by_run: dict[str, list[str]] = {}
    source_hashes: dict[str, str] = {
        "runs_ids": sha256_file(runs_ids_path),
        "atlas": sha256_file(atlas_path),
    }

    for src_run in source_run_ids:
        cs_path = out_root / src_run / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        if not cs_path.exists():
            raise FileNotFoundError(
                "Missing upstream input. "
                f"Expected: {cs_path}\n"
                "Command to regenerate upstream (example): "
                f"python mvp/pipeline.py single --run-id {src_run} --event-id <EVENT_ID> --atlas-path {atlas_path}\n"
                "Candidates detected: <none>"
            )
        cs = json.loads(cs_path.read_text(encoding="utf-8"))
        ranked = cs.get("ranked_all", [])
        ranked_ids = [str(r.get("geometry_id")) for r in ranked]
        rank = _find_rank(ranked_ids, target_geometry_id)
        top5_ids = ranked_ids[:5]

        obs = cs.get("observables", {})
        cov = cs.get("covariance_logspace", {})
        event_id = cs.get("event_id") or cs.get("metadata", {}).get("event_id") or src_run

        ranked_ids_by_run[src_run] = ranked_ids
        per_event.append(
            {
                "run_id": src_run,
                "event_id": event_id,
                "rank_target": rank,
                "n_ranked": len(ranked_ids),
                "top5_ids": top5_ids,
                "f_hz": obs.get("f_hz"),
                "Q": obs.get("Q"),
                "sigma_logf": cov.get("sigma_logf"),
                "sigma_logQ": cov.get("sigma_logQ"),
                "cov_logf_logQ": cov.get("cov_logf_logQ"),
                "d2_min": cs.get("d2_min"),
                "epsilon": cs.get("epsilon"),
                "metric": cs.get("metric"),
                "best_geometry_id": ranked_ids[0] if ranked_ids else None,
                "best_entry_id": (cs.get("atlas_posterior") or {}).get("best_entry_id"),
                "n_compatible": len(cs.get("compatible_geometries", [])),
            }
        )
        source_hashes[f"compatible_set::{src_run}"] = sha256_file(cs_path)

    support_by_k: dict[str, dict[str, int]] = {}
    top_by_k: dict[str, list[dict[str, Any]]] = {}
    in_topk_by_event: dict[str, dict[str, bool]] = {}
    for k in k_values:
        cnt: Counter[str] = Counter()
        for row in per_event:
            src_run = row["run_id"]
            ids = ranked_ids_by_run[src_run][:k]
            cnt.update(ids)
            per_event_key = f"{src_run}::{row['event_id']}"
            if per_event_key not in in_topk_by_event:
                in_topk_by_event[per_event_key] = {}
            in_topk_by_event[per_event_key][str(k)] = target_geometry_id in ids
        ordered = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        support_by_k[str(k)] = dict(ordered)
        top_by_k[str(k)] = [{"geometry_id": gid, "support": sup} for gid, sup in ordered[:10]]

    ranks = sorted(float(x["rank_target"]) for x in per_event)
    rank_summary = {
        "geometry_id": target_geometry_id,
        "n_events": len(per_event),
        "min": min(ranks) if ranks else None,
        "max": max(ranks) if ranks else None,
        "mean": mean(ranks) if ranks else None,
        "p10": _percentile(ranks, 0.10),
        "p50": _percentile(ranks, 0.50),
        "p90": _percentile(ranks, 0.90),
        "histogram": dict(sorted(Counter(int(r) for r in ranks).items(), key=lambda kv: kv[0])),
    }

    def _is_in_top_k(rank: int, k: int) -> bool:
        return rank >= 0 and rank < k

    worst_events_target: list[dict[str, Any]] = []
    for row in per_event:
        rank_target = int(row["rank_target"])
        top5_geometry_ids = list(row["top5_ids"])
        worst_events_target.append(
            {
                "run_id": row["run_id"],
                "event_id": row["event_id"],
                "rank_target": rank_target,
                "best_geometry_id": row.get("best_geometry_id"),
                "in_top3": _is_in_top_k(rank_target, 3),
                "in_top5": _is_in_top_k(rank_target, 5),
                "in_top10": _is_in_top_k(rank_target, 10),
                "observables": {"f_hz": row.get("f_hz"), "Q": row.get("Q")},
                "covariance_logspace": {
                    "sigma_logf": row.get("sigma_logf"),
                    "sigma_logQ": row.get("sigma_logQ"),
                },
                "d2_min": row.get("d2_min"),
                "epsilon": row.get("epsilon"),
                "metric": row.get("metric"),
                "top5_geometry_ids": top5_geometry_ids,
            }
        )

    def _worst_event_key(row: dict[str, Any]) -> tuple[int, str]:
        rank = int(row["rank_target"])
        rank_sort = rank if rank >= 0 else 10**9
        return (-rank_sort, str(row["run_id"]))

    worst_events_target = sorted(worst_events_target, key=_worst_event_key)[:worst_n]

    # Correlations rank(target) vs observables/covariance
    corr_inputs: dict[str, list[float]] = {"f_hz": [], "Q": [], "sigma_logf": [], "sigma_logQ": [], "d2_min": []}
    corr_rank: list[float] = []
    for row in per_event:
        rv = float(row["rank_target"])
        for key in list(corr_inputs.keys()):
            vv = row.get(key)
            if isinstance(vv, (int, float)):
                corr_inputs[key].append(float(vv))
            else:
                corr_inputs[key].append(float("nan"))
        corr_rank.append(rv)

    correlations: dict[str, float | None] = {}
    for key, values in corr_inputs.items():
        paired = [(r, v) for r, v in zip(corr_rank, values) if not math.isnan(v)]
        if not paired:
            correlations[key] = None
            continue
        rr = [x[0] for x in paired]
        vv = [x[1] for x in paired]
        correlations[key] = _pearson_corr(rr, vv)

    atlas_focus_ids = [target_geometry_id, "bK_140_a0.95_df-0.20_dQ+0.10", "bK_107_a0.80_df-0.20_dQ+0.20"]
    atlas_focus = {gid: atlas_by_id.get(gid) for gid in atlas_focus_ids}

    atlas_target = atlas_by_id.get(target_geometry_id)
    atlas_comparisons: dict[str, Any] = {}
    for comparison_id in ["bK_140_a0.95_df-0.20_dQ+0.10", "bK_107_a0.80_df-0.20_dQ+0.20"]:
        if comparison_id in atlas_by_id:
            atlas_comparisons[comparison_id] = atlas_by_id[comparison_id]

    atlas_bk141 = atlas_by_id.get(target_geometry_id)
    atlas_bk107_id = "bK_107_a0.80_df-0.20_dQ+0.20"
    atlas_bk107 = atlas_by_id.get(atlas_bk107_id)
    atlas_comparison = {
        "target": {"geometry_id": target_geometry_id, "entry": atlas_bk141},
        "comparison": {"geometry_id": atlas_bk107_id, "entry": atlas_bk107},
    }

    spin_family_counts = dict(
        sorted(Counter(_spin_family(str(row.get("best_entry_id") or "")) for row in per_event).items(), key=lambda kv: kv[0])
    )

    transitions_3_5_10: list[dict[str, Any]] = []
    has_3 = "3" in {str(k) for k in k_values}
    has_5 = "5" in {str(k) for k in k_values}
    has_10 = "10" in {str(k) for k in k_values}
    if has_3 and has_5 and has_10:
        for row in per_event:
            per_event_key = f"{row['run_id']}::{row['event_id']}"
            flags = in_topk_by_event.get(per_event_key, {})
            in3 = bool(flags.get("3", False))
            in5 = bool(flags.get("5", False))
            in10 = bool(flags.get("10", False))
            if len({in3, in5, in10}) > 1:
                transitions_3_5_10.append(
                    {
                        "run_id": row["run_id"],
                        "event_id": row["event_id"],
                        "rank_target": row["rank_target"],
                        "in_top3": in3,
                        "in_top5": in5,
                        "in_top10": in10,
                        "best_entry_id": row["best_entry_id"],
                        "top5_ids": row["top5_ids"],
                    }
                )
    transitions_3_5_10.sort(key=lambda x: (x["rank_target"], str(x["event_id"])))

    # Compare top-1/2/3 by support at each K
    support_comp: dict[str, Any] = {}
    for k, top in top_by_k.items():
        first = top[0] if len(top) > 0 else None
        second = top[1] if len(top) > 1 else None
        third = top[2] if len(top) > 2 else None
        gap_1_2 = None
        if first and second:
            gap_1_2 = first["support"] - second["support"]
        support_comp[k] = {"top1": first, "top2": second, "top3": third, "gap_1_2": gap_1_2}

    report = {
        "generated_at": utc_now_iso(),
        "target_geometry_id": target_geometry_id,
        "source_run_count": len(source_run_ids),
        "k_values": k_values,
        "support_by_k": support_by_k,
        "support_definition": (
            "support_by_k[K] counts events where geometry appears in ranked_all[:K]. "
            "For K=1 this equals the top-1 count."
        ),
        "top_by_k": top_by_k,
        "support_comparison": support_comp,
        "rank_summary": rank_summary,
        "worst_events": worst_events_target,
        "worst_events_target": worst_events_target,
        "correlations_rank_target": correlations,
        "atlas_focus": atlas_focus,
        "atlas_target": atlas_target,
        "atlas_comparisons": atlas_comparisons,
        "atlas_bk141_vs_bk107": atlas_comparison,
        "best_entry_spin_family_counts": spin_family_counts,
        "target_membership_transitions_3_5_10": transitions_3_5_10,
    }

    report_path = outputs_dir / "geometry_support_report.json"
    per_event_path = outputs_dir / "per_event_metrics.json"
    write_json_atomic(report_path, report)
    write_json_atomic(per_event_path, {"events": per_event})

    stage_summary = {
        "stage": f"experiment/{EXPERIMENT_NAME}",
        "verdict": "PASS",
        "run_id": run_id,
        "results": {
            "target_geometry_id": target_geometry_id,
            "source_run_count": len(source_run_ids),
            "support_k": {str(k): support_by_k[str(k)].get(target_geometry_id, 0) for k in k_values},
            "rank_p50": rank_summary["p50"],
            "rank_p90": rank_summary["p90"],
        },
    }
    stage_summary_path = stage_dir / "stage_summary.json"
    write_json_atomic(stage_summary_path, stage_summary)

    output_hashes = {
        "geometry_support_report.json": sha256_file(report_path),
        "per_event_metrics.json": sha256_file(per_event_path),
        "stage_summary.json": sha256_file(stage_summary_path),
    }
    manifest = {
        "schema_version": "mvp_manifest_v1",
        "stage": f"experiment/{EXPERIMENT_NAME}",
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "inputs": {
            "runs_ids": str(runs_ids_path),
            "atlas_path": str(atlas_path),
            "source_run_ids": source_run_ids,
            "sha256": source_hashes,
        },
        "artifacts": {
            "outputs": {
                "geometry_support_report": "outputs/geometry_support_report.json",
                "per_event_metrics": "outputs/per_event_metrics.json",
            },
            "stage_summary": "stage_summary.json",
        },
        "sha256": output_hashes,
    }
    manifest_path = stage_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    print(f"OUT_ROOT={out_root.resolve()}")
    print(f"STAGE_DIR={stage_dir.resolve()}")
    print(f"OUTPUTS_DIR={outputs_dir.resolve()}")
    print(f"STAGE_SUMMARY={stage_summary_path.resolve()}")
    print(f"MANIFEST={manifest_path.resolve()}")

    return {
        "stage_dir": str(stage_dir),
        "outputs_dir": str(outputs_dir),
        "manifest": str(manifest_path),
        "stage_summary": str(stage_summary_path),
        "report": report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Analyze geometry support across many BASURIN runs.")
    ap.add_argument("--run-id", required=True, help="Output run_id where experiment artifacts are written")
    ap.add_argument("--runs-ids", default="runs_50_ids.txt", help="Text file with source run_ids (one per line)")
    ap.add_argument("--atlas-path", default="docs/ringdown/atlas/atlas_real_v1_s4.json")
    ap.add_argument("--target-geometry-id", default=DEFAULT_TARGET_GEOMETRY)
    ap.add_argument("--k-values", default="1,3,5,10,20,50", help="Comma-separated K values")
    ap.add_argument("--worst-n", type=int, default=10)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    k_values = [int(x.strip()) for x in str(args.k_values).split(",") if x.strip()]
    out_root = resolve_out_root("runs")
    analyze_geometry_support(
        out_root=out_root,
        run_id=args.run_id,
        runs_ids_path=Path(args.runs_ids),
        atlas_path=Path(args.atlas_path),
        target_geometry_id=args.target_geometry_id,
        k_values=k_values,
        worst_n=args.worst_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
