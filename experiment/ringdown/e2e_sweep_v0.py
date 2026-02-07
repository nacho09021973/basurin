#!/usr/bin/env python3
"""End-to-end ringdown sweep for BASURIN.

Compatibility note:
- TESIS_PUENTE_PHI.md: thesis thresholds top1>=70% and top3>=95% at N=128, sigma=0.05.
- RELACION_QNM_BULK.md: inverse problem is ill-conditioned; QNM do not uniquely determine metric.
- README.md: information limits and degeneracy reports are scientific outcomes, not bugs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for cand in [HERE.parents[1], HERE.parents[2], HERE.parents[3]]:
    if (cand / "basurin_io.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np

from basurin_io import (
    compute_sha256,
    ensure_dir,
    load_run_valid_verdict,
    read_json,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE = "experiment/ringdown/e2e_sweep_v0"
MODEL_FAMILY = "phi_phenomenological_v0"
EPISTEMIC = "conjectural/phenomenological"


def _run_rel(run_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_root.resolve()))


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _sigma_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def _stable_offset(tag: str) -> int:
    return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _make_atlas(n: int) -> dict:
    deltas = np.linspace(1.5, 8.0, n)
    r1s = np.linspace(1.05, 1.50, n)
    geometries = []
    for i in range(n):
        geometries.append(
            {
                "geometry_index": int(i),
                "delta": float(deltas[i]),
                "r1": float(r1s[i]),
                "M2_0": 4.0,
                "L": 1.0,
            }
        )
    return {
        "version": "atlas.v0",
        "geometries": geometries,
    }


def _degeneracy_for_point(ranking: dict, atlas: dict) -> dict:
    geoms = {int(g["geometry_index"]): g for g in atlas.get("geometries", [])}
    directed: dict[tuple[int, int], int] = {}
    example_by_pair: dict[tuple[int, int], dict] = {}

    for case in ranking.get("cases", []):
        true_idx = int(case["true_geometry_index"])
        pred_idx = int(case["ranking"][0]["geometry_index"])
        if pred_idx == true_idx:
            continue
        key = (true_idx, pred_idx)
        directed[key] = directed.get(key, 0) + 1
        sym = tuple(sorted((true_idx, pred_idx)))
        if sym not in example_by_pair:
            gt = geoms.get(true_idx, {})
            gc = geoms.get(pred_idx, {})
            example_by_pair[sym] = {
                "true": true_idx,
                "confused": pred_idx,
                "case_id": case.get("case_id"),
                "delta_true": gt.get("delta"),
                "delta_confused": gc.get("delta"),
                "r1_true": gt.get("r1"),
                "r1_confused": gc.get("r1"),
            }

    confusions = {f"{i}->{j}": c for (i, j), c in sorted(directed.items(), key=lambda kv: kv[1], reverse=True)}

    sym_counts: dict[tuple[int, int], int] = {}
    for (i, j), c in directed.items():
        sym = tuple(sorted((i, j)))
        sym_counts[sym] = sym_counts.get(sym, 0) + c

    top_pairs_sorted = sorted(sym_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_pairs = [{"i": i, "j": j, "count_sym": c} for (i, j), c in top_pairs_sorted]

    examples = []
    for (i, j), _ in top_pairs_sorted[:5]:
        if (i, j) in example_by_pair:
            examples.append(example_by_pair[(i, j)])

    return {
        "confusions": confusions,
        "top_pairs": top_pairs,
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--Ns", default="8,16,32,64,128")
    ap.add_argument("--sigmas", default="0,0.01,0.05,0.10")
    ap.add_argument("--seed-base", type=int, default=12345)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--thesis-top1", type=float, default=0.70)
    ap.add_argument("--thesis-top3", type=float, default=0.95)
    args = ap.parse_args()

    runs_base = Path(args.root)
    repo_root = HERE.parents[2]
    run_root = runs_base / "runs" / args.run

    run_pass, run_valid_path = load_run_valid_verdict(run_root)
    if not run_pass:
        print("[BASURIN ABORT] RUN_VALID != PASS", file=sys.stderr)
        return 2

    ns = _parse_csv_ints(args.Ns)
    sigmas = _parse_csv_floats(args.sigmas)

    rows = []
    deg_grid = []

    for n in ns:
        for sigma_rel in sigmas:
            tag = f"N{n}_S{_sigma_tag(sigma_rel)}"
            work_base = run_root / STAGE / "work" / tag
            atlas_path = work_base / "atlas.json"
            ensure_dir(work_base)
            atlas = _make_atlas(n)
            write_json_atomic(atlas_path, atlas)

            synth_subdir = f"{STAGE}/work/{tag}/ringdown_synth"
            feat_subdir = f"{STAGE}/work/{tag}/featuremap_v0"
            sel_subdir = f"{STAGE}/work/{tag}/geometry_select_v0"

            _run(
                [
                    sys.executable,
                    str(repo_root / "stages" / "stage_ringdown_synth.py"),
                    "--root",
                    str(runs_base),
                    "--run",
                    args.run,
                    "--stage-subdir",
                    synth_subdir,
                    "--atlas-json",
                    str(atlas_path),
                    "--sigma-rel",
                    str(sigma_rel),
                    "--seed-base",
                    str(args.seed_base + _stable_offset(tag)),
                ]
            )

            synthetic_events_json = run_root / synth_subdir / "outputs" / "synthetic_events.json"
            _run(
                [
                    sys.executable,
                    str(repo_root / "stages" / "stage_featuremap_v0.py"),
                    "--root",
                    str(runs_base),
                    "--run",
                    args.run,
                    "--stage-subdir",
                    feat_subdir,
                    "--synthetic-events-json",
                    str(synthetic_events_json),
                ]
            )

            mapped_features_json = run_root / feat_subdir / "outputs" / "mapped_features.json"
            _run(
                [
                    sys.executable,
                    str(repo_root / "stages" / "stage_geometry_select_v0.py"),
                    "--root",
                    str(runs_base),
                    "--run",
                    args.run,
                    "--stage-subdir",
                    sel_subdir,
                    "--atlas-json",
                    str(atlas_path),
                    "--mapped-features-json",
                    str(mapped_features_json),
                    "--topk",
                    str(args.topk),
                    "--acc-top1-threshold",
                    "0.0",
                    "--acc-topk-threshold",
                    "0.0",
                ]
            )

            ranking_path = run_root / sel_subdir / "outputs" / "geometry_ranking.json"
            ranking = read_json(ranking_path)
            acc1 = float(ranking.get("accuracy_top1", 0.0))
            acck = float(ranking.get("accuracy_topk", 0.0))

            rows.append({"N": n, "sigma_rel": float(sigma_rel), "accuracy_top1": acc1, "accuracy_top3": acck})

            diag = _degeneracy_for_point(ranking, atlas)
            deg_grid.append({"N": n, "sigma_rel": float(sigma_rel), **diag})

    stage_dir = run_root / STAGE
    outputs_dir = stage_dir / "outputs"
    ensure_dir(outputs_dir)

    sweep_results = {
        "version": "e2e_sweep_results.v0",
        "model": {"family": MODEL_FAMILY, "epistemic_status": EPISTEMIC},
        "topk": args.topk,
        "grid": rows,
    }
    degeneracy_report = {
        "version": "degeneracy_report.v0",
        "notes": "Confusiones esperables por problema inverso mal condicionado. No implica unicidad de métrica.",
        "grid": deg_grid,
    }

    point = next((r for r in rows if r["N"] == 128 and abs(r["sigma_rel"] - 0.05) < 1e-12), None)
    point_acc1 = point["accuracy_top1"] if point else None
    point_acc3 = point["accuracy_top3"] if point else None
    thesis_gate_pass = bool(
        point is not None and point_acc1 >= args.thesis_top1 and point_acc3 >= args.thesis_top3
    )

    candidates = [r["N"] for r in rows if abs(r["sigma_rel"] - 0.05) < 1e-12 and r["accuracy_top1"] >= args.thesis_top1]
    nmax = max(candidates) if candidates else None
    verdict_str = f"PHI_BRIDGE_VALID_UP_TO_N={nmax}" if nmax is not None else "PHI_BRIDGE_REJECTED"

    if thesis_gate_pass:
        interpretation = (
            "Punto (N=128, sigma=0.05) supera umbrales en este barrido. "
            "Advertencia: QNM no determinan unívocamente la métrica; existen mimickers y la interpretación sigue siendo fenomenológica."
        )
    else:
        interpretation = (
            "El punto (N=128, sigma=0.05) no supera umbrales y se documenta como límite de información/degeneración, no como bug. "
            "Advertencia explícita: QNM no determinan unívocamente la métrica y existen mimickers."
        )

    verdict = {
        "version": "verdict.v0",
        "model": {"family": MODEL_FAMILY, "epistemic_status": EPISTEMIC},
        "point": {
            "N": 128,
            "sigma_rel": 0.05,
            "accuracy_top1": point_acc1,
            "accuracy_top3": point_acc3,
        },
        "thesis": {
            "top1": args.thesis_top1,
            "top3": args.thesis_top3,
            "thesis_gate_pass": thesis_gate_pass,
        },
        "Nmax_70_at_5pct": nmax,
        "verdict": verdict_str,
        "interpretation": interpretation,
    }

    sweep_path = outputs_dir / "sweep_results.json"
    deg_path = outputs_dir / "degeneracy_report.json"
    verdict_path = outputs_dir / "verdict.json"
    write_json_atomic(sweep_path, sweep_results)
    write_json_atomic(deg_path, degeneracy_report)
    write_json_atomic(verdict_path, verdict)

    integrity_ok = True
    try:
        _ = read_json(sweep_path)
        _ = read_json(deg_path)
        parsed_verdict = read_json(verdict_path)
        if any(r["N"] == 128 and abs(r["sigma_rel"] - 0.05) < 1e-12 for r in rows):
            if parsed_verdict.get("point", {}).get("accuracy_top1") is None:
                integrity_ok = False
    except Exception:
        integrity_ok = False

    stage_verdict = "PASS" if integrity_ok else "FAIL"

    inputs = {
        "run_valid": {
            "path": _run_rel(run_root, run_valid_path),
            "sha256": compute_sha256(run_valid_path),
        }
    }
    outputs = {
        "sweep_results_json": {"path": _run_rel(run_root, sweep_path), "sha256": compute_sha256(sweep_path)},
        "degeneracy_report_json": {"path": _run_rel(run_root, deg_path), "sha256": compute_sha256(deg_path)},
        "verdict_json": {"path": _run_rel(run_root, verdict_path), "sha256": compute_sha256(verdict_path)},
    }

    write_stage_summary(
        stage_dir,
        stage=STAGE,
        impl_module="experiment.ringdown.e2e_sweep_v0",
        params={
            "Ns": ns,
            "sigmas": sigmas,
            "seed_base": args.seed_base,
            "topk": args.topk,
            "thesis_top1": args.thesis_top1,
            "thesis_top3": args.thesis_top3,
        },
        inputs=inputs,
        outputs=outputs,
        verdict=stage_verdict,
        model_family=MODEL_FAMILY,
        epistemic_status=EPISTEMIC,
    )

    write_manifest(
        stage_dir,
        stage=STAGE,
        artifact_relpaths=[
            Path("manifest.json"),
            Path("stage_summary.json"),
            Path("outputs/sweep_results.json"),
            Path("outputs/degeneracy_report.json"),
            Path("outputs/verdict.json"),
        ],
    )
    return 0 if stage_verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
