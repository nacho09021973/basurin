#!/usr/bin/env python3
# experiment/ringdown/e2e_sweep_plot_v0.py

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from basurin_io import (
    ensure_dir,
    read_json,
    write_json_atomic,
    compute_sha256,
    load_run_valid_verdict,
    write_manifest,
    write_stage_summary,
)

MODEL_FAMILY = "phi_phenomenological_v0"
EPI_STATUS = "conjectural/phenomenological"


def _run_root(root: str | None, run_id: str) -> Path:
    base = Path(root).resolve() if root else Path.cwd().resolve()
    return base / "runs" / run_id


def _stage_dir(run_root: Path) -> Path:
    return run_root / "experiment" / "ringdown" / "e2e_sweep_plot_v0"


def _parse_grid(grid: List[Dict[str, Any]]) -> Tuple[List[int], List[float], Dict[Tuple[int, float], Dict[str, float]]]:
    """
    Returns:
      Ns_sorted, sigmas_sorted, map[(N,sigma)] = {"acc1":..., "acc3":...}
    """
    Ns = sorted({int(row["N"]) for row in grid})
    sigmas = sorted({float(row["sigma_rel"]) for row in grid})
    m: Dict[Tuple[int, float], Dict[str, float]] = {}
    for row in grid:
        N = int(row["N"])
        s = float(row["sigma_rel"])
        acc1 = float(row.get("accuracy_top1"))
        # En tu pipeline el "topk" puede ser 3; en sweep_results suele venir accuracy_top3 o accuracy_topk.
        acc3 = row.get("accuracy_top3", row.get("accuracy_topk"))
        if acc3 is None:
            raise ValueError("Missing accuracy_top3/accuracy_topk in sweep_results grid row")
        m[(N, s)] = {"acc1": float(acc1), "acc3": float(acc3)}
    return Ns, sigmas, m


def _write_csv(out_csv: Path, Ns: List[int], sigmas: List[float], m: Dict[Tuple[int, float], Dict[str, float]]) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "sigma_rel", "accuracy_top1", "accuracy_top3"])
        for s in sigmas:
            for N in Ns:
                vals = m.get((N, s))
                if vals is None:
                    continue
                w.writerow([N, s, vals["acc1"], vals["acc3"]])


def _plot_lines(out_png: Path, Ns: List[int], sigmas: List[float], m: Dict[Tuple[int, float], Dict[str, float]], key: str, title: str) -> None:
    ensure_dir(out_png.parent)
    plt.figure()
    for s in sigmas:
        ys = [m[(N, s)][key] for N in Ns if (N, s) in m]
        xs = [N for N in Ns if (N, s) in m]
        plt.plot(xs, ys, marker="o", label=f"sigma={s:g}")
    plt.xscale("log", base=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("N (tamaño del atlas)")
    plt.ylabel(key)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_heatmap(out_png: Path, Ns: List[int], sigmas: List[float], m: Dict[Tuple[int, float], Dict[str, float]], key: str, title: str) -> None:
    ensure_dir(out_png.parent)
    # matriz [len(sigmas) x len(Ns)]
    A = np.full((len(sigmas), len(Ns)), np.nan, dtype=float)
    for i, s in enumerate(sigmas):
        for j, N in enumerate(Ns):
            if (N, s) in m:
                A[i, j] = m[(N, s)][key]

    plt.figure()
    im = plt.imshow(A, aspect="auto", interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(len(Ns)), labels=[str(N) for N in Ns])
    plt.yticks(ticks=np.arange(len(sigmas)), labels=[f"{s:g}" for s in sigmas])
    plt.xlabel("N (atlas)")
    plt.ylabel("sigma_rel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=None, help="Directorio que contiene runs/ (por defecto: cwd)")
    ap.add_argument(
        "--sweep-results-json",
        default=None,
        help="Ruta a sweep_results.json (por defecto: runs/<run>/experiment/ringdown/e2e_sweep_v0/outputs/sweep_results.json)",
    )
    ap.add_argument(
        "--degeneracy-report-json",
        default=None,
        help="Ruta opcional a degeneracy_report.json (no imprescindible para plots básicos)",
    )
    args = ap.parse_args()

    run_root = _run_root(args.root, args.run)

    # Gate soberano
    ok, verdict_path = load_run_valid_verdict(run_root)
    if not ok:
        raise SystemExit(f"RUN_INVALID: RUN_VALID != PASS (checked {verdict_path})")

    stage_dir = _stage_dir(run_root)
    out_dir = stage_dir / "outputs"
    ensure_dir(out_dir)

    default_sweep = run_root / "experiment" / "ringdown" / "e2e_sweep_v0" / "outputs" / "sweep_results.json"
    sweep_path = Path(args.sweep_results_json).resolve() if args.sweep_results_json else default_sweep
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing sweep_results.json at {sweep_path}")

    sweep = read_json(sweep_path)
    if sweep.get("version") is None or sweep.get("grid") is None:
        raise ValueError("Invalid sweep_results.json schema: missing version/grid")

    grid = sweep["grid"]
    Ns, sigmas, m = _parse_grid(grid)

    # CSV tabular
    out_csv = out_dir / "summary_table.csv"
    _write_csv(out_csv, Ns, sigmas, m)

    # Plots
    p1 = out_dir / "accuracy_top1_vs_N.png"
    p3 = out_dir / "accuracy_top3_vs_N.png"
    hm = out_dir / "heatmap_top1.png"

    _plot_lines(p1, Ns, sigmas, m, key="acc1", title="Accuracy top-1 vs N (por sigma_rel)")
    _plot_lines(p3, Ns, sigmas, m, key="acc3", title="Accuracy top-3 vs N (por sigma_rel)")
    _plot_heatmap(hm, Ns, sigmas, m, key="acc1", title="Heatmap accuracy top-1 (sigma_rel x N)")

    # Opcional: copiar/registrar degeneracy_report.json como input (no lo ploteo aquí por simplicidad)
    inputs = {
        "run_valid": {"path": str(verdict_path.relative_to(run_root)), "sha256": compute_sha256(verdict_path)},
        "sweep_results": {"path": str(sweep_path.relative_to(run_root)), "sha256": compute_sha256(sweep_path)},
    }
    if args.degeneracy_report_json:
        dpath = Path(args.degeneracy_report_json).resolve()
        if dpath.exists():
            inputs["degeneracy_report"] = {"path": str(dpath.relative_to(run_root)), "sha256": compute_sha256(dpath)}

    # Outputs dict
    outputs = {
        "accuracy_top1_vs_N_png": {"path": str(p1.relative_to(run_root)), "sha256": compute_sha256(p1)},
        "accuracy_top3_vs_N_png": {"path": str(p3.relative_to(run_root)), "sha256": compute_sha256(p3)},
        "heatmap_top1_png": {"path": str(hm.relative_to(run_root)), "sha256": compute_sha256(hm)},
        "summary_table_csv": {"path": str(out_csv.relative_to(run_root)), "sha256": compute_sha256(out_csv)},
    }

    params = {
        "run": args.run,
        "sweep_results_json": str(sweep_path),
        "degeneracy_report_json": str(args.degeneracy_report_json) if args.degeneracy_report_json else None,
    }

    # stage_summary + manifest (orden importante: primero summary, luego manifest incluyendo ambos)
    summary_path = write_stage_summary(
        stage_dir,
        stage="experiment/ringdown/e2e_sweep_plot_v0",
        impl_module="experiment.ringdown.e2e_sweep_plot_v0",
        params=params,
        inputs=inputs,
        outputs=outputs,
        verdict="PASS",
        model_family=MODEL_FAMILY,
        epistemic_status=EPI_STATUS,
    )

    # manifest: relpaths desde stage_dir
    artifact_relpaths = [
        Path("outputs") / p1.name,
        Path("outputs") / p3.name,
        Path("outputs") / hm.name,
        Path("outputs") / out_csv.name,
        Path("stage_summary.json"),
        Path("manifest.json"),
    ]
    # write_manifest debe hashear tras existir manifest.json; si tu helper requiere otra convención, ajusta aquí.
    write_manifest(stage_dir, stage="experiment/ringdown/e2e_sweep_plot_v0", artifact_relpaths=artifact_relpaths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
