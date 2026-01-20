#!/usr/bin/env python3
"""BASURIN — Contrato C7 (Bridge Stability + Non-degeneracy)

Este contrato valida el resultado del stage bridge_f4_1_alignment (F4-1).

C7 define tres posibles resultados:
- PASS: evidencia de puente estable y no-aleatorio, sin degeneración severa.
- FAIL_DEGENERATE (FAIL informativo): hay señal (significancia) pero el puente es no-inyectivo/degenerado.
- FAIL_NO_BRIDGE: no hay evidencia estadística robusta vs permutación o es inestable.

IO determinista:
  runs/<run_id>/contract_C7_bridge/
    manifest.json
    stage_summary.json
    outputs/contract_C7_bridge.json

Nota: no modifica artefactos del stage de alineación; referencia sus hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

__version__ = "0.1.0"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Thresholds:
    stability_score_min: float = 0.80
    significance_ratio_min: float = 3.0
    degeneracy_index_max_pass: float = 100.0
    p_value_max: float = 0.05


@dataclass(frozen=True)
class Config:
    run: str
    out_root: str = "runs"
    alignment_stage: str = "bridge_f4_1_alignment"
    thresholds: Thresholds = Thresholds()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="BASURIN Contract C7: Bridge Stability + Non-degeneracy")
    p.add_argument("--run", required=True, type=str, help="run_id (carpeta bajo runs/<run>/)")
    p.add_argument("--out-root", default="runs", type=str)
    p.add_argument("--alignment-stage", default="bridge_f4_1_alignment", type=str)

    # thresholds
    p.add_argument("--stability-min", default=0.80, type=float)
    p.add_argument("--significance-min", default=3.0, type=float)
    p.add_argument("--degeneracy-max", default=100.0, type=float)
    p.add_argument("--pvalue-max", default=0.05, type=float)

    a = p.parse_args()
    thr = Thresholds(
        stability_score_min=float(a.stability_min),
        significance_ratio_min=float(a.significance_min),
        degeneracy_index_max_pass=float(a.degeneracy_max),
        p_value_max=float(a.pvalue_max),
    )
    return Config(run=a.run, out_root=a.out_root, alignment_stage=a.alignment_stage, thresholds=thr)


def main() -> int:
    cfg = parse_args()

    run_root = Path(cfg.out_root) / cfg.run
    in_stage_dir = run_root / cfg.alignment_stage
    in_summary = in_stage_dir / "stage_summary.json"
    in_metrics = in_stage_dir / "outputs" / "metrics.json"

    if not in_summary.exists():
        print(f"ERROR: no existe {in_summary}. Ejecuta primero el stage de alineación.", file=sys.stderr)
        return 1
    if not in_metrics.exists():
        print(f"ERROR: no existe {in_metrics}.", file=sys.stderr)
        return 1

    with open(in_summary, "r") as f:
        summary = json.load(f)
    with open(in_metrics, "r") as f:
        metrics = json.load(f)

    if summary.get("status") != "OK":
        # propagate aborts
        status = summary.get("status")
        verdict = "FAIL_NO_BRIDGE"
        reason = f"Upstream stage status={status}"
        report = {
            "contract": "C7_bridge",
            "version": __version__,
            "created": utcnow_iso(),
            "run": cfg.run,
            "verdict": verdict,
            "reason": reason,
            "thresholds": asdict(cfg.thresholds),
            "metrics": summary.get("results", {}),
            "upstream": {
                "stage": cfg.alignment_stage,
                "summary": str(in_summary),
                "hash_summary": compute_file_hash(in_summary),
            },
        }
    else:
        r = summary["results"]
        thr = cfg.thresholds

        stability = float(r.get("stability_score", 0.0))
        sig_ratio = float(r.get("significance_ratio", 0.0))
        p_value = float(r.get("p_value", 1.0))
        deg_med = float(r.get("degeneracy_index_median", float("inf")))

        # decisiones
        ok_signif = (sig_ratio >= thr.significance_ratio_min) and (p_value <= thr.p_value_max)
        ok_stable = stability >= thr.stability_score_min
        ok_degen = deg_med <= thr.degeneracy_index_max_pass

        if ok_signif and ok_stable and ok_degen:
            verdict = "PASS"
            reason = "Bridge evidence > permutation baseline, stable axes, and acceptable degeneracy."
        elif ok_signif and ok_stable and (not ok_degen):
            verdict = "FAIL_DEGENERATE"
            reason = "Significant + stable alignment but degeneracy suggests non-injectivity (many-to-one)."
        else:
            verdict = "FAIL_NO_BRIDGE"
            reason = "No robust evidence of bridge vs permutation and/or unstable axes."

        # Diagnósticos adicionales
        perm = metrics.get("permutation", {})
        boot = metrics.get("bootstrap", {})
        deg = metrics.get("degeneracy", {})

        report = {
            "contract": "C7_bridge",
            "version": __version__,
            "created": utcnow_iso(),
            "run": cfg.run,
            "verdict": verdict,
            "reason": reason,
            "thresholds": asdict(thr),
            "checks": {
                "significance_ok": bool(ok_signif),
                "stability_ok": bool(ok_stable),
                "degeneracy_ok": bool(ok_degen),
            },
            "metrics": {
                "stability_score": stability,
                "mean_axis_angle_deg": float(r.get("mean_axis_angle_deg", float("nan"))),
                "significance_ratio": sig_ratio,
                "p_value": p_value,
                "degeneracy_index_median": deg_med,
                "degeneracy_index_p90": float(r.get("degeneracy_index_p90", float("nan"))),
                "canonical_corr_mean": float(r.get("canonical_corr_mean", float("nan"))),
                "perm_score_true": float(perm.get("score_true", float("nan"))),
                "perm_score_median": float(perm.get("score_perm_median", float("nan"))),
                "bootstrap_mean_angle_deg": float(boot.get("mean_angle_deg", float("nan"))),
            },
            "diagnostics": {
                "permutation": perm,
                "bootstrap": boot,
                "degeneracy": deg,
            },
            "upstream": {
                "stage": cfg.alignment_stage,
                "summary": str(in_summary),
                "metrics": str(in_metrics),
                "hash_summary": compute_file_hash(in_summary),
                "hash_metrics": compute_file_hash(in_metrics),
            },
        }

    # Escribir stage C7
    stage_dir = run_root / "contract_C7_bridge"
    outdir = stage_dir / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / "contract_C7_bridge.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    stage_summary = {
        "stage": "contract_C7_bridge",
        "version": __version__,
        "created": utcnow_iso(),
        "run": cfg.run,
        "status": report["verdict"],
        "config": {
            "alignment_stage": cfg.alignment_stage,
            "thresholds": asdict(cfg.thresholds),
        },
        "hashes": {
            "outputs/contract_C7_bridge.json": compute_file_hash(out_path),
            "inputs/alignment_stage_summary": compute_file_hash(in_summary),
            "inputs/alignment_stage_metrics": compute_file_hash(in_metrics),
        },
    }

    with open(stage_dir / "stage_summary.json", "w") as f:
        json.dump(stage_summary, f, indent=2)

    manifest = {
        "stage": "contract_C7_bridge",
        "version": __version__,
        "run": cfg.run,
        "created": utcnow_iso(),
        "inputs": {
            "alignment_stage": cfg.alignment_stage,
            "alignment_summary": str(in_summary),
            "alignment_metrics": str(in_metrics),
        },
        "files": {
            "contract": "outputs/contract_C7_bridge.json",
            "summary": "stage_summary.json",
        },
    }
    with open(stage_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Contract C7_bridge ===")
    print(f"verdict: {report['verdict']}")
    print(f"reason: {report['reason']}")

    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
