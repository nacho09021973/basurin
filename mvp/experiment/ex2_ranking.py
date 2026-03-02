#!/usr/bin/env python3
"""Ex2 ranking experiment: rank events by discriminative power from s6c metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_json_atomic,
)

EXPERIMENT_STAGE = "experiment/ex2_ranking"
SCHEMA_VERSION = "ex2_event_ranking_v1"


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _is_valid_number(value: float) -> bool:
    return math.isfinite(value)


def _detector_score(row: dict[str, Any]) -> float:
    kappa = abs(_safe_float(row.get("kappa", row.get("R"))))
    sigma = _safe_float(row.get("sigma"))
    chi_psd = _safe_float(row.get("chi_psd"))
    if not (_is_valid_number(kappa) and _is_valid_number(sigma) and _is_valid_number(chi_psd)):
        return float("nan")
    if sigma <= 0.0:
        return float("nan")
    if chi_psd < 0.0:
        return float("nan")
    return kappa / (sigma * (1.0 + chi_psd))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def rank_events(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        event_id = str(row.get("event_id", "UNKNOWN"))
        grouped[event_id].append(row)

    ranking: list[dict[str, Any]] = []
    for event_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: str(row.get("detector", "")))
        detector_scores: list[float] = []
        kappas: list[float] = []
        sigmas: list[float] = []
        chis: list[float] = []
        n_warnings = 0
        for row in rows:
            warnings = row.get("warnings")
            if isinstance(warnings, list):
                n_warnings += len(warnings)
            score = _detector_score(row)
            if _is_valid_number(score):
                detector_scores.append(score)
                kappas.append(abs(_safe_float(row.get("kappa", row.get("R")))))
                sigmas.append(_safe_float(row.get("sigma")))
                chis.append(_safe_float(row.get("chi_psd")))

        n_det_total = len(rows)
        n_det_ok = len(detector_scores)
        coverage = (n_det_ok / n_det_total) if n_det_total > 0 else 0.0
        score_core = _median(detector_scores) if detector_scores else 0.0
        penalty = 1.0 / (1.0 + n_warnings)
        d_value = score_core * coverage * penalty

        ranking.append(
            {
                "event_id": event_id,
                "D": round(float(d_value), 12),
                "n_det_ok": n_det_ok,
                "n_det_total": n_det_total,
                "n_warnings": n_warnings,
                "features": {
                    "score_formula": "median(|kappa|/(sigma*(1+chi_psd))) * coverage * 1/(1+n_warnings)",
                    "kappa_abs_median": round(float(_median(kappas)), 12) if kappas else None,
                    "sigma_median": round(float(_median(sigmas)), 12) if sigmas else None,
                    "chi_psd_median": round(float(_median(chis)), 12) if chis else None,
                },
            }
        )

    ranking.sort(key=lambda item: (-item["D"], item["event_id"]))
    return ranking


def _resolve_input(run_dir: Path, input_override: str | None) -> Path:
    if input_override:
        p = Path(input_override)
        return p if p.is_absolute() else run_dir / p

    expected = run_dir / "s6c_brunete_psd_curvature" / "outputs" / "brunete_metrics.json"
    if expected.exists():
        return expected

    candidates = {str(p) for p in sorted(run_dir.glob("**/brunete_metrics.json"))}
    candidates.update(str(p) for p in sorted(run_dir.parent.glob("**/brunete_metrics.json")))
    cands = "\n".join(f"  - {c}" for c in sorted(candidates)[:20]) or "  - (none)"
    raise FileNotFoundError(
        "Input faltante para ex2_ranking.\n"
        f"Ruta canónica esperada: runs/{run_dir.name}/s6c_brunete_psd_curvature/outputs/brunete_metrics.json\n"
        f"Ruta esperada exacta: {expected}\n"
        "Comando exacto para regenerar upstream: "
        f"python -m mvp.s6c_brunete_psd_curvature --run-id {run_dir.name}\n"
        f"Candidatos detectados:\n{cands}"
    )


def _require_keys(metrics_rows: list[dict[str, Any]]) -> None:
    for row in metrics_rows:
        event_id = str(row.get("event_id", "UNKNOWN"))
        detector = str(row.get("detector", "UNKNOWN"))
        required_keys = ["sigma", "chi_psd"]
        if "kappa" not in row and "R" not in row:
            required_keys.append("kappa|R")

        available = sorted(str(k) for k in row.keys())
        for key in required_keys:
            if key == "kappa|R":
                continue
            if key not in row:
                raise KeyError(
                    "Schema inválido en brunete_metrics row: "
                    f"event_id={event_id}, detector={detector}, missing_key={key}, "
                    f"available_keys={available}"
                )
        if "kappa|R" in required_keys:
            raise KeyError(
                "Schema inválido en brunete_metrics row: "
                f"event_id={event_id}, detector={detector}, missing_key=kappa|R, "
                f"available_keys={available}"
            )


def run_experiment(run_id: str, input_override: str | None = None) -> int:
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)

    run_dir = out_root / run_id
    input_path = _resolve_input(run_dir, input_override)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    metrics_rows = payload.get("metrics", payload if isinstance(payload, list) else [])
    if not isinstance(metrics_rows, list):
        raise TypeError("Formato inválido: brunete_metrics debe ser lista o dict con clave 'metrics'.")
    _require_keys(metrics_rows)

    stage_dir, outputs_dir = ensure_stage_dirs(run_id, EXPERIMENT_STAGE, base_dir=out_root)

    ranking = rank_events(metrics_rows)
    ranking_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": str(input_path.relative_to(run_dir)),
        "ranking": ranking,
    }

    ranking_json = outputs_dir / "event_ranking.json"
    ranking_csv = outputs_dir / "event_ranking.csv"
    write_json_atomic(ranking_json, ranking_payload)

    with ranking_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "event_id",
                "D",
                "n_det_ok",
                "n_det_total",
                "n_warnings",
                "kappa_abs_median",
                "sigma_median",
                "chi_psd_median",
            ],
        )
        writer.writeheader()
        for item in ranking:
            feat = item.get("features", {})
            writer.writerow(
                {
                    "event_id": item["event_id"],
                    "D": f"{item['D']:.12f}",
                    "n_det_ok": item["n_det_ok"],
                    "n_det_total": item["n_det_total"],
                    "n_warnings": item["n_warnings"],
                    "kappa_abs_median": feat.get("kappa_abs_median"),
                    "sigma_median": feat.get("sigma_median"),
                    "chi_psd_median": feat.get("chi_psd_median"),
                }
            )

    summary = {
        "schema_version": "ex2_event_ranking_summary_v1",
        "stage": EXPERIMENT_STAGE,
        "run_id": run_id,
        "is_experiment": True,
        "experiment_id": "ex2_ranking",
        "upstream_stage": "s6c_brunete_psd_curvature",
        "results": {
            "n_events": len(ranking),
            "n_rows_read": len(metrics_rows),
            "n_warnings_total": sum(item["n_warnings"] for item in ranking),
            "top_event": ranking[0]["event_id"] if ranking else None,
        },
        "inputs": {
            "brunete_metrics": str(input_path.relative_to(run_dir)),
        },
        "artifacts": {
            "event_ranking_json": str(ranking_json.relative_to(run_dir)),
            "event_ranking_csv": str(ranking_csv.relative_to(run_dir)),
        },
    }
    summary_path = stage_dir / "stage_summary.json"
    write_json_atomic(summary_path, summary)

    manifest = {
        "schema_version": "mvp_manifest_v1",
        "stage": EXPERIMENT_STAGE,
        "run_id": run_id,
        "artifacts": {
            "event_ranking_json": str(ranking_json.relative_to(stage_dir)),
            "event_ranking_csv": str(ranking_csv.relative_to(stage_dir)),
            "stage_summary": "stage_summary.json",
        },
        "inputs": {
            "s6c_brunete_psd_curvature/outputs/brunete_metrics.json": sha256_file(input_path),
        },
        "hashes": {
            "event_ranking_json": sha256_file(ranking_json),
            "event_ranking_csv": sha256_file(ranking_csv),
            "stage_summary": sha256_file(summary_path),
        },
    }
    manifest_path = stage_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    print(f"OUT_ROOT={out_root}")
    print(f"STAGE_DIR={stage_dir}")
    print(f"OUTPUTS_DIR={outputs_dir}")
    print(f"STAGE_SUMMARY={summary_path}")
    print(f"MANIFEST={manifest_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ex2 ranking experiment from s6c metrics")
    ap.add_argument("--run-id", required=True, help="Run ID")
    ap.add_argument(
        "--input",
        default=None,
        help="Ruta opcional de brunete_metrics.json (abs o relativa al run dir)",
    )
    args = ap.parse_args()
    return run_experiment(run_id=args.run_id, input_override=args.input)


if __name__ == "__main__":
    raise SystemExit(main())
