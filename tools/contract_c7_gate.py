#!/usr/bin/env python3
"""BASURIN — Canonical C7 executive gate (bridge/alignment)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "1.0.0"


@dataclass(frozen=True)
class Thresholds:
    alpha_perm: float = 0.05
    min_significance_ratio: float = 3.0
    min_corr_pass: float = 0.7
    min_stability_pass: float = 0.8
    max_deg_pass: float = 100.0
    degenerate_deg_median: float = 150.0
    leakage_neg_ratio_fail: float = 1.0


@dataclass(frozen=True)
class Inputs:
    path: str
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BASURIN C7 gate (bridge/alignment)")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--bridge-stage", default=None, help="Bridge stage name")
    parser.add_argument("--alpha-perm", type=float, default=Thresholds.alpha_perm)
    parser.add_argument(
        "--min-significance-ratio", type=float, default=Thresholds.min_significance_ratio
    )
    parser.add_argument("--min-corr-pass", type=float, default=Thresholds.min_corr_pass)
    parser.add_argument(
        "--min-stability-pass", type=float, default=Thresholds.min_stability_pass
    )
    parser.add_argument("--max-deg-pass", type=float, default=Thresholds.max_deg_pass)
    parser.add_argument(
        "--degenerate-deg-median", type=float, default=Thresholds.degenerate_deg_median
    )
    parser.add_argument(
        "--leakage-neg-ratio-fail", type=float, default=Thresholds.leakage_neg_ratio_fail
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _stable_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable_dict(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_stable_dict(item) for item in value]
    return value


def _find_candidate_stages(run_dir: Path) -> list[str]:
    candidates: list[str] = []
    if not run_dir.exists():
        return candidates
    for entry in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        metrics_path = entry / "outputs" / "metrics.json"
        if metrics_path.exists():
            candidates.append(entry.name)
    return candidates


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(candidate) or math.isinf(candidate):
        return None
    return candidate


def _metric_from(payload: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in payload:
            return _to_float(payload.get(key))
    return None


def _parse_mean(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "mean" in payload:
            return _to_float(payload.get("mean"))
        if "values" in payload:
            values = payload.get("values")
            return _mean_from_values(values)
        return None
    if isinstance(payload, list):
        return _mean_from_values(payload)
    return None


def _mean_from_values(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    numeric: list[float] = []
    for item in values:
        val = _to_float(item)
        if val is not None:
            numeric.append(val)
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _sha256_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_file(path)


def _resolve_inputs(
    run_dir: Path, bridge_stage: str | None
) -> tuple[str | None, Path | None, list[str], str | None]:
    notes: list[str] = []
    if bridge_stage:
        metrics_path = run_dir / bridge_stage / "outputs" / "metrics.json"
        if not metrics_path.exists():
            notes.append(f"metrics.json faltante en {metrics_path}")
            return bridge_stage, metrics_path, notes, "MISSING_METRICS"
        return bridge_stage, metrics_path, notes, None

    candidates = _find_candidate_stages(run_dir)
    if not candidates:
        notes.append("No se encontraron stages con outputs/metrics.json")
        return None, None, notes, "MISSING_METRICS"
    if len(candidates) > 1:
        notes.append(f"Stages candidatos: {', '.join(candidates)}")
        return None, None, notes, "AMBIGUOUS_INPUTS"
    resolved_stage = candidates[0]
    metrics_path = run_dir / resolved_stage / "outputs" / "metrics.json"
    return resolved_stage, metrics_path, notes, None


def main() -> int:
    args = parse_args()
    try:
        validate_run_id(args.run, Path("runs"))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = get_run_dir(args.run)
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "C7")

    thresholds = Thresholds(
        alpha_perm=float(args.alpha_perm),
        min_significance_ratio=float(args.min_significance_ratio),
        min_corr_pass=float(args.min_corr_pass),
        min_stability_pass=float(args.min_stability_pass),
        max_deg_pass=float(args.max_deg_pass),
        degenerate_deg_median=float(args.degenerate_deg_median),
        leakage_neg_ratio_fail=float(args.leakage_neg_ratio_fail),
    )

    bridge_stage, metrics_path, notes, failure_mode = _resolve_inputs(
        run_dir, args.bridge_stage
    )

    metrics: dict[str, Any] = {}
    rules_fired: list[str] = []
    inputs: dict[str, dict[str, str]] = {}
    verdict = "UNDERDETERMINED"

    if metrics_path and metrics_path.exists():
        payload = _read_json(metrics_path)
        inputs["metrics"] = {
            "path": str(metrics_path),
            "sha256": sha256_file(metrics_path),
        }

        metrics = {
            "canonical_corr_mean": _metric_from(
                payload, ["canonical_corr_mean", "corr_mean", "cca_corr_mean"]
            ),
            "significance_ratio": _metric_from(payload, ["significance_ratio"]),
            "p_value": _metric_from(payload, ["p_value", "permutation_p"]),
            "stability_score": _metric_from(payload, ["stability_score"]),
            "mean_angle_deg": _metric_from(payload, ["mean_angle_deg", "mean_angle"]),
            "degeneracy_index_median": _metric_from(
                payload, ["degeneracy_index_median", "degeneracy_median"]
            ),
        }

        missing_keys = [k for k, v in metrics.items() if v is None]
        if missing_keys:
            verdict = "UNDERDETERMINED"
            failure_mode = "MISSING_METRICS"
            notes.append(f"Missing metrics: {', '.join(sorted(missing_keys))}")
        else:
            controls = {
                "knn_preservation_real": run_dir
                / bridge_stage
                / "outputs"
                / "knn_preservation_real.json",
                "knn_preservation_negative": run_dir
                / bridge_stage
                / "outputs"
                / "knn_preservation_negative.json",
                "knn_preservation_control_positive": run_dir
                / bridge_stage
                / "outputs"
                / "knn_preservation_control_positive.json",
            }

            control_means: dict[str, float | None] = {}
            for key, path in controls.items():
                if path.exists():
                    payload = _read_json(path)
                    control_means[key] = _parse_mean(payload)
                    inputs[key] = {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }

            real_mean = control_means.get("knn_preservation_real")
            neg_mean = control_means.get("knn_preservation_negative")
            neg_ratio = None
            if real_mean is not None and real_mean > 0 and neg_mean is not None:
                neg_ratio = neg_mean / real_mean

            metrics.update(control_means)
            metrics["neg_ratio"] = neg_ratio

            corr = metrics["canonical_corr_mean"]
            stability = metrics["stability_score"]
            deg_median = metrics["degeneracy_index_median"]
            sig_ratio = metrics["significance_ratio"]
            p_value = metrics["p_value"]

            if neg_ratio is not None and neg_ratio >= thresholds.leakage_neg_ratio_fail:
                verdict = "FAIL"
                failure_mode = "LEAKAGE"
                rules_fired.append("leakage_neg_ratio")
            elif (
                sig_ratio is not None
                and p_value is not None
                and (p_value >= thresholds.alpha_perm)
                and (sig_ratio >= thresholds.min_significance_ratio)
            ):
                verdict = "FAIL"
                failure_mode = "LEAKAGE"
                rules_fired.append("leakage_perm_significance")
            elif (
                corr is not None
                and stability is not None
                and deg_median is not None
                and corr >= thresholds.min_corr_pass
                and stability >= thresholds.min_stability_pass
                and deg_median >= thresholds.degenerate_deg_median
            ):
                verdict = "DEGENERATE"
                failure_mode = "DEGENERACY"
            elif (
                corr is not None
                and stability is not None
                and deg_median is not None
                and corr >= thresholds.min_corr_pass
                and stability >= thresholds.min_stability_pass
                and deg_median <= thresholds.max_deg_pass
            ):
                verdict = "PASS"
                failure_mode = None
            else:
                verdict = "UNDERDETERMINED"
                failure_mode = failure_mode

    report = _stable_dict(
        {
            "run": args.run,
            "stage": "C7",
            "version": __version__,
            "verdict": verdict,
            "failure_mode": failure_mode,
            "bridge_stage": bridge_stage,
            "evidence": {
                "metrics": metrics,
                "thresholds": asdict(thresholds),
                "rules_fired": rules_fired,
                "notes": notes,
            },
            "inputs": inputs,
        }
    )

    report_path = outputs_dir / "c7_report.json"
    _write_json(report_path, report)

    outputs_hashes = {
        "outputs/c7_report.json": sha256_file(report_path),
    }

    summary = _stable_dict(
        {
            "stage": "C7",
            "run": args.run,
            "version": __version__,
            "config": {
                "thresholds": asdict(thresholds),
                "bridge_stage": bridge_stage,
            },
            "inputs": inputs,
            "hashes": outputs_hashes,
            "results": {"verdict": verdict},
        }
    )

    summary_path = write_stage_summary(stage_dir, summary)
    _write_json(summary_path, _read_json(summary_path))

    manifest_path = write_manifest(
        stage_dir,
        {
            "c7_report": report_path,
            "summary": summary_path,
        },
    )
    manifest_payload = _read_json(manifest_path)
    manifest_payload.setdefault("files", {})
    manifest_payload["files"].setdefault("manifest", "manifest.json")
    _write_json(manifest_path, _stable_dict(manifest_payload))

    if verdict == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
