#!/usr/bin/env python3
"""BASURIN — C8 work relevance gate (bookkeeping vs extractable).

Motivación: inspirada en la analogía de “bookkeeping vs work extractable” de
un motor cuántico por squeezing. La idea es detectar escenarios donde un score
interno (book) parece alto, pero la porción realmente extraíble es baja, lo que
sugiere métricas espurias o leakage-like.

Ejemplos:
  python tools/c8_work_relevance_gate.py --run <run_id>
  python tools/c8_work_relevance_gate.py --run <run_id> --bridge-stage bridge_f4_1_alignment
  python tools/c8_work_relevance_gate.py --run <run_id> --book-key canonical_corr_median
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    get_run_dir,
    get_runs_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"


@dataclass(frozen=True)
class Thresholds:
    book_min_pass: float = 0.80
    ext_min_pass: float = 0.70
    gap_max: float = 0.10
    neg_max: float = 0.20
    degenerate_threshold: float = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BASURIN C8 gate: work relevance (bookkeeping vs extractable)"
    )
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--bridge-stage", default=None, help="Bridge stage name")
    parser.add_argument("--book-key", default=None, help="Metric key for book_score")
    parser.add_argument("--book-min-pass", type=float, default=Thresholds.book_min_pass)
    parser.add_argument("--ext-min-pass", type=float, default=Thresholds.ext_min_pass)
    parser.add_argument("--gap-max", type=float, default=Thresholds.gap_max)
    parser.add_argument("--neg-max", type=float, default=Thresholds.neg_max)
    parser.add_argument(
        "--degenerate-threshold",
        type=float,
        default=Thresholds.degenerate_threshold,
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _stable_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable_dict(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_stable_dict(item) for item in value]
    return value


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


def _parse_mean(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "mean" in payload:
            return _to_float(payload.get("mean"))
        if "values" in payload:
            return _mean_from_values(payload.get("values"))
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


def _resolve_metrics_path(
    run_dir: Path, bridge_stage: str | None
) -> tuple[str | None, Path | None, list[str], str | None]:
    notes: list[str] = []
    if bridge_stage:
        metrics_path = run_dir / bridge_stage / "outputs" / "metrics.json"
        if not metrics_path.exists():
            notes.append(f"metrics.json faltante en {metrics_path}")
            return bridge_stage, None, notes, "MISSING_METRICS"
        return bridge_stage, metrics_path, notes, None

    candidates = _find_candidate_stages(run_dir)
    preferred = [
        name
        for name in candidates
        if "bridge" in name.lower() or "alignment" in name.lower()
    ]
    if len(preferred) == 1:
        resolved_stage = preferred[0]
        return resolved_stage, run_dir / resolved_stage / "outputs" / "metrics.json", notes, None
    if len(preferred) > 1:
        notes.append(f"Stages puente candidatos: {', '.join(preferred)}")
        return None, None, notes, "AMBIGUOUS_INPUTS"

    if len(candidates) == 1:
        resolved_stage = candidates[0]
        return resolved_stage, run_dir / resolved_stage / "outputs" / "metrics.json", notes, None
    if len(candidates) > 1:
        notes.append(f"Stages candidatos: {', '.join(candidates)}")
        return None, None, notes, "AMBIGUOUS_INPUTS"

    dictionary_metrics = run_dir / "dictionary" / "outputs" / "metrics.json"
    if dictionary_metrics.exists():
        return "dictionary", dictionary_metrics, notes, None

    notes.append("No se encontraron métricas de bridge ni dictionary")
    return None, None, notes, "MISSING_METRICS"


def _rel_path(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _input_entry(run_dir: Path, path: Path) -> dict[str, str]:
    assert_within_runs(run_dir, path)
    return {
        "path": _rel_path(run_dir, path),
        "sha256": sha256_file(path),
    }


def _pick_book_score(
    payload: dict[str, Any], book_key: str | None
) -> tuple[float | None, str | None, list[str]]:
    notes: list[str] = []
    if book_key:
        score = _to_float(payload.get(book_key))
        if score is None:
            notes.append(f"book_key '{book_key}' no encontrado o no numérico")
            return None, None, notes
        return score, book_key, notes

    candidates = [
        "canonical_corr_median",
        "canonical_corr_mean",
        "corr_median",
        "corr_mean",
        "cca_mean",
        "cca_corr_mean",
        "cca_corr_median",
        "alignment_score",
        "alignment_score_mean",
        "alignment_score_median",
        "score",
    ]
    for key in candidates:
        score = _to_float(payload.get(key))
        if score is not None:
            return score, key, notes

    numeric_candidates: list[tuple[str, float]] = []
    for key, value in payload.items():
        score = _to_float(value)
        if score is not None:
            numeric_candidates.append((key, score))
    if numeric_candidates:
        numeric_candidates.sort(key=lambda item: item[0])
        key, score = numeric_candidates[0]
        notes.append(f"book_score seleccionado por fallback: {key}")
        return score, key, notes
    return None, None, notes


def main() -> int:
    args = parse_args()
    out_root = get_runs_root()
    try:
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = get_run_dir(args.run, base_dir=out_root)
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "C8_WORK_RELEVANCE", base_dir=out_root)

    thresholds = Thresholds(
        book_min_pass=float(args.book_min_pass),
        ext_min_pass=float(args.ext_min_pass),
        gap_max=float(args.gap_max),
        neg_max=float(args.neg_max),
        degenerate_threshold=float(args.degenerate_threshold),
    )

    bridge_stage, metrics_path, notes, failure_mode = _resolve_metrics_path(
        run_dir, args.bridge_stage
    )

    inputs: dict[str, dict[str, str]] = {}
    penalties: dict[str, Any] = {}
    verdict = "UNDERDETERMINED"
    reasons: list[str] = []
    book_score = None
    extractable_score = None
    gap = None
    evidence_strength = "weak"
    book_key_used = None

    if metrics_path and metrics_path.exists():
        inputs["metrics"] = _input_entry(run_dir, metrics_path)
        metrics_payload = _read_json(metrics_path)
        book_score, book_key_used, book_notes = _pick_book_score(
            metrics_payload, args.book_key
        )
        notes.extend(book_notes)

        degeneracy_index = _to_float(metrics_payload.get("degeneracy_index_median"))

        controls = {}
        if bridge_stage and bridge_stage != "dictionary":
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
                control_means[key] = _parse_mean(_read_json(path))
                inputs[key] = _input_entry(run_dir, path)

        neg_mean = control_means.get("knn_preservation_negative")
        real_mean = control_means.get("knn_preservation_real")
        control_pos_mean = control_means.get("knn_preservation_control_positive")

        controls_present = any(value is not None for value in control_means.values())
        if controls_present:
            evidence_strength = "strong"

        neg_penalty = 0.0
        if neg_mean is not None:
            neg_penalty = max(0.0, neg_mean - thresholds.neg_max)
        penalties["negative_control"] = {
            "value": neg_mean,
            "threshold": thresholds.neg_max,
            "penalty": neg_penalty,
        }

        control_pos_penalty = 0.0
        control_pos_gap = None
        if real_mean is not None and control_pos_mean is not None:
            control_pos_gap = control_pos_mean - real_mean
            if control_pos_mean <= real_mean:
                control_pos_penalty = real_mean - control_pos_mean
        penalties["control_positive"] = {
            "control_positive": control_pos_mean,
            "real": real_mean,
            "gap": control_pos_gap,
            "penalty": control_pos_penalty,
        }

        degeneracy_penalty = 0.0
        if degeneracy_index is not None:
            degeneracy_penalty = max(0.0, degeneracy_index - thresholds.degenerate_threshold)
        penalties["degeneracy"] = {
            "value": degeneracy_index,
            "threshold": thresholds.degenerate_threshold,
            "penalty": degeneracy_penalty,
        }

        total_penalty = neg_penalty + control_pos_penalty + degeneracy_penalty
        penalties["total"] = total_penalty

        if book_score is None:
            verdict = "UNDERDETERMINED"
            failure_mode = "MISSING_BOOK_SCORE"
            reasons.append("No se pudo inferir book_score")
        else:
            extractable_score = book_score - total_penalty
            gap = book_score - extractable_score

            if neg_mean is not None and neg_mean >= thresholds.neg_max:
                verdict = "FAIL"
                failure_mode = "LEAKAGE_NEGATIVE"
                reasons.append("kNN negative >= neg_max (leakage-like)")
            elif (
                book_score >= thresholds.book_min_pass
                and extractable_score >= thresholds.ext_min_pass
                and gap <= thresholds.gap_max
                and degeneracy_index is not None
                and degeneracy_index >= thresholds.degenerate_threshold
            ):
                verdict = "DEGENERATE"
                failure_mode = "DEGENERACY"
                reasons.append("degeneracy_index_median >= degenerate_threshold")
            elif (
                book_score >= thresholds.book_min_pass
                and extractable_score >= thresholds.ext_min_pass
                and gap <= thresholds.gap_max
            ):
                verdict = "PASS"
                failure_mode = None
                if evidence_strength == "weak":
                    reasons.append("Controles ausentes; evidencia débil")
            elif book_score >= thresholds.book_min_pass and (
                extractable_score < thresholds.ext_min_pass or gap > thresholds.gap_max
            ):
                verdict = "FAIL"
                failure_mode = "LOW_EXTRACTABLE"
                if extractable_score < thresholds.ext_min_pass:
                    reasons.append("extractable_score < ext_min_pass")
                if gap > thresholds.gap_max:
                    reasons.append("gap > gap_max")
            else:
                verdict = "UNDERDETERMINED"
                failure_mode = failure_mode
                reasons.append("No cumple reglas de decisión con la evidencia actual")

            if evidence_strength == "weak" and verdict == "FAIL":
                verdict = "UNDERDETERMINED"
                failure_mode = "WEAK_EVIDENCE"
                reasons.append("Controles ausentes; no se concluye FAIL")
    else:
        reasons.append("metrics.json faltante")

    report = _stable_dict(
        {
            "run": args.run,
            "stage": "C8_WORK_RELEVANCE",
            "created": utc_now_iso(),
            "version": __version__,
            "bridge_stage": bridge_stage,
            "inputs": inputs,
            "book_score": book_score,
            "book_key": book_key_used,
            "extractable_score": extractable_score,
            "gap": gap,
            "penalties": penalties,
            "thresholds": asdict(thresholds),
            "evidence_strength": evidence_strength,
            "verdict": verdict,
            "failure_mode": failure_mode,
            "reasons": reasons,
            "notes": notes,
        }
    )

    report_path = outputs_dir / "c8_report.json"
    _write_json(report_path, report)

    outputs_hashes = {
        "outputs/c8_report.json": sha256_file(report_path),
    }

    summary = _stable_dict(
        {
            "stage": "C8_WORK_RELEVANCE",
            "run": args.run,
            "version": __version__,
            "created": utc_now_iso(),
            "config": {
                "bridge_stage": bridge_stage,
                "book_key": args.book_key,
                "thresholds": asdict(thresholds),
            },
            "inputs": inputs,
            "hashes": outputs_hashes,
            "results": {
                "verdict": verdict,
                "book_score": book_score,
                "extractable_score": extractable_score,
                "gap": gap,
                "evidence_strength": evidence_strength,
            },
        }
    )

    summary_path = write_stage_summary(stage_dir, summary)
    _write_json(summary_path, _read_json(summary_path))

    manifest_path = write_manifest(
        stage_dir,
        {
            "c8_report": report_path,
            "summary": summary_path,
        },
        extra={"version": __version__},
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
