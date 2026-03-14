#!/usr/bin/env python3
"""Compare fixed af symbolic candidates on repeated train/test splits.

Consumes:
  - runs/<run_id>/experiment/malda_feature_table/outputs/event_features.csv

Produces under runs/<run_id>/experiment/malda_af_formula_compare/:
  - outputs/formula_compare.json
  - outputs/formula_compare_split_metrics.csv
  - outputs/formula_compare_ranking.json
  - manifest.json
  - stage_summary.json

This stage does not fit formulas. It benchmarks fixed symbolic candidates for
af on repeated random splits to decide which surrogate is more robust.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import (  # noqa: E402
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)
import mvp.contracts as contracts  # noqa: E402


STAGE_NAME = "malda_af_formula_compare"
FEATURE_TABLE_STAGE_NAME = "malda_feature_table"
TARGET_NAME = "af"

DEFAULT_FORMULAS = (
    ("additive_quadratic", "square((chi_eff * 0.3089345) + (eta + 0.5638679))"),
    ("multiplicative", "square(eta + 0.4833311) * (chi_eff + 1.237785)"),
)


def _load_validation_module() -> Any:
    module_path = REPO_ROOT / "malda" / "12_validate_formula_candidates.py"
    spec = importlib.util.spec_from_file_location("malda_12_validate_formula_candidates", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATION = _load_validation_module()


def _metric_row(formula_label: str, split_index: int, partition: str, metrics: dict[str, Any]) -> dict[str, Any]:
    fit = metrics.get("fit") or {}
    bounds = metrics.get("physics_bounds") or {}
    return {
        "formula_label": formula_label,
        "split_index": split_index,
        "partition": partition,
        "n_target_valid": metrics.get("n_target_valid"),
        "n_evaluated": metrics.get("n_evaluated"),
        "finite_fraction": metrics.get("finite_fraction"),
        "bounds_fraction": bounds.get("fraction_in_bounds"),
        "r2": fit.get("r2"),
        "rmse": fit.get("rmse"),
        "nrmse_std": fit.get("nrmse_std"),
        "median_relative_error": fit.get("median_relative_error"),
    }


def _aggregate_split_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    aggregates: dict[str, Any] = {}
    for key in ("r2", "rmse", "nrmse_std", "median_relative_error", "finite_fraction", "bounds_fraction"):
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        if not values:
            aggregates[key] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregates[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return aggregates


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "formula_label",
        "split_index",
        "partition",
        "n_target_valid",
        "n_evaluated",
        "finite_fraction",
        "bounds_fraction",
        "r2",
        "rmse",
        "nrmse_std",
        "median_relative_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_formulas(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [
        (args.label_a, args.equation_a),
        (args.label_b, args.equation_b),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Existing run_id with RUN_VALID=PASS")
    parser.add_argument(
        "--feature-table",
        default="",
        help=(
            "Path to event_features.csv. Defaults to "
            "runs/<run-id>/experiment/malda_feature_table/outputs/event_features.csv"
        ),
    )
    parser.add_argument("--label-a", default=DEFAULT_FORMULAS[0][0], help="Label for formula A")
    parser.add_argument("--equation-a", default=DEFAULT_FORMULAS[0][1], help="Formula A")
    parser.add_argument("--label-b", default=DEFAULT_FORMULAS[1][0], help="Label for formula B")
    parser.add_argument("--equation-b", default=DEFAULT_FORMULAS[1][1], help="Formula B")
    parser.add_argument("--n-splits", type=int, default=32, help="Number of repeated random train/test splits")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of rows used for test in each split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeated splits")
    parser.add_argument(
        "--include-non-bbh",
        action="store_true",
        help="Use all rows instead of filtering to BBH rows only",
    )
    args = parser.parse_args(argv)

    runs_root = resolve_out_root("runs")
    validate_run_id(args.run_id, runs_root)

    try:
        require_run_valid(runs_root, args.run_id)
    except Exception as exc:
        print(f"[13_af_compare] ERROR: RUN_VALID check failed: {exc}", file=sys.stderr)
        return 1

    if args.n_splits <= 0:
        print("[13_af_compare] ERROR: --n-splits must be > 0", file=sys.stderr)
        return 1
    if not (0.0 < args.test_fraction < 1.0):
        print("[13_af_compare] ERROR: --test-fraction must be in (0,1)", file=sys.stderr)
        return 1
    if args.label_a == args.label_b:
        print("[13_af_compare] ERROR: formula labels must be distinct", file=sys.stderr)
        return 1

    stage_dir = runs_root / args.run_id / "experiment" / STAGE_NAME
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_table_path = Path(args.feature_table) if args.feature_table else (
        runs_root / args.run_id / "experiment" / FEATURE_TABLE_STAGE_NAME / "outputs" / "event_features.csv"
    )
    if not feature_table_path.exists():
        print(
            f"[13_af_compare] ERROR: feature table not found at {feature_table_path}\n"
            f"  Run first: python malda/10_build_event_feature_table.py --run-id {args.run_id}",
            file=sys.stderr,
        )
        return 1

    bbh_only = not args.include_non_bbh
    _, columns = VALIDATION.load_feature_columns(feature_table_path, bbh_only=bbh_only)
    if TARGET_NAME not in columns:
        print(f"[13_af_compare] ERROR: target column '{TARGET_NAME}' missing from feature table", file=sys.stderr)
        return 1

    formulas = _resolve_formulas(args)
    full_target = np.asarray(columns[TARGET_NAME], dtype=np.float64)
    target_indices = np.where(np.isfinite(full_target))[0]
    if len(target_indices) < 4:
        print("[13_af_compare] ERROR: need at least 4 finite target rows to compare formulas", file=sys.stderr)
        return 1

    n_test = max(1, int(round(len(target_indices) * args.test_fraction)))
    n_test = min(n_test, len(target_indices) - 1)
    rng = np.random.default_rng(args.seed)

    results_by_formula: dict[str, dict[str, Any]] = {}
    split_rows: list[dict[str, Any]] = []

    print(f"[13_af_compare] Loaded {len(target_indices)} finite '{TARGET_NAME}' rows")
    print(f"[13_af_compare] Comparing {len(formulas)} formulas over {args.n_splits} splits")

    for label, equation in formulas:
        try:
            full_pred, variable_names = VALIDATION.evaluate_formula(equation, columns)
        except Exception as exc:
            print(f"[13_af_compare] ERROR: failed to evaluate formula '{label}': {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        full_metrics = VALIDATION.compute_target_metrics(
            target_name=TARGET_NAME,
            y_true=full_target,
            y_pred=np.asarray(full_pred, dtype=np.float64),
            bootstrap_samples=0,
            seed=args.seed,
        )

        train_rows: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        for split_index in range(args.n_splits):
            perm = rng.permutation(target_indices)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

            train_metrics = VALIDATION.compute_target_metrics(
                target_name=TARGET_NAME,
                y_true=full_target[train_idx],
                y_pred=np.asarray(full_pred, dtype=np.float64)[train_idx],
                bootstrap_samples=0,
                seed=args.seed + split_index,
            )
            test_metrics = VALIDATION.compute_target_metrics(
                target_name=TARGET_NAME,
                y_true=full_target[test_idx],
                y_pred=np.asarray(full_pred, dtype=np.float64)[test_idx],
                bootstrap_samples=0,
                seed=args.seed + split_index,
            )

            train_row = _metric_row(label, split_index, "train", train_metrics)
            test_row = _metric_row(label, split_index, "test", test_metrics)
            train_rows.append(train_row)
            test_rows.append(test_row)
            split_rows.extend([train_row, test_row])

        train_summary = _aggregate_split_rows(train_rows)
        test_summary = _aggregate_split_rows(test_rows)
        results_by_formula[label] = {
            "equation": equation,
            "variables": variable_names,
            "full_dataset_metrics": full_metrics,
            "split_metrics": {
                "train": train_summary,
                "test": test_summary,
            },
        }

        test_r2 = (test_summary.get("r2") or {}).get("mean")
        test_nrmse = (test_summary.get("nrmse_std") or {}).get("mean")
        print(
            f"[13_af_compare] {label}: "
            f"test_r2_mean={test_r2} test_nrmse_std_mean={test_nrmse}"
        )

    ranking = sorted(
        [
            {
                "formula_label": label,
                "equation": payload["equation"],
                "test_r2_mean": (payload["split_metrics"]["test"].get("r2") or {}).get("mean"),
                "test_nrmse_std_mean": (payload["split_metrics"]["test"].get("nrmse_std") or {}).get("mean"),
                "full_r2": ((payload["full_dataset_metrics"].get("fit") or {}).get("r2")),
                "full_nrmse_std": ((payload["full_dataset_metrics"].get("fit") or {}).get("nrmse_std")),
            }
            for label, payload in results_by_formula.items()
        ],
        key=lambda row: (
            float("inf") if row["test_nrmse_std_mean"] is None else row["test_nrmse_std_mean"],
            float("inf") if row["test_r2_mean"] is None else -row["test_r2_mean"],
            row["formula_label"],
        ),
    )

    winner = ranking[0]["formula_label"]
    payload = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "created": utc_now_iso(),
        "config": {
            "target": TARGET_NAME,
            "bbh_only": bbh_only,
            "n_splits": args.n_splits,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "formulas": [{"label": label, "equation": equation} for label, equation in formulas],
        },
        "inputs": {
            "feature_table": str(feature_table_path),
            "feature_table_sha256": sha256_file(feature_table_path),
        },
        "results": {
            "winner_by_test_nrmse_std_mean": winner,
            "ranking": ranking,
            "formulas": results_by_formula,
        },
    }

    compare_json_path = outputs_dir / "formula_compare.json"
    metrics_csv_path = outputs_dir / "formula_compare_split_metrics.csv"
    ranking_json_path = outputs_dir / "formula_compare_ranking.json"
    write_json_atomic(compare_json_path, payload)
    _write_metrics_csv(metrics_csv_path, split_rows)
    write_json_atomic(
        ranking_json_path,
        {
            "stage": STAGE_NAME,
            "run_id": args.run_id,
            "created": utc_now_iso(),
            "winner_by_test_nrmse_std_mean": winner,
            "ranking": ranking,
        },
    )

    artifacts: dict[str, Path] = {
        "formula_compare": compare_json_path,
        "formula_compare_split_metrics": metrics_csv_path,
        "formula_compare_ranking": ranking_json_path,
    }
    manifest_path = write_manifest(
        stage_dir,
        artifacts,
        extra={
            "run_id": args.run_id,
            "stage": STAGE_NAME,
            "inputs": payload["inputs"],
        },
    )
    output_hashes = {label: sha256_file(path) for label, path in artifacts.items()}
    stage_summary_path = write_stage_summary(
        stage_dir,
        {
            "stage": STAGE_NAME,
            "verdict": "PASS",
            "run_id": args.run_id,
            "created": utc_now_iso(),
            "config": payload["config"],
            "inputs": payload["inputs"],
            "results": {
                "target": TARGET_NAME,
                "n_rows_compared": len(target_indices),
                "n_splits": args.n_splits,
                "winner_by_test_nrmse_std_mean": winner,
                "test_nrmse_std_mean_by_formula": {
                    item["formula_label"]: item["test_nrmse_std_mean"] for item in ranking
                },
                "test_r2_mean_by_formula": {
                    item["formula_label"]: item["test_r2_mean"] for item in ranking
                },
            },
            "outputs": {label: str(path) for label, path in artifacts.items()},
            "hashes": output_hashes,
        },
    )

    print(f"[13_af_compare] Winner: {winner}")
    print(f"[13_af_compare] Compare JSON: {compare_json_path}")
    print(f"[13_af_compare] Split metrics CSV: {metrics_csv_path}")
    print(f"[13_af_compare] Ranking JSON: {ranking_json_path}")

    contracts.log_stage_paths(
        SimpleNamespace(
            out_root=runs_root,
            stage_dir=stage_dir,
            outputs_dir=outputs_dir,
            stage_summary_path=stage_summary_path,
            manifest_path=manifest_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
