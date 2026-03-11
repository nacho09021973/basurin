#!/usr/bin/env python3
"""Experimental validation of MALDA symbolic formula candidates.

Consumes:
  - runs/<run_id>/experiment/malda_feature_table/outputs/event_features.csv
  - runs/<run_id>/experiment/malda_discovery/outputs/discovery_summary.json

Produces under runs/<run_id>/experiment/malda_formula_validation/:
  - outputs/formula_validation.json
  - outputs/formula_validation_metrics.csv
  - outputs/formula_recommendations.json
  - manifest.json
  - stage_summary.json

This stage does not promote equations into the canonical pipeline. It only
benchmarks them on the current BASURIN dataset so we can decide whether they
are worth testing as experimental priors or warm-starts.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
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


STAGE_NAME = "malda_formula_validation"
DISCOVERY_STAGE_NAME = "malda_discovery"
FEATURE_TABLE_STAGE_NAME = "malda_feature_table"

TARGET_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "E_rad_frac": (0.0, 1.0),
    "af": (0.0, 1.0),
    "S_f": (0.0, None),
    "delta_S": (0.0, None),
    "S_ratio": (1.0, None),
    "Q_220": (0.0, None),
    "F_220_dimless": (0.0, None),
    "f_ratio_221_220": (0.0, None),
}

ALLOWED_FUNCS: dict[str, Any] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "square": np.square,
}
ALLOWED_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}
RECOMMENDATION_PRIORITY = {
    "EXPERIMENTAL_PRIOR_CANDIDATE": 0,
    "WARM_START_CANDIDATE": 1,
    "INTERNAL_BENCHMARK": 2,
    "REVIEW_REQUIRED": 3,
    "REWORK_OR_REJECT": 4,
    "REJECT_NONFINITE": 5,
}


class FormulaValidator(ast.NodeVisitor):
    """Restrict symbolic formulas to a small safe algebra."""

    def __init__(self) -> None:
        self.variable_names: set[str] = set()

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in formulas")
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        if node.func.id not in ALLOWED_FUNCS:
            raise ValueError(f"Function not allowed: {node.func.id}")
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ALLOWED_FUNCS or node.id in ALLOWED_CONSTS:
            return
        self.variable_names.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Constant type not allowed: {type(node.value).__name__}")

    def generic_visit(self, node: ast.AST) -> None:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.UAdd,
            ast.USub,
        )
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"AST node not allowed: {type(node).__name__}")
        super().generic_visit(node)


def load_feature_columns(path: Path, *, bbh_only: bool) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if bbh_only:
        rows = [row for row in rows if row.get("is_bbh", "0").strip() == "1"]
        print(f"[12_formula_validation] BBH filter: {len(rows)} events retained")

    if not rows:
        raise RuntimeError("No events loaded from feature table. Run step 10 first.")

    columns = list(rows[0].keys())
    data: dict[str, np.ndarray] = {}
    for column in columns:
        values: list[float] = []
        for row in rows:
            raw = row.get(column, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        data[column] = np.asarray(values, dtype=np.float64)
    return rows, data


def load_discovery_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in discovery summary: {path}")
    return payload


def normalize_formula(expression: str) -> str:
    return expression.replace("^", "**")


def compile_formula(expression: str) -> tuple[object, list[str]]:
    normalized = normalize_formula(expression)
    tree = ast.parse(normalized, mode="eval")
    validator = FormulaValidator()
    validator.visit(tree)
    return compile(tree, "<formula>", "eval"), sorted(validator.variable_names)


def resolve_variable(name: str, columns: dict[str, np.ndarray]) -> np.ndarray:
    if name in columns:
        return columns[name]
    if name.startswith("log_"):
        base = name[4:]
        if base not in columns:
            raise ValueError(f"Variable '{name}' missing and base column '{base}' not found")
        base_values = columns[base]
        with np.errstate(all="ignore"):
            return np.where(base_values > 0.0, np.log(base_values), np.nan)
    raise ValueError(f"Variable '{name}' not available in feature table")


def evaluate_formula(expression: str, columns: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    code, variable_names = compile_formula(expression)
    if not columns:
        raise ValueError("No feature columns available")

    env: dict[str, Any] = {**ALLOWED_FUNCS, **ALLOWED_CONSTS}
    for variable_name in variable_names:
        env[variable_name] = resolve_variable(variable_name, columns)

    with np.errstate(all="ignore"):
        prediction = eval(code, {"__builtins__": {}}, env)
    if np.isscalar(prediction):
        n_rows = len(next(iter(columns.values())))
        prediction = np.full(n_rows, float(prediction), dtype=np.float64)
    return np.asarray(prediction, dtype=np.float64), variable_names


def _core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    y_mean = float(np.mean(y_true))
    y_std = float(np.std(y_true))
    y_span = float(np.max(y_true) - np.min(y_true))
    sse = float(np.sum(np.square(residual)))
    sst = float(np.sum(np.square(y_true - y_mean)))
    r2 = float(1.0 - sse / sst) if sst > 0.0 else float("nan")
    baseline_rmse = float(np.sqrt(np.mean(np.square(y_true - y_mean))))

    denom = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), np.nan)
    relative_error = np.abs(residual) / denom

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse_std": float(rmse / y_std) if y_std > 0.0 else float("nan"),
        "nrmse_range": float(rmse / y_span) if y_span > 0.0 else float("nan"),
        "median_relative_error": float(np.nanmedian(relative_error)),
        "mean_relative_error": float(np.nanmean(relative_error)),
        "baseline_rmse": baseline_rmse,
        "rmse_vs_mean_baseline": float(rmse / baseline_rmse) if baseline_rmse > 0.0 else float("nan"),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]] | None:
    if samples <= 0 or len(y_true) < 3:
        return None

    rng = np.random.default_rng(seed)
    metrics_by_name = {
        "r2": [],
        "rmse": [],
        "nrmse_std": [],
        "median_relative_error": [],
    }
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_metrics = _core_metrics(y_true[indices], y_pred[indices])
        for name in metrics_by_name:
            value = sampled_metrics[name]
            if math.isfinite(value):
                metrics_by_name[name].append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for metric_name, values in metrics_by_name.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[metric_name] = {
            "median": float(np.quantile(arr, 0.5)),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }
    return summary or None


def compute_target_metrics(
    *,
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    target_mask = np.isfinite(y_true)
    finite_mask = target_mask & np.isfinite(y_pred)
    n_target_valid = int(np.sum(target_mask))
    n_eval = int(np.sum(finite_mask))

    metrics: dict[str, Any] = {
        "n_target_valid": n_target_valid,
        "n_evaluated": n_eval,
        "finite_fraction": float(n_eval / n_target_valid) if n_target_valid else 0.0,
        "n_nonfinite_predictions": int(np.sum(target_mask & ~np.isfinite(y_pred))),
    }

    bounds = TARGET_BOUNDS.get(target_name)
    if bounds is not None and n_eval > 0:
        lower, upper = bounds
        in_bounds = np.ones(n_eval, dtype=bool)
        y_pred_eval = y_pred[finite_mask]
        if lower is not None:
            in_bounds &= y_pred_eval >= lower
        if upper is not None:
            in_bounds &= y_pred_eval <= upper
        metrics["physics_bounds"] = {
            "lower": lower,
            "upper": upper,
            "fraction_in_bounds": float(np.mean(in_bounds)),
            "n_out_of_bounds": int(np.sum(~in_bounds)),
        }
    else:
        metrics["physics_bounds"] = None

    if n_eval == 0:
        metrics["fit"] = None
        metrics["bootstrap"] = None
        return metrics

    y_eval = y_true[finite_mask]
    y_pred_eval = y_pred[finite_mask]
    fit_metrics = _core_metrics(y_eval, y_pred_eval)
    fit_metrics["prediction_min"] = float(np.min(y_pred_eval))
    fit_metrics["prediction_max"] = float(np.max(y_pred_eval))
    fit_metrics["target_min"] = float(np.min(y_eval))
    fit_metrics["target_max"] = float(np.max(y_eval))
    metrics["fit"] = fit_metrics
    metrics["bootstrap"] = bootstrap_metrics(
        y_eval,
        y_pred_eval,
        samples=bootstrap_samples,
        seed=seed,
    )
    return metrics


def build_recommendation(target_name: str, formula: str, metrics: dict[str, Any]) -> dict[str, Any]:
    fit = metrics.get("fit") or {}
    bounds = metrics.get("physics_bounds")
    finite_fraction = float(metrics.get("finite_fraction", 0.0))
    r2 = float(fit.get("r2", float("nan")))
    nrmse_std = float(fit.get("nrmse_std", float("nan")))

    caution_flags: list[str] = []
    if finite_fraction < 1.0:
        caution_flags.append("nonfinite_predictions")
    if bounds and bounds["n_out_of_bounds"] > 0:
        caution_flags.append("physics_bounds_violation")
    if "/" in formula:
        caution_flags.append("contains_division")
    if "exp(" in formula:
        caution_flags.append("contains_exp")
    if "log(" in formula:
        caution_flags.append("contains_log")

    if finite_fraction < 1.0:
        label = "REJECT_NONFINITE"
        reason = "Equation produces non-finite predictions on valid target rows."
    elif bounds and bounds["fraction_in_bounds"] < 1.0:
        label = "REVIEW_REQUIRED"
        reason = "Equation violates broad physical bounds on part of the evaluation set."
    elif math.isfinite(r2) and r2 >= 0.98 and math.isfinite(nrmse_std) and nrmse_std <= 0.2:
        label = "EXPERIMENTAL_PRIOR_CANDIDATE"
        reason = "High in-sample fidelity with no domain or bounds failures."
    elif math.isfinite(r2) and r2 >= 0.95 and math.isfinite(nrmse_std) and nrmse_std <= 0.35:
        label = "WARM_START_CANDIDATE"
        reason = "Strong fit, but still better suited for warm-start or proposal use than direct prior use."
    elif math.isfinite(r2) and r2 >= 0.85:
        label = "INTERNAL_BENCHMARK"
        reason = "Useful enough for internal comparisons, but not strong enough for experimental pipeline use."
    else:
        label = "REWORK_OR_REJECT"
        reason = "Fit quality is too weak for pipeline experimentation."

    return {
        "target": target_name,
        "label": label,
        "reason": reason,
        "caution_flags": caution_flags,
    }


def infer_bbh_only(discovery_stage_summary_path: Path) -> tuple[bool, str]:
    if not discovery_stage_summary_path.exists():
        return False, "default_false"
    try:
        payload = json.loads(discovery_stage_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "default_false"
    config = payload.get("config", {})
    return bool(config.get("bbh_only", False)), "discovery_stage_summary"


def resolve_targets(summary_rows: list[dict[str, Any]], requested_targets: str) -> list[dict[str, Any]]:
    if not requested_targets:
        return list(summary_rows)
    requested = {target.strip() for target in requested_targets.split(",") if target.strip()}
    return [row for row in summary_rows if row.get("target") in requested]


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target",
        "recommendation",
        "r2",
        "rmse",
        "nrmse_std",
        "median_relative_error",
        "finite_fraction",
        "bounds_fraction",
        "complexity",
        "discovery_loss",
        "variables",
        "equation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate MALDA symbolic formulas on the BASURIN event feature table"
    )
    parser.add_argument("--run-id", required=True, help="BASURIN run identifier")
    parser.add_argument(
        "--feature-table",
        default="",
        help=(
            "Path to event_features.csv. Defaults to "
            "runs/<run-id>/experiment/malda_feature_table/outputs/event_features.csv"
        ),
    )
    parser.add_argument(
        "--discovery-summary",
        default="",
        help=(
            "Path to discovery_summary.json. Defaults to "
            "runs/<run-id>/experiment/malda_discovery/outputs/discovery_summary.json"
        ),
    )
    parser.add_argument(
        "--discovery-stage-summary",
        default="",
        help=(
            "Path to malda_discovery stage_summary.json. Defaults to "
            "runs/<run-id>/experiment/malda_discovery/stage_summary.json"
        ),
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Comma-separated subset of targets to validate (default: all targets present in discovery_summary)",
    )
    parser.add_argument(
        "--bbh-only",
        action="store_true",
        help="Force BBH-only evaluation instead of inferring it from malda_discovery stage_summary.json",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap resamples per target for metric uncertainty (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling",
    )
    args = parser.parse_args(argv)

    runs_root = resolve_out_root("runs")
    validate_run_id(args.run_id, runs_root)

    try:
        require_run_valid(runs_root, args.run_id)
    except Exception as exc:
        print(f"[12_formula_validation] ERROR: RUN_VALID check failed: {exc}", file=sys.stderr)
        return 1

    stage_dir = runs_root / args.run_id / "experiment" / STAGE_NAME
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_table_path = Path(args.feature_table) if args.feature_table else (
        runs_root / args.run_id / "experiment" / FEATURE_TABLE_STAGE_NAME / "outputs" / "event_features.csv"
    )
    if not feature_table_path.exists():
        print(
            f"[12_formula_validation] ERROR: feature table not found at {feature_table_path}\n"
            f"  Run first: python malda/10_build_event_feature_table.py --run-id {args.run_id}",
            file=sys.stderr,
        )
        return 1

    discovery_summary_path = Path(args.discovery_summary) if args.discovery_summary else (
        runs_root / args.run_id / "experiment" / DISCOVERY_STAGE_NAME / "outputs" / "discovery_summary.json"
    )
    if not discovery_summary_path.exists():
        print(
            f"[12_formula_validation] ERROR: discovery summary not found at {discovery_summary_path}\n"
            f"  Run first: python malda/11_kan_pysr_discovery.py --run-id {args.run_id}",
            file=sys.stderr,
        )
        return 1

    discovery_stage_summary_path = Path(args.discovery_stage_summary) if args.discovery_stage_summary else (
        runs_root / args.run_id / "experiment" / DISCOVERY_STAGE_NAME / "stage_summary.json"
    )
    inferred_bbh_only, bbh_source = infer_bbh_only(discovery_stage_summary_path)
    bbh_only = bool(args.bbh_only or inferred_bbh_only)

    _, columns = load_feature_columns(feature_table_path, bbh_only=bbh_only)
    summary_rows = resolve_targets(load_discovery_summary(discovery_summary_path), args.targets)
    if not summary_rows:
        print(
            "[12_formula_validation] ERROR: no targets left after applying --targets filter",
            file=sys.stderr,
        )
        return 1

    print(f"[12_formula_validation] Loaded {len(next(iter(columns.values())))} events for validation")
    print(f"[12_formula_validation] Validating {len(summary_rows)} targets")

    validation_results: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    for row in summary_rows:
        target_name = str(row.get("target", "")).strip()
        formula = str(row.get("best_equation", "")).strip()
        if not target_name:
            continue
        print(f"\n[12_formula_validation] TARGET: {target_name}")
        if not formula or formula == "—":
            validation_results[target_name] = {
                "status": "skipped",
                "reason": "missing best_equation in discovery summary",
            }
            continue
        if target_name not in columns:
            validation_results[target_name] = {
                "status": "skipped",
                "reason": f"target column '{target_name}' missing from feature table",
            }
            continue

        try:
            y_pred, variable_names = evaluate_formula(formula, columns)
            metrics = compute_target_metrics(
                target_name=target_name,
                y_true=columns[target_name],
                y_pred=y_pred,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            recommendation = build_recommendation(target_name, formula, metrics)
            validation_results[target_name] = {
                "status": "ok",
                "equation": formula,
                "variables": variable_names,
                "complexity": row.get("complexity"),
                "discovery_loss": row.get("loss"),
                "metrics": metrics,
                "recommendation": recommendation,
            }
            fit = metrics.get("fit") or {}
            bounds = metrics.get("physics_bounds") or {}
            metrics_rows.append(
                {
                    "target": target_name,
                    "recommendation": recommendation["label"],
                    "r2": fit.get("r2"),
                    "rmse": fit.get("rmse"),
                    "nrmse_std": fit.get("nrmse_std"),
                    "median_relative_error": fit.get("median_relative_error"),
                    "finite_fraction": metrics.get("finite_fraction"),
                    "bounds_fraction": bounds.get("fraction_in_bounds"),
                    "complexity": row.get("complexity"),
                    "discovery_loss": row.get("loss"),
                    "variables": ",".join(variable_names),
                    "equation": formula,
                }
            )
            recommendations.append(recommendation)
            print(
                f"  recommendation={recommendation['label']} "
                f"r2={fit.get('r2')} nrmse_std={fit.get('nrmse_std')}"
            )
        except Exception as exc:
            validation_results[target_name] = {
                "status": "failed",
                "equation": formula,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            metrics_rows.append(
                {
                    "target": target_name,
                    "recommendation": "REJECT_NONFINITE",
                    "r2": None,
                    "rmse": None,
                    "nrmse_std": None,
                    "median_relative_error": None,
                    "finite_fraction": 0.0,
                    "bounds_fraction": None,
                    "complexity": row.get("complexity"),
                    "discovery_loss": row.get("loss"),
                    "variables": "",
                    "equation": formula,
                }
            )
            recommendations.append(
                {
                    "target": target_name,
                    "label": "REJECT_NONFINITE",
                    "reason": f"formula evaluation failed: {type(exc).__name__}: {exc}",
                    "caution_flags": ["evaluation_failed"],
                }
            )
            print(f"  evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    recommendations.sort(
        key=lambda item: (
            RECOMMENDATION_PRIORITY.get(item["label"], 99),
            item["target"],
        )
    )
    metrics_rows.sort(
        key=lambda row: (
            RECOMMENDATION_PRIORITY.get(str(row["recommendation"]), 99),
            str(row["target"]),
        )
    )

    payload = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "created": utc_now_iso(),
        "config": {
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bbh_only": bbh_only,
            "bbh_only_source": bbh_source if not args.bbh_only else "cli_flag",
            "targets": [row.get("target") for row in summary_rows],
        },
        "inputs": {
            "feature_table": str(feature_table_path),
            "feature_table_sha256": sha256_file(feature_table_path),
            "discovery_summary": str(discovery_summary_path),
            "discovery_summary_sha256": sha256_file(discovery_summary_path),
            "discovery_stage_summary": (
                str(discovery_stage_summary_path) if discovery_stage_summary_path.exists() else None
            ),
            "discovery_stage_summary_sha256": (
                sha256_file(discovery_stage_summary_path) if discovery_stage_summary_path.exists() else None
            ),
        },
        "targets": validation_results,
    }

    validation_json_path = outputs_dir / "formula_validation.json"
    metrics_csv_path = outputs_dir / "formula_validation_metrics.csv"
    recommendations_path = outputs_dir / "formula_recommendations.json"
    write_json_atomic(validation_json_path, payload)
    write_metrics_csv(metrics_csv_path, metrics_rows)
    write_json_atomic(
        recommendations_path,
        {
            "stage": STAGE_NAME,
            "run_id": args.run_id,
            "created": utc_now_iso(),
            "recommendations": recommendations,
        },
    )

    artifacts: dict[str, Path] = {
        "formula_validation": validation_json_path,
        "formula_validation_metrics": metrics_csv_path,
        "formula_recommendations": recommendations_path,
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
                "n_targets_requested": len(summary_rows),
                "n_targets_evaluated": sum(1 for item in validation_results.values() if item.get("status") == "ok"),
                "n_targets_failed": sum(1 for item in validation_results.values() if item.get("status") == "failed"),
                "n_experimental_prior_candidates": sum(
                    1 for item in recommendations if item["label"] == "EXPERIMENTAL_PRIOR_CANDIDATE"
                ),
                "n_warm_start_candidates": sum(
                    1 for item in recommendations if item["label"] == "WARM_START_CANDIDATE"
                ),
                "recommendation_labels_by_target": {
                    item["target"]: item["label"] for item in recommendations
                },
            },
            "outputs": {label: str(path) for label, path in artifacts.items()},
            "hashes": output_hashes,
        },
    )

    print(f"\n[12_formula_validation] Validation JSON: {validation_json_path}")
    print(f"[12_formula_validation] Metrics CSV: {metrics_csv_path}")
    print(f"[12_formula_validation] Recommendations: {recommendations_path}")
    print(f"\n[12_formula_validation] Done -> {stage_dir}")
    print(f"OUT_ROOT={runs_root}")
    print(f"STAGE_DIR={stage_dir}")
    print(f"OUTPUTS_DIR={outputs_dir}")
    print(f"STAGE_SUMMARY={stage_summary_path}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
