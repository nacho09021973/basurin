#!/usr/bin/env python3
"""Reproducible non-canonical tool to sweep delta_lnL thresholds on s4 logic."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import write_json_atomic
from mvp.contracts import check_inputs, finalize, init_stage, log_stage_paths
from mvp.s4_geometry_filter import _filter_atlas_by_mode, _load_atlas, compute_compatible_set

STAGE = "experiment/delta_lnL_sweep"
DEFAULT_MODES = ["(2,2,0)", "(2,2,1)"]


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--deltas must include at least one value")
    if any(v < 0 for v in values):
        raise ValueError("--deltas values must be >= 0")
    return sorted(set(values))


def _resolve_atlas_path(run_dir: Path, atlas_path_cli: str | None) -> Path:
    if atlas_path_cli:
        atlas_path = Path(atlas_path_cli)
        return atlas_path if atlas_path.is_absolute() else (Path.cwd() / atlas_path).resolve()

    stage_summary = run_dir / "s4_geometry_filter" / "stage_summary.json"
    if not stage_summary.exists():
        raise FileNotFoundError(
            "Input faltante para experimento. "
            f"ruta esperada exacta: {stage_summary}. "
            "comando exacto para regenerar upstream: "
            f"python -m mvp.s4_geometry_filter --run <RUN_ID> --atlas-path <ATLAS_PATH>"
        )
    data = json.loads(stage_summary.read_text(encoding="utf-8"))
    atlas_raw = ((data.get("parameters") or {}).get("atlas_path"))
    if not atlas_raw:
        raise ValueError(
            "Input faltante para experimento. "
            f"ruta esperada exacta: {stage_summary} (parameters.atlas_path). "
            "comando exacto para regenerar upstream: "
            f"python -m mvp.s4_geometry_filter --run <RUN_ID> --atlas-path <ATLAS_PATH>"
        )
    atlas_path = Path(atlas_raw)
    return atlas_path if atlas_path.is_absolute() else (Path.cwd() / atlas_path).resolve()


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _build_row(event_id: str, mode: str, delta: float, result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") or {}
    quantiles = diagnostics.get("d2_quantiles") or {}
    return {
        "event_id": event_id,
        "mode": mode,
        "delta": float(delta),
        "n_atlas": int(result.get("n_atlas", 0)),
        "n_compatible": int(result.get("n_compatible", 0)),
        "acceptance_fraction": float(diagnostics.get("acceptance_fraction", 0.0)),
        "informative_status": diagnostics.get("informative_status"),
        "d2_min": result.get("d2_min"),
        "d2_p50": quantiles.get("p50"),
        "d2_p90": quantiles.get("p90"),
        "d2_iqr": quantiles.get("iqr"),
    }


def run_sweep(run_id: str, atlas_path_cli: str | None, deltas: list[float], modes: list[str]) -> dict[str, Any]:
    ctx = init_stage(
        run_id,
        STAGE,
        params={"deltas": deltas, "modes": modes, "atlas_path_override": atlas_path_cli},
    )

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    atlas_path = _resolve_atlas_path(ctx.run_dir, atlas_path_cli)
    check_inputs(ctx, {"estimates": estimates_path, "atlas": atlas_path})

    estimates = json.loads(estimates_path.read_text(encoding="utf-8"))
    combined = estimates.get("combined") or {}
    unc = estimates.get("combined_uncertainty") or {}
    f_obs = float(combined.get("f_hz", 0.0))
    q_obs = float(combined.get("Q", 0.0))
    if f_obs <= 0 or q_obs <= 0:
        raise ValueError(f"Invalid observables in estimates: f_hz={f_obs}, Q={q_obs}")

    sigma_logf = unc.get("sigma_logf")
    sigma_logq = unc.get("sigma_logQ")
    cov = unc.get("cov_logf_logQ")
    if sigma_logf is None or sigma_logq is None:
        raise ValueError(
            "Input faltante para experimento. "
            f"ruta esperada exacta: {estimates_path} (combined_uncertainty.sigma_logf/sigma_logQ). "
            "comando exacto para regenerar upstream: "
            f"python -m mvp.s3_ringdown_estimates --run-id {run_id}"
        )

    atlas_entries = _load_atlas(atlas_path)
    event_id = str(estimates.get("event_id", "unknown"))

    rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_atlas = _filter_atlas_by_mode(atlas_entries, mode)
        for delta in deltas:
            result = compute_compatible_set(
                f_obs,
                q_obs,
                mode_atlas,
                epsilon=5.991,
                metric="mahalanobis_log",
                metric_params={
                    "sigma_logf": sigma_logf,
                    "sigma_logQ": sigma_logq,
                    "cov_logf_logQ": cov,
                },
                threshold_mode="delta_lnL",
                threshold_params={"delta_lnL": float(delta), "source_flag": "delta_sweep"},
            )
            rows.append(_build_row(event_id, mode, delta, result))

    rows_sorted = sorted(rows, key=lambda r: (r["event_id"], r["mode"], r["delta"]))

    json_out = {
        "schema_version": "experiment_delta_lnL_sweep_v1",
        "stage": STAGE,
        "run_id": run_id,
        "event_id": event_id,
        "atlas_path": str(atlas_path),
        "deltas": deltas,
        "modes": modes,
        "rows": rows_sorted,
    }

    json_path = ctx.outputs_dir / "delta_sweep.json"
    tsv_path = ctx.outputs_dir / "delta_sweep.tsv"
    write_json_atomic(json_path, json_out)

    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "mode",
                "delta",
                "n_atlas",
                "n_compatible",
                "acceptance_fraction",
                "informative_status",
                "d2_min",
                "d2_p50",
                "d2_p90",
                "d2_iqr",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    d2_values = sorted(
        float(r["d2_min"]) for r in rows_sorted if isinstance(r.get("d2_min"), (int, float))
    )
    finalize(
        ctx,
        artifacts={"delta_sweep_json": json_path, "delta_sweep_tsv": tsv_path},
        results={
            "rows": len(rows_sorted),
            "d2_min": d2_values[0] if d2_values else None,
            "d2_p50": _quantile(d2_values, 0.5),
            "d2_p90": _quantile(d2_values, 0.9),
            "d2_iqr": (
                (_quantile(d2_values, 0.75) - _quantile(d2_values, 0.25))
                if len(d2_values) > 0 and _quantile(d2_values, 0.75) is not None and _quantile(d2_values, 0.25) is not None
                else None
            ),
        },
    )
    log_stage_paths(ctx)
    return json_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep delta_lnL thresholds using s4-compatible logic")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument("--deltas", required=True, help="Comma-separated non-negative delta values")
    ap.add_argument(
        "--modes",
        default=";".join(DEFAULT_MODES),
        help="Semicolon-separated mode filters, default '(2,2,0);(2,2,1)'",
    )
    args = ap.parse_args()

    deltas = _parse_csv_floats(args.deltas)
    modes = [m.strip() for m in args.modes.split(";") if m.strip()]
    if not modes:
        raise ValueError("--modes must include at least one mode")

    run_sweep(args.run_id, args.atlas_path, deltas, modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
