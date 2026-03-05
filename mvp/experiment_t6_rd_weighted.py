#!/usr/bin/env python3
"""Non-canonical experiment: recompute RD per-event quantiles with likelihood weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import require_run_valid, resolve_out_root, sha256_file, utc_now_iso, validate_run_id, write_json_atomic
from mvp import contracts

DEFAULT_OUT_NAME = "t6_rd_weighted"
DEFAULT_IN_REL = "experiment/area_theorem/outputs/per_event_spinmag.csv"


class InsufficientGranularityError(RuntimeError):
    pass


def _canonical(path: str) -> str:
    return path.strip().lower().replace("-", "_")


def _find_column(columns: list[str], *candidates: str) -> str | None:
    canon = {_canonical(c): c for c in columns}
    for cand in candidates:
        hit = canon.get(_canonical(cand))
        if hit:
            return hit
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    order = sorted(range(len(values)), key=lambda i: values[i])
    v = [values[i] for i in order]
    w = [weights[i] for i in order]
    total = sum(w)
    if total <= 0:
        raise ValueError("sum of normalized weights must be > 0")
    target = q * total
    csum = 0.0
    for val, wt in zip(v, w):
        csum += wt
        if csum >= target:
            return float(val)
    return float(v[-1])


def _discover_granular_candidates(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for pat in ["experiment/**/outputs/*.csv", "experiment/**/outputs/*.json", "**/outputs/*.csv", "**/outputs/*.json"]:
        for path in sorted(run_dir.glob(pat)):
            if path.name == "per_event_spinmag_rd_weighted.csv":
                continue
            if path.is_file():
                candidates.append(path)
    return candidates


def _load_granular_samples(path: Path, weight_key: str) -> dict[str, list[dict[str, float]]]:
    if path.suffix.lower() != ".csv":
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return {}
        cols = list(reader.fieldnames)
        event_col = _find_column(cols, "event_id", "event", "event_name")
        w_col = _find_column(cols, weight_key)
        rd_col = _find_column(cols, "a_f_rd_sample", "a_f_rd", "af_rd", "a_f_rd_value")
        if not event_col or not w_col or not rd_col:
            return {}
        events: dict[str, list[dict[str, float]]] = {}
        for row in reader:
            event = str(row.get(event_col, "")).strip()
            w_raw = _as_float(row.get(w_col))
            rd = _as_float(row.get(rd_col))
            if not event or w_raw is None or rd is None:
                continue
            events.setdefault(event, []).append({"delta_lnL": w_raw, "a_f_rd": rd})
        return events


def _resolve_in_per_event(run_dir: Path, in_per_event: str | None) -> Path:
    if in_per_event:
        p = Path(in_per_event)
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    default = run_dir / DEFAULT_IN_REL
    if default.exists():
        return default
    raise FileNotFoundError(
        "Input faltante para experimento. "
        f"ruta esperada exacta: {default}. "
        "comando exacto para regenerar upstream: "
        f"python -m mvp.experiment_area_theorem --run-id {run_dir.name}"
    )


def _compute_weights(delta: list[float], transform: str) -> list[float]:
    if transform == "identity":
        vals = delta
    elif transform == "exp":
        mx = max(delta)
        vals = [math.exp(d - mx) for d in delta]
    else:
        raise ValueError(f"unknown weight transform: {transform}")
    s = sum(vals)
    if s <= 0:
        raise ValueError("non-positive total weight")
    return [v / s for v in vals]


def run_experiment(
    run_id: str,
    in_per_event: str | None,
    out_name: str,
    weight_key: str,
    weight_transform: str,
    min_effective_samples: int,
) -> dict[str, Any]:
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)

    run_dir = out_root / run_id
    in_csv = _resolve_in_per_event(run_dir, in_per_event)
    if not in_csv.exists():
        raise FileNotFoundError(
            "Input faltante para experimento. "
            f"ruta esperada exacta: {in_csv}. "
            "comando exacto para regenerar upstream: "
            f"python -m mvp.experiment_area_theorem --run-id {run_id}"
        )

    with in_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"per-event CSV sin encabezados: {in_csv}")
        per_fields = list(reader.fieldnames)
        per_rows = list(reader)

    event_col = _find_column(per_fields, "event_id", "event", "event_name")
    rd_p10_col = _find_column(per_fields, "a_f_rd_p10", "af_rd_p10", "a_f_rd_q10")
    rd_p50_col = _find_column(per_fields, "a_f_rd_p50", "af_rd_p50", "a_f_rd_q50")
    rd_p90_col = _find_column(per_fields, "a_f_rd_p90", "af_rd_p90", "a_f_rd_q90")

    missing_cols = [name for name, col in {
        "event_id": event_col,
        "a_f_rd_p10": rd_p10_col,
        "a_f_rd_p50": rd_p50_col,
        "a_f_rd_p90": rd_p90_col,
    }.items() if col is None]
    if missing_cols:
        raise ValueError(
            f"Columnas requeridas faltantes en {in_csv}: {missing_cols}. "
            f"columnas disponibles: {per_fields}"
        )

    granular_by_event: dict[str, list[dict[str, float]]] = {}
    candidates = _discover_granular_candidates(run_dir)
    for cand in candidates:
        parsed = _load_granular_samples(cand, weight_key)
        for ev, rows in parsed.items():
            granular_by_event.setdefault(ev, []).extend(rows)

    if not granular_by_event:
        raise InsufficientGranularityError(
            "INSUFFICIENT_INPUT_GRANULARITY: no se encontró artefacto granular por geometría "
            f"con columnas '{weight_key}' y 'A_f^RD sample' bajo {run_dir}. "
            "Se necesita un artefacto por-evento con filas por geometría que incluya "
            "delta_lnL y (A_f^{RD} o M_f,chi_f RD) para reponderar. "
            f"Candidatos detectados: {[str(p.relative_to(run_dir)) for p in candidates[:20]]}"
        )

    event_summaries: list[dict[str, Any]] = []
    for row in per_rows:
        event = str(row[event_col]).strip()
        samples = granular_by_event.get(event, [])
        if not samples:
            raise InsufficientGranularityError(
                "INSUFFICIENT_INPUT_GRANULARITY: faltan muestras granulares para evento "
                f"'{event}'. ruta esperada exacta: <run>/**/outputs/*.csv con {weight_key}, A_f^RD sample y event_id. "
                "comando exacto para regenerar upstream: python -m mvp.experiment_area_theorem --run-id "
                f"{run_id}"
            )

        deltas = [float(s["delta_lnL"]) for s in samples]
        af_vals = [float(s["a_f_rd"]) for s in samples]
        w = _compute_weights(deltas, weight_transform)
        ess = (sum(w) ** 2) / sum((wi ** 2) for wi in w)

        row[rd_p10_col] = f"{_weighted_quantile(af_vals, w, 0.10):.12g}"
        row[rd_p50_col] = f"{_weighted_quantile(af_vals, w, 0.50):.12g}"
        row[rd_p90_col] = f"{_weighted_quantile(af_vals, w, 0.90):.12g}"

        band: list[str] = []
        if ess < float(min_effective_samples):
            band.append("rd_low_ess")
        if str(row.get("mode_221_saturated", "")).strip().lower() in {"1", "true", "yes"}:
            band.append("221_saturated")
        event_summaries.append(
            {
                "event_id": event,
                "n_samples": len(samples),
                "ess": ess,
                "event_quality": {
                    "policy": "INCLUDE_FOR_STRESS_TESTS" if ess < float(min_effective_samples) else "OK",
                    "band": band,
                },
            }
        )

    exp_dir = run_dir / "experiment" / out_name
    with tempfile.TemporaryDirectory(prefix=f"{out_name}_", dir=run_dir) as tmpdir:
        tmp_stage = Path(tmpdir) / "stage"
        tmp_outputs = tmp_stage / "outputs"
        tmp_outputs.mkdir(parents=True, exist_ok=True)

        out_csv = tmp_outputs / "per_event_spinmag_rd_weighted.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=per_fields)
            writer.writeheader()
            writer.writerows(per_rows)

        out_summary = tmp_outputs / "summary.json"
        summary_payload = {
            "schema_version": "experiment_t6_rd_weighted_v1",
            "run_id": run_id,
            "source_per_event": str(in_csv),
            "weight_key": weight_key,
            "weight_transform": weight_transform,
            "min_effective_samples": int(min_effective_samples),
            "events": sorted(event_summaries, key=lambda x: x["event_id"]),
        }
        write_json_atomic(out_summary, summary_payload)

        outputs = [out_csv, out_summary]
        output_records = [
            {
                "path": str(p.relative_to(tmp_stage)),
                "sha256": sha256_file(p),
            }
            for p in outputs
        ]
        manifest = {
            "schema_version": "mvp_manifest_v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "artifacts": output_records,
            "inputs": [
                {"path": str(in_csv), "sha256": sha256_file(in_csv)},
            ],
        }
        write_json_atomic(tmp_stage / "manifest.json", manifest)

        stage_summary = {
            "status": "PASS",
            "stage": f"experiment/{out_name}",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "inputs": manifest["inputs"],
            "outputs": output_records,
            "metrics": {
                "events": len(event_summaries),
                "ess_min": min((e["ess"] for e in event_summaries), default=None),
                "ess_max": max((e["ess"] for e in event_summaries), default=None),
            },
        }
        write_json_atomic(tmp_stage / "stage_summary.json", stage_summary)

        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_stage), str(exp_dir))

    ctx = SimpleNamespace(
        out_root=out_root,
        stage_dir=exp_dir,
        outputs_dir=exp_dir / "outputs",
    )
    contracts.log_stage_paths(ctx)
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Recompute per-event RD quantiles with likelihood weighting")
    ap.add_argument("--run-id", required=True, help="Run ID")
    ap.add_argument("--in-per-event", default=None, help="Input per-event CSV (default from area_theorem experiment)")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    ap.add_argument("--weight-key", default="delta_lnL")
    ap.add_argument("--weight-transform", choices=["exp", "identity"], default="exp")
    ap.add_argument("--min-effective-samples", type=int, default=200)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_experiment(
        run_id=args.run_id,
        in_per_event=args.in_per_event,
        out_name=args.out_name,
        weight_key=args.weight_key,
        weight_transform=args.weight_transform,
        min_effective_samples=args.min_effective_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
