#!/usr/bin/env python3
"""Non-canonical experiment: recompute T6 dA/p_violate with RD weighting from batch compatible_set."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import require_run_valid, resolve_out_root, sha256_file, utc_now_iso, validate_run_id, write_json_atomic
from mvp import contracts

DEFAULT_OUT_NAME = "t6_rd_weighted"
DEFAULT_IN_REL = "experiment/area_theorem/outputs/per_event_spinmag.csv"
DEFAULT_BATCH_220 = "batch_with_t0_220_eps2500_fixlen_20260304T160054Z"
DEFAULT_BATCH_221 = "batch_with_t0_221_eps2500_fixlen_20260304T160617Z"
DEFAULT_BATCH_RESULTS = "experiment/offline_batch/outputs/results.csv"
DEFAULT_S4_COMPATIBLE = "s4_geometry_filter/outputs/compatible_set.json"
DEFAULT_IMR_JSON_RELPATH = "external_inputs/gwtc_posteriors"


class InsufficientGranularityError(RuntimeError):
    pass


def _canonical(s: str) -> str:
    return s.strip().lower().replace("-", "_")


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
    target = q * sum(w)
    csum = 0.0
    for val, wt in zip(v, w):
        csum += wt
        if csum >= target:
            return float(val)
    return float(v[-1])


def _black_hole_area(m_solar: float, chi: float) -> float:
    return 8.0 * math.pi * m_solar * m_solar * (1.0 + math.sqrt(1.0 - chi * chi))


def _resolve_in_per_event(run_dir: Path, in_per_event: str | None) -> Path:
    if in_per_event:
        p = Path(in_per_event)
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return run_dir / DEFAULT_IN_REL


def _extract_rows(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if any(k in obj for k in ("family", "source", "M_solar", "chi", "Af", "delta_lnL")):
            rows.append(obj)
        for v in obj.values():
            rows.extend(_extract_rows(v))
    elif isinstance(obj, list):
        for it in obj:
            rows.extend(_extract_rows(it))
    return rows


def _phys_key(row: dict[str, Any]) -> tuple[str, str, float, float] | None:
    fam = str(row.get("family", "")).strip().lower()
    src = str(row.get("source", "")).strip().lower()
    m = _as_float(row.get("M_solar"))
    chi = _as_float(row.get("chi"))
    if not fam or not src or m is None or chi is None:
        return None
    return (fam, src, m, chi)


def _af_value(row: dict[str, Any]) -> float | None:
    af = _as_float(row.get("Af"))
    if af is not None:
        return af
    m = _as_float(row.get("M_solar"))
    chi = _as_float(row.get("chi"))
    if m is None or chi is None or abs(chi) >= 1:
        return None
    return _black_hole_area(m, chi)


def _load_event_to_subrun(results_csv: Path) -> dict[str, str]:
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"results.csv sin encabezados: {results_csv}")
        ev_col = _find_column(fields, "event_id", "event", "event_name")
        subrun_col = _find_column(fields, "subrun_id", "run_id", "child_run_id")
        if not ev_col or not subrun_col:
            raise ValueError(
                "No se pudieron detectar columnas event/subrun en results.csv. "
                f"ruta esperada exacta: {results_csv}. columnas disponibles: {fields}. "
                "comando exacto para regenerar upstream: python -m mvp.experiment_offline_batch --batch-run-id <BATCH_RUN_ID>"
            )
        mapping: dict[str, str] = {}
        for row in reader:
            ev = str(row.get(ev_col, "")).strip()
            sub = str(row.get(subrun_col, "")).strip()
            if ev and sub:
                mapping[ev] = sub
        return mapping


def _build_weighted_af_samples(path220: Path, path221: Path, ranked_all_limit: int) -> dict[str, Any]:
    data220 = json.loads(path220.read_text(encoding="utf-8"))
    data221 = json.loads(path221.read_text(encoding="utf-8"))

    compat220 = data220.get("compatible_geometries", data220 if isinstance(data220, list) else [])
    compat221 = data221.get("compatible_geometries", data221 if isinstance(data221, list) else [])
    ranked220_full = data220.get("ranked_all", []) if isinstance(data220, dict) else []
    ranked221_full = data221.get("ranked_all", []) if isinstance(data221, dict) else []
    ranked220 = ranked220_full if ranked_all_limit == 0 else ranked220_full[:ranked_all_limit]
    ranked221 = ranked221_full if ranked_all_limit == 0 else ranked221_full[:ranked_all_limit]

    delta220_by_gid = {
        str(r["geometry_id"]): float(r["delta_lnL"])
        for r in ranked220
        if isinstance(r, dict) and r.get("geometry_id") is not None and _as_float(r.get("delta_lnL")) is not None
    }
    delta221_by_gid = {
        str(r["geometry_id"]): float(r["delta_lnL"])
        for r in ranked221
        if isinstance(r, dict) and r.get("geometry_id") is not None and _as_float(r.get("delta_lnL")) is not None
    }

    if not delta220_by_gid:
        sample_keys = sorted(list(ranked220[0].keys())) if ranked220 and isinstance(ranked220[0], dict) else []
        raise ValueError(f"MISSING_DELTA_LNL path={path220} source=ranked_all keys(sample)={sample_keys}")
    if not delta221_by_gid:
        sample_keys = sorted(list(ranked221[0].keys())) if ranked221 and isinstance(ranked221[0], dict) else []
        raise ValueError(f"MISSING_DELTA_LNL path={path221} source=ranked_all keys(sample)={sample_keys}")

    def _enrich_with_delta(compat: list[Any], delta_by_gid: dict[str, float]) -> tuple[list[dict[str, Any]], int]:
        enriched: list[dict[str, Any]] = []
        dropped = 0
        for row in compat:
            if not isinstance(row, dict):
                dropped += 1
                continue
            gid = row.get("geometry_id")
            if gid is None:
                dropped += 1
                continue
            delta = delta_by_gid.get(str(gid))
            if delta is None:
                dropped += 1
                continue
            out_row = dict(row)
            out_row["delta_lnL"] = float(delta)
            enriched.append(out_row)
        return enriched, dropped

    def _best_by_phys(enriched: list[dict[str, Any]]) -> dict[tuple[str, str, float, float], dict[str, Any]]:
        best: dict[tuple[str, str, float, float], dict[str, Any]] = {}
        for row in enriched:
            key = _phys_key(row)
            if key is None:
                continue
            if key not in best or float(row["delta_lnL"]) > float(best[key]["delta_lnL"]):
                best[key] = row
        return best

    enriched220, dropped220 = _enrich_with_delta(compat220, delta220_by_gid)
    enriched221, dropped221 = _enrich_with_delta(compat221, delta221_by_gid)

    best220 = _best_by_phys(enriched220)
    best221 = _best_by_phys(enriched221)
    support = sorted(set(best220).intersection(best221))

    samples: list[dict[str, float]] = []
    dropped_missing_weight = 0
    for key in support:
        geom220 = best220.get(key)
        geom221 = best221.get(key)
        if geom220 is None or geom221 is None:
            dropped_missing_weight += 1
            continue
        dlnl220 = float(geom220["delta_lnL"])
        dlnl221 = float(geom221["delta_lnL"])
        geom = geom220 or geom221 or {}
        af = _af_value(geom)
        if af is None:
            continue
        samples.append({"af_rd": af, "delta_lnL": dlnl220 + dlnl221})

    if not samples:
        return {
            "status": "AF_EMPTY" if not support else "MISSING_DELTA_LNL",
            "samples": [],
            "n220": len(best220),
            "n221": len(best221),
            "n_support": len(support),
            "n_intersection": len(support),
            "n_used": 0,
            "n_dropped_missing_weight": dropped_missing_weight,
            "n_ranked_all_220": len(ranked220),
            "n_ranked_all_221": len(ranked221),
            "n_ranked_all_total_220": len(ranked220_full),
            "n_ranked_all_total_221": len(ranked221_full),
            "ranked_all_limit": ranked_all_limit,
            "n_compat_220": len(compat220) if isinstance(compat220, list) else 0,
            "n_compat_221": len(compat221) if isinstance(compat221, list) else 0,
            "n_enriched_220": len(enriched220),
            "n_enriched_221": len(enriched221),
            "n_dropped_missing_gid_or_weight_220": dropped220,
            "n_dropped_missing_gid_or_weight_221": dropped221,
            "n_support_phys": len(support),
            "join_policy": "ranked_all.geometry_id -> compatible_geometries.geometry_id; reduce=max_delta_per_phys; combine=delta220+delta221",
            "weight_source": "ranked_all.delta_lnL",
            "weight_reduce": "sum",
        }

    return {
        "status": "OK",
        "samples": samples,
        "n220": len(best220),
        "n221": len(best221),
        "n_support": len(support),
        "n_intersection": len(support),
        "n_used": len(samples),
        "n_dropped_missing_weight": dropped_missing_weight,
        "n_ranked_all_220": len(ranked220),
        "n_ranked_all_221": len(ranked221),
        "n_ranked_all_total_220": len(ranked220_full),
        "n_ranked_all_total_221": len(ranked221_full),
        "ranked_all_limit": ranked_all_limit,
        "n_compat_220": len(compat220) if isinstance(compat220, list) else 0,
        "n_compat_221": len(compat221) if isinstance(compat221, list) else 0,
        "n_enriched_220": len(enriched220),
        "n_enriched_221": len(enriched221),
        "n_dropped_missing_gid_or_weight_220": dropped220,
        "n_dropped_missing_gid_or_weight_221": dropped221,
        "n_support_phys": len(support),
        "join_policy": "ranked_all.geometry_id -> compatible_geometries.geometry_id; reduce=max_delta_per_phys; combine=delta220+delta221",
        "weight_source": "ranked_all.delta_lnL",
        "weight_reduce": "sum",
    }


def _load_imr(path: Path) -> list[dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("samples", data if isinstance(data, list) else [])
    out: list[dict[str, float]] = []
    for row in items:
        m1 = _as_float(row.get("mass_1_source", row.get("m1_source")))
        m2 = _as_float(row.get("mass_2_source", row.get("m2_source")))
        a1 = _as_float(row.get("a_1", row.get("a1", row.get("chi1"))))
        a2 = _as_float(row.get("a_2", row.get("a2", row.get("chi2"))))
        if None in (m1, m2, a1, a2):
            continue
        out.append({"m1": float(m1), "m2": float(m2), "a1": abs(float(a1)), "a2": abs(float(a2))})
    if not out:
        raise ValueError(f"IMR JSON sin muestras utilizables: {path}")
    return out


def _compute_weights(delta: list[float]) -> list[float]:
    mx = max(delta)
    w = [math.exp(d - mx) for d in delta]
    total = sum(w)
    return [x / total for x in w]


def run_experiment(
    run_id: str,
    in_per_event: str | None,
    out_name: str,
    min_effective_samples: int,
    batch_220: str | None = None,
    batch_221: str | None = None,
    batch_results_relpath: str = DEFAULT_BATCH_RESULTS,
    s4_compatible_relpath: str = DEFAULT_S4_COMPATIBLE,
    imr_json_root: str | None = None,
    ranked_all_limit: int = 50,
) -> dict[str, Any]:
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)
    if ranked_all_limit < 0:
        raise ValueError(f"--ranked-all-limit debe ser >= 0; recibido={ranked_all_limit}")

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
        per_fields = list(reader.fieldnames or [])
        per_rows = list(reader)

    event_col = _find_column(per_fields, "event_id", "event", "event_name")
    d10_col = _find_column(per_fields, "dA_p10")
    d50_col = _find_column(per_fields, "dA_p50")
    d90_col = _find_column(per_fields, "dA_p90")
    p_col = _find_column(per_fields, "p_violate")
    nmc_col = _find_column(per_fields, "n_mc")
    status_col = _find_column(per_fields, "status")
    if not all([event_col, d10_col, d50_col, d90_col, p_col, nmc_col, status_col]):
        raise ValueError(f"Columnas requeridas faltantes en {in_csv}; disponibles={per_fields}")

    for new_col in ("af_rd_p10", "af_rd_p50", "af_rd_p90", "ess_rd"):
        if new_col not in per_fields:
            per_fields.append(new_col)

    if not batch_220 or not batch_221:
        raise InsufficientGranularityError(
            "INSUFFICIENT_INPUT_GRANULARITY: faltan --batch-220/--batch-221 y el CSV agregado no contiene granularidad RD. "
            f"ruta esperada exacta: {run_dir / DEFAULT_IN_REL}. "
            f"comando exacto para regenerar upstream: python -m mvp.experiment_offline_batch --batch-run-id {DEFAULT_BATCH_220}"
        )

    require_run_valid(out_root, batch_220)
    require_run_valid(out_root, batch_221)

    map220 = _load_event_to_subrun(out_root / batch_220 / batch_results_relpath)
    map221 = _load_event_to_subrun(out_root / batch_221 / batch_results_relpath)

    imr_root = (run_dir / DEFAULT_IMR_JSON_RELPATH) if not imr_json_root else Path(imr_json_root)
    if not imr_root.is_absolute():
        imr_root = (Path.cwd() / imr_root).resolve()

    rng = random.Random(7)
    event_summaries: list[dict[str, Any]] = []
    for row in per_rows:
        event = str(row[event_col]).strip()
        sub220 = map220.get(event)
        sub221 = map221.get(event)
        if not sub220 or not sub221:
            raise InsufficientGranularityError(
                f"No se pudo mapear evento={event} en results.csv. path220={out_root / batch_220 / batch_results_relpath}; "
                f"path221={out_root / batch_221 / batch_results_relpath}; candidatos220={sorted(list(map220.keys()))[:20]}"
            )

        compat220 = out_root / sub220 / s4_compatible_relpath
        compat221 = out_root / sub221 / s4_compatible_relpath
        af_pack = _build_weighted_af_samples(compat220, compat221, ranked_all_limit=ranked_all_limit)

        if af_pack["status"] != "OK":
            row[status_col] = str(af_pack["status"])
            row["ess_rd"] = "0"
            event_summaries.append({"event_id": event, **af_pack, "ess": 0.0, "policy": "CONSERVATIVE_SKIP"})
            continue

        deltas = [x["delta_lnL"] for x in af_pack["samples"]]
        af_vals = [x["af_rd"] for x in af_pack["samples"]]
        w = _compute_weights(deltas)
        ess = (sum(w) ** 2) / sum((x * x) for x in w)

        row["af_rd_p10"] = f"{_weighted_quantile(af_vals, w, 0.1):.12g}"
        row["af_rd_p50"] = f"{_weighted_quantile(af_vals, w, 0.5):.12g}"
        row["af_rd_p90"] = f"{_weighted_quantile(af_vals, w, 0.9):.12g}"
        row["ess_rd"] = f"{ess:.12g}"

        imr_samples = _load_imr(imr_root / f"{event}.json")
        n_mc = int(float(row[nmc_col])) if str(row.get(nmc_col, "")).strip() else min(1000, len(imr_samples) * 4)
        dA: list[float] = []
        for _ in range(n_mc):
            i = rng.randrange(len(imr_samples))
            j = rng.randrange(len(af_vals))
            pre = imr_samples[i]
            a1 = _black_hole_area(pre["m1"], pre["a1"])
            a2 = _black_hole_area(pre["m2"], pre["a2"])
            dA.append(af_vals[j] - (a1 + a2))
        dA.sort()
        row[d10_col] = f"{dA[max(0, int(0.1 * (len(dA) - 1)))]:.12g}"
        row[d50_col] = f"{dA[int(0.5 * (len(dA) - 1))]:.12g}"
        row[d90_col] = f"{dA[int(0.9 * (len(dA) - 1))]:.12g}"
        row[p_col] = f"{(sum(1 for x in dA if x < 0) / len(dA)):.12g}"
        row[status_col] = "OK"

        event_summaries.append(
            {
                "event_id": event,
                "ess": ess,
                "bands": ["rd_low_ess"] if ess < float(min_effective_samples) else [],
                "policy": "INCLUDE_FOR_STRESS_TESTS" if ess < float(min_effective_samples) else "OK",
                "n220": af_pack["n220"],
                "n221": af_pack["n221"],
                "n_intersection": af_pack["n_intersection"],
                "n_support": af_pack["n_support"],
                "n_used": af_pack["n_used"],
                "n_dropped_missing_weight": af_pack["n_dropped_missing_weight"],
                "n_ranked_all_220": af_pack["n_ranked_all_220"],
                "n_ranked_all_221": af_pack["n_ranked_all_221"],
                "n_ranked_all_total_220": af_pack["n_ranked_all_total_220"],
                "n_ranked_all_total_221": af_pack["n_ranked_all_total_221"],
                "ranked_all_limit": af_pack["ranked_all_limit"],
                "n_compat_220": af_pack["n_compat_220"],
                "n_compat_221": af_pack["n_compat_221"],
                "n_enriched_220": af_pack["n_enriched_220"],
                "n_enriched_221": af_pack["n_enriched_221"],
                "n_dropped_missing_gid_or_weight_220": af_pack["n_dropped_missing_gid_or_weight_220"],
                "n_dropped_missing_gid_or_weight_221": af_pack["n_dropped_missing_gid_or_weight_221"],
                "n_support_phys": af_pack["n_support_phys"],
                "join_policy": af_pack["join_policy"],
                "weight_source": af_pack["weight_source"],
                "weight_reduce": af_pack["weight_reduce"],
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
            "schema_version": "experiment_t6_rd_weighted_v2",
            "run_id": run_id,
            "source_per_event": str(in_csv),
            "batch_220": batch_220,
            "batch_221": batch_221,
            "batch_results_relpath": batch_results_relpath,
            "s4_compatible_relpath": s4_compatible_relpath,
            "imr_json_root": str(imr_root),
            "ranked_all_limit": ranked_all_limit,
            "per_event": sorted(event_summaries, key=lambda x: x["event_id"]),
        }
        write_json_atomic(out_summary, summary_payload)

        outputs = [out_csv, out_summary]
        output_records = [{"path": str(p.relative_to(tmp_stage)), "sha256": sha256_file(p)} for p in outputs]
        manifest = {
            "schema_version": "mvp_manifest_v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "artifacts": output_records,
            "inputs": [{"path": str(in_csv), "sha256": sha256_file(in_csv)}],
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
                "ess_min": min((e.get("ess", 0.0) for e in event_summaries), default=None),
                "ess_max": max((e.get("ess", 0.0) for e in event_summaries), default=None),
            },
        }
        write_json_atomic(tmp_stage / "stage_summary.json", stage_summary)

        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_stage), str(exp_dir))

    contracts.log_stage_paths(SimpleNamespace(out_root=out_root, stage_dir=exp_dir, outputs_dir=exp_dir / "outputs"))
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Recompute T6 with RD weighted Af from compatible_set intersection")
    ap.add_argument("--run-id", required=True, help="Run ID")
    ap.add_argument("--in-per-event", default=None, help="Input per-event CSV")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    ap.add_argument("--min-effective-samples", type=int, default=200)
    ap.add_argument("--batch-220", default=DEFAULT_BATCH_220)
    ap.add_argument("--batch-221", default=DEFAULT_BATCH_221)
    ap.add_argument("--batch-results-relpath", default=DEFAULT_BATCH_RESULTS)
    ap.add_argument("--s4-compatible-relpath", default=DEFAULT_S4_COMPATIBLE)
    ap.add_argument("--imr-json-root", default=None, help="Default: runs/<run_id>/external_inputs/gwtc_posteriors")
    ap.add_argument("--ranked-all-limit", type=int, default=50, help="Max ranked_all entries per event (0 = no limit)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_experiment(
        run_id=args.run_id,
        in_per_event=args.in_per_event,
        out_name=args.out_name,
        min_effective_samples=args.min_effective_samples,
        batch_220=args.batch_220,
        batch_221=args.batch_221,
        batch_results_relpath=args.batch_results_relpath,
        s4_compatible_relpath=args.s4_compatible_relpath,
        imr_json_root=args.imr_json_root,
        ranked_all_limit=args.ranked_all_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
