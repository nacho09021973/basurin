#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from mvp import contracts

STAGE = "experiment_area_theorem"
DEFAULT_220 = "batch_with_t0_220_eps2500_fixlen_20260304T160054Z"
DEFAULT_221 = "batch_with_t0_221_eps2500_fixlen_20260304T160617Z"
DEFAULT_AF_MODE = "per_event"


def black_hole_area(m_solar: float, chi: float) -> float:
    m = float(m_solar)
    c = float(chi)
    if not math.isfinite(m) or m <= 0:
        raise ValueError(f"invalid mass m_solar={m_solar}")
    if not math.isfinite(c) or abs(c) >= 1:
        raise ValueError(f"invalid spin chi={chi}; expected |chi|<1")
    return 8.0 * math.pi * m * m * (1.0 + math.sqrt(1.0 - c * c))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile of empty list")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _load_af_candidates(batch_run_id: str, out_root: Path) -> tuple[list[dict[str, Any]], Path]:
    base = out_root / batch_run_id
    candidates = [
        base / "experiment" / "theory_survival" / "outputs" / "survivors.csv",
        base / "experiment" / "theory_survival" / "outputs" / "intersection.csv",
        base / "s6_multi_event_table" / "outputs" / "multi_event.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "No intersection CSV found. "
            f"expected_one_of={[str(p) for p in candidates]}"
        )
    csv_path = existing[0]
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    af_rows: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("family", "")).strip().lower()
        source = str(row.get("source", "")).strip().lower()
        if family != "kerr" or source != "berti":
            continue
        try:
            m_solar = float(row["M_solar"])
            chi = float(row["chi"])
        except Exception:
            continue
        af_rows.append({"M_solar": m_solar, "chi": chi})
    if not af_rows:
        raise ValueError(
            f"No physical Kerr/Berti rows found in {csv_path}; "
            "required columns: family,source,M_solar,chi"
        )
    return af_rows, csv_path


def _batch_results_csv_path(batch_run_id: str, out_root: Path) -> Path:
    base = out_root / batch_run_id / "experiment" / "offline_batch"
    candidates = [
        base / "outputs" / "results.csv",
        base / "results.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Batch results.csv not found for per-event Af extraction. "
        f"expected_one_of={[str(p) for p in candidates]}"
    )


def _load_batch_event_run_map(batch_run_id: str, out_root: Path) -> tuple[dict[str, dict[str, str]], Path]:
    csv_path = _batch_results_csv_path(batch_run_id, out_root)
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        event_id = str(row.get("event_id", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if event_id and run_id:
            out[event_id] = row
    if not out:
        raise ValueError(f"Batch results map is empty: {csv_path}")
    return out, csv_path


def _extract_af_candidates_from_compatible_set(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("compatible_geometries")
    if not isinstance(rows, list):
        raise ValueError(f"compatible_set missing list 'compatible_geometries': {path}")

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        family = str(meta.get("family", row.get("family", ""))).strip().lower()
        source = str(meta.get("source", row.get("source", ""))).strip().lower()
        if family != "kerr" or not source.startswith("berti"):
            continue

        m_raw = meta.get("M_solar", row.get("M_solar"))
        chi_raw = meta.get("chi", row.get("chi"))
        try:
            m_solar = float(m_raw)
            chi = float(chi_raw)
        except Exception:
            continue
        key = (family, source, f"{m_solar:.6f}", f"{chi:.6f}")
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "family": family,
                "source": source,
                "M_solar": m_solar,
                "chi": chi,
            }
        )
    return out


def _phys_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    family = str(row.get("family", "")).strip().lower()
    source = str(row.get("source", "")).strip().lower()
    m = float(row["M_solar"])
    chi = float(row["chi"])
    return family, source, f"{m:.6f}", f"{chi:.6f}"


def _intersect_af_rows(rows_220: list[dict[str, Any]], rows_221: list[dict[str, Any]]) -> list[dict[str, Any]]:
    map_220 = {_phys_key(row): row for row in rows_220}
    map_221 = {_phys_key(row): row for row in rows_221}
    keys = sorted(set(map_220).intersection(map_221))
    out: list[dict[str, Any]] = []
    for key in keys:
        row = map_220[key]
        out.append({"M_solar": float(row["M_solar"]), "chi": float(row["chi"])})
    return out


def _load_event_af_from_batches(
    *,
    event_id: str,
    out_root: Path,
    batch220_map: dict[str, dict[str, str]],
    batch221_map: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    row220 = batch220_map.get(event_id)
    row221 = batch221_map.get(event_id)
    if row220 is None or row221 is None:
        raise KeyError(
            f"event_id={event_id} missing in batch maps "
            f"(in220={row220 is not None}, in221={row221 is not None})"
        )

    run220 = str(row220.get("run_id", "")).strip()
    run221 = str(row221.get("run_id", "")).strip()
    if not run220 or not run221:
        raise ValueError(f"event_id={event_id} has empty run_id in batch results")

    status220 = str(row220.get("status", "")).strip().upper()
    status221 = str(row221.get("status", "")).strip().upper()
    if status220 and status220 != "PASS":
        raise ValueError(f"event_id={event_id} batch220 status={status220} != PASS")
    if status221 and status221 != "PASS":
        raise ValueError(f"event_id={event_id} batch221 status={status221} != PASS")

    s4_220 = out_root / run220 / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    s4_221 = out_root / run221 / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    if not s4_220.exists() or not s4_221.exists():
        raise FileNotFoundError(
            f"missing compatible_set for event_id={event_id}. "
            f"expected220={s4_220} expected221={s4_221}"
        )

    af_220 = _extract_af_candidates_from_compatible_set(s4_220)
    af_221 = _extract_af_candidates_from_compatible_set(s4_221)
    af_intersection = _intersect_af_rows(af_220, af_221)

    debug = {
        "event_id": event_id,
        "run220": run220,
        "run221": run221,
        "n_af_220": len(af_220),
        "n_af_221": len(af_221),
        "n_af_intersection": len(af_intersection),
    }
    paths = {
        "s4_220": s4_220,
        "s4_221": s4_221,
    }
    return af_intersection, debug, paths


def _load_event_ids(events_file: Path) -> list[str]:
    out: list[str] = []
    for line in events_file.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(item)
    return out


def _load_imr_samples(path: Path) -> list[dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Invalid IMR posterior file: {path}")
    out: list[dict[str, float]] = []
    for row in samples:
        out.append(
            {
                "m1_source": float(row["m1_source"]),
                "m2_source": float(row["m2_source"]),
                "chi1": float(row["chi1"]),
                "chi2": float(row["chi2"]),
            }
        )
    return out


def compute_delta_a_distribution(
    *,
    af_rows: list[dict[str, Any]],
    imr_samples: list[dict[str, float]],
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    af_area_pool = [black_hole_area(r["M_solar"], r["chi"]) for r in af_rows]

    deltas: list[float] = []
    for _ in range(n_draws):
        af = rng.choice(af_area_pool)
        pre = rng.choice(imr_samples)
        a1 = black_hole_area(pre["m1_source"], pre["chi1"])
        a2 = black_hole_area(pre["m2_source"], pre["chi2"])
        deltas.append(af - (a1 + a2))

    negative = sum(1 for x in deltas if x < 0)
    return {
        "P_deltaA_lt_0": negative / len(deltas),
        "deltaA_p10": _quantile(deltas, 0.10),
        "deltaA_p50": _quantile(deltas, 0.50),
        "deltaA_p90": _quantile(deltas, 0.90),
        "deltaA_mean": statistics.fmean(deltas),
        "n_draws": len(deltas),
        "n_af_candidates": len(af_area_pool),
        "n_imr_samples": len(imr_samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment T6: area theorem deltaA with IMR pre-merger posteriors")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--events-file", default=None)
    ap.add_argument("--batch220-run-id", default=DEFAULT_220)
    ap.add_argument("--batch221-run-id", default=DEFAULT_221)
    ap.add_argument(
        "--af-mode",
        choices=["per_event", "legacy_global"],
        default=DEFAULT_AF_MODE,
        help="Af source mode: per_event (default, strict no-cross-event mixing) or legacy_global.",
    )
    ap.add_argument("--mc-draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ctx = contracts.init_stage(
        args.run_id,
        STAGE,
        params={
            "events_file": args.events_file,
            "batch220_run_id": args.batch220_run_id,
            "batch221_run_id": args.batch221_run_id,
            "af_mode": args.af_mode,
            "mc_draws": args.mc_draws,
            "seed": args.seed,
        },
    )

    posterior_dir = ctx.run_dir / "external_inputs" / "gwtc_posteriors"
    if args.events_file:
        events_file = Path(args.events_file)
    else:
        pilot = posterior_dir / "pilot_events.txt"
        required = posterior_dir / "required_events.txt"
        events_file = pilot if pilot.exists() else required

    if not events_file.exists():
        contracts.abort(
            ctx,
            (
                f"Missing events list. expected={events_file}; "
                f"regen_cmd='printf ""GW150914\n"" > {posterior_dir / 'pilot_events.txt'}'; candidates=[]"
            ),
        )

    event_ids = _load_event_ids(events_file)
    if not event_ids:
        contracts.abort(ctx, f"No event IDs found in {events_file}")

    missing = [ev for ev in event_ids if not (posterior_dir / f"{ev}.json").exists()]
    if missing:
        contracts.abort(
            ctx,
            (
                f"Missing required IMR posterior JSON. expected_paths={[str(posterior_dir / f'{ev}.json') for ev in missing]}; "
                f"regen_cmd='python -m mvp.experiment_gwtc_posteriors_fetch --run-id {args.run_id} --source manual --format json'; "
                f"missing_events={missing}; candidates={sorted(str(p.name) for p in posterior_dir.glob('*.json'))}"
            ),
        )

    input_paths: dict[str, Path] = {"events_file": events_file}
    for ev in event_ids:
        input_paths[f"posterior_{ev}"] = posterior_dir / f"{ev}.json"
    debug_rows: list[dict[str, Any]] = []

    if args.af_mode == "per_event":
        try:
            batch220_map, batch220_csv = _load_batch_event_run_map(args.batch220_run_id, ctx.out_root)
            batch221_map, batch221_csv = _load_batch_event_run_map(args.batch221_run_id, ctx.out_root)
        except Exception as exc:
            contracts.abort(
                ctx,
                (
                    f"Unable to load per-event batch maps: {exc}; "
                    f"regen_cmd='python -m mvp.experiment_offline_batch --batch-run-id {args.batch220_run_id}'; "
                    f"and='python -m mvp.experiment_offline_batch --batch-run-id {args.batch221_run_id}'"
                ),
            )
        input_paths["batch220_results_csv"] = batch220_csv
        input_paths["batch221_results_csv"] = batch221_csv
    else:
        try:
            af_220_legacy, af_220_path = _load_af_candidates(args.batch220_run_id, ctx.out_root)
            af_221_legacy, af_221_path = _load_af_candidates(args.batch221_run_id, ctx.out_root)
        except Exception as exc:
            contracts.abort(
                ctx,
                (
                    f"Unable to load Af candidates from upstream batches: {exc}; "
                    f"regen_cmd='python -m mvp.experiment_offline_batch --batch-run-id {args.batch220_run_id}'; "
                    f"and='python -m mvp.experiment_offline_batch --batch-run-id {args.batch221_run_id}'"
                ),
            )
        input_paths["af_batch220"] = af_220_path
        input_paths["af_batch221"] = af_221_path

    contracts.check_inputs(ctx, input_paths)

    rows: list[dict[str, Any]] = []
    for idx, ev in enumerate(event_ids):
        if args.af_mode == "per_event":
            try:
                af_rows, dbg, extra_paths = _load_event_af_from_batches(
                    event_id=ev,
                    out_root=ctx.out_root,
                    batch220_map=batch220_map,
                    batch221_map=batch221_map,
                )
            except Exception as exc:
                contracts.abort(
                    ctx,
                    (
                        f"Per-event Af extraction failed for event_id={ev}: {exc}; "
                        f"expected batch220={args.batch220_run_id} batch221={args.batch221_run_id}. "
                        "Regenerate: python -m mvp.experiment_offline_batch --batch-run-id <BATCH_RUN_ID>"
                    ),
                )
            debug_rows.append(dbg)
            for label, path in extra_paths.items():
                key = f"{label}_{ev}"
                input_paths[key] = path
        else:
            af_rows = af_220_legacy + af_221_legacy

        samples = _load_imr_samples(posterior_dir / f"{ev}.json")
        if af_rows:
            stats = compute_delta_a_distribution(
                af_rows=af_rows,
                imr_samples=samples,
                n_draws=args.mc_draws,
                seed=args.seed + idx,
            )
        else:
            stats = {
                "P_deltaA_lt_0": None,
                "deltaA_p10": None,
                "deltaA_p50": None,
                "deltaA_p90": None,
                "deltaA_mean": None,
                "n_draws": 0,
                "n_af_candidates": 0,
                "n_imr_samples": len(samples),
            }
        rows.append({"event_id": ev, **stats})

    per_event = ctx.outputs_dir / "per_event.csv"
    cols = [
        "event_id",
        "P_deltaA_lt_0",
        "deltaA_p10",
        "deltaA_p50",
        "deltaA_p90",
        "deltaA_mean",
        "n_draws",
        "n_af_candidates",
        "n_imr_samples",
    ]
    with open(per_event, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = ctx.outputs_dir / "summary.json"
    summary_payload = {
        "stage": STAGE,
        "note": "Guardrail: mvp/gwtc_events.py exposes final medians only and is insufficient for A1+A2 pre-merger; IMR posteriors are required.",
        "run_id": args.run_id,
        "af_mode": args.af_mode,
        "events": event_ids,
        "batch_220": args.batch220_run_id,
        "batch_221": args.batch221_run_id,
        "results": rows,
        "debug_per_event_af": debug_rows,
    }
    summary.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    contracts.finalize(
        ctx,
        artifacts={"per_event": per_event, "summary": summary},
        results={"events": len(rows), "mc_draws": args.mc_draws},
    )
    contracts.log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
