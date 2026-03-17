#!/usr/bin/env python3
"""Phase4B: Hawking area-law discriminative filter.

Reads per_event_hawking_area.csv produced by phase4_hawking_area_common_support
and applies the relational filter:

    hawking_pass = (A_final >= A_initial)

where A_initial is derived from GWTC IMR posterior samples per event:

    runs/<host_run>/external_inputs/gwtc_posteriors/<EVENT_ID>.json

Schema per file (minimum):
    {
        "event_id": "GWXXXX",
        "samples": [
            {"m1_source": <Msun>, "m2_source": <Msun>, "chi1": <dim>, "chi2": <dim>},
            ...
        ]
    }

Component area formula:
    A = 8 * pi * M^2 * (1 + sqrt(1 - chi^2))
    M in solar masses, chi dimensionless.
    Units: geom_solar_mass_sq

A_initial per event = median(A1(sample) + A2(sample)) over samples.
Estimator: sample_median (p50).

Phase4 upstream is read-only.  All writes go under:

    runs/<host_run>/experiment/phase4b_hawking_area_law_filter/
"""
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
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
)
from mvp import contracts
from mvp.experiment_phase4_hawking_area_common_support import _percentile_sorted

SCHEMA_VERSION = "phase4b_hawking_area_law_filter_v1"
DEFAULT_OUT_NAME = "phase4b_hawking_area_law_filter"
PHASE4_UPSTREAM_NAME = "phase4_hawking_area_common_support"

# Units convention: A = 8*pi*M^2*(1+sqrt(1-chi^2)) with M in solar masses
REQUIRED_UNITS = "geom_solar_mass_sq"

# All Phase4 upstream artifacts that must exist
_PHASE4_REQUIRED_ARTIFACTS = [
    "stage_summary.json",
    "manifest.json",
    "outputs/per_event_hawking_area.csv",
    "outputs/per_event_common_support.csv",
    "outputs/common_support_summary.json",
    "outputs/hawking_area_summary.json",
]

# Required fields in each posterior sample
_POSTERIOR_REQUIRED_FIELDS = ("m1_source", "m2_source", "chi1", "chi2")


# ---------------------------------------------------------------------------
# Physics: Kerr area formula
# ---------------------------------------------------------------------------


def _kerr_area(M_solar: float, chi: float) -> float:
    """Kerr BH area: A = 8*pi*M^2*(1+sqrt(1-chi^2)).

    Units: M in solar masses, chi dimensionless.  Result in geom_solar_mass_sq.
    Raises ValueError if M_solar <= 0 or |chi| > 1.
    """
    if not math.isfinite(M_solar) or M_solar <= 0.0:
        raise ValueError(
            f"[phase4b] Invalid mass M_solar={M_solar}: must be finite and > 0"
        )
    if not math.isfinite(chi) or abs(chi) > 1.0:
        raise ValueError(
            f"[phase4b] Invalid spin chi={chi}: |chi| must be <= 1 and finite"
        )
    return 8.0 * math.pi * M_solar**2 * (1.0 + math.sqrt(1.0 - chi**2))


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------


def _require_phase4_upstream(phase4_dir: Path) -> dict[str, Any]:
    """Assert all Phase4 upstream artifacts exist and verdict == PASS.

    Raises FileNotFoundError for missing artifacts.
    Raises RuntimeError if verdict != PASS.
    Returns the parsed stage_summary dict.
    """
    for rel in _PHASE4_REQUIRED_ARTIFACTS:
        p = phase4_dir / rel
        if not p.exists():
            raise FileNotFoundError(
                f"[phase4b] Missing Phase4 upstream artifact: {p}"
            )

    ss_path = phase4_dir / "stage_summary.json"
    ss = json.loads(ss_path.read_text(encoding="utf-8"))
    if ss.get("verdict") != "PASS":
        raise RuntimeError(
            f"[phase4b] Phase4 upstream verdict is not PASS: {ss.get('verdict')!r}"
        )
    return ss


def _validate_posterior_sample(sample: dict, idx: int) -> None:
    """Validate a single posterior sample: required fields, numeric, finite."""
    for field in _POSTERIOR_REQUIRED_FIELDS:
        if field not in sample:
            raise ValueError(
                f"[phase4b] Sample {idx}: missing required field {field!r}"
            )
        val = sample[field]
        try:
            fval = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"[phase4b] Sample {idx}: field {field!r}={val!r} is not numeric"
            )
        if not math.isfinite(fval):
            raise ValueError(
                f"[phase4b] Sample {idx}: field {field!r}={fval} is not finite"
            )


def _load_posterior_file(path: Path) -> dict[str, Any]:
    """Load and validate a single GWTC posterior JSON file.

    Raises FileNotFoundError if path does not exist.
    Raises ValueError on schema violations (missing keys, empty samples,
    non-numeric/non-finite values, invalid spin/mass).
    """
    if not path.exists():
        raise FileNotFoundError(f"[phase4b] Posterior file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if "event_id" not in data:
        raise ValueError(f"[phase4b] Posterior file {path.name}: missing 'event_id'")
    if "samples" not in data:
        raise ValueError(f"[phase4b] Posterior file {path.name}: missing 'samples'")

    samples = data["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(
            f"[phase4b] Posterior file {path.name}: 'samples' must be a non-empty list"
        )

    for i, s in enumerate(samples):
        _validate_posterior_sample(s, i)
        # Physics constraints: abort on bad spin or mass
        m1 = float(s["m1_source"])
        m2 = float(s["m2_source"])
        chi1 = float(s["chi1"])
        chi2 = float(s["chi2"])
        if m1 <= 0.0:
            raise ValueError(
                f"[phase4b] {path.name} sample {i}: m1_source={m1} must be > 0"
            )
        if m2 <= 0.0:
            raise ValueError(
                f"[phase4b] {path.name} sample {i}: m2_source={m2} must be > 0"
            )
        if abs(chi1) > 1.0:
            raise ValueError(
                f"[phase4b] {path.name} sample {i}: |chi1|={abs(chi1):.6f} > 1"
            )
        if abs(chi2) > 1.0:
            raise ValueError(
                f"[phase4b] {path.name} sample {i}: |chi2|={abs(chi2):.6f} > 1"
            )

    return data


def _load_gwtc_posteriors(
    posteriors_dir: Path,
    required_event_ids: set[str],
) -> dict[str, list[dict]]:
    """Load and validate GWTC posterior JSON files for all required events.

    Expects: <posteriors_dir>/<event_id>.json per event.
    Returns {event_id: [sample_dicts]}.
    Raises FileNotFoundError if directory or any required file is missing.
    Raises ValueError on schema or event_id mismatch.
    """
    if not posteriors_dir.is_dir():
        raise FileNotFoundError(
            f"[phase4b] GWTC posteriors directory not found: {posteriors_dir}"
        )

    result: dict[str, list[dict]] = {}
    for eid in sorted(required_event_ids):
        path = posteriors_dir / f"{eid}.json"
        data = _load_posterior_file(path)
        file_eid = data["event_id"]
        if file_eid != eid:
            raise ValueError(
                f"[phase4b] Posterior file {path.name}: event_id={file_eid!r} "
                f"does not match expected {eid!r}"
            )
        result[eid] = data["samples"]

    return result


def _derive_initial_area_stats(
    event_id: str,
    samples: list[dict],
) -> dict[str, Any]:
    """Derive A_initial statistics from IMR posterior samples for one event.

    A_initial(sample) = A1(m1_source, chi1) + A2(m2_source, chi2)
    where A = 8*pi*M^2*(1+sqrt(1-chi^2)).

    Default estimator: sample_median (p50).
    Returns dict with n_samples, p10, p50, p90.
    """
    a_vals: list[float] = []
    for s in samples:
        A1 = _kerr_area(float(s["m1_source"]), float(s["chi1"]))
        A2 = _kerr_area(float(s["m2_source"]), float(s["chi2"]))
        a_vals.append(A1 + A2)
    sorted_vals = sorted(a_vals)
    return {
        "n_samples": len(sorted_vals),
        "p10": _percentile_sorted(sorted_vals, 10.0),
        "p50": _percentile_sorted(sorted_vals, 50.0),
        "p90": _percentile_sorted(sorted_vals, 90.0),
    }


def _check_event_coverage(
    hawking_rows: list[dict[str, Any]],
    initial_area_map: dict[str, float],
) -> None:
    """Abort if any event_id in hawking_rows lacks an entry in initial_area_map."""
    missing = {r["event_id"] for r in hawking_rows} - set(initial_area_map)
    if missing:
        raise ValueError(
            f"[phase4b] Missing A_initial for event_id(s): {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    run_id: str,
    tolerance: float = 0.0,
    out_name: str = DEFAULT_OUT_NAME,
) -> dict[str, Any]:
    """Run Phase4B under runs/<run_id>/experiment/<out_name>/."""
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)

    run_dir = out_root / run_id
    phase4_dir = run_dir / "experiment" / PHASE4_UPSTREAM_NAME
    posteriors_dir = run_dir / "external_inputs" / "gwtc_posteriors"

    # --- Gating: Phase4 upstream ---
    _require_phase4_upstream(phase4_dir)

    # --- Load Phase4 hawking area rows ---
    hawking_csv_path = phase4_dir / "outputs" / "per_event_hawking_area.csv"
    with hawking_csv_path.open(encoding="utf-8", newline="") as fh:
        hawking_rows = list(csv.DictReader(fh))

    # --- Gating: GWTC posteriors ---
    required_event_ids = {r["event_id"] for r in hawking_rows}
    posteriors_by_event = _load_gwtc_posteriors(posteriors_dir, required_event_ids)

    # --- Derive A_initial per event from posterior samples ---
    event_area_stats: dict[str, dict[str, Any]] = {}
    for eid, samples in posteriors_by_event.items():
        event_area_stats[eid] = _derive_initial_area_stats(eid, samples)

    # Default estimator: sample median (p50)
    initial_area_map: dict[str, float] = {
        eid: stats["p50"] for eid, stats in event_area_stats.items()
    }

    # --- Explicit event coverage check (belt-and-suspenders) ---
    _check_event_coverage(hawking_rows, initial_area_map)

    # --- Apply relational filter ---
    filter_rows: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    events_with_nonempty: set[str] = set()
    all_gap: list[float] = []

    for row in hawking_rows:
        eid = row["event_id"]
        # Phase4 uses column "A"; accept "A_final" for forward-compat
        A_final_raw = row.get("A") or row.get("A_final", "")
        S_final_raw = row.get("S") or row.get("S_final", "")
        try:
            A_final = float(A_final_raw)
        except (ValueError, TypeError):
            A_final = float("nan")
        try:
            S_final = float(S_final_raw)
        except (ValueError, TypeError):
            S_final = float("nan")

        A_initial = initial_area_map[eid]
        area_gap = A_final - A_initial
        # Default tolerance = 0.0 → strict (no tolerance)
        h_pass = math.isfinite(A_final) and (A_final >= A_initial - tolerance)

        if h_pass:
            n_pass += 1
            events_with_nonempty.add(eid)
        else:
            n_fail += 1

        if math.isfinite(area_gap):
            all_gap.append(area_gap)

        filter_rows.append(
            {
                "event_id": eid,
                "family": row["family"],
                "provenance": row["provenance"],
                "M_solar": row["M_solar"],
                "chi": row["chi"],
                "A_final": A_final,
                "S_final": S_final,
                "A_initial": A_initial,
                "area_gap": area_gap,
                "hawking_pass": h_pass,
            }
        )

    # --- Per-event aggregation ---
    event_stats: dict[str, dict[str, Any]] = {}
    for fr in filter_rows:
        eid = fr["event_id"]
        if eid not in event_stats:
            event_stats[eid] = {
                "n_input_rows": 0,
                "n_pass_rows": 0,
                "A_initial": initial_area_map[eid],
                "n_samples": event_area_stats[eid]["n_samples"],
            }
        event_stats[eid]["n_input_rows"] += 1
        if fr["hawking_pass"]:
            event_stats[eid]["n_pass_rows"] += 1

    support_rows = []
    n_events_total = len(event_stats)
    n_events_empty = 0
    for eid in sorted(event_stats):
        stats = event_stats[eid]
        tot = stats["n_input_rows"]
        n_p = stats["n_pass_rows"]
        n_f = tot - n_p
        pf = n_p / tot if tot > 0 else 0.0
        empty = n_p == 0
        if empty:
            n_events_empty += 1
        support_rows.append(
            {
                "event_id": eid,
                "n_input_rows": tot,
                "n_pass_rows": n_p,
                "n_fail_rows": n_f,
                "pass_fraction": pf,
                "empty_after_filter": empty,
                "A_initial": stats["A_initial"],
                "n_samples": stats["n_samples"],
            }
        )

    n_input_total = len(filter_rows)
    pass_fraction = n_pass / n_input_total if n_input_total > 0 else 0.0
    n_events_with_nonempty = len(events_with_nonempty)
    all_gap_sorted = sorted(g for g in all_gap if math.isfinite(g))

    # --- Build output payloads ---
    hawking_filter_summary: dict[str, Any] = {
        "schema_version": "hawking_filter_summary_v1",
        "host_run": run_id,
        "source_phase4_experiment": PHASE4_UPSTREAM_NAME,
        "initial_area_source": "external_inputs/gwtc_posteriors",
        "initial_area_estimator": "sample_median",
        "filter_rule": "A_final >= A_initial",
        "tolerance": tolerance,
        "units": REQUIRED_UNITS,
        "n_rows_input_common": n_input_total,
        "n_rows_hawking_pass": n_pass,
        "n_rows_hawking_fail": n_fail,
        "pass_fraction": pass_fraction,
        "n_events_total": n_events_total,
        "n_events_with_nonempty_hawking": n_events_with_nonempty,
        "n_events_empty_after_filter": n_events_empty,
        "area_gap_quantiles": {
            "p10": _percentile_sorted(all_gap_sorted, 10.0),
            "p50": _percentile_sorted(all_gap_sorted, 50.0),
            "p90": _percentile_sorted(all_gap_sorted, 90.0),
        },
    }

    hawking_filter_support_summary: dict[str, Any] = {
        "schema_version": "hawking_filter_support_summary_v1",
        "rows": support_rows,
    }

    stage_summary: dict[str, Any] = {
        "experiment_name": out_name,
        "verdict": "PASS",
        "host_run": run_id,
        "created_utc": utc_now_iso(),
        "discriminative_filter": True,
        "filter_role": "hawking_area_law_discriminative",
        "gating": {
            "host_run_valid": True,
            "phase4_upstream_present": True,
            "phase4_upstream_pass": True,
            "gwtc_posteriors_present": True,
            "gwtc_posteriors_schema_valid": True,
            "gwtc_posteriors_event_coverage_complete": True,
        },
        "inputs": {
            "phase4_stage_summary": str(phase4_dir / "stage_summary.json"),
            "phase4_manifest": str(phase4_dir / "manifest.json"),
            "phase4_per_event_hawking_area": str(hawking_csv_path),
            "gwtc_posteriors_dir": str(posteriors_dir),
        },
        "initial_area_definition": {
            "source": "gwtc_posteriors",
            "estimator": "sample_median",
            "component_formula": "8*pi*M^2*(1+sqrt(1-chi^2))",
            "units": REQUIRED_UNITS,
        },
        "filter_definition": {
            "rule": "A_final >= A_initial",
            "tolerance": tolerance,
            "stable_identifier": "phys_key",
        },
        "metrics": {
            "n_rows_input_common": n_input_total,
            "n_rows_hawking_pass": n_pass,
            "n_rows_hawking_fail": n_fail,
            "n_events_total": n_events_total,
            "n_events_with_nonempty_hawking": n_events_with_nonempty,
            "n_events_empty_after_filter": n_events_empty,
            "pass_fraction": pass_fraction,
        },
        "notes": [
            "This phase is the discriminative Hawking area-law filter.",
            "Phase4 upstream remains domain-admissibility only.",
            "Historical analysis_area_theorem runs are not used as numeric upstream.",
        ],
    }

    # --- Atomic write (tempdir + shutil.move) ---
    exp_dir = run_dir / "experiment" / out_name

    # Safety guard: exp_dir must be strictly under run_dir
    try:
        exp_dir.relative_to(run_dir)
    except ValueError:
        raise RuntimeError(
            f"[phase4b] Write-path safety violation: {exp_dir} outside {run_dir}"
        )

    with tempfile.TemporaryDirectory(prefix=f"{out_name}_", dir=run_dir) as tmpdir:
        tmp_stage = Path(tmpdir) / "stage"
        tmp_outputs = tmp_stage / "outputs"
        tmp_outputs.mkdir(parents=True, exist_ok=True)

        # 1. per_event_hawking_filter.csv
        out_filter_csv = tmp_outputs / "per_event_hawking_filter.csv"
        filter_fields = [
            "event_id", "family", "provenance", "M_solar", "chi",
            "A_final", "S_final", "A_initial", "area_gap", "hawking_pass",
        ]
        with out_filter_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=filter_fields)
            writer.writeheader()
            writer.writerows(filter_rows)

        # 2. hawking_filter_summary.json
        out_filter_summary = tmp_outputs / "hawking_filter_summary.json"
        write_json_atomic(out_filter_summary, hawking_filter_summary)

        # 3. hawking_filter_support_summary.json
        out_support_summary = tmp_outputs / "hawking_filter_support_summary.json"
        write_json_atomic(out_support_summary, hawking_filter_support_summary)

        # 4. per_event_initial_area_from_posteriors.csv (derived artifact, not primary input)
        out_initial_area_csv = tmp_outputs / "per_event_initial_area_from_posteriors.csv"
        initial_area_fields = [
            "event_id", "n_samples",
            "A_initial_p10", "A_initial_p50", "A_initial_p90",
            "source_ref", "method", "units",
        ]
        initial_area_derived_rows = []
        for eid in sorted(event_area_stats):
            stats = event_area_stats[eid]
            initial_area_derived_rows.append({
                "event_id": eid,
                "n_samples": stats["n_samples"],
                "A_initial_p10": stats["p10"],
                "A_initial_p50": stats["p50"],
                "A_initial_p90": stats["p90"],
                "source_ref": str(posteriors_dir / f"{eid}.json"),
                "method": "kerr_component_area_sum_from_imr_samples",
                "units": REQUIRED_UNITS,
            })
        with out_initial_area_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=initial_area_fields)
            writer.writeheader()
            writer.writerows(initial_area_derived_rows)

        outputs = [
            out_filter_csv, out_filter_summary, out_support_summary, out_initial_area_csv,
        ]
        output_records = [
            {"path": str(p.relative_to(tmp_stage)), "sha256": sha256_file(p)}
            for p in outputs
        ]

        # 5. manifest.json
        manifest_payload: dict[str, Any] = {
            "schema_version": "mvp_manifest_v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "artifacts": output_records,
            "inputs": [
                {
                    "path": str(hawking_csv_path),
                    "sha256": sha256_file(hawking_csv_path),
                },
            ] + [
                {
                    "path": str(posteriors_dir / f"{eid}.json"),
                    "sha256": sha256_file(posteriors_dir / f"{eid}.json"),
                }
                for eid in sorted(required_event_ids)
            ],
        }
        write_json_atomic(tmp_stage / "manifest.json", manifest_payload)

        # 6. stage_summary.json
        stage_summary["outputs"] = output_records
        write_json_atomic(tmp_stage / "stage_summary.json", stage_summary)

        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_stage), str(exp_dir))

    contracts.log_stage_paths(
        SimpleNamespace(
            out_root=out_root,
            stage_dir=exp_dir,
            outputs_dir=exp_dir / "outputs",
        )
    )

    return {
        "hawking_filter_summary": hawking_filter_summary,
        "hawking_filter_support_summary": hawking_filter_support_summary,
        "stage_summary": stage_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-4B: Hawking area-law discriminative filter on common physical support."
        )
    )
    ap.add_argument(
        "--host-run", required=True, dest="host_run", help="Host run ID"
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help=(
            "Tolerance for A_final >= A_initial (default: 0.0, no tolerance). "
            f"Units: {REQUIRED_UNITS}"
        ),
    )
    ap.add_argument(
        "--out-name",
        default=DEFAULT_OUT_NAME,
        help=f"Output experiment name (default: {DEFAULT_OUT_NAME})",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_experiment(
        run_id=args.host_run,
        tolerance=args.tolerance,
        out_name=args.out_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
