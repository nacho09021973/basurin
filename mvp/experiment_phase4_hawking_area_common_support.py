#!/usr/bin/env python3
"""Non-canonical experiment: phase4 Hawking area on common physical support.

Builds on the closed phase-3 result: computes Hawking area and entropy
for each geometry in K_common(event) = K220(event) ∩ K221(event) by phys_key.

Semantic contract:
    phys_key = (family, provenance, M_solar, chi)
    A = 8*pi*M^2*(1 + sqrt(1 - chi^2))
    S = A/4
    hawking_pass = isfinite(A) and A > 0 and isfinite(S) and S > 0 and abs(chi) <= 1

The Hawking filter is applied ONLY on K_common(event), never before.
geometry_id is NOT used as physical identifier; phys_key is the stable ID.
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

# Reuse all phase-3 helpers unchanged: phys_key extraction, batch gating,
# results.csv loading, serialisation.  These are NOT duplicated here.
from mvp.experiment_phase3_physkey_common import (
    DEFAULT_BATCH_RESULTS,
    DEFAULT_S4_COMPATIBLE,
    ROUND_CHI,
    ROUND_M,
    PhysKey,
    _extract_phys_keys,
    _keys_to_sorted_list,
    _load_event_to_subrun_filtered,
    _phys_key_to_str,
    _require_batch_pass,
)

SCHEMA_VERSION = "experiment_phase4_hawking_area_common_support_v1"
DEFAULT_OUT_NAME = "phase4_hawking_area_common_support"

# ---------------------------------------------------------------------------
# Phase-4-specific helpers
# ---------------------------------------------------------------------------


def _hawking_area_and_entropy(M_solar: float, chi: float) -> tuple[float, float]:
    """A = 8*pi*M^2*(1 + sqrt(1 - chi^2)), S = A/4.

    Returns (nan, nan) when chi^2 > 1 (unphysical).
    """
    spin_sq = 1.0 - chi ** 2
    if spin_sq < 0.0:
        return float("nan"), float("nan")
    A = 8.0 * math.pi * M_solar ** 2 * (1.0 + math.sqrt(spin_sq))
    S = A / 4.0
    return A, S


def _hawking_pass_pred(A: float, S: float, chi: float) -> bool:
    """Conservative Hawking filter (minimum predicado operacional)."""
    return (
        math.isfinite(A)
        and A > 0.0
        and math.isfinite(S)
        and S > 0.0
        and abs(chi) <= 1.0
    )


def _percentile_sorted(sorted_vals: list[float], p: float) -> float | None:
    """Linear-interpolation percentile on pre-sorted list. p in [0, 100]."""
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    run_id: str,
    batch_220: str,
    batch_221: str,
    out_name: str = DEFAULT_OUT_NAME,
    batch_results_relpath: str = DEFAULT_BATCH_RESULTS,
    s4_compatible_relpath: str = DEFAULT_S4_COMPATIBLE,
) -> dict[str, Any]:
    """Run the phase-4 Hawking-area experiment under runs/<run_id>/experiment/<out_name>/."""
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)
    _require_batch_pass(out_root, batch_220, batch_results_relpath)
    _require_batch_pass(out_root, batch_221, batch_results_relpath)

    run_dir = out_root / run_id
    results_220_path = out_root / batch_220 / batch_results_relpath
    results_221_path = out_root / batch_221 / batch_results_relpath

    # Load filtered event→subrun mappings (status=PASS + compatible_set.json present)
    stats220 = _load_event_to_subrun_filtered(results_220_path, out_root, s4_compatible_relpath)
    stats221 = _load_event_to_subrun_filtered(results_221_path, out_root, s4_compatible_relpath)
    map220 = stats220.mapping
    map221 = stats221.mapping

    valid_events_220 = set(map220.keys())
    valid_events_221 = set(map221.keys())
    common_events = sorted(valid_events_220 & valid_events_221)

    # Accumulators
    K220_global: set[PhysKey] = set()
    K221_global: set[PhysKey] = set()
    per_event_support_rows: list[dict[str, Any]] = []
    per_event_hawking_rows: list[dict[str, Any]] = []
    empty_intersection_events: list[str] = []
    non_subset_cases: list[str] = []
    n_rows_hawking_pass = 0
    n_rows_hawking_fail = 0
    events_with_nonempty_hawking: set[str] = set()
    all_A_pass: list[float] = []
    all_S_pass: list[float] = []

    for event in common_events:
        sub220 = map220[event]
        sub221 = map221[event]
        path220 = out_root / sub220 / s4_compatible_relpath
        path221 = out_root / sub221 / s4_compatible_relpath

        # Raises ValueError on schema/field errors → propagates (abort, no partial output)
        k220 = _extract_phys_keys(path220)
        k221 = _extract_phys_keys(path221)

        K220_global |= k220
        K221_global |= k221

        # K_common per event: intersection by phys_key (NOT geometry_id)
        k_common = k220 & k221

        if not k_common:
            empty_intersection_events.append(event)
        if not k220.issubset(k221):
            non_subset_cases.append(event)

        # Apply Hawking filter ONLY on k_common – never on k220 or k221 alone
        n_k_hawking = 0
        for key in sorted(k_common, key=_phys_key_to_str):
            family, provenance, M_solar, chi = key
            A, S = _hawking_area_and_entropy(M_solar, chi)
            h_pass = _hawking_pass_pred(A, S, chi)
            if h_pass:
                n_k_hawking += 1
                n_rows_hawking_pass += 1
                events_with_nonempty_hawking.add(event)
                all_A_pass.append(A)
                all_S_pass.append(S)
            else:
                n_rows_hawking_fail += 1

            per_event_hawking_rows.append(
                {
                    "event_id": event,
                    "family": family,
                    "provenance": provenance,
                    "M_solar": M_solar,
                    "chi": chi,
                    "A": A,
                    "S": S,
                    "hawking_pass": h_pass,
                }
            )

        per_event_support_rows.append(
            {
                "event_id": event,
                "run_id_220": sub220,
                "run_id_221": sub221,
                "n_k220": len(k220),
                "n_k221": len(k221),
                "n_k_common": len(k_common),
                "n_k_hawking": n_k_hawking,
            }
        )

    K220_inter_K221 = K220_global & K221_global

    # -----------------------------------------------------------------------
    # Build output payloads
    # -----------------------------------------------------------------------

    common_support_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "host_run": run_id,
        "batch_220": batch_220,
        "batch_221": batch_221,
        "phys_key_definition": {
            "fields": ["family", "provenance", "M_solar", "chi"],
            "round_decimals": ROUND_M,
        },
        "n_events_valid_220": len(valid_events_220),
        "n_events_valid_221": len(valid_events_221),
        "n_common_events": len(common_events),
        "K220": len(K220_global),
        "K221": len(K221_global),
        "K220_inter_K221": len(K220_inter_K221),
        "K220_keys": _keys_to_sorted_list(K220_global),
        "K221_keys": _keys_to_sorted_list(K221_global),
        "K220_inter_K221_keys": _keys_to_sorted_list(K220_inter_K221),
        "n_empty_intersection_events": len(empty_intersection_events),
        "empty_intersection_events": sorted(empty_intersection_events),
        "n_non_subset_cases": len(non_subset_cases),
        "non_subset_cases": sorted(non_subset_cases),
    }

    all_A_sorted = sorted(all_A_pass)
    all_S_sorted = sorted(all_S_pass)

    hawking_area_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "formula": {
            "A": "8 * pi * M^2 * (1 + sqrt(1 - chi^2))",
            "S": "A / 4",
        },
        "n_rows_input_common": len(per_event_hawking_rows),
        "n_rows_hawking_pass": n_rows_hawking_pass,
        "n_rows_hawking_fail": n_rows_hawking_fail,
        "n_events_with_nonempty_hawking": len(events_with_nonempty_hawking),
        "A_quantiles": {
            "p10": _percentile_sorted(all_A_sorted, 10.0),
            "p50": _percentile_sorted(all_A_sorted, 50.0),
            "p90": _percentile_sorted(all_A_sorted, 90.0),
        },
        "S_quantiles": {
            "p10": _percentile_sorted(all_S_sorted, 10.0),
            "p50": _percentile_sorted(all_S_sorted, 50.0),
            "p90": _percentile_sorted(all_S_sorted, 90.0),
        },
    }

    stage_summary: dict[str, Any] = {
        "experiment_name": out_name,
        "verdict": "PASS",
        "host_run": run_id,
        "batch_220": batch_220,
        "batch_221": batch_221,
        "created_utc": utc_now_iso(),
        "discriminative_filter": False,
        "filter_role": "domain_admissibility_only",
        "gating": {
            "host_run_valid": True,
            "batch_220_pass": True,
            "batch_221_pass": True,
            "results_csv_present_220": results_220_path.exists(),
            "results_csv_present_221": results_221_path.exists(),
        },
        "support_definition": {
            "kind": "phys_key_intersection",
            "phys_key_fields": ["family", "provenance", "M_solar", "chi"],
            "round_decimals": ROUND_M,
            "source_of_truth": "compatible_set.json",
        },
        "hawking_area_filter": {
            "applied_on": "K220_inter_K221",
            "scope": "per_event",
            "stable_identifier": "phys_key",
        },
        "metrics": {
            "n_common_events": len(common_events),
            "K220": len(K220_global),
            "K221": len(K221_global),
            "K220_inter_K221": len(K220_inter_K221),
            "n_empty_intersection_events": len(empty_intersection_events),
            "n_rows_hawking_pass": n_rows_hawking_pass,
        },
    }

    # -----------------------------------------------------------------------
    # Atomic write
    # -----------------------------------------------------------------------
    exp_dir = run_dir / "experiment" / out_name
    with tempfile.TemporaryDirectory(prefix=f"{out_name}_", dir=run_dir) as tmpdir:
        tmp_stage = Path(tmpdir) / "stage"
        tmp_outputs = tmp_stage / "outputs"
        tmp_outputs.mkdir(parents=True, exist_ok=True)

        # 1. per_event_common_support.csv
        out_support_csv = tmp_outputs / "per_event_common_support.csv"
        support_fields = [
            "event_id", "run_id_220", "run_id_221",
            "n_k220", "n_k221", "n_k_common", "n_k_hawking",
        ]
        with out_support_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=support_fields)
            writer.writeheader()
            writer.writerows(per_event_support_rows)

        # 2. per_event_hawking_area.csv
        out_hawking_csv = tmp_outputs / "per_event_hawking_area.csv"
        hawking_fields = [
            "event_id", "family", "provenance", "M_solar", "chi", "A", "S", "hawking_pass",
        ]
        with out_hawking_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=hawking_fields)
            writer.writeheader()
            writer.writerows(per_event_hawking_rows)

        # 3. common_support_summary.json
        out_css = tmp_outputs / "common_support_summary.json"
        write_json_atomic(out_css, common_support_summary)

        # 4. hawking_area_summary.json
        out_has = tmp_outputs / "hawking_area_summary.json"
        write_json_atomic(out_has, hawking_area_summary)

        outputs = [out_support_csv, out_hawking_csv, out_css, out_has]
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
                {"path": str(results_220_path), "sha256": sha256_file(results_220_path)},
                {"path": str(results_221_path), "sha256": sha256_file(results_221_path)},
            ],
        }
        write_json_atomic(tmp_stage / "manifest.json", manifest_payload)

        # 6. stage_summary.json
        stage_summary["inputs"] = manifest_payload["inputs"]
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
        "common_support_summary": common_support_summary,
        "hawking_area_summary": hawking_area_summary,
        "stage_summary": stage_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-4: Hawking area on common physical support "
            "(K220 ∩ K221 by phys_key, per event)."
        )
    )
    ap.add_argument("--host-run", required=True, dest="host_run", help="Host run ID")
    ap.add_argument("--batch-220", required=True, dest="batch_220", help="Batch run ID for mode-220")
    ap.add_argument("--batch-221", required=True, dest="batch_221", help="Batch run ID for mode-221")
    ap.add_argument(
        "--out-name",
        default=DEFAULT_OUT_NAME,
        help=f"Output experiment name (default: {DEFAULT_OUT_NAME})",
    )
    ap.add_argument(
        "--batch-results-relpath",
        default=DEFAULT_BATCH_RESULTS,
        help=f"Relative path to results.csv inside each batch run (default: {DEFAULT_BATCH_RESULTS})",
    )
    ap.add_argument(
        "--s4-compatible-relpath",
        default=DEFAULT_S4_COMPATIBLE,
        help=(
            f"Relative path to compatible_set.json inside each sub-run "
            f"(default: {DEFAULT_S4_COMPATIBLE})"
        ),
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_experiment(
        run_id=args.host_run,
        batch_220=args.batch_220,
        batch_221=args.batch_221,
        out_name=args.out_name,
        batch_results_relpath=args.batch_results_relpath,
        s4_compatible_relpath=args.s4_compatible_relpath,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
