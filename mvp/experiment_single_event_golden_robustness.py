"""experiment_single_event_golden_robustness — Quantify whether a "golden geometry"
singleton is stable under reasonable perturbations of:
    - threshold_220
    - threshold_221
    - sigma_scale (scales sigma_f and sigma_tau before chi2 evaluation)
    - area_tolerance

and optionally aggregate peer runs of the same event.

This experiment is NOT a canonical stage; it uses no StageContract.
Writes only under:
    runs/<run_id>/experiment/golden_robustness_<timestamp>/

Canonical stage artifacts (s4g/s4h/s4i/s4j/...) are NEVER mutated.

Usage
-----
    python -m mvp.experiment_single_event_golden_robustness \\
        --run-id <id> \\
        [--peer-run-id <id> ...] \\
        [--atlas-path PATH] \\
        [--threshold-220-values 3.5,4.605,6.0] \\
        [--threshold-221-values 3.5,4.605,6.0] \\
        [--sigma-scale-values 0.8,1.0,1.2] \\
        [--area-tolerance-values 0.0,1e-12,1e-9]
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_json_atomic,
)
from mvp.golden_geometry_spec import (
    DEFAULT_AREA_TOLERANCE,
    DEFAULT_MODE_CHI2_THRESHOLD_90,
    ROBUST_UNIQUE_MIN_SUPPORT_FRACTION,
    VERDICT_SKIPPED_221_UNAVAILABLE,
    _utc_now_iso,
    exact_intersection_geometry_ids,
    robust_unique_verdict,
    singleton_geometry_id,
)
from mvp.s4g_mode220_geometry_filter import filter_mode220, load_atlas_entries
from mvp.s4h_mode221_geometry_filter import filter_mode221
from mvp.s4i_common_geometry_intersection import compute_intersection
from mvp.s4j_hawking_area_filter import filter_area_law

EXPERIMENT_NAME = "golden_robustness"

# Canonical stage output paths (relative to run_dir)
S4G_OUTPUT_REL = "s4g_mode220_geometry_filter/outputs/mode220_filter.json"
S4H_OUTPUT_REL = "s4h_mode221_geometry_filter/outputs/mode221_filter.json"
S4J_OUTPUT_REL = "s4j_hawking_area_filter/outputs/hawking_area_filter.json"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_csv_floats(arg: str) -> list[float]:
    """Parse a comma-separated string of float values, e.g. '3.5,4.605,6.0'."""
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    result: list[float] = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            raise ValueError(f"Cannot parse float value: {p!r} in CSV argument {arg!r}")
    if not result:
        raise ValueError(f"No valid float values in: {arg!r}")
    return result


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp suitable for directory names."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Run eligibility
# ---------------------------------------------------------------------------


def _check_run_valid(out_root: Path, run_id: str) -> tuple[bool, str | None]:
    """Return (has_pass, reason_or_None). Does NOT raise."""
    verdict_path = out_root / run_id / "RUN_VALID" / "verdict.json"
    if not verdict_path.exists():
        return False, f"RUN_VALID/verdict.json not found: {verdict_path}"
    try:
        data = json.loads(verdict_path.read_text(encoding="utf-8"))
        if data.get("verdict") == "PASS":
            return True, None
        return False, f"RUN_VALID verdict is {data.get('verdict')!r}, not PASS"
    except Exception as exc:
        return False, f"Error reading RUN_VALID: {exc}"


def _inventory_source_runs(
    *,
    base_run_id: str,
    peer_run_ids: list[str],
    out_root: Path,
) -> list[dict[str, Any]]:
    """Determine eligibility of each source run.

    Returns a list of dicts with keys:
        run_id, eligible, has_run_valid_pass, exclusion_reason
    """
    all_run_ids = [base_run_id] + [r for r in peer_run_ids if r != base_run_id]
    records: list[dict[str, Any]] = []
    for run_id in all_run_ids:
        has_pass, reason = _check_run_valid(out_root, run_id)
        eligible = has_pass
        records.append(
            {
                "run_id": run_id,
                "eligible": eligible,
                "has_run_valid_pass": has_pass,
                "exclusion_reason": reason,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_source_inputs_for_geometry_filters(
    run_id: str,
    out_root: Path,
) -> dict[str, Any] | None:
    """Load canonical inputs for geometry filters from an eligible run.

    Returns a dict with:
        mode220: {obs_f_hz, obs_tau_s, sigma_f_hz, sigma_tau_s} or None
        mode221: {obs_f_hz, obs_tau_s, sigma_f_hz, sigma_tau_s} or None (if unavailable)
        area_data: {geometry_id: {area_final, area_initial}} (may be empty)
        mode221_available: bool

    Returns None if the essential mode-220 inputs cannot be loaded.
    """
    run_dir = out_root / run_id

    # --- Mode 220 (essential) ---
    s4g_path = run_dir / S4G_OUTPUT_REL
    if not s4g_path.exists():
        return None
    try:
        s4g_data = json.loads(s4g_path.read_text(encoding="utf-8"))
        obs_f_220 = s4g_data.get("obs_f_hz")
        obs_tau_220 = s4g_data.get("obs_tau_s")
        sigma_f_220 = s4g_data.get("sigma_f_hz")
        sigma_tau_220 = s4g_data.get("sigma_tau_s")
        if any(v is None for v in (obs_f_220, obs_tau_220, sigma_f_220, sigma_tau_220)):
            return None
        mode220: dict[str, float] = {
            "obs_f_hz": float(obs_f_220),
            "obs_tau_s": float(obs_tau_220),
            "sigma_f_hz": float(sigma_f_220),
            "sigma_tau_s": float(sigma_tau_220),
        }
    except Exception:
        return None

    # --- Mode 221 (optional) ---
    s4h_path = run_dir / S4H_OUTPUT_REL
    mode221: dict[str, float] | None = None
    mode221_available = False
    if s4h_path.exists():
        try:
            s4h_data = json.loads(s4h_path.read_text(encoding="utf-8"))
            if s4h_data.get("verdict") != VERDICT_SKIPPED_221_UNAVAILABLE:
                obs_f_221 = s4h_data.get("obs_f_hz")
                obs_tau_221 = s4h_data.get("obs_tau_s")
                sigma_f_221 = s4h_data.get("sigma_f_hz")
                sigma_tau_221 = s4h_data.get("sigma_tau_s")
                if not any(v is None for v in (obs_f_221, obs_tau_221, sigma_f_221, sigma_tau_221)):
                    mode221 = {
                        "obs_f_hz": float(obs_f_221),
                        "obs_tau_s": float(obs_tau_221),
                        "sigma_f_hz": float(sigma_f_221),
                        "sigma_tau_s": float(sigma_tau_221),
                    }
                    mode221_available = True
        except Exception:
            pass

    # --- Area data (from s4j output; optional) ---
    area_data: dict[str, dict[str, float]] = {}
    s4j_path = run_dir / S4J_OUTPUT_REL
    if s4j_path.exists():
        try:
            s4j_data = json.loads(s4j_path.read_text(encoding="utf-8"))
            raw_area = s4j_data.get("area_data", {})
            if isinstance(raw_area, dict):
                area_data = raw_area
        except Exception:
            pass

    return {
        "mode220": mode220,
        "mode221": mode221,
        "mode221_available": mode221_available,
        "area_data": area_data,
    }


# ---------------------------------------------------------------------------
# Scenario evaluation
# ---------------------------------------------------------------------------


def _make_scenario_id(
    source_run_id: str,
    threshold_220: float,
    threshold_221: float,
    sigma_scale: float,
    area_tolerance: float,
) -> str:
    """Build a compact deterministic scenario identifier."""

    def _fmt(v: float) -> str:
        # Use repr to avoid trailing zeros issues
        s = f"{v:.6g}"
        return s

    return (
        f"{source_run_id}"
        f"__t220-{_fmt(threshold_220)}"
        f"__t221-{_fmt(threshold_221)}"
        f"__ss-{_fmt(sigma_scale)}"
        f"__at-{_fmt(area_tolerance)}"
    )


def _evaluate_single_scenario(
    *,
    source_run_id: str,
    source_inputs: dict[str, Any],
    atlas_entries: list[dict[str, Any]],
    threshold_220: float,
    threshold_221: float,
    sigma_scale: float,
    area_tolerance: float,
) -> dict[str, Any]:
    """Evaluate one scenario in memory; returns the scenario result dict.

    Parameters
    ----------
    source_run_id  : run being evaluated.
    source_inputs  : dict from _load_source_inputs_for_geometry_filters.
    atlas_entries  : list of atlas entry dicts.
    threshold_220  : chi² threshold for mode 220.
    threshold_221  : chi² threshold for mode 221.
    sigma_scale    : multiplier applied to sigma_f and sigma_tau before chi2.
    area_tolerance : tolerance passed to area law filter.
    """
    scenario_id = _make_scenario_id(
        source_run_id, threshold_220, threshold_221, sigma_scale, area_tolerance
    )

    mode220 = source_inputs["mode220"]
    mode221: dict[str, float] | None = source_inputs.get("mode221")
    mode221_available: bool = source_inputs.get("mode221_available", False)
    area_data: dict[str, dict[str, float]] = source_inputs.get("area_data", {})

    # Apply sigma_scale
    sf_220 = mode220["sigma_f_hz"] * sigma_scale
    st_220 = mode220["sigma_tau_s"] * sigma_scale

    try:
        mode220_result = filter_mode220(
            obs_f_hz=mode220["obs_f_hz"],
            obs_tau_s=mode220["obs_tau_s"],
            sigma_f_hz=sf_220,
            sigma_tau_s=st_220,
            atlas_entries=atlas_entries,
            chi2_threshold=threshold_220,
        )
        # Compatibility across filter_mode220 return shapes used in different branches:
        # either (ids, diagnostics) tuple or plain ids list.
        if isinstance(mode220_result, tuple):
            ids_220 = mode220_result[0]
        else:
            ids_220 = mode220_result

        ids_221: list[str] | None = None
        if mode221_available and mode221 is not None:
            sf_221 = mode221["sigma_f_hz"] * sigma_scale
            st_221 = mode221["sigma_tau_s"] * sigma_scale
            ids_221 = filter_mode221(
                obs_f_hz=mode221["obs_f_hz"],
                obs_tau_s=mode221["obs_tau_s"],
                sigma_f_hz=sf_221,
                sigma_tau_s=st_221,
                atlas_entries=atlas_entries,
                chi2_threshold=threshold_221,
            )

        common_ids = compute_intersection(ids_220, ids_221)

        golden_ids = filter_area_law(
            common_geometry_ids=common_ids,
            area_data=area_data,
            area_tolerance=area_tolerance,
        )

        sg_id = singleton_geometry_id(golden_ids)

        return {
            "scenario_id": scenario_id,
            "source_run_id": source_run_id,
            "threshold_220": threshold_220,
            "threshold_221": threshold_221,
            "sigma_scale": sigma_scale,
            "area_tolerance": area_tolerance,
            "status": "evaluated",
            "status_reason": None,
            "n_geometries_220": len(ids_220),
            "n_geometries_221": len(ids_221) if ids_221 is not None else None,
            "n_common_geometries": len(common_ids),
            "n_golden_geometries": len(golden_ids),
            "golden_geometry_ids": golden_ids,
            "singleton_geometry_id": sg_id,
        }
    except Exception as exc:
        return {
            "scenario_id": scenario_id,
            "source_run_id": source_run_id,
            "threshold_220": threshold_220,
            "threshold_221": threshold_221,
            "sigma_scale": sigma_scale,
            "area_tolerance": area_tolerance,
            "status": "failed",
            "status_reason": f"{type(exc).__name__}: {exc}",
            "n_geometries_220": None,
            "n_geometries_221": None,
            "n_common_geometries": None,
            "n_golden_geometries": None,
            "golden_geometry_ids": [],
            "singleton_geometry_id": None,
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_robustness(
    *,
    base_run_id: str,
    peer_run_ids: list[str],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build robustness_summary from scenario results.

    Uses helpers from golden_geometry_spec:
    - exact_intersection_geometry_ids
    - robust_unique_verdict
    """
    evaluated = [s for s in scenarios if s["status"] == "evaluated"]
    nonempty = [s for s in evaluated if s.get("n_golden_geometries", 0) > 0]
    singleton_scenarios = [s for s in evaluated if s.get("n_golden_geometries") == 1]

    n_evaluated = len(evaluated)
    n_nonempty = len(nonempty)
    n_singleton = len(singleton_scenarios)

    # Exact intersection over non-empty scenarios
    if nonempty:
        exact_intersection = exact_intersection_geometry_ids(
            [s["golden_geometry_ids"] for s in nonempty]
        )
    else:
        exact_intersection = []

    # Singleton geometry support
    singleton_support: dict[str, int] = {}
    for s in singleton_scenarios:
        sg = s.get("singleton_geometry_id")
        if sg is not None:
            singleton_support[sg] = singleton_support.get(sg, 0) + 1

    # Robustness verdict
    singleton_ids_list = [s.get("singleton_geometry_id") for s in evaluated]
    verdict_result = robust_unique_verdict(
        singleton_ids=singleton_ids_list,
        n_valid_scenarios=n_evaluated,
    )

    notes: list[str] = []
    n_failed = sum(1 for s in scenarios if s["status"] == "failed")
    n_skipped = sum(1 for s in scenarios if s["status"] == "skipped")
    if n_failed:
        notes.append(f"{n_failed} scenario(s) failed during evaluation")
    if n_skipped:
        notes.append(f"{n_skipped} scenario(s) skipped")
    if not nonempty:
        notes.append("No evaluated scenarios produced a non-empty golden set")

    return {
        "schema_name": "single_event_golden_robustness",
        "schema_version": "v1",
        "created_utc": _utc_now_iso(),
        "base_run_id": base_run_id,
        "peer_run_ids": peer_run_ids,
        "n_source_runs": 1 + len(peer_run_ids),
        "n_scenarios_total": len(scenarios),
        "n_scenarios_evaluated": n_evaluated,
        "n_scenarios_with_nonempty_golden_set": n_nonempty,
        "n_scenarios_singleton": n_singleton,
        "exact_intersection_over_nonempty_scenarios": exact_intersection,
        "singleton_geometry_support": singleton_support,
        "robust_unique_geometry_id": verdict_result["robust_unique_geometry_id"],
        "support_fraction": verdict_result["support_fraction"],
        "robustness_verdict": verdict_result["robustness_verdict"],
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Quantify golden-geometry singleton robustness for a single event"
    )
    ap.add_argument("--run-id", required=True, help="Base run ID (mandatory)")
    ap.add_argument(
        "--peer-run-id",
        dest="peer_run_ids",
        action="append",
        default=[],
        metavar="ID",
        help="Additional run IDs from the same event (may be repeated)",
    )
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument(
        "--threshold-220-values",
        default=f"{DEFAULT_MODE_CHI2_THRESHOLD_90}",
        help="Comma-separated chi² thresholds for mode 220",
    )
    ap.add_argument(
        "--threshold-221-values",
        default=f"{DEFAULT_MODE_CHI2_THRESHOLD_90}",
        help="Comma-separated chi² thresholds for mode 221",
    )
    ap.add_argument(
        "--sigma-scale-values",
        default="1.0",
        help="Comma-separated sigma scale factors",
    )
    ap.add_argument(
        "--area-tolerance-values",
        default=f"{DEFAULT_AREA_TOLERANCE}",
        help="Comma-separated area tolerances",
    )
    args = ap.parse_args(argv)

    out_root = resolve_out_root("runs")

    # Fail-fast: base run must exist and have RUN_VALID PASS
    validate_run_id(args.run_id, out_root)
    try:
        require_run_valid(out_root, args.run_id)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: base run {args.run_id!r} is not valid: {exc}", file=sys.stderr)
        return 2

    # Parse grid parameters
    try:
        thresholds_220 = _parse_csv_floats(args.threshold_220_values)
        thresholds_221 = _parse_csv_floats(args.threshold_221_values)
        sigma_scales = _parse_csv_floats(args.sigma_scale_values)
        area_tolerances = _parse_csv_floats(args.area_tolerance_values)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Set up experiment output directory
    timestamp = _utc_timestamp()
    exp_dir = out_root / args.run_id / "experiment" / f"golden_robustness_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas
    atlas_entries: list[dict[str, Any]] = []
    atlas_path_str: str | None = args.atlas_path
    if args.atlas_path:
        atlas_path = Path(args.atlas_path)
        if not atlas_path.is_absolute():
            atlas_path = (Path.cwd() / atlas_path).resolve()
        if not atlas_path.exists():
            print(f"ERROR: atlas not found: {atlas_path}", file=sys.stderr)
            return 2
        try:
            atlas_entries = load_atlas_entries(atlas_path)
        except Exception as exc:
            print(f"ERROR loading atlas: {exc}", file=sys.stderr)
            return 2
        atlas_path_str = str(atlas_path)

    created_utc = _utc_now_iso()

    # A. Inventory
    peer_run_ids: list[str] = args.peer_run_ids
    source_run_records = _inventory_source_runs(
        base_run_id=args.run_id,
        peer_run_ids=peer_run_ids,
        out_root=out_root,
    )

    inventory: dict[str, Any] = {
        "schema_name": "golden_robustness_inventory",
        "schema_version": "v1",
        "created_utc": created_utc,
        "base_run_id": args.run_id,
        "peer_run_ids": peer_run_ids,
        "source_runs": source_run_records,
    }
    inventory_path = exp_dir / "inventory.json"
    write_json_atomic(inventory_path, inventory)

    # Load inputs for each eligible run
    eligible_runs: list[tuple[str, dict[str, Any]]] = []
    for rec in source_run_records:
        if not rec["eligible"]:
            continue
        inputs = _load_source_inputs_for_geometry_filters(rec["run_id"], out_root)
        if inputs is None:
            rec["eligible"] = False
            rec["exclusion_reason"] = (
                rec.get("exclusion_reason") or
                f"Cannot load s4g mode-220 filter outputs for run {rec['run_id']!r}"
            )
            continue
        eligible_runs.append((rec["run_id"], inputs))

    # B. Scenario results
    scenarios: list[dict[str, Any]] = []
    param_grid = list(
        itertools.product(
            [r[0] for r in eligible_runs],
            thresholds_220,
            thresholds_221,
            sigma_scales,
            area_tolerances,
        )
    )

    for (run_id, t220, t221, ss, at) in param_grid:
        inputs = next(inp for rid, inp in eligible_runs if rid == run_id)
        result = _evaluate_single_scenario(
            source_run_id=run_id,
            source_inputs=inputs,
            atlas_entries=atlas_entries,
            threshold_220=t220,
            threshold_221=t221,
            sigma_scale=ss,
            area_tolerance=at,
        )
        scenarios.append(result)

    # Mark ineligible runs as skipped scenarios (one entry per ineligible run)
    for rec in source_run_records:
        if not rec["eligible"]:
            for t220, t221, ss, at in itertools.product(
                thresholds_220, thresholds_221, sigma_scales, area_tolerances
            ):
                scenario_id = _make_scenario_id(rec["run_id"], t220, t221, ss, at)
                scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "source_run_id": rec["run_id"],
                        "threshold_220": t220,
                        "threshold_221": t221,
                        "sigma_scale": ss,
                        "area_tolerance": at,
                        "status": "skipped",
                        "status_reason": rec.get("exclusion_reason") or "run not eligible",
                        "n_geometries_220": None,
                        "n_geometries_221": None,
                        "n_common_geometries": None,
                        "n_golden_geometries": None,
                        "golden_geometry_ids": [],
                        "singleton_geometry_id": None,
                    }
                )

    scenario_results: dict[str, Any] = {
        "schema_name": "golden_robustness_scenarios",
        "schema_version": "v1",
        "created_utc": created_utc,
        "base_run_id": args.run_id,
        "atlas_path": atlas_path_str,
        "scenarios": scenarios,
    }
    scenario_results_path = exp_dir / "scenario_results.json"
    write_json_atomic(scenario_results_path, scenario_results)

    # C. Robustness summary
    summary = _aggregate_robustness(
        base_run_id=args.run_id,
        peer_run_ids=peer_run_ids,
        scenarios=scenarios,
    )
    summary_path = exp_dir / "robustness_summary.json"
    write_json_atomic(summary_path, summary)

    # D. Manifest
    output_files = {
        "inventory": inventory_path,
        "scenario_results": scenario_results_path,
        "robustness_summary": summary_path,
    }
    hashes: dict[str, str] = {}
    for label, path in output_files.items():
        if path.is_file():
            hashes[label] = sha256_file(path)

    manifest: dict[str, Any] = {
        "schema_version": "mvp_manifest_v1",
        "created": created_utc,
        "experiment": EXPERIMENT_NAME,
        "base_run_id": args.run_id,
        "experiment_dir": str(exp_dir),
        "artifacts": {label: path.name for label, path in output_files.items()},
        "hashes": hashes,
    }
    manifest_path = exp_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    # Logging (required by AGENTS.md)
    print(f"OUT_ROOT={out_root}")
    print(f"STAGE_DIR={exp_dir}")
    print(f"OUTPUTS_DIR={exp_dir}")
    print(f"STAGE_SUMMARY={summary_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"[{EXPERIMENT_NAME}] n_scenarios={len(scenarios)}", flush=True)
    print(f"[{EXPERIMENT_NAME}] verdict={summary['robustness_verdict']}", flush=True)
    print(f"[{EXPERIMENT_NAME}] robust_unique_geometry_id={summary['robust_unique_geometry_id']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
