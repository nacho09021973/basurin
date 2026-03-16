#!/usr/bin/env python3
"""Temperature stability selection: Phase 6.

Experiment role: inferential / policy-selection.
NOT a physical temperature claim.  'Temperature' here is the softmax sharpness
parameter for the score-based delta_lnL policy defined in Phase 3 / Phase 5.

Reads Phase 5 temperature-sweep outputs, detects a stability plateau in the
sweep, selects the canonical temperature T* (smallest T in the plateau), and
freezes a weight policy + consensus support at T*.

Phase 5 does NOT publish per-temperature weights (no temperature_sweep_weights_v1.json).
Weights at T* are therefore recomputed from s5_aggregate event records using the
same formula as Phase 3 / Phase 5.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

# Re-use Phase 5's proven helpers for event loading and weight computation.
# Importing private helpers is intentional: Phase 5 defines the canonical
# softmax-policy computation logic; duplicating it would violate the
# minimum-change / no-duplication constraint.
from mvp.experiment_phase5_temperature_sweep import (
    _compute_temperature_rows,
    _load_aggregate_event_runs,
    _load_event_records,
    _load_json_object,
    _normalize_input_records_to_out_root,
    _renyi_metrics_from_weights,
    _load_support_basis,
)

STAGE = "experiment/phase6_temperature_stability_selection"

SCHEMA_VERSION_TEMP_SEL = "temperature_selection_v1"
SCHEMA_VERSION_CONSENSUS = "support_consensus_v1"
SCHEMA_VERSION_WEIGHT_POLICY = "weight_policy_selected_temperature_v1"
SCHEMA_VERSION_RENYI_SELECTED = "renyi_diversity_selected_temperature_v1"
SCHEMA_VERSION_STABILITY_DIAG = "stability_diagnostics_v1"

# Default stability thresholds — document them explicitly here.
DEFAULT_TAU_TOPK = 0.90        # Jaccard(topK(Ti), topK(Tj)) >= tau_topk
DEFAULT_EPS_FAMILY = 0.05      # L1(family_mass(Ti), family_mass(Tj)) <= eps_family
DEFAULT_EPS_THEORY = 0.05      # L1(theory_mass(Ti), theory_mass(Tj)) <= eps_theory
DEFAULT_EPS_RENYI = 0.05       # |H_1(Ti) - H_1(Tj)| <= eps_renyi
DEFAULT_CONSENSUS_TOPK = 10    # k for consensus support construction

FALLBACK_PRESENT_FRACTION = 0.8  # secondary consensus rule

# Renyi proxy: H_1 (Shannon entropy = Renyi alpha=1) from temperature_sweep_metrics_v1.json.
# Phase 5 exposes H_1 per temperature but not a standalone H_alpha series file.
RENYI_PROXY_FIELD = "H_1"
RENYI_PROXY_NOTE = (
    "H_1 (Shannon entropy, Renyi alpha=1) from temperature_sweep_metrics_v1.json; "
    "Phase 5 does not publish a separate per-temperature H_alpha file."
)

# ── Input paths (relative to run_dir) ─────────────────────────────────────

_P5 = Path("experiment") / "phase5_temperature_sweep" / "outputs"
P5_METRICS_REL = _P5 / "temperature_sweep_metrics_v1.json"
P5_TOPK_REL = _P5 / "temperature_sweep_topk_v1.json"
P5_FAMILY_REL = _P5 / "temperature_sweep_family_mass_v1.json"
P5_THEORY_REL = _P5 / "temperature_sweep_theory_mass_v1.json"
P5_TOP_GEOM_REL = _P5 / "temperature_sweep_top_geometry_v1.json"

SUPPORT_INPUT_REL = (
    Path("experiment") / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
)
AGGREGATE_INPUT_REL = Path("s5_aggregate") / "outputs" / "aggregate.json"
WEIGHT_POLICY_INPUT_REL = (
    Path("experiment")
    / "phase3_weight_policy_basis"
    / "outputs"
    / "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
)
RENYI_INPUT_REL = (
    Path("experiment")
    / "phase4_renyi_diversity_baseline"
    / "outputs"
    / "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select canonical temperature T* from Phase 5 stability sweep"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tau-topk", type=float, default=DEFAULT_TAU_TOPK,
                    help=f"min Jaccard for top-k stability (default {DEFAULT_TAU_TOPK})")
    ap.add_argument("--eps-family", type=float, default=DEFAULT_EPS_FAMILY,
                    help=f"max L1 for family-mass stability (default {DEFAULT_EPS_FAMILY})")
    ap.add_argument("--eps-theory", type=float, default=DEFAULT_EPS_THEORY,
                    help=f"max L1 for theory-mass stability (default {DEFAULT_EPS_THEORY})")
    ap.add_argument("--eps-renyi", type=float, default=DEFAULT_EPS_RENYI,
                    help=f"max |H_1 delta| for Renyi stability (default {DEFAULT_EPS_RENYI})")
    ap.add_argument("--consensus-topk", type=int, default=DEFAULT_CONSENSUS_TOPK,
                    help=f"k for consensus support (default {DEFAULT_CONSENSUS_TOPK})")
    return ap.parse_args(argv)


# ── Stability helpers ──────────────────────────────────────────────────────

def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 1.0


def _l1_dist(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    keys = set(dist_a) | set(dist_b)
    return sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)


def _find_stability_plateau(
    temps: list[float],
    topk_by_temp: dict[float, set[str]],
    family_mass_by_temp: dict[float, dict[str, float]],
    theory_mass_by_temp: dict[float, dict[str, float]],
    h1_by_temp: dict[float, float],
    *,
    tau_topk: float,
    eps_family: float,
    eps_theory: float,
    eps_renyi: float,
) -> tuple[
    list[float],           # best_plateau (sorted ascending)
    dict[str, float],      # pairwise_topk_jaccard
    dict[str, float],      # pairwise_family_l1
    dict[str, float],      # pairwise_theory_l1
    dict[str, float],      # pairwise_renyi_delta
    list[float],           # plateau_candidates (for diagnostics)
]:
    """Find the largest subset of temperatures that are mutually stable.

    All-pairs stability check: for every pair (Ti, Tj) in the subset, all four
    conditions must hold simultaneously.  Selection rule: largest clique first;
    ties broken by smallest minimum temperature.

    For |temps| <= ~15 (2^15 = 32768 subsets) this is trivially fast.
    The Phase 5 default grid has 7 temperatures.
    """
    n = len(temps)
    pairwise_topk: dict[str, float] = {}
    pairwise_family: dict[str, float] = {}
    pairwise_theory: dict[str, float] = {}
    pairwise_renyi: dict[str, float] = {}
    stable_pair: dict[tuple[int, int], bool] = {}

    for i, j in itertools.combinations(range(n), 2):
        Ti, Tj = temps[i], temps[j]
        key = f"{Ti}_{Tj}"
        j_val = _jaccard(topk_by_temp.get(Ti, set()), topk_by_temp.get(Tj, set()))
        f_val = _l1_dist(family_mass_by_temp.get(Ti, {}), family_mass_by_temp.get(Tj, {}))
        t_val = _l1_dist(theory_mass_by_temp.get(Ti, {}), theory_mass_by_temp.get(Tj, {}))
        r_val = abs(h1_by_temp.get(Ti, 0.0) - h1_by_temp.get(Tj, 0.0))
        pairwise_topk[key] = j_val
        pairwise_family[key] = f_val
        pairwise_theory[key] = t_val
        pairwise_renyi[key] = r_val
        stable_pair[(i, j)] = (
            j_val >= tau_topk
            and f_val <= eps_family
            and t_val <= eps_theory
            and r_val <= eps_renyi
        )

    def _all_stable(indices: tuple[int, ...]) -> bool:
        return all(
            stable_pair.get((a, b) if a < b else (b, a), False)
            for a, b in itertools.combinations(indices, 2)
        )

    best_plateau: list[float] = []
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            if _all_stable(combo):
                candidate = sorted(temps[idx] for idx in combo)
                if not best_plateau or len(candidate) > len(best_plateau):
                    best_plateau = candidate
                break  # first valid combo of this size wins (deterministic)
        if len(best_plateau) == size:
            break

    # Temperatures appearing in at least one stable pair (diagnostics)
    plateau_candidates = sorted(
        {temps[idx] for (i, j), ok in stable_pair.items() if ok for idx in (i, j)}
    )

    return (
        best_plateau,
        pairwise_topk,
        pairwise_family,
        pairwise_theory,
        pairwise_renyi,
        plateau_candidates,
    )


# ── Consensus support ──────────────────────────────────────────────────────

def _build_consensus_support(
    *,
    plateau: list[float],
    top_geom_rows: list[dict[str, Any]],
    support_basis_rows: list[dict[str, Any]],
    k: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Build consensus support rows from top_geometry data.

    Hard rule:   present in top-k for ALL plateau temperatures.
    Fallback:    present_fraction >= FALLBACK_PRESENT_FRACTION (0.8).

    Returns (rows, fallback_used).
    """
    plateau_set = set(plateau)
    n_plateau = len(plateau)

    meta_by_raw: dict[str, dict[str, str]] = {
        row["raw_geometry_id"]: {
            "normalized_geometry_id": row["normalized_geometry_id"],
            "atlas_family": row["atlas_family"],
            "atlas_theory": row["atlas_theory"],
        }
        for row in support_basis_rows
    }

    present_count: dict[str, int] = {}
    for row in top_geom_rows:
        T = float(row["temperature"])
        if T not in plateau_set:
            continue
        if int(row["rank"]) <= k:
            gid = str(row["raw_geometry_id"])
            present_count[gid] = present_count.get(gid, 0) + 1

    hard_set = {gid for gid, cnt in present_count.items() if cnt == n_plateau}
    fallback_used = False
    if hard_set:
        selected = hard_set
    else:
        selected = {
            gid
            for gid, cnt in present_count.items()
            if n_plateau > 0 and cnt / n_plateau >= FALLBACK_PRESENT_FRACTION
        }
        fallback_used = True

    rows: list[dict[str, Any]] = []
    for gid in sorted(selected, key=lambda g: (-present_count.get(g, 0), g)):
        meta = meta_by_raw.get(gid, {})
        rows.append(
            {
                "raw_geometry_id": gid,
                "normalized_geometry_id": meta.get("normalized_geometry_id", gid),
                "atlas_family": meta.get("atlas_family", ""),
                "atlas_theory": meta.get("atlas_theory", ""),
                "present_count": present_count.get(gid, 0),
                "present_fraction": (
                    present_count.get(gid, 0) / n_plateau if n_plateau > 0 else 0.0
                ),
            }
        )
    return rows, fallback_used


# ── Entrypoint ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    params: dict[str, Any] = {
        "tau_topk": args.tau_topk,
        "eps_family": args.eps_family,
        "eps_theory": args.eps_theory,
        "eps_renyi": args.eps_renyi,
        "consensus_topk": args.consensus_topk,
        "renyi_proxy_field": RENYI_PROXY_FIELD,
        "fallback_present_fraction": FALLBACK_PRESENT_FRACTION,
    }

    ctx = init_stage(args.run_id, STAGE, params=params)
    run_dir = ctx.run_dir

    # ── 1. Declare static inputs ───────────────────────────────────────────
    static_inputs: dict[str, Path] = {
        "temperature_sweep_metrics_v1": run_dir / P5_METRICS_REL,
        "temperature_sweep_topk_v1": run_dir / P5_TOPK_REL,
        "temperature_sweep_family_mass_v1": run_dir / P5_FAMILY_REL,
        "temperature_sweep_theory_mass_v1": run_dir / P5_THEORY_REL,
        "temperature_sweep_top_geometry_v1": run_dir / P5_TOP_GEOM_REL,
        "support_ontology_basis_v1": run_dir / SUPPORT_INPUT_REL,
        "aggregate": run_dir / AGGREGATE_INPUT_REL,
        "weight_policy_phase3": run_dir / WEIGHT_POLICY_INPUT_REL,
        "renyi_baseline_phase4": run_dir / RENYI_INPUT_REL,
    }

    # Pre-load aggregate to discover event source runs for check_inputs
    aggregate_event_runs: list[dict[str, str]] = []
    if (run_dir / AGGREGATE_INPUT_REL).exists():
        aggregate_event_runs = _load_aggregate_event_runs(
            run_dir / AGGREGATE_INPUT_REL, ctx=ctx
        )

    basis = None
    dynamic_inputs: dict[str, Path] = {}
    if (run_dir / SUPPORT_INPUT_REL).exists():
        basis = _load_support_basis(run_dir / SUPPORT_INPUT_REL, ctx=ctx)
        _, _, _, dynamic_inputs = _load_event_records(
            out_root=ctx.out_root,
            aggregate_event_runs=aggregate_event_runs,
            basis_by_geometry_id=basis["basis_by_geometry_id"],
            ctx=ctx,
        )

    all_inputs = {**static_inputs, **dynamic_inputs}
    check_inputs(ctx, all_inputs)
    _normalize_input_records_to_out_root(ctx, dynamic_inputs)

    # ── 2. Load Phase 5 outputs ────────────────────────────────────────────
    metrics_data = _load_json_object(
        run_dir / P5_METRICS_REL, label="temperature_sweep_metrics_v1", ctx=ctx
    )
    family_data = _load_json_object(
        run_dir / P5_FAMILY_REL, label="temperature_sweep_family_mass_v1", ctx=ctx
    )
    theory_data = _load_json_object(
        run_dir / P5_THEORY_REL, label="temperature_sweep_theory_mass_v1", ctx=ctx
    )
    top_geom_data = _load_json_object(
        run_dir / P5_TOP_GEOM_REL, label="temperature_sweep_top_geometry_v1", ctx=ctx
    )

    top_n = int(top_geom_data.get("top_n", 0))
    if args.consensus_topk > top_n:
        abort(
            ctx,
            f"consensus_topk={args.consensus_topk} exceeds top_n={top_n} published by "
            f"Phase 5; cannot build consensus support. Use --consensus-topk <= {top_n}.",
        )

    temps: list[float] = sorted(float(t) for t in metrics_data.get("temperature_grid", []))
    if not temps:
        abort(ctx, "temperature_sweep_metrics_v1.json has empty temperature_grid")

    # ── 3. Build per-temperature structures ───────────────────────────────
    # 3a. Top-k sets (normalized_geometry_id, rank <= consensus_topk)
    topk_by_temp: dict[float, set[str]] = {t: set() for t in temps}
    for row in top_geom_data.get("rows", []):
        T = float(row["temperature"])
        if T in topk_by_temp and int(row["rank"]) <= args.consensus_topk:
            topk_by_temp[T].add(str(row["normalized_geometry_id"]))

    # 3b. Family mass distributions
    family_mass_by_temp: dict[float, dict[str, float]] = {t: {} for t in temps}
    for row in family_data.get("rows", []):
        T = float(row["temperature"])
        if T in family_mass_by_temp:
            family_mass_by_temp[T][str(row["atlas_family"])] = float(row["family_mass"])

    # 3c. Theory mass distributions
    theory_mass_by_temp: dict[float, dict[str, float]] = {t: {} for t in temps}
    for row in theory_data.get("rows", []):
        T = float(row["temperature"])
        if T in theory_mass_by_temp:
            theory_mass_by_temp[T][str(row["atlas_theory"])] = float(row["theory_mass"])

    # 3d. H_1 per temperature (Renyi proxy, documented explicitly)
    h1_by_temp: dict[float, float] = {}
    for row in metrics_data.get("rows", []):
        h1_by_temp[float(row["temperature"])] = float(row[RENYI_PROXY_FIELD])

    # ── 4. Find stability plateau ─────────────────────────────────────────
    (
        plateau,
        pairwise_topk,
        pairwise_family,
        pairwise_theory,
        pairwise_renyi,
        plateau_candidates,
    ) = _find_stability_plateau(
        temps,
        topk_by_temp,
        family_mass_by_temp,
        theory_mass_by_temp,
        h1_by_temp,
        tau_topk=args.tau_topk,
        eps_family=args.eps_family,
        eps_theory=args.eps_theory,
        eps_renyi=args.eps_renyi,
    )

    if not plateau:
        abort(
            ctx,
            "No stable temperature plateau found. All pairwise stability checks failed. "
            "Check thresholds (--tau-topk, --eps-family, --eps-theory, --eps-renyi) or Phase 5 outputs.",
        )

    T_star: float = min(plateau)
    baseline_temp: float | None = (
        1.0 if any(math.isclose(t, 1.0, rel_tol=0.0, abs_tol=1e-12) for t in temps) else None
    )

    # Worst-case decision metrics within the plateau
    plateau_pairs = list(itertools.combinations(plateau, 2))
    if plateau_pairs:
        def _pk(a: float, b: float) -> str:
            lo, hi = (a, b) if a < b else (b, a)
            return f"{lo}_{hi}"

        topk_jaccard_min = min(pairwise_topk[_pk(a, b)] for a, b in plateau_pairs)
        family_l1_max = max(pairwise_family[_pk(a, b)] for a, b in plateau_pairs)
        theory_l1_max = max(pairwise_theory[_pk(a, b)] for a, b in plateau_pairs)
        renyi_delta_max = max(pairwise_renyi[_pk(a, b)] for a, b in plateau_pairs)
    else:
        # Single-element plateau: trivially perfect stability
        topk_jaccard_min = 1.0
        family_l1_max = 0.0
        theory_l1_max = 0.0
        renyi_delta_max = 0.0

    # ── 5. Recompute weight policy at T* ──────────────────────────────────
    # Phase 5 does not publish temperature_sweep_weights_v1.json.
    # We recompute using the same formula / helpers from Phase 5.
    assert basis is not None, "basis must be loaded by now"

    event_records, _, _, _ = _load_event_records(
        out_root=ctx.out_root,
        aggregate_event_runs=aggregate_event_runs,
        basis_by_geometry_id=basis["basis_by_geometry_id"],
        ctx=ctx,
    )

    weight_rows, weight_sum_raw, weight_sum_normalized = _compute_temperature_rows(
        temperature=T_star,
        basis_rows=basis["ordered_rows"],
        event_records=event_records,
        ctx=ctx,
    )

    if not math.isclose(weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=1e-9):
        abort(ctx, f"Weight normalization failed at T*={T_star}: sum={weight_sum_normalized}")

    # ── 6. Consensus support ──────────────────────────────────────────────
    consensus_rows, fallback_used = _build_consensus_support(
        plateau=plateau,
        top_geom_rows=top_geom_data.get("rows", []),
        support_basis_rows=basis["ordered_rows"],
        k=args.consensus_topk,
    )

    # ── 7. Renyi metrics at T* ────────────────────────────────────────────
    renyi_at_T_star = _renyi_metrics_from_weights(
        [float(row["weight_normalized"]) for row in weight_rows],
        n_rows=basis["n_rows"],
    )

    # ── 8. Write outputs ──────────────────────────────────────────────────
    out = ctx.outputs_dir

    # A) temperature_selection_v1.json
    temp_selection: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_TEMP_SEL,
        "selected_temperature": T_star,
        "selection_rule": "smallest_temperature_in_stability_plateau",
        "temperature_grid": temps,
        "baseline_temperature": baseline_temp,
        "stability_plateau": plateau,
        "decision_metrics": {
            "topk_jaccard_min": topk_jaccard_min,
            "family_l1_shift_max": family_l1_max,
            "theory_l1_shift_max": theory_l1_max,
            "renyi_delta_max": renyi_delta_max,
        },
        "thresholds": {
            "tau_topk": args.tau_topk,
            "eps_family": args.eps_family,
            "eps_theory": args.eps_theory,
            "eps_renyi": args.eps_renyi,
        },
    }
    out_temp_sel = out / "temperature_selection_v1.json"
    write_json_atomic(out_temp_sel, temp_selection)

    # B) support_consensus_v1.json
    consensus_rule = (
        "present_in_topk_across_all_temperatures_in_plateau"
        if not fallback_used
        else "present_fraction_gte_0.8_across_temperatures_in_plateau"
    )
    support_consensus: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_CONSENSUS,
        "selected_temperature": T_star,
        "consensus_rule": consensus_rule,
        "k": args.consensus_topk,
        "rows": consensus_rows,
    }
    out_consensus = out / "support_consensus_v1.json"
    write_json_atomic(out_consensus, support_consensus)

    # C) weight_policy_selected_temperature_v1.json
    weight_policy_out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_WEIGHT_POLICY,
        "policy_name": "event_support_delta_lnL_softmax_selected_temperature_v1",
        "source_phase5_policy": "event_support_delta_lnL_softmax_mean_temperature_v1",
        "selected_temperature": T_star,
        "basis_name": basis["basis_name"],
        "n_rows": len(weight_rows),
        "weight_sum_normalized": weight_sum_normalized,
        "rows": weight_rows,
    }
    out_weight = out / "weight_policy_selected_temperature_v1.json"
    write_json_atomic(out_weight, weight_policy_out)

    # D) renyi_diversity_selected_temperature_v1.json
    renyi_selected_out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_RENYI_SELECTED,
        "selected_temperature": T_star,
        "metrics": renyi_at_T_star,
    }
    out_renyi = out / "renyi_diversity_selected_temperature_v1.json"
    write_json_atomic(out_renyi, renyi_selected_out)

    # E) stability_diagnostics_v1.json
    stability_diag_out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_STABILITY_DIAG,
        "pairwise_topk_jaccard": pairwise_topk,
        "pairwise_family_l1": pairwise_family,
        "pairwise_theory_l1": pairwise_theory,
        "pairwise_renyi_delta": pairwise_renyi,
        "plateau_candidates": plateau_candidates,
    }
    out_diag = out / "stability_diagnostics_v1.json"
    write_json_atomic(out_diag, stability_diag_out)

    # ── 9. Finalize ───────────────────────────────────────────────────────
    artifacts: dict[str, Path] = {
        "temperature_selection_v1": out_temp_sel,
        "support_consensus_v1": out_consensus,
        "weight_policy_selected_temperature_v1": out_weight,
        "renyi_diversity_selected_temperature_v1": out_renyi,
        "stability_diagnostics_v1": out_diag,
    }

    finalize(
        ctx,
        artifacts=artifacts,
        results={
            "selected_temperature": T_star,
            "plateau_size": len(plateau),
            "n_consensus_rows": len(consensus_rows),
            "weight_sum_normalized": weight_sum_normalized,
            "consensus_fallback_used": fallback_used,
        },
        extra_summary={
            "experiment_name": "phase6_temperature_stability_selection",
            "experiment_role": "inferential/policy-selection",
            "experiment_note": (
                "NOT a physical temperature claim. 'Temperature' is the softmax "
                "sharpness parameter for the score-based delta_lnL policy."
            ),
            "selected_temperature": T_star,
            "selection_rule": "smallest_temperature_in_stability_plateau",
            "stability_thresholds": {
                "tau_topk": args.tau_topk,
                "eps_family": args.eps_family,
                "eps_theory": args.eps_theory,
                "eps_renyi": args.eps_renyi,
            },
            "consensus_rule": consensus_rule,
            "consensus_fallback_used": fallback_used,
            "plateau_size": len(plateau),
            "n_consensus_rows": len(consensus_rows),
            "weight_sum_normalized": weight_sum_normalized,
            "renyi_proxy_field": RENYI_PROXY_FIELD,
            "renyi_proxy_note": RENYI_PROXY_NOTE,
            "phase5_weights_published": False,
            "weight_recomputation_note": (
                "Phase 5 does not publish temperature_sweep_weights_v1.json. "
                "Weights recomputed at T* from s5_aggregate event records using "
                "the same _compute_temperature_rows helper as Phase 5."
            ),
            "notes": [
                "Phase 6 is inferential/policy-selection. No physical temperature interpretation.",
                f"Renyi proxy used: {RENYI_PROXY_NOTE}",
                (
                    "Weights at T* recomputed from s5_aggregate + event source runs "
                    "(Phase 5 has no weights output)."
                ),
                f"Consensus fallback (present_fraction >= {FALLBACK_PRESENT_FRACTION}): {fallback_used}",
            ],
        },
    )

    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
