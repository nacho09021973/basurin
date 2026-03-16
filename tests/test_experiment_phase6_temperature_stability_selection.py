"""Tests for experiment/phase6_temperature_stability_selection.

Covers the 8 required regression cases:
  1. test_phase6_requires_phase5_outputs
  2. test_phase6_selects_smallest_temperature_in_stable_plateau
  3. test_phase6_support_consensus_is_deterministic
  4. test_phase6_weight_policy_selected_temperature_normalizes_to_one
  5. test_phase6_writes_only_under_runs_run_id_experiment_phase6
  6. test_phase6_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest
  7. test_phase6_fallback_consensus_rule_if_hard_intersection_empty
  8. test_phase6_recomputes_policy_if_phase5_has_no_weights_output
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase6_temperature_stability_selection.py")
STAGE = "experiment/phase6_temperature_stability_selection"

# ── Fixture constants (mirrors Phase 5 test fixtures) ─────────────────────

TEMPERATURE_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
TOP_N = 20
K_GRID = [1, 3, 5, 10]
ALPHA_GRID: list[Any] = [0, 1, 2, "inf"]
SUPPORT_SCHEMA_VERSION = "support_ontology_basis_v1"
WEIGHT_POLICY_SCHEMA_VERSION = "weight_policy_basis_v1"
RENYI_SCHEMA_VERSION = "renyi_diversity_baseline_v1"
EXISTING_POLICY_NAME = "event_support_delta_lnL_softmax_mean_v1"
SWEEP_POLICY_NAME = "event_support_delta_lnL_softmax_mean_temperature_v1"

BASIS_ROWS = [
    {
        "raw_geometry_id": "geom_alpha",
        "normalized_geometry_id": "geom_alpha_norm",
        "atlas_family": "edgb",
        "atlas_theory": "EdGB",
        "n_events_supported": 2,
        "support_fraction_events": 1.0,
    },
    {
        "raw_geometry_id": "geom_beta",
        "normalized_geometry_id": "geom_beta_norm",
        "atlas_family": "kerr_newman",
        "atlas_theory": "Kerr-Newman",
        "n_events_supported": 2,
        "support_fraction_events": 1.0,
    },
    {
        "raw_geometry_id": "geom_gamma",
        "normalized_geometry_id": "geom_gamma_norm",
        "atlas_family": "dcs",
        "atlas_theory": "dCS",
        "n_events_supported": 2,
        "support_fraction_events": 1.0,
    },
]

EVENT_SPECS = [
    {
        "event_id": "GW_ALPHA",
        "source_run_id": "run_evt_alpha",
        "final_geometry_ids": ["geom_alpha", "geom_beta", "geom_gamma"],
        "ranked_rows": [
            {"geometry_id": "geom_alpha", "delta_lnL": 0.0},
            {"geometry_id": "geom_beta", "delta_lnL": -1.0},
            {"geometry_id": "geom_gamma", "delta_lnL": -2.0},
        ],
    },
    {
        "event_id": "GW_BETA",
        "source_run_id": "run_evt_beta",
        "final_geometry_ids": ["geom_alpha", "geom_beta", "geom_gamma"],
        "ranked_rows": [
            {"geometry_id": "geom_alpha", "delta_lnL": 0.0},
            {"geometry_id": "geom_beta", "delta_lnL": -0.5},
            {"geometry_id": "geom_gamma", "delta_lnL": -1.5},
        ],
    },
]


# ── Low-level helpers ──────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_valid(runs_root: Path, run_id: str, verdict: str = "PASS") -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


# ── Fixture computation helpers ────────────────────────────────────────────

def _temperature_weights(temperature: float) -> dict[str, Any]:
    """Compute softmax weights at the given temperature over BASIS_ROWS / EVENT_SPECS."""
    raw: dict[str, float] = {row["raw_geometry_id"]: 0.0 for row in BASIS_ROWS}
    for spec in EVENT_SPECS:
        support = spec["final_geometry_ids"]
        deltas = {
            r["geometry_id"]: float(r["delta_lnL"])
            for r in spec["ranked_rows"]
            if r["geometry_id"] in support
        }
        max_d = max(deltas.values())
        scores = {g: math.exp((deltas[g] - max_d) / temperature) for g in support}
        s = math.fsum(scores.values())
        for g in support:
            raw[g] += scores[g] / s
    wsum = math.fsum(raw.values())
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in BASIS_ROWS:
        gid = row["raw_geometry_id"]
        rows_by_id[gid] = {
            "raw_geometry_id": gid,
            "normalized_geometry_id": row["normalized_geometry_id"],
            "atlas_family": row["atlas_family"],
            "atlas_theory": row["atlas_theory"],
            "weight_raw": raw[gid],
            "weight_normalized": raw[gid] / wsum,
        }
    return {"weight_sum_raw": wsum, "weight_sum_normalized": 1.0, "rows_by_id": rows_by_id}


def _renyi_metrics(weights: list[float]) -> dict[str, Any]:
    pos = [w for w in weights if w > 0.0]
    h = {
        "0": math.log(float(len(pos))),
        "1": -math.fsum(w * math.log(w) for w in pos),
        "2": -math.log(math.fsum(w * w for w in pos)),
        "inf": -math.log(max(pos)),
    }
    d = {k: math.exp(v) for k, v in h.items()}
    return {
        "H_alpha": h, "D_alpha": d,
        "p_max": max(pos),
        "n_weighted": len(pos),
        "n_unweighted": len(weights) - len(pos),
        "weight_sum_normalized": math.fsum(weights),
    }


def _sorted_rows(rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows_by_id.values(),
        key=lambda r: (-float(r["weight_normalized"]), r["raw_geometry_id"]),
    )


def _phase5_metrics_payload(temperature_grid: list[float] | None = None) -> dict:
    grid = temperature_grid if temperature_grid is not None else TEMPERATURE_GRID
    rows = []
    for T in grid:
        wp = _temperature_weights(T)
        m = _renyi_metrics([wp["rows_by_id"][r["raw_geometry_id"]]["weight_normalized"] for r in BASIS_ROWS])
        rows.append({
            "temperature": T,
            "policy_name": SWEEP_POLICY_NAME,
            "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
            "n_rows": len(BASIS_ROWS),
            "coverage_fraction": 1.0,
            "n_weighted": m["n_weighted"],
            "n_unweighted": m["n_unweighted"],
            "p_max": m["p_max"],
            "H_0": m["H_alpha"]["0"],
            "H_1": m["H_alpha"]["1"],
            "H_2": m["H_alpha"]["2"],
            "H_inf": m["H_alpha"]["inf"],
            "D_0": m["D_alpha"]["0"],
            "D_1": m["D_alpha"]["1"],
            "D_2": m["D_alpha"]["2"],
            "D_inf": m["D_alpha"]["inf"],
            "weight_sum_raw": wp["weight_sum_raw"],
            "weight_sum_normalized": 1.0,
            "criterion": "delta_lnL_softmax_per_event_temperature_sweep_over_final_support_region",
            "criterion_version": "v1",
        })
    return {
        "schema_version": "temperature_sweep_metrics_v1",
        "comparison_role": "temperature_sensitivity_of_score_based_policy",
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": grid,
        "alpha_grid": ALPHA_GRID,
        "rows": rows,
        "notes": [],
    }


def _phase5_topk_payload(temperature_grid: list[float] | None = None) -> dict:
    grid = temperature_grid if temperature_grid is not None else TEMPERATURE_GRID
    rows = []
    for T in grid:
        wp = _temperature_weights(T)
        sr = _sorted_rows(wp["rows_by_id"])
        total = math.fsum(float(r["weight_normalized"]) for r in sr)
        for k in K_GRID:
            rows.append({
                "temperature": T,
                "top_k": k,
                "cumulative_mass": math.fsum(float(r["weight_normalized"]) for r in sr[:k]),
                "normalization_check": total,
                "criterion": "cumulative_topk_mass_over_weight_normalized",
                "criterion_version": "v1",
            })
    return {
        "schema_version": "temperature_sweep_topk_v1",
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": grid,
        "k_grid": K_GRID,
        "rows": rows,
    }


def _phase5_family_payload(temperature_grid: list[float] | None = None) -> dict:
    grid = temperature_grid if temperature_grid is not None else TEMPERATURE_GRID
    rows = []
    for T in grid:
        wp = _temperature_weights(T)
        bucket: dict[str, float] = {}
        cnt: dict[str, int] = {}
        for r in wp["rows_by_id"].values():
            f = r["atlas_family"]
            bucket[f] = bucket.get(f, 0.0) + float(r["weight_normalized"])
            cnt[f] = cnt.get(f, 0) + 1
        for rank, (fam, mass) in enumerate(sorted(bucket.items(), key=lambda x: (-x[1], x[0])), 1):
            rows.append({
                "temperature": T,
                "atlas_family": fam,
                "family_mass": mass,
                "family_count": cnt[fam],
                "family_mass_rank": rank,
                "criterion": "mass_sum_by_atlas_family_over_weight_normalized",
                "criterion_version": "v1",
            })
    return {
        "schema_version": "temperature_sweep_family_mass_v1",
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": grid,
        "rows": rows,
    }


def _phase5_theory_payload(temperature_grid: list[float] | None = None) -> dict:
    grid = temperature_grid if temperature_grid is not None else TEMPERATURE_GRID
    rows = []
    for T in grid:
        wp = _temperature_weights(T)
        bucket: dict[str, float] = {}
        cnt: dict[str, int] = {}
        for r in wp["rows_by_id"].values():
            th = r["atlas_theory"]
            bucket[th] = bucket.get(th, 0.0) + float(r["weight_normalized"])
            cnt[th] = cnt.get(th, 0) + 1
        for rank, (theory, mass) in enumerate(sorted(bucket.items(), key=lambda x: (-x[1], x[0])), 1):
            rows.append({
                "temperature": T,
                "atlas_theory": theory,
                "theory_mass": mass,
                "theory_count": cnt[theory],
                "theory_mass_rank": rank,
                "criterion": "mass_sum_by_atlas_theory_over_weight_normalized",
                "criterion_version": "v1",
            })
    return {
        "schema_version": "temperature_sweep_theory_mass_v1",
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": grid,
        "rows": rows,
    }


def _phase5_top_geometry_payload(
    temperature_grid: list[float] | None = None,
    top_n: int = TOP_N,
    override_rows: list[dict] | None = None,
) -> dict:
    grid = temperature_grid if temperature_grid is not None else TEMPERATURE_GRID
    if override_rows is not None:
        rows = override_rows
    else:
        rows = []
        for T in grid:
            wp = _temperature_weights(T)
            sr = _sorted_rows(wp["rows_by_id"])
            cum = 0.0
            for rank, r in enumerate(sr[:top_n], 1):
                cum += float(r["weight_normalized"])
                rows.append({
                    "temperature": T,
                    "rank": rank,
                    "raw_geometry_id": r["raw_geometry_id"],
                    "normalized_geometry_id": r["normalized_geometry_id"],
                    "atlas_family": r["atlas_family"],
                    "atlas_theory": r["atlas_theory"],
                    "weight_normalized": float(r["weight_normalized"]),
                    "cumulative_mass": cum,
                    "criterion": "weight_normalized_descending_cumulative_mass",
                    "criterion_version": "v1",
                })
    return {
        "schema_version": "temperature_sweep_top_geometry_v1",
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": grid,
        "top_n": top_n,
        "rows": rows,
    }


def _support_payload() -> dict:
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "n_events": 2,
        "n_rows": len(BASIS_ROWS),
        "family_counts": {"dcs": 1, "edgb": 1, "kerr_newman": 1},
        "theory_counts": {"EdGB": 1, "Kerr-Newman": 1, "dCS": 1},
        "n_joint_available": 0,
        "joint_weight_sum_over_support": 0.0,
        "joint_weight_role": "copied_for_audit_only_not_a_renyi_policy",
        "rows": [
            {
                **row,
                "join_mode": "exact",
                "join_status": "resolved",
                "support_basis_name": "final_support_region_union_v1",
                "in_golden_post_hawking_union": True,
                "in_final_support_region_union": True,
                "joint_available": False,
                "joint_posterior_weight_joint": None,
                "joint_support_count": 0,
                "joint_support_fraction": 0.0,
                "joint_coverage": 0.0,
            }
            for row in BASIS_ROWS
        ],
    }


def _aggregate_payload() -> dict:
    return {
        "schema_version": "mvp_aggregate_v1",
        "events": [
            {"event_id": s["event_id"], "run_id": s["source_run_id"]}
            for s in EVENT_SPECS
        ],
    }


def _existing_weight_policy_payload() -> dict:
    wp = _temperature_weights(1.0)
    rows = []
    for row in BASIS_ROWS:
        d = wp["rows_by_id"][row["raw_geometry_id"]]
        rows.append({
            "raw_geometry_id": row["raw_geometry_id"],
            "normalized_geometry_id": row["normalized_geometry_id"],
            "atlas_family": row["atlas_family"],
            "atlas_theory": row["atlas_theory"],
            "policy_name": EXISTING_POLICY_NAME,
            "weight_raw": d["weight_raw"],
            "weight_normalized": d["weight_normalized"],
            "weight_status": "WEIGHTED",
            "support_count_events": row["n_events_supported"],
            "support_fraction_events": row["support_fraction_events"],
            "source_artifacts": [],
            "criterion": "delta_lnL_softmax_per_event_over_final_support_region",
            "criterion_version": "v1",
            "evidence": {
                "basis_name": "final_support_region_union_v1",
                "contributing_event_count": 2,
                "event_ids_sample": ["GW_ALPHA", "GW_BETA"],
                "local_score_definition": "exp(delta_lnL_i - max_delta_lnL_event)",
                "local_normalization": "per_event_softmax",
                "aggregation": "sum_over_events",
            },
        })
    return {
        "schema_version": WEIGHT_POLICY_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "policy_name": EXISTING_POLICY_NAME,
        "policy_role": "comparison_score_based",
        "coverage_fraction": 1.0,
        "n_rows": len(BASIS_ROWS),
        "n_weighted": len(BASIS_ROWS),
        "n_unweighted": 0,
        "weight_sum_raw": wp["weight_sum_raw"],
        "weight_sum_normalized": 1.0,
        "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
        "source_policy_inputs": [],
        "rows": rows,
    }


def _existing_renyi_payload() -> dict:
    wp = _temperature_weights(1.0)
    m = _renyi_metrics(
        [wp["rows_by_id"][r["raw_geometry_id"]]["weight_normalized"] for r in BASIS_ROWS]
    )
    return {
        "schema_version": RENYI_SCHEMA_VERSION,
        "metric_role": "epistemic_ensemble_diversity",
        "basis_name": "final_support_region_union_v1",
        "policy_name": EXISTING_POLICY_NAME,
        "weight_policy_file": "weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
        "n_rows": len(BASIS_ROWS),
        "coverage_fraction": 1.0,
        "alpha_grid": ALPHA_GRID,
        "metrics": m,
        "notes": [],
    }


def _write_event_artifacts(runs_root: Path, spec: dict) -> None:
    src = spec["source_run_id"]
    _write_run_valid(runs_root, src)
    _write_json(
        runs_root / src / "s4k_event_support_region" / "outputs" / "event_support_region.json",
        {"event_id": spec["event_id"], "run_id": src,
         "final_geometry_ids": spec["final_geometry_ids"]},
    )
    _write_json(
        runs_root / src / "s4_geometry_filter" / "outputs" / "ranked_all_full.json",
        spec["ranked_rows"],
    )


def _write_phase5_outputs(
    runs_root: Path,
    run_id: str,
    *,
    temperature_grid: list[float] | None = None,
    top_geometry_override: list[dict] | None = None,
    top_n: int = TOP_N,
    skip_keys: list[str] | None = None,
) -> None:
    """Write Phase 5 output JSON files under runs_root/run_id/experiment/phase5_temperature_sweep/outputs/."""
    p5_out = runs_root / run_id / "experiment" / "phase5_temperature_sweep" / "outputs"
    skip = set(skip_keys or [])
    grid = temperature_grid

    if "metrics" not in skip:
        _write_json(p5_out / "temperature_sweep_metrics_v1.json", _phase5_metrics_payload(grid))
    if "topk" not in skip:
        _write_json(p5_out / "temperature_sweep_topk_v1.json", _phase5_topk_payload(grid))
    if "family" not in skip:
        _write_json(p5_out / "temperature_sweep_family_mass_v1.json", _phase5_family_payload(grid))
    if "theory" not in skip:
        _write_json(p5_out / "temperature_sweep_theory_mass_v1.json", _phase5_theory_payload(grid))
    if "top_geometry" not in skip:
        _write_json(
            p5_out / "temperature_sweep_top_geometry_v1.json",
            _phase5_top_geometry_payload(grid, top_n=top_n, override_rows=top_geometry_override),
        )


def _write_upstream_inputs(runs_root: Path, run_id: str) -> None:
    """Write all inputs required upstream of Phase 6 (Phase 2c, aggregate, Phase 3, Phase 4, events)."""
    _write_json(runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json", _support_payload())
    _write_json(runs_root / run_id / "s5_aggregate" / "outputs" / "aggregate.json", _aggregate_payload())
    _write_json(
        runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "outputs" / "weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
        _existing_weight_policy_payload(),
    )
    _write_json(
        runs_root / run_id / "experiment" / "phase4_renyi_diversity_baseline" / "outputs" / "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json",
        _existing_renyi_payload(),
    )
    for spec in EVENT_SPECS:
        _write_event_artifacts(runs_root, spec)


def _prepare_run(
    tmp_path: Path,
    *,
    run_id: str,
    temperature_grid: list[float] | None = None,
    top_geometry_override: list[dict] | None = None,
    top_n: int = TOP_N,
    skip_phase5_keys: list[str] | None = None,
) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_run_valid(runs_root, run_id)
    _write_upstream_inputs(runs_root, run_id)
    _write_phase5_outputs(
        runs_root,
        run_id,
        temperature_grid=temperature_grid,
        top_geometry_override=top_geometry_override,
        top_n=top_n,
        skip_keys=skip_phase5_keys,
    )
    return repo_root, runs_root


def _run_script(
    repo_root: Path,
    run_id: str,
    runs_root: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    cmd = [sys.executable, str(repo_root / SCRIPT), "--run-id", run_id]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=repo_root, env=env, text=True, capture_output=True, check=False)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_contract_registered() -> None:
    contract = CONTRACTS.get(STAGE)
    assert contract is not None
    assert "experiment/phase5_temperature_sweep/outputs/temperature_sweep_metrics_v1.json" in contract.required_inputs
    assert "experiment/phase5_temperature_sweep/outputs/temperature_sweep_top_geometry_v1.json" in contract.required_inputs
    assert "s5_aggregate/outputs/aggregate.json" in contract.required_inputs
    assert "{source_run}/s4k_event_support_region/outputs/event_support_region.json" in contract.dynamic_inputs
    assert "outputs/temperature_selection_v1.json" in contract.produced_outputs
    assert "outputs/support_consensus_v1.json" in contract.produced_outputs
    assert "outputs/weight_policy_selected_temperature_v1.json" in contract.produced_outputs
    assert "outputs/renyi_diversity_selected_temperature_v1.json" in contract.produced_outputs
    assert "outputs/stability_diagnostics_v1.json" in contract.produced_outputs


# Test 1 ────────────────────────────────────────────────────────────────────
def test_phase6_requires_phase5_outputs(tmp_path: Path) -> None:
    """Abort if any Phase 5 required JSON is missing."""
    required_phase5_keys = ["metrics", "topk", "family", "theory", "top_geometry"]
    for missing_key in required_phase5_keys:
        run_id = f"p6_missing_{missing_key}"
        repo_root, runs_root = _prepare_run(
            tmp_path, run_id=run_id, skip_phase5_keys=[missing_key]
        )
        result = _run_script(repo_root, run_id, runs_root)
        assert result.returncode != 0, (
            f"Expected failure when {missing_key} is missing, got returncode 0.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# Test 2 ────────────────────────────────────────────────────────────────────
def test_phase6_selects_smallest_temperature_in_stable_plateau(tmp_path: Path) -> None:
    """With fully relaxed thresholds, plateau = all temperatures; T* = min(temps)."""
    run_id = "p6_smallest_t_in_plateau"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    # tau_topk=0.0, eps=100.0 → every pair is stable → plateau = all 7 temps → T* = 0.25
    result = _run_script(
        repo_root, run_id, runs_root,
        extra_args=["--tau-topk", "0.0", "--eps-family", "100.0",
                    "--eps-theory", "100.0", "--eps-renyi", "100.0"],
    )
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    sel = json.loads((stage_dir / "outputs" / "temperature_selection_v1.json").read_text())

    assert sel["schema_version"] == "temperature_selection_v1"
    assert sel["selection_rule"] == "smallest_temperature_in_stability_plateau"
    assert math.isclose(sel["selected_temperature"], min(TEMPERATURE_GRID), rel_tol=0.0, abs_tol=1e-12), (
        f"Expected T*={min(TEMPERATURE_GRID)}, got {sel['selected_temperature']}"
    )
    # Plateau must be sorted
    assert sel["stability_plateau"] == sorted(sel["stability_plateau"])
    # T* must be the minimum element of the plateau
    assert math.isclose(sel["selected_temperature"], sel["stability_plateau"][0], rel_tol=0.0, abs_tol=1e-12)


# Test 3 ────────────────────────────────────────────────────────────────────
def test_phase6_support_consensus_is_deterministic(tmp_path: Path) -> None:
    """Same input → same consensus support exact, two independent runs."""
    run_id_a = "p6_consensus_det_a"
    run_id_b = "p6_consensus_det_b"

    repo_root, runs_root_a = _prepare_run(tmp_path / "a", run_id=run_id_a)
    _, runs_root_b = _prepare_run(tmp_path / "b", run_id=run_id_b)

    args = ["--tau-topk", "0.0", "--eps-family", "100.0",
            "--eps-theory", "100.0", "--eps-renyi", "100.0"]

    r_a = _run_script(repo_root, run_id_a, runs_root_a, extra_args=args)
    r_b = _run_script(repo_root, run_id_b, runs_root_b, extra_args=args)
    assert r_a.returncode == 0, r_a.stderr + r_a.stdout
    assert r_b.returncode == 0, r_b.stderr + r_b.stdout

    consensus_a = json.loads(
        (runs_root_a / run_id_a / STAGE / "outputs" / "support_consensus_v1.json").read_text()
    )
    consensus_b = json.loads(
        (runs_root_b / run_id_b / STAGE / "outputs" / "support_consensus_v1.json").read_text()
    )

    rows_a = sorted(consensus_a["rows"], key=lambda r: r["raw_geometry_id"])
    rows_b = sorted(consensus_b["rows"], key=lambda r: r["raw_geometry_id"])
    assert rows_a == rows_b, "Consensus support must be identical for identical inputs"
    assert consensus_a["selected_temperature"] == consensus_b["selected_temperature"]


# Test 4 ────────────────────────────────────────────────────────────────────
def test_phase6_weight_policy_selected_temperature_normalizes_to_one(tmp_path: Path) -> None:
    """weight_sum_normalized must equal 1.0 in the weight policy output."""
    run_id = "p6_weight_norm"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(
        repo_root, run_id, runs_root,
        extra_args=["--tau-topk", "0.0", "--eps-family", "100.0",
                    "--eps-theory", "100.0", "--eps-renyi", "100.0"],
    )
    assert result.returncode == 0, result.stderr + result.stdout

    wp = json.loads(
        (runs_root / run_id / STAGE / "outputs" / "weight_policy_selected_temperature_v1.json").read_text()
    )
    assert math.isclose(float(wp["weight_sum_normalized"]), 1.0, rel_tol=0.0, abs_tol=1e-9), (
        f"weight_sum_normalized={wp['weight_sum_normalized']} != 1.0"
    )
    row_sum = math.fsum(float(r["weight_normalized"]) for r in wp["rows"])
    assert math.isclose(row_sum, 1.0, rel_tol=0.0, abs_tol=1e-9), (
        f"sum of row weight_normalized={row_sum} != 1.0"
    )


# Test 5 ────────────────────────────────────────────────────────────────────
def test_phase6_writes_only_under_runs_run_id_experiment_phase6(tmp_path: Path) -> None:
    """All output files must live under runs/<run_id>/experiment/phase6_temperature_stability_selection/."""
    run_id = "p6_path_isolation"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(
        repo_root, run_id, runs_root,
        extra_args=["--tau-topk", "0.0", "--eps-family", "100.0",
                    "--eps-theory", "100.0", "--eps-renyi", "100.0"],
    )
    assert result.returncode == 0, result.stderr + result.stdout

    expected_stage_dir = runs_root / run_id / STAGE
    assert expected_stage_dir.exists()

    # Verify all produced artifacts are within the expected stage dir
    for output_name in [
        "outputs/temperature_selection_v1.json",
        "outputs/support_consensus_v1.json",
        "outputs/weight_policy_selected_temperature_v1.json",
        "outputs/renyi_diversity_selected_temperature_v1.json",
        "outputs/stability_diagnostics_v1.json",
        "stage_summary.json",
        "manifest.json",
    ]:
        artifact_path = expected_stage_dir / output_name
        assert artifact_path.exists(), f"Expected artifact missing: {artifact_path}"
        # Verify it's under the run root, not outside
        assert artifact_path.resolve().is_relative_to(runs_root.resolve()), (
            f"Artifact {artifact_path} escapes runs_root {runs_root}"
        )


# Test 6 ────────────────────────────────────────────────────────────────────
def test_phase6_logs_out_root_stage_dir_outputs_dir_stage_summary_manifest(tmp_path: Path) -> None:
    """Script stdout must contain all 5 canonical path tokens."""
    run_id = "p6_log_paths"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(
        repo_root, run_id, runs_root,
        extra_args=["--tau-topk", "0.0", "--eps-family", "100.0",
                    "--eps-theory", "100.0", "--eps-renyi", "100.0"],
    )
    assert result.returncode == 0, result.stderr + result.stdout

    combined = result.stdout + result.stderr
    for token in ("OUT_ROOT=", "STAGE_DIR=", "OUTPUTS_DIR=", "STAGE_SUMMARY=", "MANIFEST="):
        assert token in combined, f"Missing log token: {token!r}"


# Test 7 ────────────────────────────────────────────────────────────────────
def test_phase6_fallback_consensus_rule_if_hard_intersection_empty(tmp_path: Path) -> None:
    """When hard top-k intersection is empty, fallback (present_fraction>=0.8) is used
    and documented in stage_summary.json.

    Design: 5 plateau temperatures; consensus_topk=1.
    geom_alpha_norm is top-1 at 4 temps, geom_beta_norm at 1 temp.
    Hard intersection (present in all 5): empty.
    Fallback: geom_alpha_norm has 4/5=0.8 >= 0.8 → selected.
    """
    run_id = "p6_fallback_consensus"

    # Build synthetic top_geometry where geom_beta_norm is top-1 at T=50.0
    # and geom_alpha_norm is top-1 at all other temps.
    # We use 5 temps from the standard grid: [0.5, 1.0, 2.0, 5.0, 50.0]
    five_temps = [0.5, 1.0, 2.0, 5.0, 50.0]

    # Build top_geometry rows with synthetic ranks (geom_beta tops at T=50)
    synthetic_top_geom_rows: list[dict] = []
    for T in five_temps:
        if math.isclose(T, 50.0, rel_tol=0.0, abs_tol=1e-9):
            order = ["geom_beta", "geom_alpha", "geom_gamma"]
        else:
            order = ["geom_alpha", "geom_beta", "geom_gamma"]
        meta = {r["raw_geometry_id"]: r for r in BASIS_ROWS}
        cum = 0.0
        # We give synthetic weights (equal so family/theory L1 = 0)
        for rank, gid in enumerate(order, 1):
            w = 1.0 / len(order)
            cum += w
            synthetic_top_geom_rows.append({
                "temperature": T,
                "rank": rank,
                "raw_geometry_id": gid,
                "normalized_geometry_id": meta[gid]["normalized_geometry_id"],
                "atlas_family": meta[gid]["atlas_family"],
                "atlas_theory": meta[gid]["atlas_theory"],
                "weight_normalized": w,
                "cumulative_mass": cum,
                "criterion": "weight_normalized_descending_cumulative_mass",
                "criterion_version": "v1",
            })

    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        temperature_grid=five_temps,
        top_geometry_override=synthetic_top_geom_rows,
        top_n=3,
    )

    result = _run_script(
        repo_root, run_id, runs_root,
        extra_args=[
            "--tau-topk", "0.0",    # Jaccard always passes
            "--eps-family", "100.0",
            "--eps-theory", "100.0",
            "--eps-renyi", "100.0",
            "--consensus-topk", "1",  # only top-1 per temperature
        ],
    )
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE

    consensus = json.loads((stage_dir / "outputs" / "support_consensus_v1.json").read_text())
    assert "present_fraction_gte_0.8" in consensus["consensus_rule"], (
        f"Expected fallback rule in consensus_rule, got: {consensus['consensus_rule']!r}"
    )
    # geom_alpha_norm should be selected (4/5 = 0.8 >= 0.8)
    selected_gids = {r["raw_geometry_id"] for r in consensus["rows"]}
    assert "geom_alpha" in selected_gids, (
        f"geom_alpha missing from fallback consensus. rows={consensus['rows']}"
    )
    # geom_beta should NOT be selected (1/5 = 0.2 < 0.8)
    assert "geom_beta" not in selected_gids, (
        f"geom_beta should not be in fallback consensus (present_fraction=0.2). rows={consensus['rows']}"
    )

    summary = json.loads((stage_dir / "stage_summary.json").read_text())
    assert summary.get("consensus_fallback_used") is True, (
        "stage_summary.json must document consensus_fallback_used=True"
    )
    # Fallback note must appear in the notes list
    notes_text = " ".join(str(n) for n in summary.get("notes", []))
    assert "fallback" in notes_text.lower() or "present_fraction" in notes_text.lower(), (
        f"stage_summary.json notes do not mention fallback. notes={summary.get('notes')}"
    )


# Test 8 ────────────────────────────────────────────────────────────────────
def test_phase6_recomputes_policy_if_phase5_has_no_weights_output(tmp_path: Path) -> None:
    """Phase 5 never publishes temperature_sweep_weights_v1.json.
    Phase 6 must successfully recompute the weight policy at T* from
    s5_aggregate event records.  Verified by checking the output exists
    and weight_sum_normalized == 1.0.
    """
    run_id = "p6_recompute_policy"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    # Confirm Phase 5 weights file does NOT exist (as expected in production)
    weights_path = (
        runs_root / run_id
        / "experiment" / "phase5_temperature_sweep" / "outputs"
        / "temperature_sweep_weights_v1.json"
    )
    assert not weights_path.exists(), (
        "Phase 5 must NOT publish temperature_sweep_weights_v1.json "
        "(this test verifies Phase 6 recomputes it)"
    )

    result = _run_script(
        repo_root, run_id, runs_root,
        extra_args=["--tau-topk", "0.0", "--eps-family", "100.0",
                    "--eps-theory", "100.0", "--eps-renyi", "100.0"],
    )
    assert result.returncode == 0, result.stderr + result.stdout

    wp = json.loads(
        (runs_root / run_id / STAGE / "outputs" / "weight_policy_selected_temperature_v1.json").read_text()
    )
    assert wp["schema_version"] == "weight_policy_selected_temperature_v1"
    assert math.isclose(float(wp["weight_sum_normalized"]), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert wp["n_rows"] == len(BASIS_ROWS)

    # Verify the stage_summary documents the recomputation note
    summary = json.loads(
        (runs_root / run_id / STAGE / "stage_summary.json").read_text()
    )
    assert summary.get("phase5_weights_published") is False, (
        "stage_summary must record phase5_weights_published=False"
    )
    notes_text = " ".join(str(n) for n in summary.get("notes", []) + [summary.get("weight_recomputation_note", "")])
    assert "recomput" in notes_text.lower() or "recompute" in notes_text.lower() or "s5_aggregate" in notes_text.lower(), (
        f"stage_summary notes must mention recomputation. notes text: {notes_text!r}"
    )
