from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase5_temperature_sweep.py")
STAGE = "experiment/phase5_temperature_sweep"
ALPHA_GRID = [0, 1, 2, "inf"]
K_GRID = [1, 3, 5, 10]
TOP_N = 20


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_valid(runs_root: Path, run_id: str, verdict: str = "PASS") -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


def _renyi_metrics(positive_weights: list[float]) -> dict[str, Any]:
    n_weighted = len(positive_weights)
    p_max = max(positive_weights)
    h_alpha = {
        "0": math.log(float(n_weighted)),
        "1": -math.fsum(p * math.log(p) for p in positive_weights),
        "2": -math.log(math.fsum(p * p for p in positive_weights)),
        "inf": -math.log(p_max),
    }
    d_alpha = {key: math.exp(value) for key, value in h_alpha.items()}
    return {
        "H_alpha": h_alpha,
        "D_alpha": d_alpha,
        "p_max": p_max,
        "n_weighted": n_weighted,
    }


def _run_script(repo_root: Path, run_id: str, runs_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    return subprocess.run(
        [sys.executable, str(repo_root / SCRIPT), "--run-id", run_id],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _make_basis_payload(
    geometry_specs: list[tuple[str, str, str, str]],
    n_events: int = 2,
) -> dict[str, Any]:
    rows = []
    for raw_id, norm_id, family, theory in geometry_specs:
        rows.append({
            "raw_geometry_id": raw_id,
            "normalized_geometry_id": norm_id,
            "atlas_family": family,
            "atlas_theory": theory,
            "join_mode": "exact_match_v1",
            "join_status": "RESOLVED",
            "support_basis_name": "final_support_region_union_v1",
            "n_events_supported": n_events,
            "support_fraction_events": n_events / 47.0,
            "in_golden_post_hawking_union": True,
            "in_final_support_region_union": True,
            "joint_available": False,
            "joint_posterior_weight_joint": None,
            "joint_support_count": None,
            "joint_support_fraction": None,
            "joint_coverage": None,
        })
    return {
        "schema_version": "support_ontology_basis_v1",
        "basis_name": "final_support_region_union_v1",
        "n_events": 47,
        "n_rows": len(rows),
        "family_counts": {},
        "theory_counts": {},
        "rows": rows,
    }


def _make_aggregate_payload(event_runs: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "events": [
            {"event_id": event_id, "run_id": run_id}
            for event_id, run_id in event_runs
        ]
    }


def _make_event_support(geometry_ids: list[str], event_id: str, run_id: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "run_id": run_id,
        "final_geometry_ids": geometry_ids,
    }


def _make_ranked_all_full(geometry_deltas: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"geometry_id": gid, "delta_lnL": delta}
        for gid, delta in geometry_deltas.items()
    ]


def _compute_expected_weights_at_t(
    temperature: float,
    geometry_specs: list[tuple[str, str, str, str]],
    events: list[dict[str, float]],
) -> dict[str, float]:
    """Compute expected weight_normalized for each geometry at temperature T."""
    gids = [spec[0] for spec in geometry_specs]
    raw_weight = {gid: 0.0 for gid in gids}

    for event_deltas in events:
        restricted = {gid: d for gid, d in event_deltas.items() if gid in raw_weight}
        max_d = max(restricted.values())
        scores = {gid: math.exp((d - max_d) / temperature) for gid, d in restricted.items()}
        score_sum = math.fsum(scores.values())
        for gid, s in scores.items():
            raw_weight[gid] += s / score_sum

    total = math.fsum(raw_weight.values())
    return {gid: raw_weight[gid] / total for gid in gids}


def _make_weight_policy_from_weights(
    weight_normalized: dict[str, float],
    geometry_specs: list[tuple[str, str, str, str]],
) -> dict[str, Any]:
    rows = []
    for raw_id, norm_id, family, theory in geometry_specs:
        w = weight_normalized[raw_id]
        rows.append({
            "raw_geometry_id": raw_id,
            "normalized_geometry_id": norm_id,
            "atlas_family": family,
            "atlas_theory": theory,
            "policy_name": "event_support_delta_lnL_softmax_mean_v1",
            "weight_raw": w * len(geometry_specs),
            "weight_normalized": w,
            "weight_status": "WEIGHTED",
            "support_count_events": 2,
            "support_fraction_events": 2 / 47.0,
            "source_artifacts": [],
            "criterion": "delta_lnL_softmax_per_event_over_final_support_region",
            "criterion_version": "v1",
            "evidence": {},
        })
    return {
        "schema_version": "weight_policy_basis_v1",
        "basis_name": "final_support_region_union_v1",
        "policy_name": "event_support_delta_lnL_softmax_mean_v1",
        "policy_role": "comparison_score_based",
        "coverage_fraction": 1.0,
        "n_rows": len(rows),
        "n_weighted": len(rows),
        "n_unweighted": 0,
        "weight_sum_raw": math.fsum(r["weight_raw"] for r in rows),
        "weight_sum_normalized": 1.0,
        "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
        "source_policy_inputs": [],
        "rows": rows,
    }


def _make_renyi_from_weights(weight_normalized: dict[str, float]) -> dict[str, Any]:
    positive = [w for w in weight_normalized.values() if w > 0.0]
    metrics = _renyi_metrics(positive)
    metrics["n_unweighted"] = 0
    metrics["weight_sum_normalized"] = 1.0
    return {
        "schema_version": "renyi_diversity_baseline_v1",
        "metric_role": "epistemic_ensemble_diversity",
        "basis_name": "final_support_region_union_v1",
        "policy_name": "event_support_delta_lnL_softmax_mean_v1",
        "weight_policy_file": "weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
        "n_rows": len(weight_normalized),
        "coverage_fraction": 1.0,
        "alpha_grid": ALPHA_GRID,
        "metrics": metrics,
        "notes": [],
    }


# --- Fixture geometry specs ---
GEOMETRY_SPECS = [
    ("geo_a", "geo_a_norm", "edgb", "EdGB"),
    ("geo_b", "geo_b_norm", "edgb", "EdGB"),
    ("geo_c", "geo_c_norm", "kerr_newman", "Kerr-Newman"),
    ("geo_d", "geo_d_norm", "dcs", "dCS"),
]

EVENT_RUNS = [
    ("evt_alpha", "run_evt_alpha"),
    ("evt_beta", "run_evt_beta"),
]

# Event deltas: geo_a is best in both events
EVENT_DELTAS = [
    {"geo_a": 0.0, "geo_b": -1.0, "geo_c": -2.0, "geo_d": -5.0},
    {"geo_a": 0.0, "geo_b": -0.5, "geo_c": -1.5, "geo_d": -4.0},
]


def _prepare_run(
    tmp_path: Path,
    *,
    run_id: str = "phase5_test",
    geometry_specs: list[tuple[str, str, str, str]] | None = None,
    event_runs: list[tuple[str, str]] | None = None,
    event_deltas: list[dict[str, float]] | None = None,
    omit_event_support: str | None = None,
    omit_ranked: str | None = None,
    omit_delta_for_geometry: str | None = None,
) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    specs = geometry_specs or GEOMETRY_SPECS
    eruns = event_runs or EVENT_RUNS
    edeltas = event_deltas or EVENT_DELTAS

    _write_run_valid(runs_root, run_id)

    # phase2c basis
    basis_path = runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    _write_json(basis_path, _make_basis_payload(specs, n_events=len(eruns)))

    # aggregate
    agg_path = runs_root / run_id / "s5_aggregate" / "outputs" / "aggregate.json"
    _write_json(agg_path, _make_aggregate_payload(eruns))

    # per-event inputs
    gids = [s[0] for s in specs]
    for idx, (event_id, source_run_id) in enumerate(eruns):
        _write_run_valid(runs_root, source_run_id)

        deltas = edeltas[idx]

        if omit_event_support != source_run_id:
            esp = runs_root / source_run_id / "s4k_event_support_region" / "outputs" / "event_support_region.json"
            _write_json(esp, _make_event_support(gids, event_id, source_run_id))

        if omit_ranked != source_run_id:
            rp = runs_root / source_run_id / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"
            if omit_delta_for_geometry:
                deltas = {k: v for k, v in deltas.items() if k != omit_delta_for_geometry}
            _write_json(rp, _make_ranked_all_full(deltas))

    # existing weight policy at T=1
    t1_weights = _compute_expected_weights_at_t(1.0, specs, edeltas)
    wp_path = (
        runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "outputs"
        / "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
    )
    _write_json(wp_path, _make_weight_policy_from_weights(t1_weights, specs))

    # existing renyi baseline at T=1
    rp = (
        runs_root / run_id / "experiment" / "phase4_renyi_diversity_baseline" / "outputs"
        / "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json"
    )
    _write_json(rp, _make_renyi_from_weights(t1_weights))

    return repo_root, runs_root


# ── Tests ────────────────────────────────────────────────────────────────


def test_contract_registered() -> None:
    contract = CONTRACTS.get(STAGE)
    assert contract is not None
    assert "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json" in contract.required_inputs
    assert "s5_aggregate/outputs/aggregate.json" in contract.required_inputs
    assert contract.produced_outputs == [
        "outputs/temperature_sweep_metrics_v1.json",
        "outputs/temperature_sweep_topk_v1.json",
        "outputs/temperature_sweep_family_mass_v1.json",
        "outputs/temperature_sweep_theory_mass_v1.json",
        "outputs/temperature_sweep_top_geometry_v1.json",
    ]


def test_happy_path_two_temperatures_two_events(tmp_path: Path) -> None:
    """Happy path: synthetic with 4 geometries, 2 events, full T_GRID."""
    run_id = "phase5_happy"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    metrics_path = stage_dir / "outputs" / "temperature_sweep_metrics_v1.json"
    topk_path = stage_dir / "outputs" / "temperature_sweep_topk_v1.json"
    family_path = stage_dir / "outputs" / "temperature_sweep_family_mass_v1.json"
    theory_path = stage_dir / "outputs" / "temperature_sweep_theory_mass_v1.json"
    top_geom_path = stage_dir / "outputs" / "temperature_sweep_top_geometry_v1.json"
    summary_path = stage_dir / "stage_summary.json"

    assert metrics_path.exists()
    assert topk_path.exists()
    assert family_path.exists()
    assert theory_path.exists()
    assert top_geom_path.exists()
    assert summary_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    topk = json.loads(topk_path.read_text(encoding="utf-8"))
    family = json.loads(family_path.read_text(encoding="utf-8"))
    theory = json.loads(theory_path.read_text(encoding="utf-8"))
    top_geom = json.loads(top_geom_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Schema checks
    assert metrics["schema_version"] == "temperature_sweep_metrics_v1"
    assert metrics["comparison_role"] == "temperature_sensitivity_of_score_based_policy"
    assert metrics["basis_name"] == "final_support_region_union_v1"
    assert metrics["temperature_grid"] == [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    assert metrics["alpha_grid"] == ALPHA_GRID

    assert topk["schema_version"] == "temperature_sweep_topk_v1"
    assert topk["k_grid"] == K_GRID

    assert family["schema_version"] == "temperature_sweep_family_mass_v1"
    assert theory["schema_version"] == "temperature_sweep_theory_mass_v1"
    assert top_geom["schema_version"] == "temperature_sweep_top_geometry_v1"
    assert top_geom["top_n"] == TOP_N

    # Each temperature produces one metrics row
    assert len(metrics["rows"]) == 7
    for mr in metrics["rows"]:
        assert mr["n_rows"] == 4
        assert mr["coverage_fraction"] == 1.0
        assert math.isclose(mr["weight_sum_normalized"], 1.0, abs_tol=1e-9)
        assert "D_1" in mr
        assert "H_1" in mr
        assert "p_max" in mr

    # Gates
    assert "gates" in metrics
    gates = metrics["gates"]
    assert gates["overall_gate_status"] in ("PASS", "FAIL")
    assert "per_temperature_gate_status" in gates

    # Summary fields
    assert summary["t1_weight_match_with_existing_policy"] is True
    assert summary["t1_renyi_match_with_existing_baseline"] is True
    assert summary["tolerance_weight"] == 1e-9
    assert summary["tolerance_metric"] == 1e-6
    assert summary["verdict"] == "PASS"


def test_concentration_monotonicity(tmp_path: Path) -> None:
    """p_max(T_small) >= p_max(T_medium) >= p_max(T_large) and top-k mass decreases with T."""
    run_id = "phase5_monotonicity"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    metrics = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_metrics_v1.json").read_text(encoding="utf-8")
    )
    topk = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_topk_v1.json").read_text(encoding="utf-8")
    )

    # p_max should be non-increasing with T
    pmax_by_t = {mr["temperature"]: mr["p_max"] for mr in metrics["rows"]}
    temperatures = sorted(pmax_by_t.keys())
    for i in range(len(temperatures) - 1):
        t_low = temperatures[i]
        t_high = temperatures[i + 1]
        assert pmax_by_t[t_low] >= pmax_by_t[t_high] - 1e-12, (
            f"p_max not monotonically non-increasing: T={t_low} p_max={pmax_by_t[t_low]} "
            f"> T={t_high} p_max={pmax_by_t[t_high]}"
        )

    # top-1 and top-3 mass should decrease with T
    for k in [1, 3]:
        mass_by_t = {
            row["temperature"]: row["cumulative_mass"]
            for row in topk["rows"]
            if row["top_k"] == k
        }
        for i in range(len(temperatures) - 1):
            t_low = temperatures[i]
            t_high = temperatures[i + 1]
            assert mass_by_t[t_low] >= mass_by_t[t_high] - 1e-12, (
                f"top-{k} mass not monotonically non-increasing: "
                f"T={t_low}={mass_by_t[t_low]} T={t_high}={mass_by_t[t_high]}"
            )


def test_fails_if_event_support_region_missing(tmp_path: Path) -> None:
    run_id = "phase5_missing_esr"
    repo_root, runs_root = _prepare_run(
        tmp_path, run_id=run_id, omit_event_support="run_evt_alpha"
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "event_support_region.json" in (result.stderr + result.stdout)


def test_fails_if_ranked_all_full_missing(tmp_path: Path) -> None:
    run_id = "phase5_missing_ranked"
    repo_root, runs_root = _prepare_run(
        tmp_path, run_id=run_id, omit_ranked="run_evt_beta"
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "ranked_all_full.json" in (result.stderr + result.stdout)


def test_fails_if_delta_lnl_missing_for_supported_geometry(tmp_path: Path) -> None:
    run_id = "phase5_missing_delta"
    repo_root, runs_root = _prepare_run(
        tmp_path, run_id=run_id, omit_delta_for_geometry="geo_a"
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "Missing delta_lnL" in (result.stderr + result.stdout)


def test_t1_reproduces_existing_policy(tmp_path: Path) -> None:
    """T=1.0 must match the materialized phase3 policy within tolerance."""
    run_id = "phase5_t1_check"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    summary = json.loads(
        (stage_dir / "stage_summary.json").read_text(encoding="utf-8")
    )
    assert summary["t1_weight_match_with_existing_policy"] is True
    assert summary["t1_renyi_match_with_existing_baseline"] is True


def test_t1_cross_check_fails_on_mismatch(tmp_path: Path) -> None:
    """If the existing policy has different weights, T=1 cross-check must fail."""
    run_id = "phase5_t1_mismatch"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    # Corrupt the existing weight policy with wrong weights
    wp_path = (
        runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "outputs"
        / "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
    )
    wp = json.loads(wp_path.read_text(encoding="utf-8"))
    # Assign uniform weights (wrong)
    n = len(wp["rows"])
    for row in wp["rows"]:
        row["weight_normalized"] = 1.0 / n
    wp["weight_sum_normalized"] = 1.0
    _write_json(wp_path, wp)

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "cross-check FAILED" in (result.stderr + result.stdout)


def test_snapshot_metrics_json_structure(tmp_path: Path) -> None:
    """Normalized snapshot of output JSON structure."""
    run_id = "phase5_snapshot"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    metrics = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_metrics_v1.json").read_text(encoding="utf-8")
    )

    # Verify all required top-level fields
    required_top_level = [
        "schema_version", "comparison_role", "basis_name",
        "temperature_grid", "alpha_grid", "rows", "gates", "notes",
    ]
    for field in required_top_level:
        assert field in metrics, f"Missing top-level field: {field}"

    # Verify row fields
    row = metrics["rows"][0]
    required_row_fields = [
        "temperature", "policy_name", "normalization_method", "n_rows",
        "coverage_fraction", "n_weighted", "n_unweighted", "p_max",
        "H_0", "H_1", "H_2", "H_inf",
        "D_0", "D_1", "D_2", "D_inf",
        "weight_sum_raw", "weight_sum_normalized",
        "criterion", "criterion_version",
    ]
    for field in required_row_fields:
        assert field in row, f"Missing row field: {field}"

    # Verify gates fields
    gates = metrics["gates"]
    required_gate_fields = [
        "central_temperature_range", "d1_reference_at_t1",
        "d1_lower_bound", "d1_upper_bound",
        "edgb_plus_kerr_newman_min_mass", "dcs_max_mass",
        "per_temperature_gate_status", "overall_gate_status",
    ]
    for field in required_gate_fields:
        assert field in gates, f"Missing gates field: {field}"

    # Verify topk row fields
    topk = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_topk_v1.json").read_text(encoding="utf-8")
    )
    topk_row = topk["rows"][0]
    for field in ["temperature", "top_k", "cumulative_mass", "normalization_check", "criterion", "criterion_version"]:
        assert field in topk_row, f"Missing topk row field: {field}"

    # Verify family mass row fields
    fam = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_family_mass_v1.json").read_text(encoding="utf-8")
    )
    fam_row = fam["rows"][0]
    for field in ["temperature", "atlas_family", "family_mass", "family_count", "family_mass_rank", "criterion", "criterion_version"]:
        assert field in fam_row, f"Missing family row field: {field}"

    # Verify theory mass row fields
    th = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_theory_mass_v1.json").read_text(encoding="utf-8")
    )
    th_row = th["rows"][0]
    for field in ["temperature", "atlas_theory", "theory_mass", "theory_count", "theory_mass_rank", "criterion", "criterion_version"]:
        assert field in th_row, f"Missing theory row field: {field}"

    # Verify top geometry row fields
    tg = json.loads(
        (stage_dir / "outputs" / "temperature_sweep_top_geometry_v1.json").read_text(encoding="utf-8")
    )
    tg_row = tg["rows"][0]
    for field in [
        "temperature", "rank", "raw_geometry_id", "normalized_geometry_id",
        "atlas_family", "atlas_theory", "weight_normalized", "cumulative_mass",
        "criterion", "criterion_version",
    ]:
        assert field in tg_row, f"Missing top geometry row field: {field}"
