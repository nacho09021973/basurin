from __future__ import annotations

import json
from pathlib import Path


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_event_run(batch_dir: Path, event_run_id: str, geometries: list[dict], verdict=None, estimates=None):
    run_dir = batch_dir / "run_batch" / "event_runs" / event_run_id
    _write(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS", "run_id": event_run_id})
    _write(run_dir / "s4_geometry_filter" / "stage_summary.json", {"verdict": "PASS", "run_id": event_run_id})
    _write(run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json", {"geometries": geometries})
    _write(run_dir / "s3b_multimode_estimates" / "estimates.json", estimates or {"frequency_220": 1.0})
    _write(run_dir / "verdict.json", verdict or {"family_verdicts": {"edgb": {"verdict": "SUPPORTED"}}})
    return run_dir


def _make_classify_run(runs_root: Path):
    classify_run_id = "classify_demo"
    classify_stage = runs_root / classify_run_id / "classify_geometries"
    _write(classify_stage / "RUN_VALID" / "verdict.json", {"verdict": "PASS", "run_id": classify_run_id})
    batch220 = "batch220_demo"
    batch221 = "batch221_demo"
    rows = []
    g1 = [{"geometry_id": "edgb_001", "family": "edgb", "mahalanobis_d2": 1.0, "delta_lnL": 0.5, "saturation_221": 0.1}]
    g2 = [{"geometry_id": "edgb_001", "family": "edgb", "mahalanobis_d2": 2.0, "delta_lnL": 0.4, "saturation_221": 0.2}]
    g3 = [{"geometry_id": "edgb_001", "family": "edgb", "mahalanobis_d2": 1.5, "delta_lnL": 0.3, "saturation_221": 0.3}]
    verdict = {"family_verdicts": {"edgb": {"verdict": "SUPPORTED"}}}
    classify_rows = [
        {
            "event_id": "GW000001",
            "event_run_id_220": "GW000001_220",
            "event_run_id_221": "GW000001_221",
            "classification": "common_nonempty_both_221_support_multi",
            "has_joint_support": True,
            "support_region_status_221": "SUPPORT_REGION_AVAILABLE",
            "support_region_n_final_221": 2,
        },
        {
            "event_id": "GW000002",
            "event_run_id_220": "GW000002_220",
            "event_run_id_221": "GW000002_221",
            "classification": "common_nonempty_both_221_no_common_region",
            "has_joint_support": False,
            "support_region_status_221": "NO_COMMON_REGION",
            "support_region_n_final_221": 0,
        },
        {
            "event_id": "GW000003",
            "event_run_id_220": "GW000003_220",
            "event_run_id_221": "GW000003_221",
            "classification": "common_nonempty_both_221_support_singleton",
            "has_joint_support": True,
            "support_region_status_221": "SUPPORT_REGION_AVAILABLE",
            "support_region_n_final_221": 1,
        },
    ]
    for row, geoms in zip(classify_rows, (g1, g2, g3), strict=True):
        _make_event_run(runs_root / batch220, row["event_run_id_220"], geoms, verdict=verdict)
        _make_event_run(runs_root / batch221, row["event_run_id_221"], geoms, verdict=verdict)
        rows.append(row)
    _write(
        classify_stage / "outputs" / "geometry_summary.json",
        {"batch_220_run_id": batch220, "batch_221_run_id": batch221, "rows": rows},
    )
    _write(runs_root / batch220 / "run_batch" / "outputs" / "results.json", {"results": []})
    _write(runs_root / batch221 / "run_batch" / "outputs" / "results.json", {"results": []})
    return classify_run_id


def test_base_contract_resolves_event_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    classify_run_id = _make_classify_run(runs_root)

    from brunete.experiment.base_contract import resolve_event_run_ids

    run_ids = resolve_event_run_ids(classify_run_id, "220", runs_root)
    assert run_ids == ["GW000001_220", "GW000002_220", "GW000003_220"]


def test_b5f_uses_classify_has_joint_support_without_recomputing(tmp_path: Path):
    runs_root = tmp_path / "runs"
    classify_run_id = _make_classify_run(runs_root)

    from brunete.experiment.b5f import run

    result = run(classify_run_id, runs_root=str(runs_root), dry_run=True)
    assert result["n_events"] == 3
    assert result["n_joint_support_events"] == 2
    assert result["family_joint_support_rates"]["edgb"]["joint_support_rate"] == 0.6667
    assert result["events"][1]["support_region_status_221"] == "NO_COMMON_REGION"
    assert result["events"][1]["family_verdicts"]["edgb"]["counted_as_joint_support"] is False


def test_b5a_and_b5h_use_mode_specific_event_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    classify_run_id = _make_classify_run(runs_root)

    from brunete.experiment.b5a import run as run_b5a
    from brunete.experiment.b5h import run as run_b5h

    agg = run_b5a(classify_run_id, "220", runs_root=str(runs_root), dry_run=True)
    pred = run_b5h(classify_run_id, "221", runs_root=str(runs_root), dry_run=True)

    assert agg["intersection_count"] == 1
    assert agg["mode"] == "220"
    assert pred["n_events"] == 3
    assert pred["mode"] == "221"
