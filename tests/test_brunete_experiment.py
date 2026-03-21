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
    for idx, geoms in enumerate((g1, g2, g3), 1):
        event_id = f"GW{idx:06d}"
        ev220 = f"{event_id}_220"
        ev221 = f"{event_id}_221"
        _make_event_run(runs_root / batch220, ev220, geoms)
        _make_event_run(runs_root / batch221, ev221, geoms)
        rows.append({"event_id": event_id, "event_run_id_220": ev220, "event_run_id_221": ev221})
    _write(classify_stage / "outputs" / "geometry_summary.json", {"batch_220_run_id": batch220, "batch_221_run_id": batch221, "rows": rows})
    return classify_run_id


def test_base_contract_resolves_event_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    classify_run_id = _make_classify_run(runs_root)

    from brunete.experiment.base_contract import resolve_event_run_ids

    run_ids = resolve_event_run_ids(classify_run_id, "220", runs_root)
    assert run_ids == ["GW000001_220", "GW000002_220", "GW000003_220"]


def test_b5f_aggregates_from_classify_run(tmp_path: Path):
    runs_root = tmp_path / "runs"
    classify_run_id = _make_classify_run(runs_root)

    from brunete.experiment.b5f import run

    result = run(classify_run_id, runs_root=str(runs_root), dry_run=True)
    assert result["n_events_aggregated"] == 6
    assert result["family_support_rates"]["edgb"]["rate"] == 1.0


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
