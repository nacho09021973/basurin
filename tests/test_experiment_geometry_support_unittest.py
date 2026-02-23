from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.analyze_geometry_support import analyze_geometry_support


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mk_compatible(event_id: str, ranked_ids: list[str], f_hz: float, q: float, sigma_f: float, sigma_q: float) -> dict:
    return {
        "event_id": event_id,
        "ranked_all": [{"geometry_id": g, "d2": float(i + 1)} for i, g in enumerate(ranked_ids)],
        "d2_min": 1.0,
        "epsilon": 0.01,
        "metric": "d2",
        "observables": {"f_hz": f_hz, "Q": q},
        "covariance_logspace": {"sigma_logf": sigma_f, "sigma_logQ": sigma_q, "cov_logf_logQ": 0.0},
        "atlas_posterior": {"best_entry_id": ranked_ids[0] if ranked_ids else None},
        "compatible_geometries": [],
    }


def _build_atlas(path: Path, target: str) -> None:
    _write_json(
        path,
        {
            "entries": [
                {"geometry_id": target, "spin": 0.95, "delta_f_frac": -0.20, "delta_Q_frac": 0.20},
                {"geometry_id": "bK_140_a0.95_df-0.20_dQ+0.10", "spin": 0.95},
                {"geometry_id": "bK_107_a0.80_df-0.20_dQ+0.20", "spin": 0.80},
            ]
        },
    )


def test_report_completeness_and_worst_events_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    src_runs = ["r1", "r2", "r3"]
    runs_list = tmp_path / "runs_50_ids.txt"
    runs_list.write_text("\n".join(src_runs) + "\n", encoding="utf-8")

    target = "bK_141_a0.95_df-0.20_dQ+0.20"
    _write_json(
        runs_root / "r1" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV1", [target, "g2", "g3", "g4"], 250.0, 4.0, 0.1, 0.2),
    )
    _write_json(
        runs_root / "r2" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV2", ["g2", "g3", target, "g4"], 255.0, 4.2, 0.2, 0.3),
    )
    _write_json(
        runs_root / "r3" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV3", ["g3", "g2", "g4", "g5"], 260.0, 4.4, 0.3, 0.4),
    )

    atlas_path = tmp_path / "atlas.json"
    _build_atlas(atlas_path, target)

    result = analyze_geometry_support(
        out_root=runs_root,
        run_id="audit_geom_v1",
        runs_ids_path=runs_list,
        atlas_path=atlas_path,
        target_geometry_id=target,
        k_values=[1, 3, 5, 10],
        worst_n=3,
    )

    stage_dir = Path(result["stage_dir"])
    report = json.loads((stage_dir / "outputs" / "geometry_support_report.json").read_text(encoding="utf-8"))

    assert report["atlas_target"] is not None
    assert target == report["atlas_target"]["geometry_id"]
    assert report["atlas_comparisons"]["bK_140_a0.95_df-0.20_dQ+0.10"] is not None
    assert report["atlas_comparisons"]["bK_107_a0.80_df-0.20_dQ+0.20"] is not None

    assert report["support_definition"].startswith("support_by_k[K] counts events")

    worst = report["worst_events_target"]
    assert len(worst) == 3
    assert [x["run_id"] for x in worst] == ["r3", "r2", "r1"]

    by_event = {x["event_id"]: x for x in worst}
    assert by_event["EV1"]["rank_target"] == 0
    assert by_event["EV1"]["in_top3"] is True
    assert by_event["EV1"]["in_top5"] is True
    assert by_event["EV1"]["in_top10"] is True

    assert by_event["EV2"]["rank_target"] == 2
    assert by_event["EV2"]["in_top3"] is True
    assert by_event["EV2"]["in_top5"] is True
    assert by_event["EV2"]["in_top10"] is True

    assert by_event["EV3"]["rank_target"] == -1
    assert by_event["EV3"]["in_top3"] is False
    assert by_event["EV3"]["in_top5"] is False
    assert by_event["EV3"]["in_top10"] is False


def test_report_json_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr("experiment.analyze_geometry_support.utc_now_iso", lambda: "2026-01-01T00:00:00Z")

    src_runs = ["r1", "r2", "r3"]
    runs_list = tmp_path / "runs_50_ids.txt"
    runs_list.write_text("\n".join(src_runs) + "\n", encoding="utf-8")
    target = "bK_141_a0.95_df-0.20_dQ+0.20"

    _write_json(
        runs_root / "r1" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV1", [target, "g2", "g3"], 250.0, 4.0, 0.1, 0.2),
    )
    _write_json(
        runs_root / "r2" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV2", ["g2", "g3", target], 255.0, 4.2, 0.2, 0.3),
    )
    _write_json(
        runs_root / "r3" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV3", ["g3", "g2", "g4"], 260.0, 4.4, 0.3, 0.4),
    )
    atlas_path = tmp_path / "atlas.json"
    _build_atlas(atlas_path, target)

    analyze_geometry_support(
        out_root=runs_root,
        run_id="audit_geom_v1",
        runs_ids_path=runs_list,
        atlas_path=atlas_path,
        target_geometry_id=target,
        k_values=[1, 3, 5, 10],
        worst_n=3,
    )
    report_path = runs_root / "audit_geom_v1" / "experiment" / "geometry_support_v1" / "outputs" / "geometry_support_report.json"
    report_1 = json.loads(report_path.read_text(encoding="utf-8"))

    analyze_geometry_support(
        out_root=runs_root,
        run_id="audit_geom_v1",
        runs_ids_path=runs_list,
        atlas_path=atlas_path,
        target_geometry_id=target,
        k_values=[1, 3, 5, 10],
        worst_n=3,
    )
    report_2 = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_1 == report_2


def test_missing_runs_list_has_actionable_error(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_root"
    runs_root.mkdir(parents=True, exist_ok=True)
    atlas_path = tmp_path / "atlas.json"
    _write_json(atlas_path, {"entries": []})

    with pytest.raises(FileNotFoundError) as exc:
        analyze_geometry_support(
            out_root=runs_root,
            run_id="audit_geom_v1",
            runs_ids_path=tmp_path / "missing_runs_50_ids.txt",
            atlas_path=atlas_path,
            target_geometry_id="bK_141_a0.95_df-0.20_dQ+0.20",
            k_values=[1],
            worst_n=1,
        )
    msg = str(exc.value)
    assert "Expected:" in msg
    assert "Command to regenerate upstream" in msg
