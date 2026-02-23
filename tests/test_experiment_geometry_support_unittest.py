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
        "observables": {"f_hz": f_hz, "Q": q},
        "covariance_logspace": {"sigma_logf": sigma_f, "sigma_logQ": sigma_q, "cov_logf_logQ": 0.0},
        "atlas_posterior": {"best_entry_id": ranked_ids[0]},
        "compatible_geometries": [],
    }


def test_analyze_geometry_support_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

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
        _mk_compatible("EV2", ["g2", target, "g3"], 255.0, 4.2, 0.2, 0.3),
    )
    _write_json(
        runs_root / "r3" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV3", ["g3", "g2", target], 260.0, 4.4, 0.3, 0.4),
    )

    atlas_path = tmp_path / "atlas.json"
    _write_json(
        atlas_path,
        {
            "entries": [
                {"geometry_id": target, "spin": 0.95, "delta_f_frac": -0.20, "delta_Q_frac": 0.20},
                {"geometry_id": "bK_140_a0.95_df-0.20_dQ+0.10", "spin": 0.95},
                {"geometry_id": "bK_107_a0.80_df-0.20_dQ+0.20", "spin": 0.80},
            ]
        },
    )

    result = analyze_geometry_support(
        out_root=runs_root,
        run_id="audit_geom_v1",
        runs_ids_path=runs_list,
        atlas_path=atlas_path,
        target_geometry_id=target,
        k_values=[1, 3],
        worst_n=2,
    )

    stage_dir = Path(result["stage_dir"])
    report = json.loads((stage_dir / "outputs" / "geometry_support_report.json").read_text(encoding="utf-8"))
    assert report["support_by_k"]["1"][target] == 1
    assert report["support_by_k"]["3"][target] == 3
    assert report["rank_summary"]["p50"] == 2.0
    assert len(report["worst_events"]) == 2
    assert report["atlas_bk141_vs_bk107"]["target"]["geometry_id"] == target
    assert report["atlas_bk141_vs_bk107"]["comparison"]["geometry_id"] == "bK_107_a0.80_df-0.20_dQ+0.20"
    assert report["best_entry_spin_family_counts"]["a0.95"] == 1
    assert report["best_entry_spin_family_counts"]["other"] == 2


def test_topk_transitions_3_5_10(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    src_runs = ["r1", "r2", "r3"]
    runs_list = tmp_path / "runs_50_ids.txt"
    runs_list.write_text("\n".join(src_runs) + "\n", encoding="utf-8")
    target = "bK_141_a0.95_df-0.20_dQ+0.20"

    _write_json(
        runs_root / "r1" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV1", ["g1", "g2", "g3", target, "g5"], 250.0, 4.0, 0.1, 0.2),
    )
    _write_json(
        runs_root / "r2" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV2", ["g1", "g2", "g3", "g4", "g5", "g6", target], 250.0, 4.0, 0.1, 0.2),
    )
    _write_json(
        runs_root / "r3" / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        _mk_compatible("EV3", [target, "g2", "g3"], 250.0, 4.0, 0.1, 0.2),
    )
    atlas_path = tmp_path / "atlas.json"
    _write_json(atlas_path, {"entries": [{"geometry_id": target}, {"geometry_id": "bK_107_a0.80_df-0.20_dQ+0.20"}]})

    result = analyze_geometry_support(
        out_root=runs_root,
        run_id="audit_geom_v1",
        runs_ids_path=runs_list,
        atlas_path=atlas_path,
        target_geometry_id=target,
        k_values=[3, 5, 10],
        worst_n=2,
    )
    stage_dir = Path(result["stage_dir"])
    report = json.loads((stage_dir / "outputs" / "geometry_support_report.json").read_text(encoding="utf-8"))
    events = {x["event_id"] for x in report["target_membership_transitions_3_5_10"]}
    assert events == {"EV1", "EV2"}


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
