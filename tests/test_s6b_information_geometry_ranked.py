from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"


def test_s6b_exports_ranked_non_empty(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s6b_ranked"

    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    s4_stage = runs_root / run_id / "s4_geometry_filter"
    s6_out = runs_root / run_id / "s6_information_geometry" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    s4_stage.mkdir(parents=True, exist_ok=True)
    s6_out.mkdir(parents=True, exist_ok=True)

    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_text(json.dumps([
        {"id": "g0", "name": "G0"},
        {"id": "g1", "name": "G1"},
        {"id": "g2", "name": "G2"},
    ]), encoding="utf-8")
    (s4_stage / "stage_summary.json").write_text(json.dumps({
        "parameters": {"atlas_path": str(atlas_path)},
    }), encoding="utf-8")

    (s4_out / "compatible_set.json").write_text(json.dumps({
        "event_id": "GWTEST",
        "n_atlas": 3,
        "compatible_geometries": [{"geometry_id": "g0"}, {"geometry_id": "g2"}],
    }), encoding="utf-8")
    (s6_out / "curvature.json").write_text(json.dumps({
        "event_id": "GWTEST",
        "reranked_geometries": [
            {"geometry_id": "g0", "d_conformal": 0.1, "metadata": {"atlas_index": 0}},
            {"geometry_id": "g1", "d_conformal": 0.2, "metadata": {"atlas_index": 1}},
            {"geometry_id": "g2", "d_conformal": 0.3, "metadata": {"atlas_index": 2}},
        ],
    }), encoding="utf-8")

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [sys.executable, str(MVP_DIR / "s6b_information_geometry_ranked.py"), "--run", run_id]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    ranked_path = runs_root / run_id / "s6b_information_geometry_ranked" / "outputs" / "ranked_geometries.json"
    assert ranked_path.exists()
    payload = json.loads(ranked_path.read_text(encoding="utf-8"))
    assert payload["ranked"]
    assert payload["compatible"]


def test_s6b_uses_real_atlas_index_from_geometry_id(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "test_s6b_real_atlas_index"

    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_stage = runs_root / run_id / "s4_geometry_filter"
    s4_out = s4_stage / "outputs"
    s6_out = runs_root / run_id / "s6_information_geometry" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    s6_out.mkdir(parents=True, exist_ok=True)

    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_text(json.dumps([
        {"id": "geomA", "name": "A"},
        {"id": "geomB", "name": "B"},
        {"id": "geomC", "name": "C"},
    ]), encoding="utf-8")
    (s4_stage / "stage_summary.json").write_text(json.dumps({
        "parameters": {"atlas_path": str(atlas_path)},
    }), encoding="utf-8")

    (s4_out / "compatible_set.json").write_text(json.dumps({
        "event_id": "GWTEST",
        "n_atlas": 3,
        "compatible_geometries": [{"geometry_id": "geomC"}, {"geometry_id": "geomA"}],
    }), encoding="utf-8")

    (s6_out / "curvature.json").write_text(json.dumps({
        "event_id": "GWTEST",
        "reranked_geometries": [
            {"geometry_id": "geomC", "d_conformal": 0.1},
            {"geometry_id": "geomA", "d_conformal": 0.2},
            {"geometry_id": "geomB", "d_conformal": 0.3},
        ],
    }), encoding="utf-8")

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [sys.executable, str(MVP_DIR / "s6b_information_geometry_ranked.py"), "--run", run_id]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((runs_root / run_id / "s6b_information_geometry_ranked" / "outputs" / "ranked_geometries.json").read_text(encoding="utf-8"))
    ranked = payload["ranked"]
    assert [row["geometry_id"] for row in ranked] == ["geomC", "geomA", "geomB"]
    assert [row["atlas_index"] for row in ranked] == [2, 0, 1]
