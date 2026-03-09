from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_s6b(run_id: str, runs_root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [sys.executable, "-m", "mvp.s6b_information_geometry_ranked", "--run", run_id, "--top-k", "10"]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_run(tmp_path: Path, *, include_unknown_geometry: bool = False) -> tuple[Path, str]:
    runs_root = tmp_path / "runs"
    run_id = "run_s6b_atlas_index"
    run_dir = runs_root / run_id

    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    atlas_entries = [
        {"id": "A", "f_hz": 100.0, "Q": 2.0},
        {"id": "B", "f_hz": 110.0, "Q": 2.2},
        {"id": "C", "f_hz": 120.0, "Q": 2.4},
    ]
    atlas_path = tmp_path / "atlas.json"
    _write_json(atlas_path, atlas_entries)

    _write_json(
        run_dir / "s4_geometry_filter" / "stage_summary.json",
        {
            "stage": "s4_geometry_filter",
            "parameters": {"atlas_path": str(atlas_path)},
            "verdict": "PASS",
        },
    )
    _write_json(
        run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        {
            "event_id": "GW-test",
            "atlas_id": "atlas-test",
            "n_atlas": 3,
            "compatible_geometries": [
                {"geometry_id": "A"},
                {"geometry_id": "B"},
                {"geometry_id": "C"},
            ],
        },
    )

    rows = [
        {"geometry_id": "A", "d_conformal": 3.0},
        {"geometry_id": "B", "d_conformal": 1.0},
        {"geometry_id": "C", "d_conformal": 2.0},
    ]
    if include_unknown_geometry:
        rows.append({"geometry_id": "Z", "d_conformal": 0.5})

    _write_json(
        run_dir / "s6_information_geometry" / "outputs" / "curvature.json",
        {
            "event_id": "GW-test",
            "reranked_geometries": rows,
        },
    )

    return runs_root, run_id


def test_s6b_emits_geometry_id_with_real_atlas_index(tmp_path: Path) -> None:
    runs_root, run_id = _prepare_run(tmp_path)

    proc = _run_s6b(run_id, runs_root)
    assert proc.returncode == 0, proc.stderr

    out_path = runs_root / run_id / "s6b_information_geometry_ranked" / "outputs" / "ranked_geometries.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    expected_atlas_index = {"A": 0, "B": 1, "C": 2}
    for item in payload["ranked"]:
        assert item["geometry_id"] in expected_atlas_index
        assert item["atlas_index"] == expected_atlas_index[item["geometry_id"]]
        assert "score" in item

    assert all("geometry_id" in item for item in payload["ranked"])
    assert all("geometry_id" in item for item in payload["compatible"])



def test_s6b_aborts_when_geometry_not_found_in_atlas(tmp_path: Path) -> None:
    runs_root, run_id = _prepare_run(tmp_path, include_unknown_geometry=True)

    proc = _run_s6b(run_id, runs_root)
    assert proc.returncode != 0
    assert "not found in atlas" in proc.stderr
