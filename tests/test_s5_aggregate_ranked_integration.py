from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mvp.s5_aggregate import aggregate_compatible_sets

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"


def _mk_run(runs_root: Path, run_id: str, *, ranked: list[int] | None, compatible: list[int] | None) -> None:
    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    (s4_out / "compatible_set.json").write_text(json.dumps({
        "event_id": run_id,
        "metric": "euclidean_log",
        "n_atlas": 5,
        "ranked_all": [{"geometry_id": f"g{i}"} for i in range(5)],
        "compatible_geometries": [{"geometry_id": "g0"}],
    }), encoding="utf-8")

    if ranked is not None and compatible is not None:
        s6b_out = runs_root / run_id / "s6b_information_geometry_ranked" / "outputs"
        s6b_out.mkdir(parents=True, exist_ok=True)
        (s6b_out / "ranked_geometries.json").write_text(json.dumps({
            "schema_version": "mvp_s6b_ranked_v1",
            "event_id": run_id,
            "atlas_id": "unknown",
            "n_atlas": 5,
            "ranked": [{"atlas_index": i, "score": 1.0 / (1 + j)} for j, i in enumerate(ranked)],
            "compatible": [{"atlas_index": i, "score": 1.0} for i in compatible],
            "compatibility_criterion": {"name": "test", "params": {}},
        }), encoding="utf-8")


def test_s5_aggregate_uses_s6b_and_warns_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1, 2], compatible=[1, 2])
    _mk_run(runs_root, "run_b", ranked=[1, 2, 3], compatible=[2, 3])
    _mk_run(runs_root, "run_c", ranked=None, compatible=None)

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_test",
        "--source-runs",
        "run_a,run_b,run_c",
        "--top-k",
        "3",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_test" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))

    assert len(payload["events"]) == 3
    assert all("ranked" in ev and "compatible" in ev for ev in payload["events"])
    assert payload["n_common_geometries"] >= 0
    assert payload["n_common_compatible"] >= 0
    assert any(w.startswith("MISSING_S6B_RANKED:") for w in payload["warnings"])
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" not in payload["warnings"]


def test_s5_aggregate_sets_no_common_warning_only_when_data_present(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0])
    _mk_run(runs_root, "run_b", ranked=[1, 2], compatible=[2])

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_test_2",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_test_2" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" in payload["warnings"]


def test_s6b_common_geometries_match_common_compatible_when_events_match() -> None:
    source_data = [
        {
            "run_id": "run_a",
            "event_id": "event_a",
            "s6b_present": True,
            "ranked_indices": [0, 1, 2],
            "compatible_indices": [0, 1, 2],
        },
        {
            "run_id": "run_b",
            "event_id": "event_b",
            "s6b_present": True,
            "ranked_indices": [0, 1, 2],
            "compatible_indices": [0, 1, 2],
        },
    ]

    agg = aggregate_compatible_sets(source_data, min_coverage=1.0, top_k=3)

    assert all(isinstance(x, int) for x in agg["common_geometries"])
    assert set(agg["common_compatible_geometries"]) == set(agg["common_geometries"])
