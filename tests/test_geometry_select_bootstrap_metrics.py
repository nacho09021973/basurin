from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _prepare_run(tmp_path: Path, run_id: str) -> Path:
    run_root = tmp_path / "runs" / run_id
    (run_root / "RUN_VALID").mkdir(parents=True)
    (run_root / "RUN_VALID" / "verdict.json").write_text(
        json.dumps({"version": "run_valid.v1", "verdict": "PASS"}),
        encoding="utf-8",
    )

    inputs_dir = run_root / "inputs"
    inputs_dir.mkdir(parents=True)
    atlas = {
        "version": "atlas.v0",
        "geometries": [
            {"geometry_index": i, "M2_0": 4.0, "r1": float(r), "L": 1.0, "delta": None}
            for i, r in enumerate([1.05, 1.08, 1.12, 1.2, 1.35, 1.5])
        ],
    }
    (inputs_dir / "atlas.json").write_text(json.dumps(atlas, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "stages/stage_ringdown_synth.py",
            "--root",
            str(tmp_path),
            "--run",
            run_id,
            "--atlas-json",
            f"runs/{run_id}/inputs/atlas.json",
            "--sigma-rel",
            "0",
            "--seed-base",
            "12345",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "stages/stage_featuremap_v0.py",
            "--root",
            str(tmp_path),
            "--run",
            run_id,
            "--synthetic-events-json",
            f"runs/{run_id}/ringdown_synth/outputs/synthetic_events.json",
        ],
        check=True,
    )
    return run_root


def _run_selector(tmp_path: Path, run_id: str, bootstrap_k: int) -> dict:
    subprocess.run(
        [
            sys.executable,
            "stages/stage_geometry_select_v0.py",
            "--root",
            str(tmp_path),
            "--run",
            run_id,
            "--atlas-json",
            f"runs/{run_id}/inputs/atlas.json",
            "--mapped-features-json",
            f"runs/{run_id}/experiment/ringdown/featuremap_v0/outputs/mapped_features.json",
            "--topk",
            "3",
            "--acc-top1-threshold",
            "0.0",
            "--acc-topk-threshold",
            "0.0",
            "--bootstrap-k",
            str(bootstrap_k),
            "--bootstrap-seed",
            "123",
        ],
        check=True,
    )
    out_path = tmp_path / "runs" / run_id / "experiment" / "ringdown" / "geometry_select_v0" / "outputs" / "geometry_ranking.json"
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_geometry_select_bootstrap_metrics_deterministic(tmp_path: Path) -> None:
    run_id = "RUN_BOOTSTRAP"
    _prepare_run(tmp_path, run_id)

    first = _run_selector(tmp_path, run_id, bootstrap_k=10)
    second = _run_selector(tmp_path, run_id, bootstrap_k=10)

    assert "bootstrap" in first
    assert set(first["bootstrap"].keys()) == {"k", "seed", "winner_stability", "overlap_topk", "rank_variance_true"}
    assert first["bootstrap"] == second["bootstrap"]


def test_geometry_select_bootstrap_zero_keeps_schema_compatible(tmp_path: Path) -> None:
    run_id = "RUN_BOOTSTRAP_ZERO"
    _prepare_run(tmp_path, run_id)

    out = _run_selector(tmp_path, run_id, bootstrap_k=0)
    assert "bootstrap" not in out
