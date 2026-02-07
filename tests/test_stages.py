from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_three_stage_ringdown_pipeline(tmp_path: Path) -> None:
    run_id = "RUN"
    run_root = tmp_path / "runs" / run_id
    (run_root / "RUN_VALID").mkdir(parents=True)
    (run_root / "RUN_VALID" / "verdict.json").write_text(
        json.dumps({"version": "run_valid.v1", "verdict": "PASS"}),
        encoding="utf-8",
    )

    inputs_dir = run_root / "inputs"
    inputs_dir.mkdir(parents=True)
    r1_list = np.linspace(1.05, 1.50, 16)
    atlas = {
        "version": "atlas.v0",
        "geometries": [
            {
                "geometry_index": i,
                "M2_0": 4.0,
                "r1": float(r1),
                "L": 1.0,
                "delta": None,
            }
            for i, r1 in enumerate(r1_list)
        ],
    }
    atlas_path = inputs_dir / "atlas.json"
    atlas_path.write_text(json.dumps(atlas, indent=2), encoding="utf-8")

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
            "1.0",
            "--acc-topk-threshold",
            "1.0",
        ],
        check=True,
    )

    stage_dirs = [
        run_root / "ringdown_synth",
        run_root / "experiment" / "ringdown" / "featuremap_v0",
        run_root / "experiment" / "ringdown" / "geometry_select_v0",
    ]
    outputs = [
        "outputs/synthetic_events.json",
        "outputs/mapped_features.json",
        "outputs/geometry_ranking.json",
    ]

    for stage_dir, out_rel in zip(stage_dirs, outputs):
        manifest_path = stage_dir / "manifest.json"
        summary_path = stage_dir / "stage_summary.json"
        out_path = stage_dir / out_rel
        assert manifest_path.exists()
        assert summary_path.exists()
        assert out_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["version"] == "manifest.v1"
        assert "manifest.json" in manifest["artifacts"]
        assert "stage_summary.json" in manifest["artifacts"]
        assert out_rel in manifest["artifacts"]

        assert len(manifest["artifacts"]["manifest.json"]["sha256"]) == 64
        assert manifest["artifacts"]["stage_summary.json"]["sha256"] == _sha(summary_path)
        assert manifest["artifacts"][out_rel]["sha256"] == _sha(out_path)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["model"]["family"] == "phi_phenomenological_v0"

    ranking = json.loads(
        (run_root / "experiment" / "ringdown" / "geometry_select_v0" / "outputs" / "geometry_ranking.json").read_text(
            encoding="utf-8"
        )
    )
    assert ranking["accuracy_top1"] == 1.0
    assert ranking["verdict"] == "PASS"
