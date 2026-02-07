from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_edc_01_degeneracy_audit_smoke(tmp_path: Path) -> None:
    run_id = "RUN_EDC_01"
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
            for i, r in enumerate([1.02, 1.03, 1.05, 1.08, 1.12, 1.18, 1.25, 1.35, 1.5])
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
            "0.01",
            "--seed-base",
            "12345",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "experiment/ringdown/edc_01_degeneracy_audit.py",
            "--out-root",
            str(tmp_path / "runs"),
            "--run",
            run_id,
            "--k",
            "3",
            "--m-per-group",
            "2",
            "--n-cases-per-geom",
            "1",
            "--bootstrap-k",
            "5",
            "--bootstrap-seed",
            "123",
        ],
        check=True,
    )

    stage_dir = run_root / "experiment" / "ringdown" / "EDC_01__degeneracy_audit"
    out_path = stage_dir / "outputs" / "edc_results.json"
    manifest_path = stage_dir / "manifest.json"
    summary_path = stage_dir / "stage_summary.json"

    assert out_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["verdict"] in {
        "MODEL_MISSPECIFIED",
        "MODEL_HALLUCINATING",
        "DEGENERACY_INEVITABLE",
        "INCONCLUSIVE",
    }

    for info in payload["inputs"].values():
        assert not Path(info["path"]).is_absolute()
        assert len(info["sha256"]) == 64

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relpath in manifest["files"].values():
        assert not Path(relpath).is_absolute()
    assert all(len(v) == 64 for v in manifest["hashes"].values())

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for info in summary["inputs"].values():
        assert not Path(info["path"]).is_absolute()
        assert len(info["sha256"]) == 64
