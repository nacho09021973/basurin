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
    s6_out = runs_root / run_id / "s6_information_geometry" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    s6_out.mkdir(parents=True, exist_ok=True)

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
