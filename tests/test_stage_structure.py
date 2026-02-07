# tests/test_stage_structure.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


RUN_ID = os.environ.get("BASURIN_RUN", "2026-02-03__REAL_GW150914_FETCH")
OUT_ROOT = Path(os.environ.get("BASURIN_OUT_ROOT", "runs"))


def _stage_dir(run_id: str, stage_name: str) -> Path:
    return OUT_ROOT / run_id / stage_name


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_stage_structure(stage_dir: Path) -> tuple[Path, Path, Path]:
    manifest = stage_dir / "manifest.json"
    summary = stage_dir / "stage_summary.json"
    outputs = stage_dir / "outputs"
    assert manifest.exists(), f"missing: {manifest}"
    assert summary.exists(), f"missing: {summary}"
    assert outputs.exists() and outputs.is_dir(), f"missing dir: {outputs}"
    return manifest, summary, outputs


def _assert_manifest_relative_to_stage_dir(stage_dir: Path, manifest_path: Path) -> None:
    m = _load_json(manifest_path)
    assert isinstance(m, dict) and m, "manifest must be a non-empty dict"
    for k, rel in m.items():
        assert isinstance(k, str) and k, "manifest keys must be non-empty strings"
        assert isinstance(rel, str) and rel, "manifest values must be non-empty strings"
        p = Path(rel)

        # Contract: stored path must be relative to stage_dir (not absolute).
        assert not p.is_absolute(), f"manifest path must be relative to stage_dir: {k} -> {rel}"

        # Contract: must resolve under stage_dir (not under run root or elsewhere).
        abs_p = stage_dir / p
        assert abs_p.exists(), f"manifest points to missing file: {k} -> {abs_p}"

        # Stronger: should normally live under outputs/ (allow exceptions if ever needed).
        # If you want strictness, uncomment the next line.
        # assert abs_p.is_relative_to(stage_dir / "outputs"), f"{k} not under outputs/: {abs_p}"


def _assert_stage_summary_inputs_have_hashes(summary_path: Path) -> None:
    s = _load_json(summary_path)
    assert isinstance(s, dict), "stage_summary must be a dict"
    assert "inputs" in s and isinstance(s["inputs"], dict), "stage_summary.inputs must be a dict"
    for name, spec in s["inputs"].items():
        assert isinstance(spec, dict), f"inputs.{name} must be a dict"
        assert "path" in spec and isinstance(spec["path"], str) and spec["path"], f"inputs.{name}.path missing"
        assert "sha256" in spec and isinstance(spec["sha256"], str) and spec["sha256"], f"inputs.{name}.sha256 missing"


# -----------------------------
# ringdown_featuremap_v0 checks
# -----------------------------

@pytest.mark.skipif(
    not _stage_dir(RUN_ID, "ringdown_featuremap_v0").exists(),
    reason="integration test: stage dir missing (run not present)",
)
def test_ringdown_featuremap_v0_structure_and_manifest():
    stage_dir = _stage_dir(RUN_ID, "ringdown_featuremap_v0")
    manifest, summary, outputs = _assert_stage_structure(stage_dir)
    _assert_manifest_relative_to_stage_dir(stage_dir, manifest)
    _assert_stage_summary_inputs_have_hashes(summary)

    out = outputs / "mapped_features.json"
    assert out.exists(), f"missing output: {out}"
    payload = _load_json(out)
    assert payload.get("schema_version") == "ringdown_featuremap_v0"
    assert isinstance(payload.get("cases"), list) and len(payload["cases"]) >= 1


# --------------------------
# geometry_select_v0 checks
# --------------------------

@pytest.mark.skipif(
    not _stage_dir(RUN_ID, "geometry_select_v0").exists(),
    reason="integration test: stage dir missing (run not present)",
)
def test_geometry_select_v0_structure_and_manifest():
    stage_dir = _stage_dir(RUN_ID, "geometry_select_v0")
    manifest, summary, outputs = _assert_stage_structure(stage_dir)
    _assert_manifest_relative_to_stage_dir(stage_dir, manifest)
    _assert_stage_summary_inputs_have_hashes(summary)

    out = outputs / "geometry_ranking.json"
    assert out.exists(), f"missing output: {out}"
    payload = _load_json(out)
    assert payload.get("schema_version") == "geometry_select_v0"
    assert isinstance(payload.get("rankings"), list) and len(payload["rankings"]) >= 1
