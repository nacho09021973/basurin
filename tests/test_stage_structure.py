# tests/test_stage_structure.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


RUN_ID = os.environ.get("BASURIN_RUN", "2026-02-01__qnm_validation")
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
    assert isinstance(m, dict) and m, f"manifest must be a non-empty dict: {manifest_path}"
    # Manifest can include metadata entries that are not artifact paths.
    META_KEYS = {"created"}
    n_artifacts = 0
    for k, rel in m.items():
        assert isinstance(k, str) and k, f"manifest key must be a non-empty string in {manifest_path}: {k!r}"
        if k in META_KEYS:
            continue
        assert isinstance(rel, str) and rel, f"manifest value must be a non-empty string in {manifest_path}: {k}"
        p = Path(rel)

        # Contract: stored path must be relative to stage_dir (not absolute).
        assert not p.is_absolute(), f"manifest path must be relative to stage_dir {stage_dir}: {k} -> {rel}"

        # Contract: must resolve under stage_dir (not under run root or elsewhere).
        abs_p = stage_dir / p
        assert abs_p.exists(), f"manifest points to missing file from {manifest_path}: {k} -> {abs_p}"
        n_artifacts += 1

        # Stronger: should normally live under outputs/ (allow exceptions if ever needed).
        # If you want strictness, uncomment the next line.
        # assert abs_p.is_relative_to(stage_dir / "outputs"), f"{k} not under outputs/: {abs_p}"
    assert n_artifacts >= 1, f"manifest must contain at least one artifact entry: {manifest_path}"


def _assert_stage_summary_inputs_have_hashes(summary_path: Path) -> None:
    s = _load_json(summary_path)
    assert isinstance(s, dict), f"stage_summary must be a dict: {summary_path}"
    assert "inputs" in s and isinstance(s["inputs"], dict), f"stage_summary.inputs must be a dict: {summary_path}"
    for name, spec in s["inputs"].items():
        assert isinstance(spec, dict), f"inputs.{name} must be a dict in {summary_path}"
        assert "path" in spec and isinstance(spec["path"], str) and spec["path"], f"inputs.{name}.path missing in {summary_path}"
        assert "sha256" in spec and isinstance(spec["sha256"], str) and spec["sha256"], f"inputs.{name}.sha256 missing in {summary_path}"


# ----------------
# geometry checks
# ----------------

@pytest.mark.skipif(
    not _stage_dir(RUN_ID, "geometry").exists(),
    reason="integration test: stage dir missing (run not present)",
)
def test_geometry_structure_and_manifest():
    stage_dir = _stage_dir(RUN_ID, "geometry")
    manifest, summary, outputs = _assert_stage_structure(stage_dir)
    _assert_manifest_relative_to_stage_dir(stage_dir, manifest)
    _assert_stage_summary_inputs_have_hashes(summary)

    out = outputs / "ads_puro.h5"
    assert out.exists(), f"missing output for stage {stage_dir}: {out}"


# ----------------
# spectrum checks
# ----------------

@pytest.mark.skipif(
    not _stage_dir(RUN_ID, "spectrum").exists(),
    reason="integration test: stage dir missing (run not present)",
)
def test_spectrum_structure_and_manifest():
    stage_dir = _stage_dir(RUN_ID, "spectrum")
    manifest, summary, outputs = _assert_stage_structure(stage_dir)
    _assert_manifest_relative_to_stage_dir(stage_dir, manifest)
    _assert_stage_summary_inputs_have_hashes(summary)

    spectrum_out = outputs / "spectrum.h5"
    assert spectrum_out.exists(), f"missing output for stage {stage_dir}: {spectrum_out}"

    validation_out = outputs / "validation.json"
    assert validation_out.exists(), f"missing output for stage {stage_dir}: {validation_out}"
    payload = _load_json(validation_out)
    assert isinstance(payload, dict), f"validation output must be a dict JSON: {validation_out}"
