"""Unit tests for basurin_io.py — direct coverage of public functions.

Gaps addressed (from test_coverage_proposal.md Gap 1):
  - validate_run_id:   empty, 128-char boundary, 129-char (fail), path-traversal chars
  - sha256_file:       known hash assertion
  - require_run_valid: missing file, wrong verdict, malformed JSON, PASS
  - resolve_out_root:  BASURIN_RUNS_ROOT env var vs cwd/runs default
  - write_manifest:    schema keys present, hashes populated for existing files
  - _coerce_paths:     Path→str, nested dict, list
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import (
    _coerce_paths,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
)


# ---------------------------------------------------------------------------
# validate_run_id
# ---------------------------------------------------------------------------


def test_validate_run_id_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_run_id("", Path("/tmp"))


def test_validate_run_id_whitespace_only_raises() -> None:
    # Whitespace characters are not in [A-Za-z0-9._-], so "invalid characters" is raised
    with pytest.raises(ValueError):
        validate_run_id("   ", Path("/tmp"))


def test_validate_run_id_at_boundary_128_passes() -> None:
    # Exactly 128 characters: should not raise
    validate_run_id("a" * 128, Path("/tmp"))


def test_validate_run_id_too_long_129_raises() -> None:
    with pytest.raises(ValueError, match="too long"):
        validate_run_id("a" * 129, Path("/tmp"))


def test_validate_run_id_path_traversal_slash_raises() -> None:
    with pytest.raises(ValueError):
        validate_run_id("../escape", Path("/tmp"))


def test_validate_run_id_path_traversal_double_dot_raises() -> None:
    with pytest.raises(ValueError):
        validate_run_id("foo/../bar", Path("/tmp"))


def test_validate_run_id_unicode_raises() -> None:
    with pytest.raises(ValueError):
        validate_run_id("GW150914_ñ", Path("/tmp"))


def test_validate_run_id_valid_alphanumeric_passes() -> None:
    validate_run_id("GW150914", Path("/tmp"))


def test_validate_run_id_valid_with_dots_dashes_underscores_passes() -> None:
    validate_run_id("mvp_GW150914_20240101T120000Z", Path("/tmp"))


def test_validate_run_id_single_char_passes() -> None:
    validate_run_id("a", Path("/tmp"))


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_known_hash(tmp_path: Path) -> None:
    content = b'{"a":1}'
    f = tmp_path / "test.json"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_large_content(tmp_path: Path) -> None:
    # Verifies chunked reading works correctly (> 64 KiB)
    content = b"x" * (65_536 * 2 + 17)
    f = tmp_path / "large.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_returns_hex_string(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_bytes(b"hello")
    result = sha256_file(f)
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# require_run_valid
# ---------------------------------------------------------------------------


def test_require_run_valid_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="RUN_VALID"):
        require_run_valid(tmp_path, "nonexistent_run")


def test_require_run_valid_wrong_verdict_raises(tmp_path: Path) -> None:
    run_id = "test_run"
    rv_dir = tmp_path / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text('{"verdict": "FAIL"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="not PASS"):
        require_run_valid(tmp_path, run_id)


def test_require_run_valid_malformed_json_raises(tmp_path: Path) -> None:
    run_id = "test_run"
    rv_dir = tmp_path / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text("NOT JSON {{{", encoding="utf-8")
    with pytest.raises(Exception):
        require_run_valid(tmp_path, run_id)


def test_require_run_valid_null_verdict_raises(tmp_path: Path) -> None:
    run_id = "test_run"
    rv_dir = tmp_path / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text('{"verdict": null}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        require_run_valid(tmp_path, run_id)


def test_require_run_valid_pass_succeeds(tmp_path: Path) -> None:
    run_id = "test_run"
    rv_dir = tmp_path / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text('{"verdict": "PASS"}', encoding="utf-8")
    # Should not raise
    require_run_valid(tmp_path, run_id)


def test_require_run_valid_missing_key_raises(tmp_path: Path) -> None:
    run_id = "test_run"
    rv_dir = tmp_path / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True)
    (rv_dir / "verdict.json").write_text('{"status": "ok"}', encoding="utf-8")
    # Missing "verdict" key → verdict is None → not PASS
    with pytest.raises(RuntimeError):
        require_run_valid(tmp_path, run_id)


# ---------------------------------------------------------------------------
# resolve_out_root
# ---------------------------------------------------------------------------


def test_resolve_out_root_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_root = tmp_path / "custom_runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(custom_root))
    result = resolve_out_root("runs")
    assert result == custom_root.resolve()
    assert result.is_dir()


def test_resolve_out_root_uses_cwd_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASURIN_RUNS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    result = resolve_out_root("runs")
    assert result == (tmp_path / "runs").resolve()
    assert result.is_dir()


def test_resolve_out_root_creates_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    new_root = tmp_path / "new" / "nested" / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(new_root))
    result = resolve_out_root("runs")
    assert result.is_dir()


def test_resolve_out_root_custom_root_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASURIN_RUNS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    result = resolve_out_root("my_output")
    assert result == (tmp_path / "my_output").resolve()


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------


def test_write_manifest_schema_keys_present(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    artifact = stage_dir / "output.json"
    artifact.write_text('{"result": 1}', encoding="utf-8")

    manifest_path = write_manifest(stage_dir, {"output": artifact})

    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "mvp_manifest_v1"
    assert "created" in data
    assert "artifacts" in data
    assert "hashes" in data


def test_write_manifest_hashes_existing_file(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    content = b'{"x": 42}'
    artifact = stage_dir / "data.json"
    artifact.write_bytes(content)

    write_manifest(stage_dir, {"data": artifact})
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))

    expected_hash = hashlib.sha256(content).hexdigest()
    assert manifest["hashes"]["data"] == expected_hash


def test_write_manifest_missing_file_has_no_hash(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    nonexistent = stage_dir / "missing.json"

    write_manifest(stage_dir, {"missing": nonexistent})
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))

    # Non-existent file: no hash entry
    assert "missing" not in manifest["hashes"]


def test_write_manifest_extra_fields_merged(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    write_manifest(stage_dir, {}, extra={"custom_key": "custom_value"})
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["custom_key"] == "custom_value"


# ---------------------------------------------------------------------------
# _coerce_paths
# ---------------------------------------------------------------------------


def test_coerce_paths_converts_path_to_str() -> None:
    p = Path("/some/path/file.json")
    result = _coerce_paths(p)
    assert result == str(p)
    assert isinstance(result, str)


def test_coerce_paths_nested_dict() -> None:
    data = {"key": Path("/a/b"), "nested": {"inner": Path("/c/d")}}
    result = _coerce_paths(data)
    assert result == {"key": "/a/b", "nested": {"inner": "/c/d"}}


def test_coerce_paths_list_of_paths() -> None:
    data = [Path("/x"), Path("/y"), "already_str"]
    result = _coerce_paths(data)
    assert result == ["/x", "/y", "already_str"]


def test_coerce_paths_tuple_treated_as_list() -> None:
    data = (Path("/a"), 42)
    result = _coerce_paths(data)
    assert result == ["/a", 42]


def test_coerce_paths_non_path_scalars_unchanged() -> None:
    assert _coerce_paths(42) == 42
    assert _coerce_paths("hello") == "hello"
    assert _coerce_paths(3.14) == 3.14
    assert _coerce_paths(None) is None


def test_coerce_paths_mixed_nested_structure() -> None:
    data = {
        "a": [Path("/p1"), {"b": Path("/p2")}],
        "c": "unchanged",
    }
    result = _coerce_paths(data)
    assert result["a"][0] == "/p1"
    assert result["a"][1]["b"] == "/p2"
    assert result["c"] == "unchanged"
