"""basurin_io – Deterministic IO helpers for the BASURIN MVP pipeline.

All writes go under runs/<run_id>/ relative to the repo root (or the
directory indicated by the BASURIN_RUNS_ROOT environment variable).
JSON writes are atomic (write-to-tmp then os.replace).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_RUN_ID_MAX_LEN = 128


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def resolve_out_root(root_name: str = "runs") -> Path:
    """Return the absolute path of the output root directory.

    Uses the BASURIN_RUNS_ROOT env var when set, otherwise resolves
    *root_name* relative to the current working directory.
    The directory is created if it does not exist.
    """
    env = os.environ.get("BASURIN_RUNS_ROOT")
    if env:
        root = Path(env).resolve()
    else:
        root = Path.cwd() / root_name
    assert_no_symlink_ancestors(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def assert_no_symlink_ancestors(path: Path, *, stop_at: Path | None = None) -> None:
    """Fail if *path* or any existing ancestor is a symlink.

    Traverses from *path* up to *stop_at* (inclusive) when provided,
    otherwise to the filesystem root.
    """
    current = Path(os.path.abspath(path))
    boundary = Path(os.path.abspath(stop_at)) if stop_at is not None else None

    while True:
        if current.is_symlink():
            raise RuntimeError(f"symlink ancestor forbidden: {current}")
        if boundary is not None and current == boundary:
            break
        if current.parent == current:
            break
        current = current.parent


def validate_run_id(run_id: str, out_root: Path) -> None:
    """Validate *run_id* format (no filesystem side-effects).

    Rules: non-empty, <= 128 chars, alphanumeric plus ``_``, ``-``, ``.``.
    Raises ``ValueError`` on invalid input.
    """
    if not run_id:
        raise ValueError("run_id must not be empty")
    if len(run_id) > _RUN_ID_MAX_LEN:
        raise ValueError(
            f"run_id too long ({len(run_id)} > {_RUN_ID_MAX_LEN}): {run_id!r}"
        )
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"run_id contains invalid characters: {run_id!r}  "
            f"(allowed: A-Z a-z 0-9 . _ -)"
        )


def require_run_valid(out_root: Path, run_id: str) -> None:
    """Assert that ``runs/<run_id>/RUN_VALID/verdict.json`` exists and is PASS.

    Raises ``FileNotFoundError`` if the file is missing and ``RuntimeError``
    if the verdict is not ``"PASS"``.
    """
    verdict_path = out_root / run_id / "RUN_VALID" / "verdict.json"
    if not verdict_path.exists():
        raise FileNotFoundError(
            f"RUN_VALID verdict not found: {verdict_path}"
        )
    data = json.loads(verdict_path.read_text(encoding="utf-8"))
    if data.get("verdict") != "PASS":
        raise RuntimeError(
            f"RUN_VALID verdict is not PASS: {data.get('verdict')!r}"
        )


def ensure_stage_dirs(
    run_id: str,
    stage_name: str,
    base_dir: Path,
) -> tuple[Path, Path]:
    """Create and return ``(stage_dir, outputs_dir)`` under *base_dir*.

    Creates::
        <base_dir>/<run_id>/<stage_name>/
        <base_dir>/<run_id>/<stage_name>/outputs/
    """
    stage_dir = base_dir / run_id / stage_name
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir, outputs_dir


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65_536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# JSON writing (atomic)
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, data: Any) -> Path:
    """Atomically write *data* as pretty-printed JSON to *path*.

    Writes to a temporary file in the same directory then replaces, so
    readers never see a partial file.  Returns *path*.
    """
    path = Path(path)
    assert_no_symlink_ancestors(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Coerce Path values to str so json.dumps doesn't choke
    data = _coerce_paths(data)

    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, text.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        os.close(fd) if not _fd_closed(fd) else None
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_manifest(
    stage_dir: Path,
    artifacts: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` inside *stage_dir*.

    The manifest records schema version, creation timestamp, and a dict
    of artifact labels to their string representations (Paths → str).
    *extra* fields are merged at the top level.
    """
    artifact_strings = {k: str(v) for k, v in artifacts.items()}
    artifact_hashes: dict[str, str] = {}
    for label, value in artifacts.items():
        path = Path(value)
        if path.is_file():
            artifact_hashes[label] = sha256_file(path)

    payload: dict[str, Any] = {
        "schema_version": "mvp_manifest_v1",
        "created": utc_now_iso(),
        "artifacts": artifact_strings,
        "hashes": artifact_hashes,
    }
    if extra:
        payload.update(extra)
    return write_json_atomic(stage_dir / "manifest.json", payload)


def write_stage_summary(
    stage_dir: Path,
    summary: dict[str, Any],
) -> Path:
    """Write ``stage_summary.json`` inside *stage_dir*.  Returns the path."""
    return write_json_atomic(stage_dir / "stage_summary.json", summary)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _coerce_paths(obj: Any) -> Any:
    """Recursively convert Path instances to str for JSON serialisation."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _coerce_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_paths(v) for v in obj]
    return obj


def _fd_closed(fd: int) -> bool:
    """Return True if the file descriptor is already closed."""
    try:
        os.fstat(fd)
        return False
    except OSError:
        return True
