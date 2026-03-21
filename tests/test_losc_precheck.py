from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "losc_precheck.py"


def _run(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_losc_precheck_pass(tmp_path: Path) -> None:
    event_dir = tmp_path / "losc" / "GW150914"
    event_dir.mkdir(parents=True)
    (event_dir / "H-H1_GWOSC.hdf5").write_bytes(b"h1")
    (event_dir / "L-L1_GWOSC.h5").write_bytes(b"l1")

    proc = _run(["--event-id", "GW150914", "--losc-root", str(tmp_path / "losc")])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "LOSC_ROOT_EFFECTIVE=" in out
    assert "EVENT_DIR=" in out
    assert "h5_count=2" in out
    assert "match_count_H1=1" in out
    assert "match_count_L1=1" in out
    assert "match_count_V1=0" in out


def test_losc_precheck_fail_includes_actionable_diagnostics(tmp_path: Path) -> None:
    event_dir = tmp_path / "losc" / "GW190521"
    event_dir.mkdir(parents=True)
    (event_dir / "weird_name_a.h5").write_bytes(b"x")

    proc = _run(["--event-id", "GW190521", "--losc-root", str(tmp_path / "losc")])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 2, out
    assert "LOSC_ROOT_EFFECTIVE=" in out
    assert "EVENT_DIR=" in out
    assert "h5_count=1" in out
    assert "match_count_H1=0" in out
    assert "match_count_L1=0" in out
    assert "match_count_V1=0" in out
    assert "rama B" in out


def test_losc_precheck_passes_with_l1_v1_when_h1_is_missing(tmp_path: Path) -> None:
    event_dir = tmp_path / "losc" / "GW200112_155838"
    event_dir.mkdir(parents=True)
    (event_dir / "L-L1_GWOSC.hdf5").write_bytes(b"l1")
    (event_dir / "V-V1_GWOSC.h5").write_bytes(b"v1")

    proc = _run(["--event-id", "GW200112_155838", "--losc-root", str(tmp_path / "losc")])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "match_count_H1=0" in out
    assert "match_count_L1=1" in out
    assert "match_count_V1=1" in out


def test_losc_precheck_finds_hdf5_in_detector_subdirectory(tmp_path: Path) -> None:
    """rglob discovers HDF5 files placed inside detector sub-directories."""
    losc_root = tmp_path / "losc"
    h1_dir = losc_root / "GW170814" / "H1"
    l1_dir = losc_root / "GW170814" / "L1"
    h1_dir.mkdir(parents=True)
    l1_dir.mkdir(parents=True)
    (h1_dir / "H-H1_GWOSC.hdf5").write_bytes(b"h1")
    (l1_dir / "L-L1_GWOSC.h5").write_bytes(b"l1")

    proc = _run(["--event-id", "GW170814", "--losc-root", str(losc_root)])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "h5_count=2" in out
    assert "match_count_H1=1" in out
    assert "match_count_L1=1" in out


def test_losc_precheck_subprocess_is_read_only_and_cwd_independent(tmp_path: Path) -> None:
    losc_root = tmp_path / "external" / "losc"
    event_dir = losc_root / "GW170104"
    event_dir.mkdir(parents=True)
    (event_dir / "H1.h5").write_bytes(b"h1")
    (event_dir / "L1.hdf5").write_bytes(b"l1")

    run_cwd = tmp_path / "cwd"
    run_cwd.mkdir()

    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    proc = _run(["--event-id", "GW170104", "--losc-root", str(losc_root)], cwd=run_cwd)
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert before == after


def test_losc_precheck_finds_hdf5_recursively(tmp_path: Path) -> None:
    event_dir = tmp_path / "losc" / "GW150914"
    (event_dir / "H1").mkdir(parents=True)
    (event_dir / "L1").mkdir(parents=True)
    (event_dir / "H1" / "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5").write_bytes(b"h1")
    (event_dir / "L1" / "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5").write_bytes(b"l1")

    proc = _run(["--event-id", "GW150914", "--losc-root", str(tmp_path / "losc")])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "h5_count=2" in out
    assert "match_count_H1=1" in out
    assert "match_count_L1=1" in out


def test_losc_precheck_keeps_root_level_matching(tmp_path: Path) -> None:
    event_dir = tmp_path / "losc" / "GW150914"
    event_dir.mkdir(parents=True)
    (event_dir / "H-H1_GWOSC.hdf5").write_bytes(b"h1")
    (event_dir / "L-L1_GWOSC.hdf5").write_bytes(b"l1")

    proc = _run(["--event-id", "GW150914", "--losc-root", str(tmp_path / "losc")])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "h5_count=2" in out
    assert "match_count_H1=1" in out
    assert "match_count_L1=1" in out


LIST_SCRIPT = REPO_ROOT / "tools" / "list_losc_events.py"


def _run_list(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(LIST_SCRIPT), *args],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_list_losc_events_returns_sorted_visible_event_dirs(tmp_path: Path) -> None:
    losc_root = tmp_path / "losc"
    (losc_root / "GW190412").mkdir(parents=True)
    (losc_root / "GW150914").mkdir()
    (losc_root / "README.txt").write_text("not a dir", encoding="utf-8")

    proc = _run_list(["--losc-root", str(losc_root)])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "LOSC_ROOT_EFFECTIVE=" in out
    assert "event_dir_count=2" in out
    assert out.splitlines()[-2:] == ["GW150914", "GW190412"]


def test_list_losc_events_check_nonempty_fails_with_actionable_hint(tmp_path: Path) -> None:
    losc_root = tmp_path / "losc"
    losc_root.mkdir(parents=True)

    proc = _run_list(["--losc-root", str(losc_root), "--check-nonempty"])
    out = proc.stdout + proc.stderr

    assert proc.returncode == 2, out
    assert "event_dir_count=0" in out
    assert "data/losc/<EVENT_ID>/" in out
    assert "gw_events/strain/<EVENT_ID>/" in out
