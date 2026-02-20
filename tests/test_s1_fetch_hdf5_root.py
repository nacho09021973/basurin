from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp import s1_fetch_strain


def test_autoresolve_hdf5_root_unique_files(tmp_path: Path) -> None:
    event_dir = tmp_path / "data" / "losc" / "GW150914"
    event_dir.mkdir(parents=True)
    h1 = event_dir / "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
    l1 = event_dir / "L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
    h1.write_bytes(b"h1")
    l1.write_bytes(b"l1")

    resolved = s1_fetch_strain._resolve_event_hdf5_or_die(
        hdf5_root=tmp_path / "data" / "losc",
        event_id="GW150914",
        detectors=["H1", "L1"],
    )

    assert resolved["H1"] == h1.resolve()
    assert resolved["L1"] == l1.resolve()


def test_autoresolve_hdf5_root_missing_files_has_actionable_error(tmp_path: Path) -> None:
    root = tmp_path / "data" / "losc"
    (root / "GW170104").mkdir(parents=True)

    with pytest.raises(ValueError) as exc:
        s1_fetch_strain._resolve_event_hdf5_or_die(
            hdf5_root=root,
            event_id="GW170104",
            detectors=["H1", "L1"],
        )

    msg = str(exc.value)
    assert str(root / "GW170104") in msg
    assert "*H1*.hdf5" in msg
    assert "*L1*.hdf5" in msg
    assert "find" in msg
    assert "--local-hdf5" in msg


def test_main_does_not_write_outside_runs_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    external_root = tmp_path / "data" / "losc"
    event_dir = external_root / "GW150914"
    event_dir.mkdir(parents=True)
    (event_dir / "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5").write_bytes(b"h1")
    (event_dir / "L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5").write_bytes(b"l1")

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    monkeypatch.setattr(
        s1_fetch_strain,
        "_load_local_hdf5",
        lambda _path: (np.array([0.0, 1.0]), 4096.0, 1126259462.0, "stub"),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            "run_det",
            "--event-id",
            "GW150914",
            "--detectors",
            "H1,L1",
            "--duration-s",
            "2",
            "--hdf5-root",
            str(external_root),
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.stat().st_size >= 0]
    outside_runs = [p for p in written if runs_root not in p.parents and p != runs_root]
    outside_runs = [p for p in outside_runs if not str(p).startswith(str(external_root))]
    assert outside_runs == []
