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

    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", lambda _event_id: 1126259462.0)

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


def test_match_hdf5_files_is_deterministic_and_filters_extensions(tmp_path: Path) -> None:
    event_dir = tmp_path / "GW150914"
    event_dir.mkdir()
    (event_dir / "b_L1.h5").write_bytes(b"1")
    (event_dir / "a_H1.hdf5").write_bytes(b"2")
    (event_dir / "c_V1.hdf5").write_bytes(b"3")
    (event_dir / "c_H1.txt").write_text("x", encoding="utf-8")

    matches_1 = s1_fetch_strain.match_hdf5_files(event_dir)
    matches_2 = s1_fetch_strain.match_hdf5_files(event_dir)

    assert [p.name for p in matches_1["all"]] == ["a_H1.hdf5", "b_L1.h5", "c_V1.hdf5"]
    assert [p.name for p in matches_1["H1"]] == ["a_H1.hdf5"]
    assert [p.name for p in matches_1["L1"]] == ["b_L1.h5"]
    assert [p.name for p in matches_1["V1"]] == ["c_V1.hdf5"]
    assert [p.name for p in matches_2["all"]] == [p.name for p in matches_1["all"]]


def test_match_hdf5_files_no_throw_on_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    matches = s1_fetch_strain.match_hdf5_files(missing)
    assert matches == {"all": [], "H1": [], "L1": [], "V1": []}


def test_match_hdf5_files_finds_files_in_nested_subdirectory(tmp_path: Path) -> None:
    """match_hdf5_files uses rglob and finds HDF5 in detector sub-directories."""
    event_dir = tmp_path / "GW170814"
    h1_dir = event_dir / "H1"
    l1_dir = event_dir / "L1"
    h1_dir.mkdir(parents=True)
    l1_dir.mkdir(parents=True)
    (h1_dir / "H-H1_GWOSC.hdf5").write_bytes(b"h1")
    (l1_dir / "L-L1_GWOSC.h5").write_bytes(b"l1")

    matches = s1_fetch_strain.match_hdf5_files(event_dir)

    assert len(matches["all"]) == 2
    assert len(matches["H1"]) == 1
    assert len(matches["L1"]) == 1
    assert matches["H1"][0].name == "H-H1_GWOSC.hdf5"
    assert matches["L1"][0].name == "L-L1_GWOSC.h5"


def test_autoresolve_hdf5_root_accepts_l1_v1_pair(tmp_path: Path) -> None:
    event_dir = tmp_path / "data" / "losc" / "GW200112_155838"
    event_dir.mkdir(parents=True)
    l1 = event_dir / "L-L1_GWOSC_4KHZ_R1-1262877888-4096.hdf5"
    v1 = event_dir / "V-V1_GWOSC_4KHZ_R1-1262877888-4096.hdf5"
    l1.write_bytes(b"l1")
    v1.write_bytes(b"v1")

    resolved = s1_fetch_strain._resolve_event_hdf5_or_die(
        hdf5_root=tmp_path / "data" / "losc",
        event_id="GW200112_155838",
        detectors=["L1", "V1"],
    )

    assert resolved["L1"] == l1.resolve()
    assert resolved["V1"] == v1.resolve()


def test_match_hdf5_files_finds_nested_files(tmp_path: Path) -> None:
    event_dir = tmp_path / "GW150914"
    (event_dir / "H1").mkdir(parents=True)
    (event_dir / "L1").mkdir(parents=True)
    (event_dir / "H1" / "a_H1.hdf5").write_bytes(b"1")
    (event_dir / "L1" / "b_L1.h5").write_bytes(b"2")

    matches = s1_fetch_strain.match_hdf5_files(event_dir)

    assert [p.name for p in matches["all"]] == ["a_H1.hdf5", "b_L1.h5"]
    assert [p.name for p in matches["H1"]] == ["a_H1.hdf5"]
    assert [p.name for p in matches["L1"]] == ["b_L1.h5"]
