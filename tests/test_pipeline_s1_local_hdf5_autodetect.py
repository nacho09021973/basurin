from __future__ import annotations

import os
from pathlib import Path


def _extract_local_hdf5_paths(args: list[str]) -> list[str]:
    paths: list[str] = []
    for idx, arg in enumerate(args):
        if arg == "--local-hdf5":
            det, path = args[idx + 1].split("=", 1)
            assert det in {"H1", "L1", "V1"}
            paths.append(path)
    return paths


def test_build_s1_fetch_args_autodetects_canonical_losc_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    event_id = "GW150914"
    losc_dir = tmp_path / "data" / "losc" / event_id
    losc_dir.mkdir(parents=True)
    (losc_dir / "H1.h5").write_text("dummy")
    (losc_dir / "L1.h5").write_text("dummy")

    from mvp.pipeline import _build_s1_fetch_args

    args = _build_s1_fetch_args(
        run_id="run123",
        event_id=event_id,
        duration_s=32.0,
        synthetic=False,
        reuse_strain=False,
        local_hdf5=None,
        offline=False,
    )

    assert args.count("--local-hdf5") == 2
    assert f"H1={(losc_dir / 'H1.h5').resolve().as_posix()}" in args
    assert f"L1={(losc_dir / 'L1.h5').resolve().as_posix()}" in args
    assert all(os.path.isabs(path) for path in _extract_local_hdf5_paths(args))


def test_build_s1_fetch_args_respects_basurin_losc_root(monkeypatch, tmp_path):
    event_id = "GW150914"
    losc_root = tmp_path / "external_losc"
    event_dir = losc_root / event_id
    event_dir.mkdir(parents=True)
    (event_dir / "H1.hdf5").write_text("dummy")
    (event_dir / "L1.h5").write_text("dummy")

    monkeypatch.setenv("BASURIN_LOSC_ROOT", str(losc_root))

    from mvp.pipeline import _build_s1_fetch_args

    args = _build_s1_fetch_args(
        run_id="run123",
        event_id=event_id,
        duration_s=32.0,
        synthetic=False,
        reuse_strain=False,
        local_hdf5=None,
        offline=False,
    )

    assert args.count("--local-hdf5") == 2
    assert f"H1={Path(losc_root / event_id / 'H1.hdf5').resolve().as_posix()}" in args
    assert f"L1={Path(losc_root / event_id / 'L1.h5').resolve().as_posix()}" in args
    assert all(os.path.isabs(path) for path in _extract_local_hdf5_paths(args))


def test_build_s0_oracle_args_offline_autodetects_h1_l1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    event_id = "GWX"
    losc_dir = tmp_path / "data" / "losc" / event_id
    losc_dir.mkdir(parents=True)

    # H1: must pick largest (then by name)
    (losc_dir / "aaa_H1_small.hdf5").write_bytes(b"1")
    (losc_dir / "zzz_H1_big.h5").write_bytes(b"12345")
    # L1: must pick largest (then by name)
    (losc_dir / "bbb_L1_small.h5").write_bytes(b"12")
    (losc_dir / "aaa_L1_big.hdf5").write_bytes(b"1234")

    from mvp.pipeline import _build_s0_oracle_args

    args = _build_s0_oracle_args(
        run_id="run123",
        event_id=event_id,
        local_hdf5=None,
        offline=True,
    )

    assert "--require-offline" in args
    assert args.count("--local-hdf5") == 2
    assert f"H1={(losc_dir / 'zzz_H1_big.h5').resolve().as_posix()}" in args
    assert f"L1={(losc_dir / 'aaa_L1_big.hdf5').resolve().as_posix()}" in args
    assert all(os.path.isabs(path) for path in _extract_local_hdf5_paths(args))


def test_build_s1_fetch_args_falls_back_to_l1_v1_when_h1_is_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    event_id = "GW200112_155838"
    losc_dir = tmp_path / "data" / "losc" / event_id
    losc_dir.mkdir(parents=True)
    (losc_dir / "L-L1_GWOSC_4KHZ_R1-1262877888-4096.hdf5").write_text("dummy")
    (losc_dir / "V-V1_GWOSC_4KHZ_R1-1262877888-4096.hdf5").write_text("dummy")

    from mvp.pipeline import _build_s1_fetch_args

    args = _build_s1_fetch_args(
        run_id="run123",
        event_id=event_id,
        duration_s=32.0,
        synthetic=False,
        reuse_strain=False,
        local_hdf5=None,
        offline=False,
    )

    assert args.count("--local-hdf5") == 2
    assert f"L1={(losc_dir / 'L-L1_GWOSC_4KHZ_R1-1262877888-4096.hdf5').resolve().as_posix()}" in args
    assert f"V1={(losc_dir / 'V-V1_GWOSC_4KHZ_R1-1262877888-4096.hdf5').resolve().as_posix()}" in args


def test_build_s0_oracle_args_offline_falls_back_to_l1_v1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    event_id = "GW200112_155838"
    losc_dir = tmp_path / "data" / "losc" / event_id
    losc_dir.mkdir(parents=True)
    (losc_dir / "L1.h5").write_text("dummy")
    (losc_dir / "V1.h5").write_text("dummy")

    from mvp.pipeline import _build_s0_oracle_args

    args = _build_s0_oracle_args(
        run_id="run123",
        event_id=event_id,
        local_hdf5=None,
        offline=True,
    )

    assert "--require-offline" in args
    assert args.count("--local-hdf5") == 2
    assert f"L1={(losc_dir / 'L1.h5').resolve().as_posix()}" in args
    assert f"V1={(losc_dir / 'V1.h5').resolve().as_posix()}" in args
