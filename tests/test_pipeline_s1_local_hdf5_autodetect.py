from __future__ import annotations

from pathlib import Path


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
    assert "H1=data/losc/GW150914/H1.h5" in args
    assert "L1=data/losc/GW150914/L1.h5" in args


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
    assert f"H1={Path(losc_root / event_id / 'H1.hdf5').as_posix()}" in args
    assert f"L1={Path(losc_root / event_id / 'L1.h5').as_posix()}" in args
