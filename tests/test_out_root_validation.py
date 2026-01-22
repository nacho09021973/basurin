import pytest

from basurin_io import resolve_out_root, validate_run_id


def test_resolve_out_root_accepts_runs(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_out_root("runs") == runs.resolve()


def test_resolve_out_root_accepts_subdir(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    subdir = runs / "bridge"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert resolve_out_root("runs/bridge") == subdir.resolve()


def test_resolve_out_root_rejects_outside(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        resolve_out_root("../outside")


def test_validate_run_id_rejects_traversal(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.chdir(tmp_path)
    out_root = resolve_out_root("runs")
    with pytest.raises(ValueError):
        validate_run_id("../escape", out_root)
    with pytest.raises(ValueError):
        validate_run_id("nested/../escape", out_root)
    validate_run_id("ok_run", out_root)
