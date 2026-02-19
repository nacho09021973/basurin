from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")
from mvp import s1_fetch_strain


def test_parse_local_hdf5_accepts_glob_single_match(tmp_path: Path):
    h5 = tmp_path / "H-H1_GWOSC_4KHZ.h5"
    h5.write_bytes(b"dummy")

    resolved = s1_fetch_strain._resolve_local_hdf5_mappings(
        [f"H1={tmp_path / 'H-H1_GWOSC_*.h5'}"],
        event_id="GW150914",
    )

    assert resolved["H1"] == h5.resolve()


def test_parse_local_hdf5_glob_ambiguous_fails(tmp_path: Path):
    (tmp_path / "H-H1_GWOSC_a.h5").write_bytes(b"a")
    (tmp_path / "H-H1_GWOSC_b.h5").write_bytes(b"b")

    with pytest.raises(ValueError) as exc:
        s1_fetch_strain._resolve_local_hdf5_mappings(
            [f"H1={tmp_path / 'H-H1_GWOSC_*.h5'}"],
            event_id="GW150914",
        )

    msg = str(exc.value)
    assert "Ambiguous HDF5" in msg
    assert "Candidates" in msg


def test_offline_disallows_gwosc(monkeypatch, tmp_path: Path):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def _boom(*_args, **_kwargs):
        raise AssertionError("GWOSC fetch must not be called in offline mode")

    monkeypatch.setattr(s1_fetch_strain, "_fetch_via_gwpy", _boom)
    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", _boom)

    with pytest.raises(SystemExit) as exc:
        monkeypatch.setattr(
            "sys.argv",
            [
                "s1_fetch_strain.py",
                "--run",
                "offline_run",
                "--event-id",
                "GW150914",
                "--detectors",
                "H1",
                "--duration-s",
                "4",
                "--offline",
            ],
        )
        s1_fetch_strain.main()

    assert exc.value.code == 2


def test_reuse_if_present_skips_fetch(monkeypatch, tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_id = "reuse_run"
    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    np.savez(
        out / "strain.npz",
        sample_rate_hz=np.float64(4096.0),
        gps_start=np.float64(0.0),
        duration_s=np.float64(4.0),
        H1=arr,
    )
    prov = {
        "event_id": "GW150914",
        "detectors": ["H1"],
        "duration_s": 4.0,
        "sha256_per_detector": {"H1": s1_fetch_strain._sha256_array(arr)},
    }
    (out / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise AssertionError("fetch must not be called when reuse is valid")

    monkeypatch.setattr(s1_fetch_strain, "_fetch_via_gwpy", _boom)
    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", _boom)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            run_id,
            "--event-id",
            "GW150914",
            "--detectors",
            "H1",
            "--duration-s",
            "4",
            "--reuse-if-present",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0
