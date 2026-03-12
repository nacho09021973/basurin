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


def test_sanitize_strain_array_interpolates_small_nonfinite_spans() -> None:
    strain = np.array([0.0, 1.0, np.nan, np.inf, 4.0, 5.0], dtype=np.float64)

    sanitized, details = s1_fetch_strain._sanitize_strain_array(
        strain,
        detector="H1",
        max_nonfinite_fraction=0.5,
    )

    assert np.isfinite(sanitized).all()
    assert sanitized.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert details["applied"] is True
    assert details["nonfinite_count_raw"] == 2
    assert details["method"] == "linear_interp_nonfinite"


def test_crop_local_strain_to_requested_window_uses_32s_target() -> None:
    strain = np.arange(20.0, dtype=np.float64)

    cropped, gps_start_out, details = s1_fetch_strain._crop_local_strain_to_requested_window(
        strain,
        detector="H1",
        sample_rate_hz=2.0,
        gps_start_local=100.0,
        gps_start_target=103.0,
        duration_s=4.0,
    )

    assert cropped.tolist() == pytest.approx([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    assert gps_start_out == pytest.approx(103.0)
    assert details["applied"] is True
    assert details["source_n_samples"] == 20
    assert details["output_n_samples"] == 8
    assert details["reason"] == "cropped_to_requested_window"


def test_reuse_if_present_rejects_cached_nonfinite_and_refetches_local(monkeypatch, tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_id = "reuse_nonfinite"
    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    cached = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    np.savez(
        out / "strain.npz",
        sample_rate_hz=np.float64(4096.0),
        gps_start=np.float64(0.0),
        duration_s=np.float64(4.0),
        H1=cached,
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "event_id": "GW150914",
                "detectors": ["H1"],
                "duration_s": 4.0,
                "source": "local_hdf5",
                "sha256_per_detector": {"H1": s1_fetch_strain._sha256_array(cached)},
                "local_input_sha256": {},
            }
        ),
        encoding="utf-8",
    )

    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    local_h1 = tmp_path / "H1_good.hdf5"
    local_h1.write_bytes(b"placeholder")

    def _fake_load(_path):
        return np.array([10.0, 11.0, 12.0], dtype=np.float64), 4096.0, 100.0, "stub"

    monkeypatch.setattr(s1_fetch_strain, "_load_local_hdf5", _fake_load)
    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", lambda *_a, **_k: 102.0)
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
            "--local-hdf5",
            f"H1={local_h1}",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    payload = np.load(out / "strain.npz")
    assert np.isfinite(payload["H1"]).all()
    assert payload["H1"].tolist() == pytest.approx([10.0, 11.0, 12.0])


def test_local_hdf5_is_cropped_before_nonfinite_checks(monkeypatch, tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_id = "crop_before_sanitize"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    local_h1 = tmp_path / "H1_windowed.hdf5"
    local_h1.write_bytes(b"placeholder")

    def _fake_load(_path):
        arr = np.arange(20.0, dtype=np.float64)
        arr[:4] = np.nan
        arr[-4:] = np.inf
        return arr, 2.0, 100.0, "stub"

    monkeypatch.setattr(s1_fetch_strain, "_load_local_hdf5", _fake_load)
    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", lambda *_a, **_k: 106.0)
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
            "--local-hdf5",
            f"H1={local_h1}",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert payload["H1"].tolist() == pytest.approx([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert np.isfinite(payload["H1"]).all()

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    crop = provenance["local_window_crop"]["H1"]
    assert crop["applied"] is True
    assert crop["output_n_samples"] == 8
    assert crop["output_gps_start"] == pytest.approx(104.0)
    assert provenance["strain_sanitization"]["H1"]["applied"] is False


def test_local_hdf5_nonfinite_samples_are_sanitized_and_recorded(monkeypatch, tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_id = "sanitize_local"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    local_h1 = tmp_path / "H1_nan.hdf5"
    local_h1.write_bytes(b"placeholder")

    def _fake_load(_path):
        return np.array([0.0, np.nan, 2.0, np.inf, 4.0], dtype=np.float64), 4096.0, 100.0, "stub"

    monkeypatch.setattr(s1_fetch_strain, "_load_local_hdf5", _fake_load)
    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", lambda *_a, **_k: 102.0)
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
            "--local-hdf5",
            f"H1={local_h1}",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert np.isfinite(payload["H1"]).all()
    assert payload["H1"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    info = provenance["strain_sanitization"]["H1"]
    assert info["applied"] is True
    assert info["nonfinite_count_raw"] == 2
