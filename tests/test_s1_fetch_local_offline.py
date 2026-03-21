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


def _write_strain_hdf5(path: Path, values: np.ndarray, *, xstart: float, sample_rate_hz: float) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset("strain/Strain", data=np.asarray(values, dtype=np.float64))
        ds.attrs["Xstart"] = float(xstart)
        ds.attrs["Xspacing"] = float(1.0 / sample_rate_hz)


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


def test_default_synthetic_gps_center_fallback_produces_identical_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def _gps_fail(_event_id: str) -> float:
        raise RuntimeError("gps unavailable")

    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", _gps_fail)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            "run_default",
            "--event-id",
            "GW200225",
            "--detectors",
            "H1",
            "--duration-s",
            "4",
            "--synthetic",
        ],
    )
    rc = s1_fetch_strain.main()
    assert rc == 0

    out_default = runs_root / "run_default" / "s1_fetch_strain"
    provenance_default = json.loads((out_default / "outputs" / "provenance.json").read_text(encoding="utf-8"))
    summary_default = json.loads((out_default / "stage_summary.json").read_text(encoding="utf-8"))

    assert provenance_default["gps_center"] == pytest.approx(1126259462.4204)
    assert summary_default["results"]["synthetic_gps_center_fallback"] == pytest.approx(1126259462.4204)


def test_synthetic_gps_center_fallback_override_changes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def _gps_fail(_event_id: str) -> float:
        raise RuntimeError("gps unavailable")

    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", _gps_fail)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            "run_override",
            "--event-id",
            "GW200225",
            "--detectors",
            "H1",
            "--duration-s",
            "4",
            "--synthetic",
            "--synthetic-gps-center-fallback",
            "1000.25",
        ],
    )
    rc = s1_fetch_strain.main()
    assert rc == 0

    out_override = runs_root / "run_override" / "s1_fetch_strain"
    provenance_override = json.loads((out_override / "outputs" / "provenance.json").read_text(encoding="utf-8"))
    summary_override = json.loads((out_override / "stage_summary.json").read_text(encoding="utf-8"))

    assert provenance_override["gps_center"] == pytest.approx(1000.25)
    assert summary_override["results"]["synthetic_gps_center_fallback"] == pytest.approx(1000.25)


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


def test_fetch_gps_center_accepts_scalar_reference_catalog_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_catalog = tmp_path / "gwtc_events_t0.json"
    reference_catalog.write_text(
        json.dumps({"GW250114_082203": 1420878141.2}),
        encoding="utf-8",
    )
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    gps = s1_fetch_strain._fetch_gps_center("GW250114_082203")

    assert gps == pytest.approx(1420878141.2)


def test_local_hdf5_scalar_catalog_entry_crops_before_nonfinite_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id = "gw250114_local_scalar_catalog"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    reference_catalog = tmp_path / "gwtc_events_t0.json"
    reference_catalog.write_text(
        json.dumps({"GW250114_082203": 1420878141.2}),
        encoding="utf-8",
    )
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    sample_rate_hz = 4096.0
    duration_s = 32.0
    total_duration_s = 64.0
    total_n = int(sample_rate_hz * total_duration_s)
    output_n = int(sample_rate_hz * duration_s)
    gps_start_target = 1420878141.2 - duration_s / 2.0
    gps_start_local = gps_start_target - 16.0
    i_start = int(round((gps_start_target - gps_start_local) * sample_rate_hz))
    i_end = i_start + output_n

    h1_values = np.full(total_n, np.nan, dtype=np.float64)
    l1_values = np.full(total_n, np.inf, dtype=np.float64)
    h1_values[i_start:i_end] = np.linspace(1.0, 2.0, output_n, dtype=np.float64)
    l1_values[i_start:i_end] = np.linspace(3.0, 4.0, output_n, dtype=np.float64)

    h1_path = tmp_path / "H-H1_GWOSC_O4b3Disc_4KHZ_R1-test.hdf5"
    l1_path = tmp_path / "L-L1_GWOSC_O4b3Disc_4KHZ_R1-test.hdf5"
    _write_strain_hdf5(h1_path, h1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)
    _write_strain_hdf5(l1_path, l1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            run_id,
            "--event-id",
            "GW250114_082203",
            "--detectors",
            "H1,L1",
            "--duration-s",
            "32",
            "--local-hdf5",
            f"H1={h1_path}",
            "--local-hdf5",
            f"L1={l1_path}",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert payload["H1"].shape == (output_n,)
    assert payload["L1"].shape == (output_n,)
    assert np.isfinite(payload["H1"]).all()
    assert np.isfinite(payload["L1"]).all()
    assert payload["H1"].tolist() == pytest.approx(h1_values[i_start:i_end].tolist())
    assert payload["L1"].tolist() == pytest.approx(l1_values[i_start:i_end].tolist())

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["gps_center"] == pytest.approx(1420878141.2)
    assert provenance["local_window_crop"]["H1"]["applied"] is True
    assert provenance["local_window_crop"]["L1"]["applied"] is True
    assert provenance["local_window_crop"]["H1"]["output_n_samples"] == output_n
    assert provenance["local_window_crop"]["L1"]["output_n_samples"] == output_n
    assert provenance["strain_sanitization"]["H1"]["applied"] is False
    assert provenance["strain_sanitization"]["L1"]["applied"] is False


def test_local_hdf5_legacy_dict_catalog_entry_still_supports_32s_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id = "gw150914_local_legacy_catalog"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    reference_catalog = tmp_path / "gwtc_events_t0.json"
    reference_catalog.write_text(
        json.dumps({"GW150914": {"GPS": 1126259462.4}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    sample_rate_hz = 4096.0
    duration_s = 32.0
    total_duration_s = 64.0
    total_n = int(sample_rate_hz * total_duration_s)
    output_n = int(sample_rate_hz * duration_s)
    gps_start_target = 1126259462.4 - duration_s / 2.0
    gps_start_local = gps_start_target - 16.0
    i_start = int(round((gps_start_target - gps_start_local) * sample_rate_hz))
    i_end = i_start + output_n

    h1_values = np.linspace(-1.0, 1.0, total_n, dtype=np.float64)
    l1_values = np.linspace(2.0, 5.0, total_n, dtype=np.float64)

    h1_path = tmp_path / "H-H1_GWOSC_4KHZ_R1-legacy.hdf5"
    l1_path = tmp_path / "L-L1_GWOSC_4KHZ_R1-legacy.hdf5"
    _write_strain_hdf5(h1_path, h1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)
    _write_strain_hdf5(l1_path, l1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            run_id,
            "--event-id",
            "GW150914",
            "--detectors",
            "H1,L1",
            "--duration-s",
            "32",
            "--local-hdf5",
            f"H1={h1_path}",
            "--local-hdf5",
            f"L1={l1_path}",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert payload["H1"].shape == (output_n,)
    assert payload["L1"].shape == (output_n,)
    assert payload["H1"].tolist() == pytest.approx(h1_values[i_start:i_end].tolist())
    assert payload["L1"].tolist() == pytest.approx(l1_values[i_start:i_end].tolist())

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["gps_center"] == pytest.approx(1126259462.4)
    assert provenance["local_window_crop"]["H1"]["applied"] is True
    assert provenance["local_window_crop"]["L1"]["applied"] is True
    assert provenance["strain_sanitization"]["H1"]["applied"] is False
    assert provenance["strain_sanitization"]["L1"]["applied"] is False


def test_local_hdf5_event_metadata_fallback_supports_32s_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id = "gw150914_local_event_metadata"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    reference_catalog = tmp_path / "gwtc_events_t0.json"
    reference_catalog.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    metadata_dir = tmp_path / "event_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "GW150914_metadata.json").write_text(
        json.dumps({"t0_gps": 1126259462.4204}),
        encoding="utf-8",
    )
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_EVENT_METADATA_DIR", metadata_dir)

    sample_rate_hz = 4096.0
    duration_s = 32.0
    total_duration_s = 64.0
    total_n = int(sample_rate_hz * total_duration_s)
    output_n = int(sample_rate_hz * duration_s)
    gps_start_target = 1126259462.4204 - duration_s / 2.0
    gps_start_local = gps_start_target - 16.0
    i_start = int(round((gps_start_target - gps_start_local) * sample_rate_hz))
    i_end = i_start + output_n

    h1_values = np.linspace(-1.0, 1.0, total_n, dtype=np.float64)
    l1_values = np.linspace(2.0, 5.0, total_n, dtype=np.float64)

    h1_path = tmp_path / "H-H1_GWOSC_4KHZ_R1-metadata.hdf5"
    l1_path = tmp_path / "L-L1_GWOSC_4KHZ_R1-metadata.hdf5"
    _write_strain_hdf5(h1_path, h1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)
    _write_strain_hdf5(l1_path, l1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            run_id,
            "--event-id",
            "GW150914",
            "--detectors",
            "H1,L1",
            "--duration-s",
            "32",
            "--local-hdf5",
            f"H1={h1_path}",
            "--local-hdf5",
            f"L1={l1_path}",
            "--offline",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert float(payload["gps_start"]) == pytest.approx(gps_start_target)
    assert payload["H1"].tolist() == pytest.approx(h1_values[i_start:i_end].tolist())
    assert payload["L1"].tolist() == pytest.approx(l1_values[i_start:i_end].tolist())


def test_fetch_gps_center_accepts_t_coalescence_gps_event_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_WINDOW_CATALOG", tmp_path / "missing_window_catalog_v1.json")

    metadata_dir = tmp_path / "event_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "GW170814_metadata.json").write_text(
        json.dumps({"event_id": "GW170814", "t_coalescence_gps": 1186741861.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_EVENT_METADATA_DIR", metadata_dir)

    gps = s1_fetch_strain._fetch_gps_center("GW170814")

    assert gps == pytest.approx(1186741861.0)


def test_local_hdf5_window_catalog_fallback_supports_32s_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id = "gw170814_local_window_catalog"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")

    metadata_dir = tmp_path / "event_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_EVENT_METADATA_DIR", metadata_dir)

    window_catalog = tmp_path / "window_catalog_v1.json"
    window_catalog.write_text(
        json.dumps({"GW170814": {"t0_gps": 1186741861.5}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(s1_fetch_strain, "DEFAULT_WINDOW_CATALOG", window_catalog)

    sample_rate_hz = 4096.0
    duration_s = 32.0
    total_duration_s = 64.0
    total_n = int(sample_rate_hz * total_duration_s)
    output_n = int(sample_rate_hz * duration_s)
    gps_start_target = 1186741861.5 - duration_s / 2.0
    gps_start_local = gps_start_target - 16.0
    i_start = int(round((gps_start_target - gps_start_local) * sample_rate_hz))
    i_end = i_start + output_n

    h1_values = np.linspace(-1.0, 1.0, total_n, dtype=np.float64)
    l1_values = np.linspace(2.0, 5.0, total_n, dtype=np.float64)

    h1_path = tmp_path / "H-H1_GWOSC_4KHZ_R1-window.hdf5"
    l1_path = tmp_path / "L-L1_GWOSC_4KHZ_R1-window.hdf5"
    _write_strain_hdf5(h1_path, h1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)
    _write_strain_hdf5(l1_path, l1_values, xstart=gps_start_local, sample_rate_hz=sample_rate_hz)

    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            run_id,
            "--event-id",
            "GW170814",
            "--detectors",
            "H1,L1",
            "--duration-s",
            "32",
            "--local-hdf5",
            f"H1={h1_path}",
            "--local-hdf5",
            f"L1={l1_path}",
            "--offline",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert float(payload["gps_start"]) == pytest.approx(gps_start_target)
    assert payload["H1"].tolist() == pytest.approx(h1_values[i_start:i_end].tolist())
    assert payload["L1"].tolist() == pytest.approx(l1_values[i_start:i_end].tolist())


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


def test_local_hdf5_entirely_nonfinite_detector_is_dropped_if_other_detector_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_id = "drop_bad_h1_keep_l1"
    run_valid = runs_root / run_id / "RUN_VALID"
    run_valid.mkdir(parents=True, exist_ok=True)
    (run_valid / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    h1_path = tmp_path / "H1_all_nan.hdf5"
    l1_path = tmp_path / "L1_good.hdf5"
    h1_path.write_bytes(b"placeholder")
    l1_path.write_bytes(b"placeholder")

    def _fake_load(path: Path):
        if path.name.startswith("H1"):
            return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64), 4096.0, 100.0, "stub"
        return np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64), 4096.0, 100.0, "stub"

    monkeypatch.setattr(s1_fetch_strain, "_load_local_hdf5", _fake_load)
    monkeypatch.setattr(s1_fetch_strain, "_fetch_gps_center", lambda *_a, **_k: 102.0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "s1_fetch_strain.py",
            "--run",
            run_id,
            "--event-id",
            "GW190910_112807",
            "--detectors",
            "H1,L1",
            "--duration-s",
            "4",
            "--local-hdf5",
            f"H1={h1_path}",
            "--local-hdf5",
            f"L1={l1_path}",
        ],
    )

    rc = s1_fetch_strain.main()
    assert rc == 0

    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    payload = np.load(out / "strain.npz")
    assert "H1" not in payload.files
    assert payload["L1"].tolist() == pytest.approx([10.0, 11.0, 12.0, 13.0])

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["detectors"] == ["L1"]
    assert provenance["detectors_requested"] == ["H1", "L1"]
    assert provenance["detector_rejections"]["H1"]["policy"] == (
        "drop_entirely_nonfinite_detector_if_others_available"
    )
