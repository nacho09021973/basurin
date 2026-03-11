from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "malda" / "10_build_event_feature_table.py"
_SPEC = importlib.util.spec_from_file_location("malda_10_build_event_feature_table", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_catalog(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event",
        "version",
        "catalog",
        "GPS",
        "snr",
        "p_astro",
        "far_yr",
        "m1_source",
        "m2_source",
        "M_total_source",
        "chirp_mass_source",
        "chi_eff",
        "final_mass_source",
        "luminosity_distance",
        "redshift",
        "glitch_mitigated",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_t0(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_row_falls_back_to_catalog_threshold_for_bbh() -> None:
    row = _MODULE.build_row(
        {
            "event": "GW_TEST_BBH",
            "m1_source": "34.5",
            "m2_source": "29.0",
            "chi_eff": "0.1",
            "final_mass_source": "60.0",
            "catalog": "synthetic",
        },
        {},
        {"source_class": None},
    )

    assert row["is_bbh"] == 1
    assert row["is_bns"] == 0
    assert row["is_nsbh"] == 0
    assert row["classification_source"] == "catalog_mass_threshold"


def test_build_row_classifies_bns_from_catalog_threshold() -> None:
    row = _MODULE.build_row(
        {
            "event": "GW170817",
            "m1_source": "1.46",
            "m2_source": "1.27",
            "chi_eff": "0.0",
            "final_mass_source": "2.8",
            "catalog": "synthetic",
        },
        {},
        {"source_class": None},
    )

    assert row["is_bbh"] == 0
    assert row["is_bns"] == 1
    assert row["is_nsbh"] == 0
    assert row["classification_source"] == "catalog_mass_threshold"


def test_bbh_only_zero_rows_writes_fail_summary(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "custom_runs"
    catalog_path = tmp_path / "catalog.csv"
    t0_path = tmp_path / "t0.json"
    losc_root = tmp_path / "data" / "losc"
    (losc_root / "GW170817").mkdir(parents=True)

    _write_catalog(
        catalog_path,
        [
            {
                "event": "GW170817",
                "version": "3",
                "catalog": "synthetic",
                "GPS": "1187008882.4",
                "snr": "33.0",
                "p_astro": "1.0",
                "far_yr": "1e-07",
                "m1_source": "1.46",
                "m2_source": "1.27",
                "M_total_source": "2.73",
                "chirp_mass_source": "1.186",
                "chi_eff": "0.0",
                "final_mass_source": "2.8",
                "luminosity_distance": "40.0",
                "redshift": "0.01",
                "glitch_mitigated": "False",
            }
        ],
    )
    _write_t0(t0_path, {"GW170817": {"GPS": 1187008882.4}})

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(_MODULE, "CATALOG_CSV", catalog_path)
    monkeypatch.setattr(_MODULE, "T0_JSON", t0_path)
    monkeypatch.setattr(_MODULE, "LOSC_DIR", losc_root)
    monkeypatch.setattr(_MODULE, "META_DIR", tmp_path / "docs" / "ringdown" / "event_metadata")
    monkeypatch.setattr(_MODULE, "write_hdf5", lambda rows, path: None)

    run_id = "malda_zero_rows"
    rc = _MODULE.main(["--run-id", run_id, "--bbh-only"])
    stage_dir = runs_root / run_id / "experiment" / "malda_feature_table"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert summary["verdict"] == "FAIL"
    assert summary["reason"] == "bbh_filter_yielded_zero_rows"
    assert manifest["verdict"] == "FAIL"
    assert manifest["reason"] == "bbh_filter_yielded_zero_rows"
    assert (stage_dir / "outputs" / "event_features.csv").exists()
    assert (stage_dir / "outputs" / "feature_catalog.json").exists()


def test_main_respects_temp_runs_root_and_writes_only_inside_stage_dir(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "isolated_runs"
    catalog_path = tmp_path / "catalog.csv"
    t0_path = tmp_path / "t0.json"
    losc_root = tmp_path / "data" / "losc"
    (losc_root / "GW_TEST_BBH").mkdir(parents=True)

    _write_catalog(
        catalog_path,
        [
            {
                "event": "GW_TEST_BBH",
                "version": "1",
                "catalog": "synthetic",
                "GPS": "1234567890.0",
                "snr": "20.0",
                "p_astro": "1.0",
                "far_yr": "1e-06",
                "m1_source": "34.5",
                "m2_source": "29.0",
                "M_total_source": "63.5",
                "chirp_mass_source": "27.0",
                "chi_eff": "0.1",
                "final_mass_source": "60.0",
                "luminosity_distance": "500.0",
                "redshift": "0.1",
                "glitch_mitigated": "False",
            }
        ],
    )
    _write_t0(t0_path, {"GW_TEST_BBH": {"GPS": 1234567890.0}})

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(_MODULE, "CATALOG_CSV", catalog_path)
    monkeypatch.setattr(_MODULE, "T0_JSON", t0_path)
    monkeypatch.setattr(_MODULE, "LOSC_DIR", losc_root)
    monkeypatch.setattr(_MODULE, "META_DIR", tmp_path / "docs" / "ringdown" / "event_metadata")
    monkeypatch.setattr(_MODULE, "write_hdf5", lambda rows, path: None)

    run_id = "malda_path_safety"
    rc = _MODULE.main(["--run-id", run_id, "--bbh-only"])
    stage_dir = runs_root / run_id / "experiment" / "malda_feature_table"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((stage_dir / "outputs" / "event_features.csv").open("r", encoding="utf-8")))

    assert rc == 0
    assert summary["verdict"] == "PASS"
    assert len(rows) == 1
    assert rows[0]["is_bbh"] == "1"
    assert rows[0]["classification_source"] == "catalog_mass_threshold"
    assert stage_dir.exists()
    assert not ((Path(__file__).resolve().parents[1] / "runs" / run_id).exists())
