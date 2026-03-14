from __future__ import annotations

import csv
import importlib.util
import json
import math
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
    )

    assert row["is_bbh"] == 1
    assert row["is_bns"] == 0
    assert row["is_nsbh"] == 0
    assert row["classification_source"] == "catalog_mass_threshold"
    assert row["has_multimessenger"] == 0


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
    )

    assert row["is_bbh"] == 0
    assert row["is_bns"] == 1
    assert row["is_nsbh"] == 0
    assert row["classification_source"] == "catalog_mass_threshold"


def test_build_row_classifies_nsbh_from_catalog_threshold() -> None:
    row = _MODULE.build_row(
        {
            "event": "GW_TEST_NSBH",
            "m1_source": "7.2",
            "m2_source": "1.8",
            "chi_eff": "0.0",
            "final_mass_source": "8.7",
            "catalog": "synthetic",
        },
        {},
    )

    assert row["is_bbh"] == 0
    assert row["is_bns"] == 0
    assert row["is_nsbh"] == 1
    assert row["classification_source"] == "catalog_mass_threshold"


def test_build_row_marks_unknown_when_catalog_masses_are_invalid() -> None:
    row = _MODULE.build_row(
        {
            "event": "GW_TEST_UNKNOWN",
            "m1_source": "",
            "m2_source": "nan",
            "chi_eff": "0.0",
            "final_mass_source": "8.7",
            "catalog": "synthetic",
        },
        {},
    )

    assert row["is_bbh"] == 0
    assert row["is_bns"] == 0
    assert row["is_nsbh"] == 0
    assert row["classification_source"] == "unknown"


def test_build_row_computes_horizon_thermo_and_distance_features() -> None:
    row = _MODULE.build_row(
        {
            "event": "GW_TEST_BBH",
            "m1_source": "34.5",
            "m2_source": "29.0",
            "chi_eff": "0.1",
            "final_mass_source": "60.0",
            "luminosity_distance": "500.0",
            "redshift": "0.1",
            "catalog": "synthetic",
        },
        {},
    )

    assert math.isclose(float(row["DL_Gly"]), 500.0 * _MODULE.MPC_TO_GLY, rel_tol=1e-12)
    assert math.isclose(float(row["Mf_kg"]), 60.0 * _MODULE.MSUN_KG, rel_tol=1e-12)
    assert 0.0 < float(row["T_H_K"]) < 1e-6
    assert 0.0 < float(row["f_horizon_hz"]) < float(row["f_220_hz"])
    assert 0.0 < float(row["E_rot_frac"]) < 1.0
    assert float(row["M_irr_Msun"]) < float(row["Mf"])
    assert float(row["r_plus_km"]) > float(row["r_g_km"])
    assert float(row["A_horizon_km2"]) > 0.0


def test_build_row_sets_new_remnant_features_to_nan_without_valid_final_mass() -> None:
    row = _MODULE.build_row(
        {
            "event": "GW_TEST_MISSING_MF",
            "m1_source": "34.5",
            "m2_source": "29.0",
            "chi_eff": "0.1",
            "final_mass_source": "",
            "luminosity_distance": "500.0",
            "catalog": "synthetic",
        },
        {},
    )

    assert math.isnan(float(row["Mf_kg"]))
    assert math.isnan(float(row["M_irr_Msun"]))
    assert math.isnan(float(row["E_rot_Msun"]))
    assert math.isnan(float(row["E_rot_frac"]))
    assert math.isnan(float(row["r_g_km"]))
    assert math.isnan(float(row["r_plus_km"]))
    assert math.isnan(float(row["A_horizon_km2"]))
    assert math.isnan(float(row["T_H_K"]))
    assert math.isnan(float(row["f_horizon_hz"]))


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
    assert set(summary["config"]) == {"bbh_only", "catalog_csv", "t0_json", "losc_inventory_root"}
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
    assert rows[0]["has_multimessenger"] == "0"
    assert set(summary["config"]) == {"bbh_only", "catalog_csv", "t0_json", "losc_inventory_root"}
    assert stage_dir.exists()
    assert not ((Path(__file__).resolve().parents[1] / "runs" / run_id).exists())
