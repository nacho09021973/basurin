from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from basurin_io import sha256_file
from mvp.experiment_t6_rd_weighted import InsufficientGranularityError, run_experiment


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mk_run(tmp_path: Path, run_id: str) -> Path:
    run_dir = tmp_path / run_id
    rv = run_dir / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    return run_dir


def test_insufficient_granularity_fails_without_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_id = "run_no_granular"
    run_dir = _mk_run(tmp_path, run_id)
    in_csv = run_dir / "experiment" / "area_theorem" / "outputs" / "per_event_spinmag.csv"
    _write_csv(
        in_csv,
        rows=[{"event_id": "GW150914", "status": "OK", "p_violate": "0.1", "dA_p10": "1", "dA_p50": "2", "dA_p90": "3", "n_mc": "100"}],
        fields=["event_id", "status", "p_violate", "dA_p10", "dA_p50", "dA_p90", "n_mc"],
    )

    with pytest.raises(InsufficientGranularityError, match="INSUFFICIENT_INPUT_GRANULARITY"):
        run_experiment(
            run_id=run_id,
            in_per_event=str(in_csv),
            out_name="t6_rd_weighted",
            min_effective_samples=200,
            batch_220=None,
            batch_221=None,
        )

    exp_dir = run_dir / "experiment" / "t6_rd_weighted"
    assert not exp_dir.exists()


def test_batch_intersection_weighted_quantiles_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    analysis = _mk_run(tmp_path, "run_weighted_ok")
    batch220 = _mk_run(tmp_path, "batch220")
    batch221 = _mk_run(tmp_path, "batch221")
    sub220 = _mk_run(tmp_path, "sub220_e1")
    sub221 = _mk_run(tmp_path, "sub221_e1")

    in_csv = analysis / "experiment" / "area_theorem" / "outputs" / "per_event_spinmag.csv"
    _write_csv(
        in_csv,
        rows=[{"event_id": "E1", "status": "OK", "p_violate": "0", "dA_p10": "0", "dA_p50": "0", "dA_p90": "0", "n_mc": "50"}],
        fields=["event_id", "status", "p_violate", "dA_p10", "dA_p50", "dA_p90", "n_mc"],
    )

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_e1"}],
        fields=["event_id", "subrun_id"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_e1"}],
        fields=["event_id", "subrun_id"],
    )

    c220 = {
        "compatible_geometries": [
            {"geometry_id": "g220_1", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000},
            {"geometry_id": "g220_2", "family": "kerr", "source": "berti", "M_solar": 11, "chi": 0.2, "Af": 1100},
            {"geometry_id": "g220_3", "family": "kerr", "source": "berti", "M_solar": 12, "chi": 0.3, "Af": 1200},
        ],
        "ranked_all": [
            {"geometry_id": "g220_1", "delta_lnL": 0.0},
            {"geometry_id": "g220_2", "delta_lnL": -1.0},
            {"geometry_id": "g220_3", "delta_lnL": -2.0},
        ],
    }
    c221 = {
        "compatible_geometries": [
            {"geometry_id": "g221_1", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000},
            {"geometry_id": "g221_2", "family": "kerr", "source": "berti", "M_solar": 11, "chi": 0.2, "Af": 1100},
            {"geometry_id": "g221_3", "family": "kerr", "source": "berti", "M_solar": 12, "chi": 0.3, "Af": 1200},
        ],
        "ranked_all": [
            {"geometry_id": "g221_1", "delta_lnL": 0.0},
            {"geometry_id": "g221_2", "delta_lnL": 0.0},
            {"geometry_id": "g221_3", "delta_lnL": 0.0},
        ],
    }
    (sub220 / "s4_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (sub221 / "s4_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (sub220 / "s4_geometry_filter" / "outputs" / "compatible_set.json").write_text(json.dumps(c220), encoding="utf-8")
    (sub221 / "s4_geometry_filter" / "outputs" / "compatible_set.json").write_text(json.dumps(c221), encoding="utf-8")

    imr = {
        "samples": [
            {"mass_1_source": 10, "mass_2_source": 8, "a_1": 0.1, "a_2": 0.2},
            {"mass_1_source": 11, "mass_2_source": 7, "a_1": 0.1, "a_2": 0.2},
            {"mass_1_source": 12, "mass_2_source": 6, "a_1": 0.1, "a_2": 0.2},
        ]
    }
    (analysis / "external_inputs" / "gwtc_posteriors").mkdir(parents=True, exist_ok=True)
    (analysis / "external_inputs" / "gwtc_posteriors" / "E1.json").write_text(json.dumps(imr), encoding="utf-8")

    run_experiment(
        run_id="run_weighted_ok",
        in_per_event=str(in_csv),
        out_name="t6_rd_weighted",
        min_effective_samples=2,
        batch_220="batch220",
        batch_221="batch221",
    )

    exp_dir = analysis / "experiment" / "t6_rd_weighted"
    out_csv = exp_dir / "outputs" / "per_event_spinmag_rd_weighted.csv"
    out_summary = exp_dir / "outputs" / "summary.json"
    manifest = exp_dir / "manifest.json"
    stage_summary = exp_dir / "stage_summary.json"
    assert out_csv.exists() and out_summary.exists() and manifest.exists() and stage_summary.exists()

    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    assert rows[0]["af_rd_p10"] == "1000"
    assert rows[0]["af_rd_p50"] == "1000"
    assert rows[0]["af_rd_p90"] == "1100"

    w = [1.0, math.exp(-1), math.exp(-2)]
    ess_expected = (sum(w) ** 2) / sum(x * x for x in w)
    assert float(rows[0]["ess_rd"]) == pytest.approx(ess_expected)

    first_csv_sha = sha256_file(out_csv)
    first_summary_sha = sha256_file(out_summary)
    run_experiment(
        run_id="run_weighted_ok",
        in_per_event=str(in_csv),
        out_name="t6_rd_weighted",
        min_effective_samples=2,
        batch_220="batch220",
        batch_221="batch221",
    )
    assert first_csv_sha == sha256_file(out_csv)
    assert first_summary_sha == sha256_file(out_summary)


def test_missing_delta_lnl_in_ranked_all_fails_with_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    analysis = _mk_run(tmp_path, "run_weighted_missing_delta")
    batch220 = _mk_run(tmp_path, "batch220_missing")
    batch221 = _mk_run(tmp_path, "batch221_missing")
    sub220 = _mk_run(tmp_path, "sub220_missing")
    sub221 = _mk_run(tmp_path, "sub221_missing")

    in_csv = analysis / "experiment" / "area_theorem" / "outputs" / "per_event_spinmag.csv"
    _write_csv(
        in_csv,
        rows=[{"event_id": "E1", "status": "OK", "p_violate": "0", "dA_p10": "0", "dA_p50": "0", "dA_p90": "0", "n_mc": "50"}],
        fields=["event_id", "status", "p_violate", "dA_p10", "dA_p50", "dA_p90", "n_mc"],
    )

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_missing"}],
        fields=["event_id", "subrun_id"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_missing"}],
        fields=["event_id", "subrun_id"],
    )

    c220 = {
        "compatible_geometries": [{"geometry_id": "g220", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000}],
        "ranked_all": [{"geometry_id": "g220"}],
    }
    c221 = {
        "compatible_geometries": [{"geometry_id": "g221", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000}],
        "ranked_all": [{"geometry_id": "g221", "delta_lnL": 0.0}],
    }
    (sub220 / "s4_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (sub221 / "s4_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (sub220 / "s4_geometry_filter" / "outputs" / "compatible_set.json").write_text(json.dumps(c220), encoding="utf-8")
    (sub221 / "s4_geometry_filter" / "outputs" / "compatible_set.json").write_text(json.dumps(c221), encoding="utf-8")

    (analysis / "external_inputs" / "gwtc_posteriors").mkdir(parents=True, exist_ok=True)
    (analysis / "external_inputs" / "gwtc_posteriors" / "E1.json").write_text(
        json.dumps({"samples": [{"mass_1_source": 10, "mass_2_source": 8, "a_1": 0.1, "a_2": 0.2}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"MISSING_DELTA_LNL.*source=ranked_all"):
        run_experiment(
            run_id="run_weighted_missing_delta",
            in_per_event=str(in_csv),
            out_name="t6_rd_weighted",
            min_effective_samples=2,
            batch_220="batch220_missing",
            batch_221="batch221_missing",
        )


def test_geometry_id_join_and_max_delta_per_phys_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    analysis = _mk_run(tmp_path, "run_weighted_gid_join")
    batch220 = _mk_run(tmp_path, "batch220_gid")
    batch221 = _mk_run(tmp_path, "batch221_gid")
    sub220 = _mk_run(tmp_path, "sub220_gid")
    sub221 = _mk_run(tmp_path, "sub221_gid")

    in_csv = analysis / "experiment" / "area_theorem" / "outputs" / "per_event_spinmag.csv"
    _write_csv(
        in_csv,
        rows=[{"event_id": "E1", "status": "OK", "p_violate": "0", "dA_p10": "0", "dA_p50": "0", "dA_p90": "0", "n_mc": "20"}],
        fields=["event_id", "status", "p_violate", "dA_p10", "dA_p50", "dA_p90", "n_mc"],
    )

    _write_csv(
        batch220 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub220_gid"}],
        fields=["event_id", "subrun_id"],
    )
    _write_csv(
        batch221 / "experiment" / "offline_batch" / "outputs" / "results.csv",
        rows=[{"event_id": "E1", "subrun_id": "sub221_gid"}],
        fields=["event_id", "subrun_id"],
    )

    c220 = {
        "compatible_geometries": [
            {"geometry_id": "g220_hi", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000},
            {"geometry_id": "g220_lo", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000},
            {"family": "kerr", "source": "berti", "M_solar": 12, "chi": 0.2, "Af": 1200},
        ],
        "ranked_all": [
            {"geometry_id": "g220_hi", "delta_lnL": 0.0},
            {"geometry_id": "g220_lo", "delta_lnL": -8.0},
        ],
    }
    c221 = {
        "compatible_geometries": [
            {"geometry_id": "g221", "family": "kerr", "source": "berti", "M_solar": 10, "chi": 0.1, "Af": 1000},
        ],
        "ranked_all": [
            {"geometry_id": "g221", "delta_lnL": 0.0},
        ],
    }
    (sub220 / "s4_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (sub221 / "s4_geometry_filter" / "outputs").mkdir(parents=True, exist_ok=True)
    (sub220 / "s4_geometry_filter" / "outputs" / "compatible_set.json").write_text(json.dumps(c220), encoding="utf-8")
    (sub221 / "s4_geometry_filter" / "outputs" / "compatible_set.json").write_text(json.dumps(c221), encoding="utf-8")

    (analysis / "external_inputs" / "gwtc_posteriors").mkdir(parents=True, exist_ok=True)
    (analysis / "external_inputs" / "gwtc_posteriors" / "E1.json").write_text(
        json.dumps({"samples": [{"mass_1_source": 10, "mass_2_source": 8, "a_1": 0.1, "a_2": 0.2}]}),
        encoding="utf-8",
    )

    run_experiment(
        run_id="run_weighted_gid_join",
        in_per_event=str(in_csv),
        out_name="t6_rd_weighted",
        min_effective_samples=1,
        batch_220="batch220_gid",
        batch_221="batch221_gid",
    )

    out_summary = analysis / "experiment" / "t6_rd_weighted" / "outputs" / "summary.json"
    event = json.loads(out_summary.read_text(encoding="utf-8"))["per_event"][0]
    assert event["n_dropped_missing_gid_or_weight_220"] == 1
    assert event["n_support_phys"] == 1
    assert event["join_policy"].startswith("ranked_all.geometry_id")
