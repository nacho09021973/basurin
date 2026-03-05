from __future__ import annotations

import csv
import json
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
        rows=[{"event_id": "GW150914", "a_f_rd_p10": "0.65", "a_f_rd_p50": "0.7", "a_f_rd_p90": "0.75"}],
        fields=["event_id", "a_f_rd_p10", "a_f_rd_p50", "a_f_rd_p90"],
    )

    with pytest.raises(InsufficientGranularityError, match="INSUFFICIENT_INPUT_GRANULARITY"):
        run_experiment(
            run_id=run_id,
            in_per_event=str(in_csv),
            out_name="t6_rd_weighted",
            weight_key="delta_lnL",
            weight_transform="exp",
            min_effective_samples=200,
        )

    exp_dir = run_dir / "experiment" / "t6_rd_weighted"
    assert not exp_dir.exists()


def test_generates_artifacts_and_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(tmp_path))
    run_id = "run_weighted_ok"
    run_dir = _mk_run(tmp_path, run_id)

    in_csv = run_dir / "experiment" / "area_theorem" / "outputs" / "per_event_spinmag.csv"
    _write_csv(
        in_csv,
        rows=[
            {"event_id": "E1", "a_f_rd_p10": "0.0", "a_f_rd_p50": "0.0", "a_f_rd_p90": "0.0"},
            {"event_id": "E2", "a_f_rd_p10": "0.0", "a_f_rd_p50": "0.0", "a_f_rd_p90": "0.0"},
        ],
        fields=["event_id", "a_f_rd_p10", "a_f_rd_p50", "a_f_rd_p90"],
    )

    granular = run_dir / "experiment" / "toy_source" / "outputs" / "samples.csv"
    _write_csv(
        granular,
        rows=[
            {"event_id": "E1", "delta_lnL": "0.0", "a_f_rd_sample": "0.60"},
            {"event_id": "E1", "delta_lnL": "-1.0", "a_f_rd_sample": "0.80"},
            {"event_id": "E2", "delta_lnL": "0.0", "a_f_rd_sample": "0.50"},
            {"event_id": "E2", "delta_lnL": "-0.5", "a_f_rd_sample": "0.70"},
        ],
        fields=["event_id", "delta_lnL", "a_f_rd_sample"],
    )

    run_experiment(
        run_id=run_id,
        in_per_event=str(in_csv),
        out_name="t6_rd_weighted",
        weight_key="delta_lnL",
        weight_transform="exp",
        min_effective_samples=2,
    )

    exp_dir = run_dir / "experiment" / "t6_rd_weighted"
    out_csv = exp_dir / "outputs" / "per_event_spinmag_rd_weighted.csv"
    out_summary = exp_dir / "outputs" / "summary.json"
    manifest = exp_dir / "manifest.json"
    stage_summary = exp_dir / "stage_summary.json"

    assert out_csv.exists()
    assert out_summary.exists()
    assert manifest.exists()
    assert stage_summary.exists()

    first_csv_sha = sha256_file(out_csv)
    first_summary_sha = sha256_file(out_summary)

    run_experiment(
        run_id=run_id,
        in_per_event=str(in_csv),
        out_name="t6_rd_weighted",
        weight_key="delta_lnL",
        weight_transform="exp",
        min_effective_samples=2,
    )

    assert first_csv_sha == sha256_file(out_csv)
    assert first_summary_sha == sha256_file(out_summary)
