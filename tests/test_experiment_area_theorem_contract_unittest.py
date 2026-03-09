from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _init_run_valid(runs_root: Path, run_id: str) -> None:
    verdict = runs_root / run_id / "RUN_VALID" / "verdict.json"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text('{"verdict":"PASS"}\n', encoding="utf-8")


def _make_batch_csv(runs_root: Path, run_id: str) -> None:
    out = runs_root / run_id / "experiment" / "theory_survival" / "outputs" / "survivors.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "family,source,M_solar,chi\n"
        "kerr,berti,65.0,0.68\n"
        "kerr,berti,70.0,0.72\n",
        encoding="utf-8",
    )


def _make_batch_results_csv(
    runs_root: Path,
    run_id: str,
    rows: list[tuple[str, str, str]],
) -> None:
    """Write offline_batch results.csv with (event_id, source_run_id, status)."""
    out = runs_root / run_id / "experiment" / "offline_batch" / "outputs" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event_id", "run_id", "status"])
        for event_id, source_run_id, status in rows:
            writer.writerow([event_id, source_run_id, status])


def _make_compatible_set(
    runs_root: Path,
    source_run_id: str,
    points: list[tuple[float, float]],
) -> None:
    out = runs_root / source_run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    geos = []
    for idx, (m_solar, chi) in enumerate(points):
        geos.append(
            {
                "geometry_id": f"g{idx}",
                "metadata": {
                    "family": "kerr",
                    "source": "berti_2009_fit",
                    "M_solar": float(m_solar),
                    "chi": float(chi),
                },
            }
        )
    _write_json(
        out,
        {
            "compatible_geometries": geos,
            "n_compatible": len(geos),
        },
    )


def test_manual_fetch_aborts_with_missing_list(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "analysis_test_t6_missing"
    _init_run_valid(runs_root, run_id)
    req = runs_root / run_id / "external_inputs" / "gwtc_posteriors" / "required_events.txt"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text("GW150914\nGW170104\n", encoding="utf-8")

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    proc = subprocess.run(
        [sys.executable, "-m", "mvp.experiment_gwtc_posteriors_fetch", "--run-id", run_id, "--source", "manual"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "missing_events=['GW150914', 'GW170104']" in proc.stderr


def test_area_theorem_writes_outputs_and_columns(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    analysis_run = "analysis_test_t6_ok"
    batch220 = "batch220_ok"
    batch221 = "batch221_ok"

    _init_run_valid(runs_root, analysis_run)

    req = runs_root / analysis_run / "external_inputs" / "gwtc_posteriors" / "required_events.txt"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text("GW150914\n", encoding="utf-8")

    _write_json(
        runs_root / analysis_run / "external_inputs" / "gwtc_posteriors" / "GW150914.json",
        {
            "event_id": "GW150914",
            "source": {"kind": "manual", "citation": "synthetic", "url_or_id": "local"},
            "samples": [
                {"m1_source": 36.0, "m2_source": 29.0, "chi1": 0.2, "chi2": 0.1},
                {"m1_source": 34.0, "m2_source": 28.0, "chi1": 0.3, "chi2": 0.15},
            ],
        },
    )

    _make_batch_csv(runs_root, batch220)
    _make_batch_csv(runs_root, batch221)

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp.experiment_area_theorem_deltaA",
            "--run-id",
            analysis_run,
            "--batch220-run-id",
            batch220,
            "--batch221-run-id",
            batch221,
            "--af-mode",
            "legacy_global",
            "--mc-draws",
            "200",
            "--seed",
            "11",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    per_event = runs_root / analysis_run / "experiment_area_theorem" / "outputs" / "per_event.csv"
    assert per_event.exists()
    with open(per_event, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    expected_cols = {
        "event_id",
        "P_deltaA_lt_0",
        "deltaA_p10",
        "deltaA_p50",
        "deltaA_p90",
        "deltaA_mean",
        "n_draws",
        "n_af_candidates",
        "n_imr_samples",
    }
    assert set(rows[0].keys()) == expected_cols


def test_area_theorem_per_event_uses_only_event_af_intersection(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    analysis_run = "analysis_test_t6_per_event"
    batch220 = "batch220_per_event"
    batch221 = "batch221_per_event"
    event_id = "GW170817"

    _init_run_valid(runs_root, analysis_run)

    req = runs_root / analysis_run / "external_inputs" / "gwtc_posteriors" / "required_events.txt"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(f"{event_id}\n", encoding="utf-8")

    _write_json(
        runs_root / analysis_run / "external_inputs" / "gwtc_posteriors" / f"{event_id}.json",
        {
            "event_id": event_id,
            "source": {"kind": "manual", "citation": "synthetic", "url_or_id": "local"},
            "samples": [
                {"m1_source": 1.0, "m2_source": 1.0, "chi1": 0.1, "chi2": 0.1},
                {"m1_source": 1.2, "m2_source": 0.9, "chi1": 0.2, "chi2": 0.1},
            ],
        },
    )

    # target event runs (intersection size should be exactly 1)
    run220_evt = "src220_evt"
    run221_evt = "src221_evt"
    _make_compatible_set(runs_root, run220_evt, [(10.0, 0.1), (20.0, 0.2)])
    _make_compatible_set(runs_root, run221_evt, [(20.0, 0.2), (30.0, 0.3)])

    # extra event runs to detect accidental global mixing
    run220_other = "src220_other"
    run221_other = "src221_other"
    _make_compatible_set(runs_root, run220_other, [(70.0, 0.7)])
    _make_compatible_set(runs_root, run221_other, [(70.0, 0.7)])

    _make_batch_results_csv(
        runs_root,
        batch220,
        [(event_id, run220_evt, "PASS"), ("GW150914", run220_other, "PASS")],
    )
    _make_batch_results_csv(
        runs_root,
        batch221,
        [(event_id, run221_evt, "PASS"), ("GW150914", run221_other, "PASS")],
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp.experiment_area_theorem_deltaA",
            "--run-id",
            analysis_run,
            "--batch220-run-id",
            batch220,
            "--batch221-run-id",
            batch221,
            "--mc-draws",
            "50",
            "--seed",
            "7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    per_event = runs_root / analysis_run / "experiment_area_theorem" / "outputs" / "per_event.csv"
    with open(per_event, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["event_id"] == event_id
    assert int(rows[0]["n_af_candidates"]) == 1
