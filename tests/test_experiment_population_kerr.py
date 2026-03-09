from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from basurin_io import sha256_file, write_json_atomic
import mvp.experiment_population_kerr as exp


def _make_synthetic_run(
    runs_root: Path,
    run_id: str,
    *,
    run_valid: bool = True,
    has_s4d: bool = True,
    has_s7: bool = True,
    epsilon_f: float | None = None,
    epsilon_tau: float | None = None,
    chi2: float | None = None,
    verdict: str | None = None,
) -> Path:
    run_dir = runs_root / run_id
    if run_valid:
        write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    if has_s4d:
        write_json_atomic(
            run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json",
            {
                "schema_name": "kerr_extraction",
                "schema_version": "v1",
                "verdict": "PASS",
                "M_final_Msun": 68.0,
                "chi_final": 0.69,
            },
        )

    if has_s7:
        if verdict is None:
            verdict = "GR_CONSISTENT"
        if epsilon_f is None:
            epsilon_f = 0.0
        if epsilon_tau is None:
            epsilon_tau = 0.0
        if chi2 is None:
            chi2 = 1.0
        write_json_atomic(
            run_dir / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json",
            {
                "schema_name": "beyond_kerr_score",
                "schema_version": "v1",
                "verdict": verdict,
                "chi2_kerr_2dof": chi2,
                "epsilon_f": epsilon_f,
                "epsilon_tau": epsilon_tau,
            },
        )
    return run_dir


def test_inventory_skips_runs_without_run_valid(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(runs_root, "run_no_valid", run_valid=False, has_s4d=True, has_s7=False)

    inventory = exp._inventory_runs(runs_root, experiment_run_id="host", batch_prefix=None)
    row = next(r for r in inventory["runs"] if r["run_id"] == "run_no_valid")
    assert row["eligible"] is False
    assert row["has_run_valid_pass"] is False
    assert row["exclusion_reason"] == "RUN_VALID_NOT_PASS"


def test_inventory_marks_existing_s7_correctly(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(runs_root, "run_has_s7", run_valid=True, has_s4d=True, has_s7=True)

    inventory = exp._inventory_runs(runs_root, experiment_run_id="host", batch_prefix=None)
    row = next(r for r in inventory["runs"] if r["run_id"] == "run_has_s7")
    assert row["eligible"] is True
    assert row["has_s7_beyond_kerr_score"] is True


def test_run_phase_records_subprocess_failure_without_aborting(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(runs_root, "run_fail", run_valid=True, has_s4d=True, has_s7=False)
    _make_synthetic_run(runs_root, "run_ok", run_valid=True, has_s4d=True, has_s7=False)
    inventory = exp._inventory_runs(runs_root, experiment_run_id="host", batch_prefix=None)

    def _fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_id = cmd[-1]
        if run_id == "run_ok":
            write_json_atomic(
                runs_root / run_id / "s7_beyond_kerr_deviation_score" / "outputs" / "beyond_kerr_score.json",
                {
                    "schema_name": "beyond_kerr_score",
                    "schema_version": "v1",
                    "verdict": "GR_CONSISTENT",
                    "chi2_kerr_2dof": 0.2,
                    "epsilon_f": 0.01,
                    "epsilon_tau": -0.01,
                },
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(cmd, 2, stdout="bad", stderr="boom")

    monkeypatch.setattr(exp.subprocess, "run", _fake_subprocess_run)
    run_log = exp._run_missing_s7(runs_root, experiment_run_id="host", inventory_payload=inventory)
    assert run_log["n_runs_logged"] == 2
    assert run_log["n_runs_failed"] == 1
    fail_entry = next(e for e in run_log["entries"] if e["run_id"] == "run_fail")
    assert fail_entry["status"] == "failed"
    assert fail_entry["returncode"] == 2


def test_aggregate_3_runs_gr_consistent(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(
        runs_root,
        "r1",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=0.01,
        epsilon_tau=0.00,
        chi2=0.5,
        verdict="GR_CONSISTENT",
    )
    _make_synthetic_run(
        runs_root,
        "r2",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=-0.01,
        epsilon_tau=0.01,
        chi2=0.7,
        verdict="GR_CONSISTENT",
    )
    _make_synthetic_run(
        runs_root,
        "r3",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=0.00,
        epsilon_tau=-0.01,
        chi2=0.8,
        verdict="GR_CONSISTENT",
    )

    inventory = exp._inventory_runs(runs_root, experiment_run_id="host", batch_prefix=None)
    summary = exp._aggregate_population(runs_root, experiment_run_id="host", inventory_payload=inventory)
    assert summary["n_events_aggregated"] == 3
    assert summary["n_events_gr_consistent"] == 3
    assert summary["population_verdict"] == "GR_CONSISTENT_POPULATION"


def test_aggregate_detects_outlier(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(
        runs_root,
        "r_cons_1",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=0.01,
        epsilon_tau=0.00,
        chi2=0.3,
        verdict="GR_CONSISTENT",
    )
    _make_synthetic_run(
        runs_root,
        "r_cons_2",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=-0.01,
        epsilon_tau=0.01,
        chi2=0.6,
        verdict="GR_CONSISTENT",
    )
    _make_synthetic_run(
        runs_root,
        "r_outlier",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=0.8,
        epsilon_tau=-0.7,
        chi2=20.0,
        verdict="GR_INCONSISTENT",
    )

    inventory = exp._inventory_runs(runs_root, experiment_run_id="host", batch_prefix=None)
    summary = exp._aggregate_population(runs_root, experiment_run_id="host", inventory_payload=inventory)
    assert summary["n_events_gr_inconsistent"] == 1
    assert summary["outlier_runs"] == ["r_outlier"]


def test_aggregate_empty_eligible_runs_does_not_crash(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(runs_root, "ineligible_a", run_valid=False, has_s4d=True, has_s7=False)
    _make_synthetic_run(runs_root, "ineligible_b", run_valid=False, has_s4d=False, has_s7=False)

    inventory = exp._inventory_runs(runs_root, experiment_run_id="host", batch_prefix=None)
    summary = exp._aggregate_population(runs_root, experiment_run_id="host", inventory_payload=inventory)
    assert summary["n_events_aggregated"] == 0
    assert summary["population_verdict"] == "NO_DATA"


def test_manifest_contains_hashes_for_all_outputs(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    _make_synthetic_run(
        runs_root,
        "evt001",
        run_valid=True,
        has_s4d=True,
        has_s7=True,
        epsilon_f=0.01,
        epsilon_tau=-0.01,
        chi2=0.4,
        verdict="GR_CONSISTENT",
    )

    rc = exp.main(["--experiment-run-id", "exp_host", "--phase", "all"])
    assert rc == 0
    experiment_parent = runs_root / "exp_host" / "experiment"
    experiment_dirs = sorted(experiment_parent.glob("population_kerr_*"))
    assert len(experiment_dirs) == 1

    experiment_dir = experiment_dirs[0]
    inventory_path = experiment_dir / "inventory.json"
    run_log_path = experiment_dir / "run_log.json"
    summary_path = experiment_dir / "population_kerr_summary.json"
    manifest_path = experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["hashes"]["inventory"] == sha256_file(inventory_path)
    assert manifest["hashes"]["run_log"] == sha256_file(run_log_path)
    assert manifest["hashes"]["population_kerr_summary"] == sha256_file(summary_path)
