from __future__ import annotations

import json
from pathlib import Path

from basurin_io import write_json_atomic
from mvp.contracts import CONTRACTS
from mvp.s4_geometry_filter import main as s4_main


def _write_run_valid(run_dir: Path) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})


def test_contracts_do_not_hardcode_estimates_for_override_paths() -> None:
    s4_req = CONTRACTS["s4_geometry_filter"].required_inputs
    s6_req = CONTRACTS["s6_information_geometry"].required_inputs

    assert "s3_ringdown_estimates/outputs/estimates.json" not in s4_req
    assert "s3_ringdown_estimates/outputs/estimates.json" not in s6_req


def test_s4_stage_summary_records_actual_overridden_estimates_input(tmp_path: Path, monkeypatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_contract_alignment"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    atlas_path = tmp_path / "atlas.json"
    write_json_atomic(atlas_path, [{"geometry_id": "g1", "f_hz": 250.0, "Q": 3.14}])

    alt_est = run_dir / "s3_spectral_estimates" / "outputs" / "spectral_estimates.json"
    write_json_atomic(
        alt_est,
        {
            "event_id": "EVT",
            "combined": {"f_hz": 250.0, "Q": 3.14, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_logf": 0.1, "sigma_logQ": 0.2},
        },
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s4_geometry_filter.py",
            "--run",
            run_id,
            "--atlas-path",
            str(atlas_path),
            "--estimates-path",
            "s3_spectral_estimates/outputs/spectral_estimates.json",
        ],
    )

    rc = s4_main()
    assert rc == 0

    summary = json.loads((run_dir / "s4_geometry_filter" / "stage_summary.json").read_text(encoding="utf-8"))
    inp = {row["label"]: row for row in summary["inputs"]}
    assert inp["estimates"]["path"] == "s3_spectral_estimates/outputs/spectral_estimates.json"


def test_s6_cli_exposes_estimates_path_override() -> None:
    src = Path("mvp/s6_information_geometry.py").read_text(encoding="utf-8")
    assert "--estimates-path" in src
