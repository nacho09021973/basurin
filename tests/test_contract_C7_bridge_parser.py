import json
import sys
from pathlib import Path

import pytest

from experiment.bridge import contract_C7_bridge


def _write_alignment_inputs(
    base_dir: Path,
    run_id: str,
    alignment_subdir: Path,
    metrics_payload: dict,
    summary_payload: dict | None = None,
) -> None:
    stage_dir = base_dir / run_id / alignment_subdir
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary = summary_payload or {"status": "OK", "results": {}}
    (stage_dir / "stage_summary.json").write_text(json.dumps(summary))
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics_payload))


def _run_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_id: str, alignment_stage: str) -> Path:
    monkeypatch.chdir(tmp_path)
    out_root = "runs_tmp"
    monkeypatch.setattr(
        sys,
        "argv",
        ["contract_C7_bridge.py", "--run", run_id, "--out-root", out_root, "--alignment-stage", alignment_stage],
    )
    contract_C7_bridge.main()
    return tmp_path / out_root / run_id / "contract_C7_bridge" / "outputs" / "contract_C7_bridge.json"


def _base_metrics() -> dict:
    return {
        "stability_score": 0.9,
        "significance_ratio": 3.5,
        "p_value": 0.01,
        "degeneracy_index_median": 10.0,
        "degeneracy_index_p90": 12.0,
        "varX_trace_ratio_median": 1.1,
        "canonical_corr_mean": 0.42,
        "mean_axis_angle_deg": 5.0,
        "knn_preservation_mean": 0.2,
        "knn_preservation_negative_mean": 0.1,
        "knn_preservation_ratio": 2.5,
        "control_positive_status": "OK",
        "control_positive_overlap_mean": 0.5,
    }


def test_parser_accepts_top_level_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "2026-01-27__f4_1_smoke"
    out_root = tmp_path / "runs_tmp"
    metrics = _base_metrics()
    _write_alignment_inputs(out_root, run_id, Path("bridge_f4_1_alignment"), metrics)

    report_path = _run_contract(monkeypatch, tmp_path, run_id, "bridge_f4_1_alignment")
    report = json.loads(report_path.read_text())

    assert report["verdict"] != "UNDERDETERMINED"


def test_missing_gate_sets_underdetermined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "2026-01-27__f4_1_smoke"
    out_root = tmp_path / "runs_tmp"
    metrics = _base_metrics()
    metrics.pop("canonical_corr_mean")
    _write_alignment_inputs(out_root, run_id, Path("bridge_f4_1_alignment"), metrics)

    report_path = _run_contract(monkeypatch, tmp_path, run_id, "bridge_f4_1_alignment")
    report = json.loads(report_path.read_text())

    assert report["verdict"] == "UNDERDETERMINED"
    assert "canonical_corr_mean" in report["missing_metrics"]


def test_resolves_experiment_alignment_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "2026-01-27__f4_1_smoke"
    out_root = tmp_path / "runs_tmp"
    metrics = _base_metrics()
    _write_alignment_inputs(out_root, run_id, Path("experiment") / "bridge_f4_1_alignment", metrics)

    report_path = _run_contract(monkeypatch, tmp_path, run_id, "bridge_f4_1_alignment")
    report = json.loads(report_path.read_text())

    assert report["verdict"] != "UNDERDETERMINED"
