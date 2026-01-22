import json
import subprocess
import sys
from pathlib import Path


def _run_cmd(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, cwd=cwd)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_gate(tmp_path: Path, run_id: str, extra: list[str]) -> subprocess.CompletedProcess:
    script = _repo_root() / "tools" / "contract_c7_gate.py"
    return _run_cmd([sys.executable, str(script), "--run", run_id, *extra], cwd=tmp_path)


def _report_path(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "runs" / run_id / "C7" / "outputs" / "c7_report.json"


def test_c7_gate_leakage_fail(tmp_path: Path) -> None:
    run_id = "2026-02-01__leakage"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(
        metrics_path,
        {
            "canonical_corr_mean": 0.9,
            "significance_ratio": 4.0,
            "p_value": 0.01,
            "stability_score": 0.9,
            "mean_angle_deg": 10.0,
            "degeneracy_index_median": 50.0,
        },
    )
    _write_json(
        tmp_path / "runs" / run_id / bridge / "outputs" / "knn_preservation_real.json",
        {"mean": 0.2},
    )
    _write_json(
        tmp_path
        / "runs"
        / run_id
        / bridge
        / "outputs"
        / "knn_preservation_negative.json",
        {"mean": 0.5},
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--leakage-neg-ratio-fail",
            "2.0",
            "--min-corr-pass",
            "0.5",
            "--min-stability-pass",
            "0.5",
        ],
    )
    assert result.returncode == 2

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "FAIL"
    assert report["failure_mode"] == "LEAKAGE"


def test_c7_gate_degeneracy(tmp_path: Path) -> None:
    run_id = "2026-02-01__degenerate"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(
        metrics_path,
        {
            "canonical_corr_mean": 0.9,
            "significance_ratio": 4.0,
            "p_value": 0.01,
            "stability_score": 0.9,
            "mean_angle_deg": 12.0,
            "degeneracy_index_median": 200.0,
        },
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--min-corr-pass",
            "0.8",
            "--min-stability-pass",
            "0.8",
            "--degenerate-deg-median",
            "150",
            "--max-deg-pass",
            "100",
        ],
    )
    assert result.returncode == 0

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "DEGENERATE"
    assert report["failure_mode"] == "DEGENERACY"


def test_c7_gate_pass(tmp_path: Path) -> None:
    run_id = "2026-02-01__pass"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(
        metrics_path,
        {
            "canonical_corr_mean": 0.9,
            "significance_ratio": 4.0,
            "p_value": 0.01,
            "stability_score": 0.9,
            "mean_angle_deg": 9.0,
            "degeneracy_index_median": 80.0,
        },
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--min-corr-pass",
            "0.8",
            "--min-stability-pass",
            "0.8",
            "--max-deg-pass",
            "100",
        ],
    )
    assert result.returncode == 0

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "PASS"
    assert report["failure_mode"] is None


def test_c7_gate_missing_metrics(tmp_path: Path) -> None:
    run_id = "2026-02-01__missing"
    (tmp_path / "runs" / run_id).mkdir(parents=True, exist_ok=True)

    result = _run_gate(tmp_path, run_id, [])
    assert result.returncode == 0

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "UNDERDETERMINED"
    assert report["failure_mode"] == "MISSING_METRICS"
