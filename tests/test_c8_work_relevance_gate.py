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
    script = _repo_root() / "tools" / "c8_work_relevance_gate.py"
    return _run_cmd([sys.executable, str(script), "--run", run_id, *extra], cwd=tmp_path)


def _report_path(tmp_path: Path, run_id: str) -> Path:
    return (
        tmp_path
        / "runs"
        / run_id
        / "C8_WORK_RELEVANCE"
        / "outputs"
        / "c8_report.json"
    )


def test_c8_missing_metrics_underdetermined(tmp_path: Path) -> None:
    run_id = "2026-02-01__missing"
    (tmp_path / "runs" / run_id).mkdir(parents=True, exist_ok=True)

    result = _run_gate(tmp_path, run_id, [])
    assert result.returncode == 0

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "UNDERDETERMINED"
    assert report["failure_mode"] == "MISSING_METRICS"


def test_c8_pass(tmp_path: Path) -> None:
    run_id = "2026-02-01__pass"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(
        metrics_path,
        {
            "canonical_corr_median": 0.92,
            "degeneracy": {"degeneracy_index_median": 0.2},
        },
    )
    _write_json(
        tmp_path / "runs" / run_id / bridge / "outputs" / "knn_preservation_real.json",
        {"mean": 0.4},
    )
    _write_json(
        tmp_path / "runs" / run_id / bridge / "outputs" / "knn_preservation_negative.json",
        {"mean": 0.05},
    )
    _write_json(
        tmp_path
        / "runs"
        / run_id
        / bridge
        / "outputs"
        / "knn_preservation_control_positive.json",
        {"mean": 0.6},
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--book-min-pass",
            "0.8",
            "--ext-min-pass",
            "0.7",
            "--gap-max",
            "0.1",
            "--neg-max",
            "0.2",
            "--degenerate-threshold",
            "0.5",
        ],
    )
    assert result.returncode == 0

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "PASS"
    assert report["failure_mode"] is None


def test_c8_fail_negative_control(tmp_path: Path) -> None:
    run_id = "2026-02-01__fail"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(metrics_path, {"canonical_corr_median": 0.91})
    _write_json(
        tmp_path / "runs" / run_id / bridge / "outputs" / "knn_preservation_negative.json",
        {"mean": 0.3},
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--book-min-pass",
            "0.8",
            "--ext-min-pass",
            "0.7",
            "--gap-max",
            "0.1",
            "--neg-max",
            "0.2",
        ],
    )
    assert result.returncode == 2

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "FAIL"
    assert report["failure_mode"] == "LEAKAGE_NEGATIVE"


def test_c8_dotpath_book_score_and_degeneracy(tmp_path: Path) -> None:
    run_id = "2026-02-01__dotpath"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(
        metrics_path,
        {
            "results": {"canonical_corr_mean": 0.99},
            "degeneracy": {"degeneracy_index_median": 1.0},
            "structure_preservation": {
                "negative": {"overlap_mean": 0.5},
                "real": {"overlap_mean": 0.9},
            },
            "control_positive": {"overlap_mean": 1.0},
        },
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--book-key",
            "results.canonical_corr_mean",
        ],
    )
    assert result.returncode == 2

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "FAIL"
    assert report["book_score"] == 0.99
    assert report["penalties"]["degeneracy"]["value"] == 1.0


def test_c8_negative_control_fallback_from_metrics(tmp_path: Path) -> None:
    run_id = "2026-02-01__fallback-controls"
    bridge = "bridge_stage"
    metrics_path = tmp_path / "runs" / run_id / bridge / "outputs" / "metrics.json"
    _write_json(
        metrics_path,
        {
            "results": {"canonical_corr_mean": 0.99},
            "degeneracy": {"degeneracy_index_median": 1.0},
            "structure_preservation": {
                "negative": {"overlap_mean": 0.5},
                "real": {"overlap_mean": 0.9},
            },
            "control_positive": {"overlap_mean": 1.0},
        },
    )

    result = _run_gate(
        tmp_path,
        run_id,
        [
            "--bridge-stage",
            bridge,
            "--book-key",
            "results.canonical_corr_mean",
        ],
    )
    assert result.returncode == 2

    report = json.loads(_report_path(tmp_path, run_id).read_text())
    assert report["verdict"] == "FAIL"
    assert "kNN negative >= neg_max (leakage-like)" in report["reasons"]
    assert "Controles ausentes; no se concluye FAIL" not in report["reasons"]
