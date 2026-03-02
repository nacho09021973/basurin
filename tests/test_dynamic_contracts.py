from __future__ import annotations

from pathlib import Path

import pytest

from basurin_io import write_json_atomic
from mvp.contracts import check_inputs, init_stage


def _write_run_valid(run_dir: Path) -> None:
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})


def test_s3_contract_fails_without_s2_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_missing_s2"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    ctx = init_stage(run_id, "s3_ringdown_estimates")

    with pytest.raises(SystemExit) as exc:
        check_inputs(ctx, {})

    assert exc.value.code == 2
    summary = (run_dir / "s3_ringdown_estimates" / "stage_summary.json").read_text(encoding="utf-8")
    assert "s2_ringdown_window/outputs/*_rd.npz" in summary
    assert "python -m mvp.s2_ringdown_window --run-id <RUN_ID>" in summary


def test_s3_contract_passes_with_mock_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "runs"
    run_id = "run_with_s2"
    run_dir = out_root / run_id
    _write_run_valid(run_dir)

    mock_npz = run_dir / "s2_ringdown_window" / "outputs" / "H1_rd.npz"
    mock_npz.parent.mkdir(parents=True, exist_ok=True)
    mock_npz.write_bytes(b"fake")

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(out_root))
    ctx = init_stage(run_id, "s3_ringdown_estimates")

    records = check_inputs(ctx, {})

    assert records == []
    assert ctx.check_inputs_info["discovered_inputs"] == ["s2_ringdown_window/outputs/H1_rd.npz"]
