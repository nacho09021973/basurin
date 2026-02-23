from __future__ import annotations

import json
from pathlib import Path

from mvp import find_artifact


def test_parse_args_requires_run_id_for_estimates() -> None:
    args = find_artifact.parse_args(["--what", "pass"])
    assert args.what == "pass"


def test_estimates_returns_2_when_missing(tmp_path: Path) -> None:
    rc = find_artifact.main(
        [
            "--runs-root",
            str(tmp_path),
            "--run-id",
            "run_missing",
            "--what",
            "estimates",
        ]
    )
    assert rc == 2


def test_estimates_returns_0_when_present(tmp_path: Path) -> None:
    estimates = tmp_path / "run_ok" / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    estimates.parent.mkdir(parents=True, exist_ok=True)
    estimates.write_text(json.dumps({"ok": True}), encoding="utf-8")

    rc = find_artifact.main(
        [
            "--runs-root",
            str(tmp_path),
            "--run-id",
            "run_ok",
            "--what",
            "estimates",
        ]
    )
    assert rc == 0


def test_gating_reads_verdict(tmp_path: Path) -> None:
    verdict = tmp_path / "run_ok" / "RUN_VALID" / "verdict.json"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    rc = find_artifact.main(
        [
            "--runs-root",
            str(tmp_path),
            "--run-id",
            "run_ok",
            "--what",
            "gating",
        ]
    )
    assert rc == 0
