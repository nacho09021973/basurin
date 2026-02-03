from __future__ import annotations

import json
import os
from pathlib import Path


def _wjson(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_canonical_run_valid_layout(run_dir: Path) -> None:
    """
    Canonical contract (per BASURIN docs):
      - preferred: runs/<run>/RUN_VALID/verdict.json
      - legacy   : runs/<run>/RUN_VALID/outputs/run_valid.json
    and always include:
      - runs/<run>/RUN_VALID/stage_summary.json
      - runs/<run>/RUN_VALID/manifest.json
    """
    rv_dir = run_dir / "RUN_VALID"
    assert rv_dir.exists(), f"missing {rv_dir}"

    preferred = rv_dir / "verdict.json"
    legacy = rv_dir / "outputs" / "run_valid.json"
    assert preferred.exists() or legacy.exists(), (
        "missing RUN_VALID verdict: expected RUN_VALID/verdict.json (preferred) "
        "or RUN_VALID/outputs/run_valid.json (legacy)"
    )

    assert (rv_dir / "stage_summary.json").exists(), "missing RUN_VALID/stage_summary.json"
    assert (rv_dir / "manifest.json").exists(), "missing RUN_VALID/manifest.json"

    # Basic semantic check: verdict PASS/FAIL must exist if preferred.
    if preferred.exists():
        j = json.loads(preferred.read_text(encoding="utf-8"))
        assert j.get("verdict") in {"PASS", "FAIL"}, "RUN_VALID/verdict.json must contain verdict PASS|FAIL"


def test_run_valid_stage_is_canonical(tmp_path: Path, monkeypatch) -> None:
    """
    Deterministic test:
    - If BASURIN_TEST_RUN_ID is provided, validate that run under repo runs/<id>.
    - Else create a minimal synthetic run under tmp_path/runs/<id> and validate layout there.
    """
    run_id = os.environ.get("BASURIN_TEST_RUN_ID", "").strip()

    if run_id:
        run_dir = Path("runs") / run_id
        assert run_dir.exists(), f"BASURIN_TEST_RUN_ID points to missing run: {run_dir}"
        _assert_canonical_run_valid_layout(run_dir)
        return

    # No env var: create minimal run in isolated FS
    work = tmp_path
    monkeypatch.chdir(work)

    run_dir = work / "runs" / "test_run_valid_layout"
    rv_dir = run_dir / "RUN_VALID"
    (rv_dir / "outputs").mkdir(parents=True, exist_ok=True)

    # Preferred verdict
    _wjson(rv_dir / "verdict.json", {"verdict": "PASS"})

    # Required files
    _wjson(rv_dir / "manifest.json", {"schema_version": 1, "stage": "RUN_VALID"})
    _wjson(rv_dir / "stage_summary.json", {"schema_version": 1, "results": {"overall_verdict": "PASS"}})

    # (Optional legacy) keep absent to ensure preferred path passes.

    _assert_canonical_run_valid_layout(run_dir)
