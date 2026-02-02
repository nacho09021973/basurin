from pathlib import Path


def test_ops_docs_reference_basurin_where():
    repo_root = Path(__file__).resolve().parents[1]
    ringdown_plan = repo_root / "docs" / "ringdown" / "PLAN_PRE_REAL_V0_v1.md"
    entrypoints = repo_root / "docs" / "ops" / "ENTRYPOINTS_AND_CANONICAL_PATHS.md"

    ringdown_text = ringdown_plan.read_text(encoding="utf-8")
    entrypoints_text = entrypoints.read_text(encoding="utf-8")

    assert "tools/basurin_where.py" in ringdown_text
    assert "tools/basurin_where.py" in entrypoints_text
