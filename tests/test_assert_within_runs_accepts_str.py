import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basurin_io import assert_within_runs


def test_assert_within_runs_accepts_str(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "a").mkdir(parents=True)
    ok_path = run_dir / "a" / "x.txt"
    ok_path.write_text("x", encoding="utf-8")

    assert_within_runs(run_dir, str(ok_path))


def test_assert_within_runs_rejects_outside(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        assert_within_runs(run_dir, str(outside))
