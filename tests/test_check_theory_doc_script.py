from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_check_theory_doc_script_runs_ok() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "ci" / "check_theory_doc.py")]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "[PASS]" in proc.stdout
