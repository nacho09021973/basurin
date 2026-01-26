from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


def _abort(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def validate_out_root(out_root: Union[str, Path], repo_runs: Path) -> Path:
    repo_runs_resolved = repo_runs.resolve()
    candidate = Path(out_root)
    if candidate == Path("runs"):
        return repo_runs_resolved
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.name == "runs":
        return candidate
    try:
        candidate.relative_to(repo_runs_resolved)
        return candidate
    except ValueError:
        _abort(
            "ERROR: out_root must be 'runs', a directory named 'runs', "
            f"or a subdir under {repo_runs_resolved}"
        )
    raise SystemExit(2)
