from __future__ import annotations

import os
from pathlib import Path


def test_only_root_readme_is_canonical() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ignored_dirs = {".git", ".venv", ".pytest_cache", "__pycache__", "runs"}

    offenders: list[str] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        root_path = Path(root)
        for filename in files:
            if not filename.startswith("README"):
                continue
            path = root_path / filename
            if path == repo_root / "README.md":
                continue
            offenders.append(str(path.relative_to(repo_root)))

    offenders.sort()
    assert offenders == [], (
        "Only README.md at repo root is canonical. Offenders: "
        + ", ".join(offenders)
    )
