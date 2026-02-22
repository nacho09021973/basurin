"""Shared path-resolution helpers for run-scoped stage inputs."""
from __future__ import annotations

from pathlib import Path


def resolve_run_scoped_input(
    run_dir: Path,
    override: str | None,
    *,
    default_rel: str,
    arg_name: str,
) -> Path:
    """Resolve an optional override path under run_dir and block escapes.

    If override is None, returns run_dir/default_rel.
    Otherwise supports both relative and absolute override paths, but requires
    that the resolved path stays under run_dir.
    """
    run_dir_real = run_dir.resolve()
    if override is None:
        return (run_dir_real / default_rel).resolve()

    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = run_dir_real / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_dir_real):
        raise ValueError(
            f"Invalid {arg_name}: resolved path escapes run directory "
            f"({resolved} not under {run_dir_real})"
        )
    return resolved
