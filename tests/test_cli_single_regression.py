"""Regression tests for FIX-A: sp_single must expose threshold-mode and delta-lnL args.

If these fail it means pipeline.py single subcommand cannot forward s4 threshold
configuration, reproducing the AttributeError: 'Namespace' object has no attribute
'threshold_mode' that was reported in the E2E validation session.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import argparse


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = REPO_ROOT / "mvp" / "pipeline.py"


def _parse_single_args(extra: list[str]) -> argparse.Namespace:
    """Parse pipeline single args in-process using the same parser as main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline", PIPELINE)
    assert spec is not None
    mod = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
    raise AssertionError("should not reach here; use subprocess")


# ---------------------------------------------------------------------------
# Smoke tests via subprocess --help (no network, no atlas)
# ---------------------------------------------------------------------------

def test_single_help_exposes_threshold_mode():
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), "single", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 0, f"single --help exited {proc.returncode}: {proc.stderr}"
    assert "--threshold-mode" in proc.stdout, (
        "--threshold-mode not found in 'pipeline.py single --help' output; "
        "sp_single subparser is missing this arg (FIX-A regression)"
    )


def test_single_help_exposes_delta_lnL_220():
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), "single", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 0
    assert "--delta-lnL-220" in proc.stdout, (
        "--delta-lnL-220 not found in 'pipeline.py single --help' output"
    )


def test_single_help_exposes_delta_lnL_221():
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), "single", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 0
    assert "--delta-lnL-221" in proc.stdout, (
        "--delta-lnL-221 not found in 'pipeline.py single --help' output"
    )


# ---------------------------------------------------------------------------
# Namespace attribute tests: argparse must produce attrs without crash
# ---------------------------------------------------------------------------

def _build_single_parser() -> argparse.ArgumentParser:
    """Reconstruct just the 'single' subparser by importing the live module."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT))
    from mvp import pipeline  # noqa: PLC0415

    # pipeline.main() builds the parser internally; replicate just enough
    # to get the single subparser namespace.
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode")
    # We need to invoke main() parse path; easiest is to call parse_known_args
    # on a fake invocation with --help suppressed.
    return ap


def test_single_namespace_has_threshold_mode_attr():
    """Simulate 'pipeline.py single ...' parse and assert attrs exist (no AttributeError)."""
    sys.path.insert(0, str(REPO_ROOT))
    import importlib
    import mvp.pipeline as _pipeline  # noqa: PLC0415

    # Re-import to pick up any cached state
    _pipeline = importlib.reload(_pipeline)

    # Build the parser the same way main() does
    import argparse as _ap

    parser = _ap.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    # We invoke main()'s internal parser-building by running with --help suppressed
    # Instead, use the subprocess approach to parse a minimal valid-looking invocation
    # The key test: argparse must NOT raise AttributeError when we access .threshold_mode
    proc = subprocess.run(
        [
            sys.executable, str(PIPELINE),
            "single",
            "--event-id", "GW150914",
            "--atlas-default",
            "--threshold-mode", "d2",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
        timeout=10,
        env={**__import__("os").environ, "BASURIN_RUNS_ROOT": "/tmp/_test_threshold_regression"},
    )
    # The run may fail for operational reasons (no atlas, no network) but must NOT
    # produce an AttributeError about threshold_mode.
    combined = proc.stdout + "\n" + proc.stderr
    assert "AttributeError" not in combined, (
        f"AttributeError detected when running single with --threshold-mode=d2:\n{combined}"
    )
    assert "'Namespace' object has no attribute 'threshold_mode'" not in combined, (
        "Exact FIX-A regression: Namespace missing threshold_mode attr"
    )
