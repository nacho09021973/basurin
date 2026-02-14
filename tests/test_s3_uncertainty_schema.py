"""Tests for s3_ringdown_estimates uncertainty propagation (Paso 1).

Test 1 — Schema additive: new uncertainty keys exist alongside originals.
Test 2 — Determinism: same input → same SHA-256 of estimates.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_deterministic_ringdown_npz(path: Path, *, seed: int = 42) -> None:
    """Write a synthetic damped-sinusoid NPZ that s3 can consume."""
    fs = 4096.0
    duration = 0.1  # 100 ms window
    n = int(fs * duration)
    t = np.arange(n) / fs

    f0 = 251.0
    tau = 0.004
    signal = np.exp(-t / tau) * np.cos(2.0 * np.pi * f0 * t)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, n)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=(signal + noise).astype(np.float64),
             sample_rate_hz=np.float64(fs))


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )


def _create_s2_outputs(runs_root: Path, run_id: str, *, seed: int = 42) -> None:
    """Create mock s2 outputs (H1_rd.npz, L1_rd.npz, window_meta.json)."""
    out_dir = runs_root / run_id / "s2_ringdown_window" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_deterministic_ringdown_npz(out_dir / "H1_rd.npz", seed=seed)
    _make_deterministic_ringdown_npz(out_dir / "L1_rd.npz", seed=seed + 1)
    (out_dir / "window_meta.json").write_text(
        json.dumps({"event_id": "TEST_EVENT"}), encoding="utf-8"
    )
    # s2 stage_summary so contracts don't complain
    stage_dir = runs_root / run_id / "s2_ringdown_window"
    (stage_dir / "stage_summary.json").write_text(
        json.dumps({"stage": "s2_ringdown_window", "verdict": "PASS"}),
        encoding="utf-8",
    )


def _run_s3(run_id: str, runs_root: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(MVP_DIR / "s3_ringdown_estimates.py"),
        "--run", run_id,
        "--band-low", "100", "--band-high", "500",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(REPO_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Test 1: Schema additive ─────────────────────────────────────────────

class TestSchemaAdditive:
    """New uncertainty keys exist AND original keys are untouched."""

    def test_combined_keys_preserved(self, tmp_path):
        """combined contains f_hz, tau_s, Q (original contract)."""
        runs_root = tmp_path / "runs"
        run_id = "test_schema"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root)
        assert r.returncode == 0, f"s3 failed: {r.stderr}"

        est = json.loads(
            (runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json")
            .read_text(encoding="utf-8")
        )

        # Original combined keys must still be present
        assert "combined" in est
        for key in ("f_hz", "tau_s", "Q"):
            assert key in est["combined"], f"Missing combined.{key}"
            assert isinstance(est["combined"][key], float)

    def test_combined_uncertainty_exists(self, tmp_path):
        """combined_uncertainty block exists with all required sigma fields."""
        runs_root = tmp_path / "runs"
        run_id = "test_unc_block"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root)
        assert r.returncode == 0, f"s3 failed: {r.stderr}"

        est = json.loads(
            (runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json")
            .read_text(encoding="utf-8")
        )

        assert "combined_uncertainty" in est, "Missing combined_uncertainty block"
        unc = est["combined_uncertainty"]
        for key in ("sigma_f_hz", "sigma_tau_s", "sigma_Q", "cov_logf_logQ"):
            assert key in unc, f"Missing combined_uncertainty.{key}"

    def test_sigmas_finite_and_positive(self, tmp_path):
        """All sigma_* values are finite and > 0."""
        runs_root = tmp_path / "runs"
        run_id = "test_sigma_pos"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root)
        assert r.returncode == 0, f"s3 failed: {r.stderr}"

        est = json.loads(
            (runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json")
            .read_text(encoding="utf-8")
        )
        unc = est["combined_uncertainty"]

        for key in ("sigma_f_hz", "sigma_tau_s", "sigma_Q"):
            val = unc[key]
            assert math.isfinite(val), f"{key} is not finite: {val}"
            assert val > 0, f"{key} must be > 0, got {val}"

    def test_per_detector_has_uncertainty_fields(self, tmp_path):
        """Per-detector estimates include sigma_* fields."""
        runs_root = tmp_path / "runs"
        run_id = "test_det_sigma"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root)
        assert r.returncode == 0, f"s3 failed: {r.stderr}"

        est = json.loads(
            (runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json")
            .read_text(encoding="utf-8")
        )

        for det, det_est in est["per_detector"].items():
            if "error" in det_est:
                continue
            for key in ("sigma_f_hz", "sigma_tau_s", "sigma_Q", "cov_logf_logQ"):
                assert key in det_est, f"{det} missing {key}"
                assert math.isfinite(det_est[key]), f"{det}.{key} not finite"

    def test_q_relation_preserved(self, tmp_path):
        """Q == pi * f * tau still holds for the combined estimate."""
        runs_root = tmp_path / "runs"
        run_id = "test_q_rel"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root)
        assert r.returncode == 0, f"s3 failed: {r.stderr}"

        est = json.loads(
            (runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json")
            .read_text(encoding="utf-8")
        )
        c = est["combined"]
        assert abs(c["Q"] - math.pi * c["f_hz"] * c["tau_s"]) < 1e-6


# ── Test 2: Determinism ─────────────────────────────────────────────────

class TestDeterminism:
    """Same input → identical estimates.json (SHA-256 golden hash)."""

    def test_two_runs_produce_identical_hash(self, tmp_path):
        runs_root = tmp_path / "runs"

        hashes = []
        for i in range(2):
            run_id = f"det_run_{i}"
            _create_run_valid(runs_root, run_id)
            _create_s2_outputs(runs_root, run_id, seed=42)

            r = _run_s3(run_id, runs_root)
            assert r.returncode == 0, f"s3 run {i} failed: {r.stderr}"

            est_path = (
                runs_root / run_id / "s3_ringdown_estimates"
                / "outputs" / "estimates.json"
            )
            hashes.append(_sha256(est_path))

        assert hashes[0] == hashes[1], (
            f"Non-deterministic: hash0={hashes[0]}, hash1={hashes[1]}"
        )
