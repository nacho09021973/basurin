"""Tests for the epsilon-sweep experiment (experiment_eps_sweep.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"
ATLAS_FIXTURE = MVP_DIR / "test_atlas_fixture.json"


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")


def _create_s3_estimates(
    runs_root: Path,
    run_id: str,
    f_hz: float = 251.0,
    Q: float = 3.14,
    include_uncertainty: bool = False,
) -> Path:
    """Create synthetic s3 estimates as if s3 had run."""
    stage_dir = runs_root / run_id / "s3_ringdown_estimates"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    estimates = {
        "schema_version": "mvp_estimates_v1",
        "event_id": "GW150914",
        "method": "hilbert_envelope",
        "combined": {"f_hz": f_hz, "tau_s": Q / (3.14159 * f_hz), "Q": Q},
        "per_detector": {"H1": {"f_hz": f_hz, "tau_s": 0.004, "Q": Q, "snr_peak": 8.0}},
        "n_detectors_valid": 1,
    }
    if include_uncertainty:
        estimates["combined_uncertainty"] = {
            "sigma_logf": 0.2,
            "sigma_logQ": 0.5,
            "cov_logf_logQ": 0.0,
        }
    est_path = outputs_dir / "estimates.json"
    est_path.write_text(json.dumps(estimates, indent=2), encoding="utf-8")

    (stage_dir / "stage_summary.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates", "run": run_id, "verdict": "PASS"}),
        encoding="utf-8",
    )
    (stage_dir / "manifest.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8",
    )
    return est_path


class TestEpsSweep:
    """Tests for run_eps_sweep."""

    def test_sweep_produces_artifacts(self, tmp_path: Path) -> None:
        """Sweep creates per-epsilon dirs, compatible_set.json, manifests, and summary."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import EXPERIMENT_TAG, run_eps_sweep

        run_id = "test_sweep_001"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        epsilons = [0.10, 0.30, 0.50]
        summary = run_eps_sweep(run_id, ATLAS_FIXTURE, epsilons, runs_root=tmp_path)

        sweep_dir = tmp_path / run_id / "experiment" / EXPERIMENT_TAG

        # Top-level artifacts
        assert (sweep_dir / "sweep_summary.json").exists()
        assert (sweep_dir / "manifest.json").exists()

        # Per-epsilon artifacts
        for eps in epsilons:
            eps_dir = sweep_dir / f"eps_{eps:.3f}"
            assert eps_dir.exists(), f"Missing dir for eps={eps}"
            assert (eps_dir / "compatible_set.json").exists()
            assert (eps_dir / "manifest.json").exists()

        # Summary structure
        assert summary["schema_version"] == "mvp_eps_sweep_v1"
        assert summary["experiment_tag"] == EXPERIMENT_TAG
        assert summary["n_epsilons"] == 3
        assert len(summary["rows"]) == 3

    def test_monotonic_n_compatible(self, tmp_path: Path) -> None:
        """n_compatible must be non-decreasing as epsilon grows."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import run_eps_sweep

        run_id = "test_sweep_mono"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        epsilons = [0.05, 0.10, 0.20, 0.30, 0.50]
        summary = run_eps_sweep(run_id, ATLAS_FIXTURE, epsilons, runs_root=tmp_path)

        n_vals = [row["n_compatible"] for row in summary["rows"]]
        for i in range(1, len(n_vals)):
            assert n_vals[i] >= n_vals[i - 1], (
                f"n_compatible decreased: eps={epsilons[i - 1]}->{epsilons[i]}, "
                f"n={n_vals[i - 1]}->{n_vals[i]}"
            )

    def test_bits_excluded_decreases(self, tmp_path: Path) -> None:
        """bits_excluded must be non-increasing as epsilon grows."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import run_eps_sweep

        run_id = "test_sweep_bits"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        epsilons = [0.05, 0.10, 0.20, 0.30, 0.50]
        summary = run_eps_sweep(run_id, ATLAS_FIXTURE, epsilons, runs_root=tmp_path)

        bits_vals = [row["bits_excluded"] for row in summary["rows"]]
        for i in range(1, len(bits_vals)):
            assert bits_vals[i] <= bits_vals[i - 1], (
                f"bits_excluded increased: eps={epsilons[i - 1]}->{epsilons[i]}, "
                f"bits={bits_vals[i - 1]}->{bits_vals[i]}"
            )

    def test_compatible_set_json_structure(self, tmp_path: Path) -> None:
        """Each per-epsilon compatible_set.json has the expected schema."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import EXPERIMENT_TAG, run_eps_sweep

        run_id = "test_sweep_schema"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        epsilons = [0.20]
        run_eps_sweep(run_id, ATLAS_FIXTURE, epsilons, runs_root=tmp_path)

        cs_path = tmp_path / run_id / "experiment" / EXPERIMENT_TAG / "eps_0.200" / "compatible_set.json"
        cs = json.loads(cs_path.read_text(encoding="utf-8"))

        assert cs["schema_version"] == "mvp_compatible_set_v1"
        assert cs["epsilon"] == 0.20
        assert "n_atlas" in cs
        assert "n_compatible" in cs
        assert "bits_excluded" in cs
        assert "compatible_geometries" in cs
        assert "ranked_all" in cs
        assert cs["event_id"] == "GW150914"
        assert cs["run_id"] == run_id

    def test_missing_estimates_raises(self, tmp_path: Path) -> None:
        """Sweep raises if s3 estimates don't exist."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import run_eps_sweep

        run_id = "test_sweep_noest"
        _create_run_valid(tmp_path, run_id)
        # No s3 estimates created

        with pytest.raises(FileNotFoundError, match="s3 estimates not found"):
            run_eps_sweep(run_id, ATLAS_FIXTURE, [0.30], runs_root=tmp_path)

    def test_missing_atlas_raises(self, tmp_path: Path) -> None:
        """Sweep raises if atlas doesn't exist."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import run_eps_sweep

        run_id = "test_sweep_noatlas"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        with pytest.raises(FileNotFoundError, match="Atlas not found"):
            run_eps_sweep(run_id, Path("/nonexistent/atlas.json"), [0.30], runs_root=tmp_path)

    def test_sweep_summary_sha256(self, tmp_path: Path) -> None:
        """Sweep manifest includes SHA256 hashes."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))

        from mvp.experiment_eps_sweep import EXPERIMENT_TAG, run_eps_sweep

        run_id = "test_sweep_hash"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        run_eps_sweep(run_id, ATLAS_FIXTURE, [0.10, 0.30], runs_root=tmp_path)

        manifest_path = tmp_path / run_id / "experiment" / EXPERIMENT_TAG / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert "hashes" in manifest
        assert len(manifest["hashes"]) > 0
        # Every hash should be a 64-char hex string
        for label, h in manifest["hashes"].items():
            assert len(h) == 64, f"Bad hash for {label}: {h}"

    def test_cli_mahalanobis_requires_uncertainties(self, tmp_path: Path) -> None:
        run_id = "test_cli_mah_requires_unc"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id, include_uncertainty=False)

        env = dict(**os.environ, BASURIN_RUNS_ROOT=str(tmp_path))
        proc = subprocess.run(
            [
                sys.executable,
                str(MVP_DIR / "experiment_eps_sweep.py"),
                "--run", run_id,
                "--atlas-path", str(ATLAS_FIXTURE),
                "--metric", "mahalanobis_log",
                "--epsilons", "0.1",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 2
        assert "requires uncertainties" in proc.stderr

    def test_cli_mahalanobis_with_cli_sigmas_succeeds(self, tmp_path: Path) -> None:
        run_id = "test_cli_mah_sigmas"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id, include_uncertainty=False)

        env = dict(**os.environ, BASURIN_RUNS_ROOT=str(tmp_path))
        proc = subprocess.run(
            [
                sys.executable,
                str(MVP_DIR / "experiment_eps_sweep.py"),
                "--run", run_id,
                "--atlas-path", str(ATLAS_FIXTURE),
                "--metric", "mahalanobis_log",
                "--epsilons", "0.1",
                "--sigma-lnf", "0.2",
                "--sigma-lnQ", "0.5",
                "--correlation", "0.0",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0
