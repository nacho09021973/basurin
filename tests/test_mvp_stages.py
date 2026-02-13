"""MVP Pipeline — one test per stage + integration test.

All tests use synthetic data (no network access required).
Each test validates the BASURIN contract: manifest.json + stage_summary.json + outputs/.
"""
from __future__ import annotations

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
ATLAS_FIXTURE = MVP_DIR / "test_atlas_fixture.json"


def _run_stage(script: str, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=str(REPO_ROOT))


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")


def _create_synthetic_strain(runs_root: Path, run_id: str, event_id: str = "GW150914") -> None:
    """Create synthetic strain data as if s1 had run, bypassing network."""
    stage_dir = runs_root / run_id / "s1_fetch_strain"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    fs = 4096.0
    duration = 32.0
    gps_start = 1126259462.4204 - duration / 2.0
    n = int(fs * duration)
    t = np.arange(n) / fs

    # Place ringdown at center (coalescence time)
    t_ring = duration / 2.0
    dt = t - t_ring
    f0 = 251.0
    tau = 0.004
    # Only compute exponential for dt >= 0 to avoid overflow
    decay = np.zeros(n)
    mask = dt >= 0
    decay[mask] = np.exp(-dt[mask] / tau) * np.cos(2 * np.pi * f0 * dt[mask])
    signal = decay

    rng = np.random.default_rng(42)
    noise_h1 = rng.normal(0, 0.01, n)
    noise_l1 = rng.normal(0, 0.01, n)

    np.savez(
        outputs_dir / "strain.npz",
        H1=(signal + noise_h1).astype(np.float64),
        L1=(signal + noise_l1).astype(np.float64),
        sample_rate_hz=np.float64(fs),
        gps_start=np.float64(gps_start),
        duration_s=np.float64(duration),
    )

    provenance = {
        "event_id": event_id, "source": "synthetic_test",
        "gps_start": gps_start, "duration_s": duration, "sample_rate_hz": fs,
    }
    (outputs_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    # Stage summary + manifest
    summary = {"stage": "s1_fetch_strain", "run": run_id, "verdict": "PASS"}
    (stage_dir / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage_dir / "manifest.json").write_text(json.dumps({"stage": "s1_fetch_strain"}), encoding="utf-8")


def _assert_stage_contract(runs_root: Path, run_id: str, stage_name: str) -> dict:
    """Validate that a stage produced the required contract artifacts."""
    stage_dir = runs_root / run_id / stage_name
    assert stage_dir.exists(), f"Stage dir missing: {stage_dir}"

    summary_path = stage_dir / "stage_summary.json"
    assert summary_path.exists(), f"stage_summary.json missing in {stage_dir}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    manifest_path = stage_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json missing in {stage_dir}"

    outputs_dir = stage_dir / "outputs"
    assert outputs_dir.exists(), f"outputs/ missing in {stage_dir}"

    return summary


# ── Test Stage 1: Fetch Strain ────────────────────────────────────────────

class TestS1FetchStrain:
    def test_synthetic_mode_produces_contract(self, tmp_path):
        """s1 --synthetic produces strain.npz + provenance.json + contract files."""
        runs_root = tmp_path / "runs"
        run_id = "test_s1"
        _create_run_valid(runs_root, run_id)

        result = _run_stage(
            "s1_fetch_strain.py",
            ["--run", run_id, "--event-id", "GW150914", "--synthetic", "--duration-s", "4"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s1 failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, run_id, "s1_fetch_strain")
        assert summary["verdict"] == "PASS"

        # Check strain.npz content
        npz = np.load(runs_root / run_id / "s1_fetch_strain" / "outputs" / "strain.npz")
        assert "H1" in npz.files
        assert "L1" in npz.files
        assert "sample_rate_hz" in npz.files
        assert npz["H1"].ndim == 1
        assert npz["H1"].size > 0

    def test_local_hdf5_mode_produces_contract(self, tmp_path):
        """s1 --local-hdf5 writes inputs copy + strain/provenance with local_hdf5 source."""
        h5py = pytest.importorskip("h5py")

        runs_root = tmp_path / "runs"
        run_id = "test_s1_local"
        _create_run_valid(runs_root, run_id)

        fs = 256.0
        duration = 4.0
        n = int(fs * duration)
        gps_start = 1126259446.0

        local_h1 = tmp_path / "H-H1_local.hdf5"
        local_l1 = tmp_path / "L-L1_local.hdf5"
        for p, seed in [(local_h1, 1), (local_l1, 2)]:
            with h5py.File(p, "w") as h5:
                grp = h5.create_group("strain")
                data = np.random.default_rng(seed).normal(0.0, 1e-21, n).astype(np.float64)
                ds = grp.create_dataset("Strain", data=data)
                ds.attrs["Xspacing"] = 1.0 / fs
                ds.attrs["Xstart"] = gps_start

        result = _run_stage(
            "s1_fetch_strain.py",
            [
                "--run", run_id,
                "--event-id", "GW150914",
                "--duration-s", str(duration),
                "--detectors", "H1,L1",
                "--local-hdf5", f"H1={local_h1}",
                "--local-hdf5", f"L1={local_l1}",
            ],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s1 local failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, run_id, "s1_fetch_strain")
        assert summary["verdict"] == "PASS"

        stage_dir = runs_root / run_id / "s1_fetch_strain"
        assert (stage_dir / "outputs" / "strain.npz").exists()
        prov_path = stage_dir / "outputs" / "provenance.json"
        assert prov_path.exists()
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov["source"] == "local_hdf5"
        assert set(prov["local_inputs"].keys()) == {"H1", "L1"}
        assert set(prov["local_input_sha256"].keys()) == {"H1", "L1"}
        assert (stage_dir / "inputs" / local_h1.name).exists()
        assert (stage_dir / "inputs" / local_l1.name).exists()


# ── Test Stage 2: Ringdown Window ─────────────────────────────────────────

class TestS2RingdownWindow:
    def test_window_crop_produces_contract(self, tmp_path):
        """s2 crops correct window and produces H1_rd.npz, L1_rd.npz, window_meta.json."""
        runs_root = tmp_path / "runs"
        run_id = "test_s2"
        _create_run_valid(runs_root, run_id)
        _create_synthetic_strain(runs_root, run_id)

        result = _run_stage(
            "s2_ringdown_window.py",
            ["--run", run_id, "--event-id", "GW150914",
             "--dt-start-s", "0.003", "--duration-s", "0.06"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s2 failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, run_id, "s2_ringdown_window")
        assert summary["verdict"] == "PASS"

        outputs = runs_root / run_id / "s2_ringdown_window" / "outputs"
        assert (outputs / "H1_rd.npz").exists()
        assert (outputs / "L1_rd.npz").exists()
        assert (outputs / "window_meta.json").exists()

        # Check window size (round matches the stage's int(round(...)))
        h1 = np.load(outputs / "H1_rd.npz")
        expected_samples = int(round(0.06 * 4096))
        assert h1["strain"].size == expected_samples

    def test_window_out_of_range_aborts(self, tmp_path):
        """s2 aborts if window falls outside strain range."""
        runs_root = tmp_path / "runs"
        run_id = "test_s2_oor"
        _create_run_valid(runs_root, run_id)
        _create_synthetic_strain(runs_root, run_id)

        result = _run_stage(
            "s2_ringdown_window.py",
            ["--run", run_id, "--event-id", "GW150914",
             "--dt-start-s", "100.0", "--duration-s", "0.06"],  # way past end
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2, "Should abort with exit code 2"


# ── Test Stage 3: Ringdown Estimates ──────────────────────────────────────

class TestS3RingdownEstimates:
    def test_estimates_from_synthetic_ringdown(self, tmp_path):
        """s3 estimates f ~ 251 Hz and Q > 0 from synthetic ringdown signal."""
        runs_root = tmp_path / "runs"
        run_id = "test_s3"
        _create_run_valid(runs_root, run_id)
        _create_synthetic_strain(runs_root, run_id)

        # Run s2 first to create windowed data
        _run_stage(
            "s2_ringdown_window.py",
            ["--run", run_id, "--event-id", "GW150914",
             "--dt-start-s", "0.0", "--duration-s", "0.1"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )

        result = _run_stage(
            "s3_ringdown_estimates.py",
            ["--run", run_id, "--band-low", "100", "--band-high", "500"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s3 failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, run_id, "s3_ringdown_estimates")
        assert summary["verdict"] == "PASS"

        # Check estimates
        est_path = runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json"
        est = json.loads(est_path.read_text(encoding="utf-8"))
        assert "combined" in est
        f_hz = est["combined"]["f_hz"]
        Q = est["combined"]["Q"]
        tau_s = est["combined"]["tau_s"]

        # Frequency should be near 251 Hz (within 30% for noisy synthetic)
        assert 150 < f_hz < 400, f"f_hz={f_hz} out of expected range"
        assert Q > 0, f"Q must be > 0, got {Q}"
        assert tau_s > 0, f"tau_s must be > 0, got {tau_s}"
        assert abs(Q - math.pi * f_hz * tau_s) < 1e-6, "Q != pi*f*tau"


# ── Test Stage 4: Geometry Filter ─────────────────────────────────────────

class TestS4GeometryFilter:
    def test_filter_finds_compatible_geometries(self, tmp_path):
        """s4 finds geometries near observed (f, Q) in test atlas."""
        runs_root = tmp_path / "runs"
        run_id = "test_s4"
        _create_run_valid(runs_root, run_id)

        # Create mock estimates (as if s3 ran)
        est_dir = runs_root / run_id / "s3_ringdown_estimates"
        est_out = est_dir / "outputs"
        est_out.mkdir(parents=True, exist_ok=True)
        estimates = {
            "schema_version": "mvp_estimates_v1",
            "event_id": "GW150914",
            "combined": {"f_hz": 251.0, "tau_s": 0.004, "Q": math.pi * 251.0 * 0.004},
        }
        (est_out / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
        (est_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )

        result = _run_stage(
            "s4_geometry_filter.py",
            ["--run", run_id, "--atlas-path", str(ATLAS_FIXTURE), "--epsilon", "0.3"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s4 failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, run_id, "s4_geometry_filter")
        assert summary["verdict"] == "PASS"

        cs_path = runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        cs = json.loads(cs_path.read_text(encoding="utf-8"))
        assert cs["n_atlas"] == 20
        assert cs["n_compatible"] >= 1, "At least one geometry should be compatible"
        assert cs["epsilon"] == 0.3
        assert all("geometry_id" in g for g in cs["compatible_geometries"])

    def test_empty_atlas_aborts(self, tmp_path):
        """s4 aborts on empty atlas."""
        runs_root = tmp_path / "runs"
        run_id = "test_s4_empty"
        _create_run_valid(runs_root, run_id)

        # Mock estimates
        est_dir = runs_root / run_id / "s3_ringdown_estimates"
        est_out = est_dir / "outputs"
        est_out.mkdir(parents=True, exist_ok=True)
        (est_out / "estimates.json").write_text(
            json.dumps({"combined": {"f_hz": 251.0, "Q": 3.0}}), encoding="utf-8"
        )
        (est_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )

        empty_atlas = tmp_path / "empty_atlas.json"
        empty_atlas.write_text(json.dumps({"entries": []}), encoding="utf-8")

        result = _run_stage(
            "s4_geometry_filter.py",
            ["--run", run_id, "--atlas-path", str(empty_atlas), "--epsilon", "0.3"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2


# ── Test Stage 5: Aggregate ───────────────────────────────────────────────

class TestS5Aggregate:
    def test_intersection_of_two_events(self, tmp_path):
        """s5 finds common geometries across two events."""
        runs_root = tmp_path / "runs"

        # Create two mock event runs with overlapping compatible sets
        for run_id, gids in [("event_A", ["geo_005", "geo_006", "geo_007"]),
                              ("event_B", ["geo_005", "geo_007", "geo_010"])]:
            _create_run_valid(runs_root, run_id)
            s4_dir = runs_root / run_id / "s4_geometry_filter"
            s4_out = s4_dir / "outputs"
            s4_out.mkdir(parents=True, exist_ok=True)
            cs = {
                "event_id": run_id,
                "compatible_geometries": [
                    {"geometry_id": g, "distance": 0.1, "compatible": True}
                    for g in gids
                ],
            }
            (s4_out / "compatible_set.json").write_text(json.dumps(cs), encoding="utf-8")
            (s4_dir / "stage_summary.json").write_text(
                json.dumps({"verdict": "PASS"}), encoding="utf-8"
            )

        result = _run_stage(
            "s5_aggregate.py",
            ["--out-run", "agg_test", "--source-runs", "event_A,event_B", "--min-coverage", "1.0"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s5 failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, "agg_test", "s5_aggregate")
        assert summary["verdict"] == "PASS"

        agg_path = runs_root / "agg_test" / "s5_aggregate" / "outputs" / "aggregate.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        assert agg["n_events"] == 2
        common_ids = {g["geometry_id"] for g in agg["common_geometries"]}
        assert common_ids == {"geo_005", "geo_007"}, f"Expected intersection, got {common_ids}"


# ── Integration Test ──────────────────────────────────────────────────────

class TestIntegration:
    def test_full_single_event_pipeline_synthetic(self, tmp_path):
        """Full pipeline runs end-to-end with synthetic data."""
        runs_root = tmp_path / "runs"
        run_id = "integration_test"

        env = {"BASURIN_RUNS_ROOT": str(runs_root)}

        # Stage 1
        _create_run_valid(runs_root, run_id)
        r = _run_stage("s1_fetch_strain.py",
                        ["--run", run_id, "--event-id", "GW150914",
                         "--synthetic", "--duration-s", "4"], env=env)
        assert r.returncode == 0, f"s1: {r.stderr}"

        # Stage 2
        r = _run_stage("s2_ringdown_window.py",
                        ["--run", run_id, "--event-id", "GW150914",
                         "--dt-start-s", "0.0", "--duration-s", "0.1"], env=env)
        assert r.returncode == 0, f"s2: {r.stderr}"

        # Stage 3
        r = _run_stage("s3_ringdown_estimates.py",
                        ["--run", run_id, "--band-low", "100", "--band-high", "500"], env=env)
        assert r.returncode == 0, f"s3: {r.stderr}"

        # Stage 4
        r = _run_stage("s4_geometry_filter.py",
                        ["--run", run_id, "--atlas-path", str(ATLAS_FIXTURE),
                         "--epsilon", "0.5"], env=env)
        assert r.returncode == 0, f"s4: {r.stderr}"

        # Verify all stages produced contracts
        for stage in ("s1_fetch_strain", "s2_ringdown_window",
                       "s3_ringdown_estimates", "s4_geometry_filter"):
            summary = _assert_stage_contract(runs_root, run_id, stage)
            assert summary["verdict"] == "PASS", f"{stage} verdict != PASS"

        # Verify final compatible set exists
        cs_path = runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        cs = json.loads(cs_path.read_text(encoding="utf-8"))
        assert cs["n_atlas"] > 0
        assert isinstance(cs["compatible_geometries"], list)


# ── Test --reuse-if-present ──────────────────────────────────────────────

class TestS1ReuseIfPresent:
    def test_reuse_skips_fetch_when_outputs_match(self, tmp_path):
        """--reuse-if-present skips fetch when existing outputs match params."""
        runs_root = tmp_path / "runs"
        run_id = "test_reuse"
        _create_run_valid(runs_root, run_id)
        env = {"BASURIN_RUNS_ROOT": str(runs_root)}

        # First run: generate outputs normally
        r1 = _run_stage(
            "s1_fetch_strain.py",
            ["--run", run_id, "--event-id", "GW150914", "--synthetic", "--duration-s", "4"],
            env=env,
        )
        assert r1.returncode == 0, f"first run failed: {r1.stderr}"

        # Record file mod-time of strain.npz
        npz_path = runs_root / run_id / "s1_fetch_strain" / "outputs" / "strain.npz"
        mtime_before = npz_path.stat().st_mtime

        # Second run with --reuse-if-present (same run_id, outputs already there)
        r2 = _run_stage(
            "s1_fetch_strain.py",
            ["--run", run_id, "--event-id", "GW150914", "--synthetic",
             "--duration-s", "4", "--reuse-if-present"],
            env=env,
        )
        assert r2.returncode == 0, f"reuse run failed: {r2.stderr}"
        assert "reuse: outputs valid, skipping fetch" in r2.stdout

        # strain.npz should NOT have been rewritten
        mtime_after = npz_path.stat().st_mtime
        assert mtime_before == mtime_after, "strain.npz was rewritten despite reuse"

        # stage_summary should record reused=True
        summary = json.loads(
            (runs_root / run_id / "s1_fetch_strain" / "stage_summary.json").read_text()
        )
        assert summary.get("reused") is True

    def test_reuse_falls_back_on_param_mismatch(self, tmp_path):
        """--reuse-if-present re-fetches when duration_s differs."""
        runs_root = tmp_path / "runs"
        run_id = "test_reuse_mismatch"
        _create_run_valid(runs_root, run_id)
        env = {"BASURIN_RUNS_ROOT": str(runs_root)}

        # First run with duration=4
        r1 = _run_stage(
            "s1_fetch_strain.py",
            ["--run", run_id, "--event-id", "GW150914", "--synthetic", "--duration-s", "4"],
            env=env,
        )
        assert r1.returncode == 0

        # Second run with duration=8 + reuse flag — should NOT reuse
        r2 = _run_stage(
            "s1_fetch_strain.py",
            ["--run", run_id, "--event-id", "GW150914", "--synthetic",
             "--duration-s", "8", "--reuse-if-present"],
            env=env,
        )
        assert r2.returncode == 0
        assert "duration_s mismatch" in r2.stdout

    def test_reuse_falls_back_on_missing_outputs(self, tmp_path):
        """--reuse-if-present fetches normally when no prior outputs exist."""
        runs_root = tmp_path / "runs"
        run_id = "test_reuse_empty"
        _create_run_valid(runs_root, run_id)
        env = {"BASURIN_RUNS_ROOT": str(runs_root)}

        r = _run_stage(
            "s1_fetch_strain.py",
            ["--run", run_id, "--event-id", "GW150914", "--synthetic",
             "--duration-s", "4", "--reuse-if-present"],
            env=env,
        )
        assert r.returncode == 0
        assert "outputs not found, will fetch" in r.stdout
