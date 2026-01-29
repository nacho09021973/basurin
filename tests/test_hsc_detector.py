"""
Tests for HSC Detector stage.

Tests cover:
  - PASS on valid HSC-like spectrum and OPE hierarchy
  - FAIL on generic/dense CFT spectrum
  - Determinism (same input → same output hash)
  - IO contract (all required files exist and are valid JSON)
  - Edge cases (empty data, missing conventions)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to stage script
STAGE_PATH = Path(__file__).resolve().parents[1] / "experiment" / "hsc_detector" / "stage_hsc_detector.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_test_out_root(tmp_path: Path) -> Path:
    """Get test output root following BASURIN conventions."""
    out_root_abs = tmp_path / "runs"
    out_root_abs.mkdir(parents=True, exist_ok=True)
    return out_root_abs


def _create_hsc_input(
    run_dir: Path,
    operators: list[dict],
    ope_coefficients: dict[str, float],
    d: float = 3.0,
    light_ops: list[str] | None = None,
    tower_prefixes: list[str] | None = None,
) -> Path:
    """Create an HSC input.json file for testing."""
    input_dir = run_dir / "inputs" / "hsc"
    input_dir.mkdir(parents=True, exist_ok=True)

    input_data = {
        "metadata": {
            "schema_version": "hsc_input_v1",
            "theory_name": "Test Theory",
            "d": d,
            "source": "Test Generator",
            "conventions": {
                "light_ops": light_ops or ["sigma", "epsilon"],
                "tower_ops_prefix": tower_prefixes or ["[sigma sigma]_", "[epsilon epsilon]_"],
            },
        },
        "spectrum": {
            "operators": operators,
        },
        "ope_coefficients": ope_coefficients,
    }

    input_path = input_dir / "input.json"
    input_path.write_text(json.dumps(input_data, indent=2), encoding="utf-8")
    return input_path


def _write_run_valid(run_dir: Path, verdict: str = "PASS") -> None:
    outputs_dir = run_dir / "RUN_VALID" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_valid_payload = {"run": run_dir.name, "verdict": verdict}
    (outputs_dir / "run_valid.json").write_text(
        json.dumps(run_valid_payload, indent=2), encoding="utf-8"
    )


def _run_stage(
    run_id: str,
    input_path: Path,
    out_root_abs: Path,
    thresholds_path: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run the HSC detector stage via subprocess."""
    cmd = [
        sys.executable,
        str(STAGE_PATH),
        "--run", run_id,
        "--input", str(input_path),
        "--out-root", str(out_root_abs),
    ]
    if thresholds_path:
        cmd.extend(["--thresholds", str(thresholds_path)])

    # Set BASURIN_RUNS_ROOT to allow writing to test directories
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(out_root_abs)}

    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture
def test_run_dir(tmp_path: Path):
    """Fixture to create and cleanup test run directory."""
    out_root_abs = _get_test_out_root(tmp_path)
    yield out_root_abs


class TestHSCDetectorPassOnHSC:
    """Test that HSC detector passes on valid holographic-like spectrum."""

    def test_passes_on_sparse_spectrum_with_hierarchy(self, test_run_dir: Path) -> None:
        """Sparse spectrum + strong OPE hierarchy → PASS_LOCAL_BULK_CANDIDATE."""
        out_root_abs = test_run_dir
        run_id = "test_pass_hsc"
        run_dir = out_root_abs / run_id

        # HSC-like spectrum: sparse, few low-lying scalars
        operators = [
            {"id": "0", "dim": 0.0, "spin": 0, "degeneracy": 1},  # identity
            {"id": "sigma", "dim": 2.017, "spin": 0, "degeneracy": 1},
            {"id": "epsilon", "dim": 4.132, "spin": 0, "degeneracy": 1},
            {"id": "[sigma sigma]_0", "dim": 4.523, "spin": 0, "degeneracy": 1},
        ]

        # HSC-like OPE: strong hierarchy (light-light-light >> light-light-tower)
        ope_coefficients = {
            "sigma_sigma_sigma": 0.451,
            "sigma_epsilon_epsilon": 0.312,
            "epsilon_epsilon_epsilon": 0.012,
            "sigma_sigma_[sigma sigma]_0": 0.021,
            "sigma_sigma_[sigma sigma]_1": 0.011,
            "epsilon_epsilon_[epsilon epsilon]_0": 0.015,
        }

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)
        result = _run_stage(run_id, input_path, out_root_abs)

        assert result.returncode == 0, f"Stage failed: {result.stderr}"
        assert "PASS_LOCAL_BULK_CANDIDATE" in result.stdout

        # Verify verdict.json
        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        assert verdict_path.exists()
        verdict = json.loads(verdict_path.read_text())
        assert verdict["overall_verdict"] == "PASS_LOCAL_BULK_CANDIDATE"
        assert verdict["phase_summary"]["phase_1"]["verdict"] == "PASS"
        assert verdict["phase_summary"]["phase_2"]["verdict"] == "PASS"


class TestHSCDetectorFailOnGeneric:
    """Test that HSC detector fails on generic/dense CFT."""

    def test_fails_on_dense_spectrum(self, test_run_dir: Path) -> None:
        """Dense spectrum → FAIL (even if OPE might pass)."""
        out_root_abs = test_run_dir
        run_id = "test_fail_dense"
        run_dir = out_root_abs / run_id

        # Generic CFT: dense spectrum with many low-lying scalars
        operators = [
            {"id": "0", "dim": 0.0, "spin": 0, "degeneracy": 1},
            {"id": "sigma", "dim": 1.5, "spin": 0, "degeneracy": 1},
            {"id": "op2", "dim": 2.0, "spin": 0, "degeneracy": 1},
            {"id": "op3", "dim": 2.5, "spin": 0, "degeneracy": 1},
            {"id": "op4", "dim": 3.0, "spin": 0, "degeneracy": 1},
            {"id": "op5", "dim": 3.2, "spin": 0, "degeneracy": 1},
            {"id": "op6", "dim": 3.5, "spin": 0, "degeneracy": 1},
            {"id": "op7", "dim": 3.8, "spin": 0, "degeneracy": 1},
            {"id": "op8", "dim": 4.0, "spin": 0, "degeneracy": 1},
            {"id": "op9", "dim": 4.2, "spin": 0, "degeneracy": 1},
            {"id": "op10", "dim": 4.5, "spin": 0, "degeneracy": 1},
            {"id": "op11", "dim": 4.8, "spin": 0, "degeneracy": 1},
        ]

        ope_coefficients = {
            "sigma_sigma_sigma": 0.5,
            "sigma_sigma_[sigma sigma]_0": 0.02,
        }

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)
        result = _run_stage(run_id, input_path, out_root_abs)

        assert result.returncode == 0
        assert "FAIL_LOCAL_BULK" in result.stdout

        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        verdict = json.loads(verdict_path.read_text())
        assert verdict["overall_verdict"] == "FAIL_LOCAL_BULK"
        assert verdict["phase_summary"]["phase_1"]["verdict"] == "FAIL"

    def test_fails_on_no_ope_hierarchy(self, test_run_dir: Path) -> None:
        """No OPE hierarchy (ratio <= 1) → FAIL."""
        out_root_abs = test_run_dir
        run_id = "test_fail_no_hierarchy"
        run_dir = out_root_abs / run_id

        # Sparse spectrum (would pass phase 1)
        operators = [
            {"id": "0", "dim": 0.0, "spin": 0, "degeneracy": 1},
            {"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1},
            {"id": "epsilon", "dim": 4.0, "spin": 0, "degeneracy": 1},
        ]

        # No hierarchy: tower coefficients >= light coefficients
        ope_coefficients = {
            "sigma_sigma_sigma": 0.01,  # small light-light-light
            "sigma_epsilon_epsilon": 0.02,
            "sigma_sigma_[sigma sigma]_0": 0.5,  # large light-light-tower
            "sigma_sigma_[sigma sigma]_1": 0.4,
        }

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)
        result = _run_stage(run_id, input_path, out_root_abs)

        assert result.returncode == 0
        assert "FAIL_LOCAL_BULK" in result.stdout

        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        verdict = json.loads(verdict_path.read_text())
        assert verdict["overall_verdict"] == "FAIL_LOCAL_BULK"
        assert verdict["phase_summary"]["phase_2"]["verdict"] == "FAIL"


class TestHSCDetectorDeterminism:
    """Test that same input produces identical outputs."""

    def test_deterministic_output(self, test_run_dir: Path) -> None:
        """Running twice with same input → same verdict hash."""
        out_root_abs = test_run_dir

        operators = [
            {"id": "0", "dim": 0.0, "spin": 0, "degeneracy": 1},
            {"id": "sigma", "dim": 2.017, "spin": 0, "degeneracy": 1},
        ]
        ope_coefficients = {
            "sigma_sigma_sigma": 0.451,
            "sigma_sigma_[sigma sigma]_0": 0.021,
        }

        # Run 1
        run_id_1 = "test_determinism_1"
        run_dir_1 = out_root_abs / run_id_1
        _write_run_valid(run_dir_1, verdict="PASS")
        input_path_1 = _create_hsc_input(run_dir_1, operators, ope_coefficients)
        result_1 = _run_stage(run_id_1, input_path_1, out_root_abs)
        assert result_1.returncode == 0

        # Run 2
        run_id_2 = "test_determinism_2"
        run_dir_2 = out_root_abs / run_id_2
        _write_run_valid(run_dir_2, verdict="PASS")
        input_path_2 = _create_hsc_input(run_dir_2, operators, ope_coefficients)
        result_2 = _run_stage(run_id_2, input_path_2, out_root_abs)
        assert result_2.returncode == 0

        # Compare verdicts (excluding run-specific fields)
        verdict_1 = json.loads(
            (
                out_root_abs
                / run_id_1
                / "experiment"
                / "hsc_detector"
                / "outputs"
                / "verdict.json"
            ).read_text()
        )
        verdict_2 = json.loads(
            (
                out_root_abs
                / run_id_2
                / "experiment"
                / "hsc_detector"
                / "outputs"
                / "verdict.json"
            ).read_text()
        )

        # Verdicts should match
        assert verdict_1["overall_verdict"] == verdict_2["overall_verdict"]
        assert verdict_1["phase_summary"] == verdict_2["phase_summary"]
        # Input hashes should match (same content)
        assert verdict_1["input_data_hash"] == verdict_2["input_data_hash"]


class TestHSCDetectorIOContract:
    """Test that all required output files exist and are valid JSON."""

    def test_all_required_files_exist(self, test_run_dir: Path) -> None:
        """Verify manifest.json, stage_summary.json, and all outputs exist."""
        out_root_abs = test_run_dir
        run_id = "test_io_contract"
        run_dir = out_root_abs / run_id

        operators = [
            {"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1},
        ]
        ope_coefficients = {"sigma_sigma_sigma": 0.5}
        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)

        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        stage_dir = out_root_abs / run_id / "experiment" / "hsc_detector"
        legacy_stage_dir = out_root_abs / run_id / "hsc_detector"

        # Required files
        required_files = [
            "manifest.json",
            "stage_summary.json",
            "outputs/report.json",
            "outputs/verdict.json",
        ]

        assert not legacy_stage_dir.exists(), "Legacy hsc_detector dir should not exist"

        for rel_path in required_files:
            file_path = stage_dir / rel_path
            assert file_path.exists(), f"Missing required file: {rel_path}"
            # Verify valid JSON
            content = json.loads(file_path.read_text())
            assert isinstance(content, dict), f"Invalid JSON in {rel_path}"

    def test_manifest_contains_all_artifacts(self, test_run_dir: Path) -> None:
        """Manifest should list all generated output files."""
        out_root_abs = test_run_dir
        run_id = "test_manifest"
        run_dir = out_root_abs / run_id

        operators = [{"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1}]
        ope_coefficients = {"sigma_sigma_sigma": 0.5}
        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)

        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        manifest_path = (
            out_root_abs / run_id / "experiment" / "hsc_detector" / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())

        assert "files" in manifest
        assert "verdict" in manifest["files"]
        assert "report" in manifest["files"]

        assert "hashes" in manifest
        # Each file should have a hash
        for label, rel_path in manifest["files"].items():
            assert rel_path in manifest["hashes"], f"Missing hash for {label}"

    def test_stage_summary_structure(self, test_run_dir: Path) -> None:
        """Stage summary should have required fields."""
        out_root_abs = test_run_dir
        run_id = "test_summary"
        run_dir = out_root_abs / run_id

        operators = [{"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1}]
        ope_coefficients = {"sigma_sigma_sigma": 0.5}
        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)

        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        summary_path = (
            out_root_abs / run_id / "experiment" / "hsc_detector" / "stage_summary.json"
        )
        summary = json.loads(summary_path.read_text())

        assert summary["stage"] == "hsc_detector"
        assert summary["run"] == run_id
        assert "created" in summary
        assert "config" in summary
        assert "inputs" in summary
        assert "outputs" in summary
        assert "verdict" in summary
        assert summary["verdict"]["overall"] in [
            "PASS_LOCAL_BULK_CANDIDATE",
            "FAIL_LOCAL_BULK",
            "UNDERDETERMINED",
        ]

    def test_no_writes_outside_runs_root(self, test_run_dir: Path) -> None:
        """Stage should not write outside runs_root."""
        out_root_abs = test_run_dir
        run_id = "test_no_writes_outside"
        run_dir = out_root_abs / run_id

        operators = [{"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1}]
        ope_coefficients = {"sigma_sigma_sigma": 0.5}
        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)

        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        tmp_root = out_root_abs.parent
        for path in tmp_root.rglob("*"):
            if path.is_file():
                assert out_root_abs in path.parents, f"Unexpected write outside runs_root: {path}"


class TestHSCDetectorRunValidGate:
    """Test RUN_VALID gating behavior."""

    def test_abort_when_run_valid_not_pass(self, test_run_dir: Path) -> None:
        """RUN_VALID != PASS should abort and avoid outputs."""
        out_root_abs = test_run_dir
        run_id = "test_run_valid_fail"
        run_dir = out_root_abs / run_id

        operators = [{"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1}]
        ope_coefficients = {"sigma_sigma_sigma": 0.5}
        _write_run_valid(run_dir, verdict="FAIL")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)

        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode != 0

        stage_dir = out_root_abs / run_id / "experiment" / "hsc_detector"
        assert not stage_dir.exists()


class TestHSCDetectorEdgeCases:
    """Test edge cases and error handling."""

    def test_underdetermined_on_missing_conventions(self, test_run_dir: Path) -> None:
        """Missing conventions → UNDERDETERMINED for phase 2."""
        out_root_abs = test_run_dir
        run_id = "test_no_conventions"
        run_dir = out_root_abs / run_id

        # Create input without conventions
        input_dir = run_dir / "inputs" / "hsc"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_data = {
            "metadata": {
                "schema_version": "hsc_input_v1",
                "theory_name": "Test",
                "d": 3,
            },
            "spectrum": {
                "operators": [
                    {"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1},
                ],
            },
            "ope_coefficients": {
                "sigma_sigma_sigma": 0.5,
            },
        }

        input_path = input_dir / "input.json"
        input_path.write_text(json.dumps(input_data, indent=2))

        _write_run_valid(run_dir, verdict="PASS")
        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        verdict = json.loads(verdict_path.read_text())

        # Phase 2 should be UNDERDETERMINED due to missing conventions
        assert verdict["phase_summary"]["phase_2"]["verdict"] == "UNDERDETERMINED"

    def test_underdetermined_on_empty_spectrum(self, test_run_dir: Path) -> None:
        """Empty spectrum → UNDERDETERMINED for phase 1."""
        out_root_abs = test_run_dir
        run_id = "test_empty_spectrum"
        run_dir = out_root_abs / run_id

        operators: list[dict] = []  # No operators
        ope_coefficients = {"sigma_sigma_sigma": 0.5}

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)
        result = _run_stage(run_id, input_path, out_root_abs)

        assert result.returncode == 0

        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        verdict = json.loads(verdict_path.read_text())
        assert verdict["phase_summary"]["phase_1"]["verdict"] == "UNDERDETERMINED"

    def test_custom_thresholds(self, test_run_dir: Path) -> None:
        """Custom thresholds should override defaults."""
        out_root_abs = test_run_dir
        run_id = "test_custom_thresholds"
        run_dir = out_root_abs / run_id

        # Spectrum that would FAIL with default max_density=5
        # but PASS with higher threshold
        operators = [
            {"id": "0", "dim": 0.0, "spin": 0, "degeneracy": 1},
        ] + [
            {"id": f"op{i}", "dim": 1.0 + i * 0.3, "spin": 0, "degeneracy": 1}
            for i in range(8)  # 8 operators in window
        ]

        ope_coefficients = {
            "sigma_sigma_sigma": 0.5,
            "sigma_sigma_[sigma sigma]_0": 0.02,
        }

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(
            run_dir,
            operators,
            ope_coefficients,
            light_ops=["sigma", "op0", "op1"],
        )

        # Create custom thresholds with higher max_density
        thresholds_path = run_dir / "inputs" / "hsc" / "thresholds.json"
        thresholds_data = {
            "phase_1": {"max_density": 20},  # Much higher threshold
        }
        thresholds_path.write_text(json.dumps(thresholds_data, indent=2))

        result = _run_stage(run_id, input_path, out_root_abs, thresholds_path)
        assert result.returncode == 0

        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        verdict = json.loads(verdict_path.read_text())

        # With higher threshold, phase 1 should pass
        assert verdict["phase_summary"]["phase_1"]["verdict"] == "PASS"
        assert verdict["phase_summary"]["phase_1"]["thresholds_applied"]["max_density"] == 20

    def test_error_on_missing_input(self, test_run_dir: Path) -> None:
        """Missing input file should return error code."""
        out_root_abs = test_run_dir
        run_id = "test_missing_input"
        run_dir = out_root_abs / run_id
        fake_input = out_root_abs / "nonexistent.json"

        _write_run_valid(run_dir, verdict="PASS")
        result = _run_stage(run_id, fake_input, out_root_abs)
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_error_on_invalid_schema(self, test_run_dir: Path) -> None:
        """Missing required fields should return contract error."""
        out_root_abs = test_run_dir
        run_id = "test_invalid_schema"
        run_dir = out_root_abs / run_id

        input_dir = run_dir / "inputs" / "hsc"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Invalid input: missing metadata
        input_data = {
            "spectrum": {"operators": []},
            "ope_coefficients": {},
        }

        input_path = input_dir / "input.json"
        input_path.write_text(json.dumps(input_data, indent=2))

        _write_run_valid(run_dir, verdict="PASS")
        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 2  # Contract violation

    def test_underdetermined_on_insufficient_inputs(self, test_run_dir: Path) -> None:
        """Missing spectrum and OPE coefficients should return insufficient_inputs."""
        out_root_abs = test_run_dir
        run_id = "test_insufficient_inputs"
        run_dir = out_root_abs / run_id

        input_dir = run_dir / "inputs" / "hsc"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_data = {
            "metadata": {
                "schema_version": "hsc_input_v1",
                "theory_name": "Features Only",
                "d": 3,
            },
            "features": {
                "some_feature": 1.0,
            },
        }

        input_path = input_dir / "input.json"
        input_path.write_text(json.dumps(input_data, indent=2))

        _write_run_valid(run_dir, verdict="PASS")
        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        verdict_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "verdict.json"
        )
        verdict = json.loads(verdict_path.read_text())
        assert verdict["overall_verdict"] == "UNDERDETERMINED"
        assert verdict["reason"] == "insufficient_inputs"
        assert verdict["missing_inputs"] == ["spectrum", "ope_coefficients"]
        assert verdict["phase_summary"]["phase_1"]["verdict"] == "UNDERDETERMINED"
        assert verdict["phase_summary"]["phase_2"]["verdict"] == "UNDERDETERMINED"
        assert "Missing OPE classes per conventions" not in json.dumps(verdict)


class TestOPEKeyParsing:
    """Test OPE coefficient key parsing logic."""

    def test_simple_keys(self, test_run_dir: Path) -> None:
        """Simple operator names parse correctly."""
        out_root_abs = test_run_dir
        run_id = "test_simple_keys"
        run_dir = out_root_abs / run_id

        operators = [
            {"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1},
            {"id": "epsilon", "dim": 4.0, "spin": 0, "degeneracy": 1},
        ]
        ope_coefficients = {
            "sigma_sigma_sigma": 0.5,
            "sigma_sigma_epsilon": 0.3,
            "sigma_epsilon_epsilon": 0.2,
        }

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)
        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        report_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "report.json"
        )
        report = json.loads(report_path.read_text())
        # All 3 should be classified as lambda_OOO
        assert report["summary"]["phase_2"]["features"]["lambda_OOO_count"] == 3

    def test_bracket_keys(self, test_run_dir: Path) -> None:
        """Composite operator names with brackets parse correctly."""
        out_root_abs = test_run_dir
        run_id = "test_bracket_keys"
        run_dir = out_root_abs / run_id

        operators = [
            {"id": "sigma", "dim": 2.0, "spin": 0, "degeneracy": 1},
            {"id": "[sigma sigma]_0", "dim": 4.0, "spin": 0, "degeneracy": 1},
        ]
        ope_coefficients = {
            "sigma_sigma_sigma": 0.5,
            "sigma_sigma_[sigma sigma]_0": 0.02,
            "sigma_sigma_[sigma sigma]_1": 0.01,
        }

        _write_run_valid(run_dir, verdict="PASS")
        input_path = _create_hsc_input(run_dir, operators, ope_coefficients)
        result = _run_stage(run_id, input_path, out_root_abs)
        assert result.returncode == 0

        report_path = (
            out_root_abs
            / run_id
            / "experiment"
            / "hsc_detector"
            / "outputs"
            / "report.json"
        )
        report = json.loads(report_path.read_text())
        assert report["summary"]["phase_2"]["features"]["lambda_OOO_count"] == 1
        assert report["summary"]["phase_2"]["features"]["lambda_tower_count"] == 2
