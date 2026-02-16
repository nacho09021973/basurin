"""Regression tests for mvp/contracts.py — the centralized contract module.

Tests:
    1. CONTRACTS registry completeness: all 5 stages registered, no unknown stages.
    2. init_stage rejects unknown stage names and invalid run_ids.
    3. check_inputs aborts on missing files, records SHA256 on present files.
    4. finalize produces manifest.json + stage_summary.json with correct structure.
    5. abort produces FAIL verdict and exits with code 2.
    6. enforce_outputs detects missing declared outputs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.contracts import (
    CONTRACTS,
    StageContract,
    StageContext,
    init_stage,
    check_inputs,
    finalize,
    abort,
    enforce_outputs,
)


# ── Test 1: Registry completeness ─────────────────────────────────────────

class TestContractRegistry:
    EXPECTED_STAGES = {"s1_fetch_strain", "s2_ringdown_window", "s3_ringdown_estimates",
                       "s3b_multimode_estimates",
                       "s4_geometry_filter", "s4b_spectral_curvature",
                       "s4c_kerr_consistency",
                       "s5_aggregate", "s6_information_geometry"}

    def test_all_stages_registered(self):
        """Every MVP stage has a contract entry."""
        assert set(CONTRACTS.keys()) == self.EXPECTED_STAGES

    def test_every_contract_has_required_fields(self):
        """Each contract has name, produced_outputs, and is a StageContract."""
        for name, contract in CONTRACTS.items():
            assert isinstance(contract, StageContract)
            assert contract.name == name
            assert isinstance(contract.produced_outputs, list)
            assert isinstance(contract.upstream_stages, list)

    def test_upstream_stages_are_valid(self):
        """Upstream references point to real stages."""
        for name, contract in CONTRACTS.items():
            for upstream in contract.upstream_stages:
                assert upstream in CONTRACTS, \
                    f"{name} references unknown upstream '{upstream}'"

    def test_no_circular_dependencies(self):
        """Pipeline DAG has no cycles."""
        visited: set[str] = set()
        def _check(name: str, path: list[str]) -> None:
            if name in path:
                raise AssertionError(f"Circular dependency: {' -> '.join(path + [name])}")
            if name in visited:
                return
            for upstream in CONTRACTS[name].upstream_stages:
                _check(upstream, path + [name])
            visited.add(name)
        for name in CONTRACTS:
            _check(name, [])


# ── Test 2: init_stage validation ──────────────────────────────────────────

class TestInitStage:
    def test_rejects_unknown_stage(self, tmp_path):
        """init_stage exits 2 for unregistered stage names."""
        env = {"BASURIN_RUNS_ROOT": str(tmp_path / "runs")}
        with pytest.raises(SystemExit) as exc_info:
            os.environ.update(env)
            try:
                init_stage("test_run", "nonexistent_stage")
            finally:
                for k in env:
                    os.environ.pop(k, None)
        assert exc_info.value.code == 2

    def test_creates_stage_dirs(self, tmp_path):
        """init_stage creates stage_dir and outputs_dir."""
        runs_root = tmp_path / "runs"
        run_id = "test_init"
        # Create RUN_VALID (s1 doesn't check it, so use s1)
        rv = runs_root / run_id / "RUN_VALID"
        rv.mkdir(parents=True)
        (rv / "verdict.json").write_text('{"verdict":"PASS"}')

        old = os.environ.get("BASURIN_RUNS_ROOT")
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            ctx = init_stage(run_id, "s1_fetch_strain", params={"key": "val"})
        finally:
            if old is None:
                os.environ.pop("BASURIN_RUNS_ROOT", None)
            else:
                os.environ["BASURIN_RUNS_ROOT"] = old

        assert ctx.stage_dir.exists()
        assert ctx.outputs_dir.exists()
        assert ctx.params == {"key": "val"}
        assert ctx.run_id == run_id
        assert ctx.stage_name == "s1_fetch_strain"


# ── Test 3: check_inputs validation ───────────────────────────────────────

class TestCheckInputs:
    def _make_ctx(self, tmp_path: Path) -> StageContext:
        runs_root = tmp_path / "runs"
        run_id = "test_check"
        run_dir = runs_root / run_id
        stage_dir = run_dir / "test_stage"
        outputs_dir = stage_dir / "outputs"
        outputs_dir.mkdir(parents=True)
        return StageContext(
            run_id=run_id, stage_name="test_stage",
            contract=CONTRACTS["s1_fetch_strain"],
            out_root=runs_root, run_dir=run_dir,
            stage_dir=stage_dir, outputs_dir=outputs_dir,
        )

    def test_records_sha256_for_existing_files(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        test_file = tmp_path / "input.txt"
        test_file.write_text("hello")
        records = check_inputs(ctx, {"my_input": test_file})
        assert len(records) == 1
        assert records[0]["label"] == "my_input"
        assert len(records[0]["sha256"]) == 64  # SHA256 hex

    def test_aborts_on_missing_required(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(SystemExit) as exc_info:
            check_inputs(ctx, {"required": missing})
        assert exc_info.value.code == 2

    def test_optional_files_not_fatal(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        existing = tmp_path / "exists.txt"
        existing.write_text("data")
        records = check_inputs(ctx, {"req": existing}, optional={"opt": tmp_path / "nope"})
        assert len(records) == 1  # Only required recorded (optional missing)


# ── Test 4: finalize writes correct contract artifacts ─────────────────────

class TestFinalize:
    def _make_ctx(self, tmp_path: Path) -> StageContext:
        runs_root = tmp_path / "runs"
        run_id = "test_final"
        run_dir = runs_root / run_id
        stage_dir = run_dir / "my_stage"
        outputs_dir = stage_dir / "outputs"
        outputs_dir.mkdir(parents=True)
        return StageContext(
            run_id=run_id, stage_name="my_stage",
            contract=CONTRACTS["s1_fetch_strain"],
            out_root=runs_root, run_dir=run_dir,
            stage_dir=stage_dir, outputs_dir=outputs_dir,
            params={"test": True},
        )

    def test_produces_manifest_and_summary(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        out_file = ctx.outputs_dir / "result.json"
        out_file.write_text('{"data": 1}')

        finalize(ctx, artifacts={"result": out_file}, results={"value": 42})

        # Check stage_summary.json
        summary = json.loads((ctx.stage_dir / "stage_summary.json").read_text())
        assert summary["verdict"] == "PASS"
        assert summary["stage"] == "my_stage"
        assert summary["run"] == "test_final"
        assert summary["results"]["value"] == 42
        assert "created" in summary

        # Check manifest.json
        manifest = json.loads((ctx.stage_dir / "manifest.json").read_text())
        assert "hashes" in manifest or "files" in manifest


# ── Test 5: abort produces FAIL and exits ──────────────────────────────────

class TestAbort:
    def _make_ctx(self, tmp_path: Path) -> StageContext:
        runs_root = tmp_path / "runs"
        run_id = "test_abort"
        run_dir = runs_root / run_id
        stage_dir = run_dir / "fail_stage"
        outputs_dir = stage_dir / "outputs"
        outputs_dir.mkdir(parents=True)
        return StageContext(
            run_id=run_id, stage_name="fail_stage",
            contract=CONTRACTS["s1_fetch_strain"],
            out_root=runs_root, run_dir=run_dir,
            stage_dir=stage_dir, outputs_dir=outputs_dir,
        )

    def test_abort_writes_fail_and_exits_2(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            abort(ctx, "something went wrong")
        assert exc_info.value.code == 2

        summary = json.loads((ctx.stage_dir / "stage_summary.json").read_text())
        assert summary["verdict"] == "FAIL"
        assert "something went wrong" in summary["error"]


# ── Test 6: enforce_outputs detects missing declared outputs ───────────────

class TestEnforceOutputs:
    def test_detects_missing_outputs(self, tmp_path):
        runs_root = tmp_path / "runs"
        run_dir = runs_root / "test_enforce"
        stage_dir = run_dir / "s1_fetch_strain"
        outputs_dir = stage_dir / "outputs"
        outputs_dir.mkdir(parents=True)

        ctx = StageContext(
            run_id="test_enforce", stage_name="s1_fetch_strain",
            contract=CONTRACTS["s1_fetch_strain"],
            out_root=runs_root, run_dir=run_dir,
            stage_dir=stage_dir, outputs_dir=outputs_dir,
        )

        missing = enforce_outputs(ctx)
        # strain.npz and provenance.json should be missing
        assert "outputs/strain.npz" in missing
        assert "outputs/provenance.json" in missing

    def test_no_missing_when_outputs_present(self, tmp_path):
        runs_root = tmp_path / "runs"
        run_dir = runs_root / "test_enforce2"
        stage_dir = run_dir / "s1_fetch_strain"
        outputs_dir = stage_dir / "outputs"
        outputs_dir.mkdir(parents=True)

        # Create the expected outputs
        (outputs_dir / "strain.npz").write_text("fake")
        (outputs_dir / "provenance.json").write_text("{}")

        ctx = StageContext(
            run_id="test_enforce2", stage_name="s1_fetch_strain",
            contract=CONTRACTS["s1_fetch_strain"],
            out_root=runs_root, run_dir=run_dir,
            stage_dir=stage_dir, outputs_dir=outputs_dir,
        )

        missing = enforce_outputs(ctx)
        assert missing == []
