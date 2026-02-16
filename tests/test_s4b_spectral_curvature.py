"""Tests for MVP Stage 4b: Spectral curvature diagnostic.

Tests:
    1. Minimal atlas (1 geometry, 3 overtones) computes exactly one kappa point.
    2. kappa matches manual computation for complex omega.
    3. Bound check passes when kappa small, fails when large.
    4. Deterministic output: same inputs -> identical sha256.
    5. Diagnostics sorted by geometry_id.
    6. Missing overtones -> geometry marked insufficient and not counted usable.
    7. Atlas supports both list and {"entries": ...} container.
    8. Omega specified as {"re","im"} and as [re,im] both supported.
    9. Non-finite omega triggers abort (exit code 2).
   10. CLI end-to-end: verify outputs and stage_summary verdict PASS.
   11. No writes outside runs/<run_id>/ (output paths under runs root).
   12. Contract integration: CONTRACTS has s4b_spectral_curvature and produced_outputs.
   13. Empty atlas triggers abort.
   14. Atlas entry missing geometry_id triggers abort.
   15. Geometry with exactly min_overtones is usable; one fewer is insufficient.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"

# Ensure repo root on path for imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s4b_spectral_curvature import (
    _parse_omega,
    _extract_mode_family,
    compute_spectral_diagnostics,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_atlas_entry(
    gid: str,
    overtones: list[tuple[int, float, float]],
    l_mode: int = 2,
    m_mode: int = 2,
) -> dict[str, Any]:
    """Build an atlas entry with QNM overtones.

    overtones: [(n, re, im), ...]
    """
    modes = [
        {"n": n, "omega": {"re": re, "im": im}}
        for n, re, im in overtones
    ]
    return {
        "geometry_id": gid,
        "qnm": {f"({l_mode},{m_mode})": modes},
    }


def _make_atlas_entry_array_omega(
    gid: str,
    overtones: list[tuple[int, float, float]],
    l_mode: int = 2,
    m_mode: int = 2,
) -> dict[str, Any]:
    """Build atlas entry with omega as [re, im] arrays."""
    modes = [
        {"n": n, "omega": [re, im]}
        for n, re, im in overtones
    ]
    return {
        "geometry_id": gid,
        "qnm": {f"({l_mode},{m_mode})": modes},
    }


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )


def _create_s3_estimates(
    runs_root: Path, run_id: str,
    f_hz: float = 251.0, Q: float = 3.14,
) -> None:
    """Create mock s3 estimates for the stage to consume."""
    stage_dir = runs_root / run_id / "s3_ringdown_estimates"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    estimates = {
        "schema_version": "mvp_estimates_v1",
        "event_id": "GW150914",
        "combined": {"f_hz": f_hz, "tau_s": Q / (math.pi * f_hz), "Q": Q},
    }
    (outputs_dir / "estimates.json").write_text(
        json.dumps(estimates, indent=2), encoding="utf-8"
    )
    (stage_dir / "stage_summary.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates", "verdict": "PASS"}),
        encoding="utf-8",
    )
    (stage_dir / "manifest.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8"
    )


def _write_atlas(path: Path, entries: list[dict[str, Any]]) -> None:
    """Write atlas JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def _run_stage(script: str, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=str(REPO_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Shared atlas data ────────────────────────────────────────────────────

# Three overtones: n=0,1,2 → exactly one kappa point at n=1
BASIC_OVERTONES = [
    (0, 0.5, -0.1),
    (1, 0.8, -0.15),
    (2, 1.2, -0.2),
]


# ── Test 1: Minimal atlas computes exactly one kappa point ───────────────

class TestMinimalAtlas:
    def test_three_overtones_one_kappa(self):
        """3 overtones (n=0,1,2) produce exactly one kappa at n=1."""
        entry = _make_atlas_entry("geo_001", BASIC_OVERTONES)
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)

        assert len(result) == 1
        diag = result[0]
        assert diag["geometry_id"] == "geo_001"
        assert diag["n_overtones"] == 3
        assert len(diag["kappa"]) == 1
        assert diag["kappa"][0]["n"] == 1
        assert len(diag["kappa_bound"]) == 1


# ── Test 2: kappa matches manual computation ─────────────────────────────

class TestKappaComputation:
    def test_kappa_matches_manual(self):
        """kappa_1 = (w2 - 2*w1 + w0) / w1 for complex omega."""
        w0 = complex(0.5, -0.1)
        w1 = complex(0.8, -0.15)
        w2 = complex(1.2, -0.2)
        expected = (w2 - 2.0 * w1 + w0) / w1

        entry = _make_atlas_entry("geo_manual", BASIC_OVERTONES)
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=10.0)

        k = result[0]["kappa"][0]
        assert abs(k["re"] - expected.real) < 1e-12
        assert abs(k["im"] - expected.imag) < 1e-12
        assert abs(k["abs"] - abs(expected)) < 1e-12

    def test_five_overtones_three_kappa_points(self):
        """5 overtones (n=0..4) produce 3 kappa points at n=1,2,3."""
        overtones = [
            (0, 0.5, -0.1),
            (1, 0.8, -0.15),
            (2, 1.2, -0.2),
            (3, 1.7, -0.25),
            (4, 2.3, -0.3),
        ]
        entry = _make_atlas_entry("geo_5ot", overtones)
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=10.0)

        assert len(result[0]["kappa"]) == 3
        ns = [k["n"] for k in result[0]["kappa"]]
        assert ns == [1, 2, 3]


# ── Test 3: Bound check pass/fail ────────────────────────────────────────

class TestBoundCheck:
    def test_passes_when_kappa_small(self):
        """Small kappa passes the bound C * n^(-2)."""
        # Construct omega so that kappa ~ 0 (nearly linear spacing)
        overtones = [
            (0, 1.0, -0.1),
            (1, 2.0, -0.2),
            (2, 3.0, -0.3),
        ]
        entry = _make_atlas_entry("geo_linear", overtones)
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)
        diag = result[0]
        assert diag["passes_bound"] is True
        assert diag["max_violation_ratio"] <= 1.0

    def test_fails_when_kappa_large(self):
        """Large kappa violates the bound."""
        # Construct omega with very non-linear spacing
        overtones = [
            (0, 1.0, 0.0),
            (1, 1.0, 0.0),
            (2, 100.0, 0.0),  # huge jump → large kappa
        ]
        entry = _make_atlas_entry("geo_jump", overtones)
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)
        diag = result[0]
        assert diag["passes_bound"] is False
        assert diag["max_violation_ratio"] > 1.0


# ── Test 4: Deterministic output ─────────────────────────────────────────

class TestDeterminism:
    def test_identical_runs_produce_same_hash(self, tmp_path):
        """Two calls with identical inputs produce byte-identical JSON."""
        entries = [
            _make_atlas_entry("geo_A", BASIC_OVERTONES),
            _make_atlas_entry("geo_B", [
                (0, 1.0, -0.2), (1, 1.5, -0.3), (2, 2.1, -0.4),
            ]),
        ]
        hashes = []
        for i in range(2):
            result = compute_spectral_diagnostics(entries, l_mode=2, m_mode=2, C=0.1)
            j = json.dumps(result, sort_keys=True, separators=(",", ":"))
            hashes.append(hashlib.sha256(j.encode()).hexdigest())

        assert hashes[0] == hashes[1]


# ── Test 5: Sorting by geometry_id ───────────────────────────────────────

class TestSorting:
    def test_diagnostics_sorted_by_geometry_id(self):
        """Output diagnostics are sorted alphabetically by geometry_id."""
        entries = [
            _make_atlas_entry("geo_C", BASIC_OVERTONES),
            _make_atlas_entry("geo_A", BASIC_OVERTONES),
            _make_atlas_entry("geo_B", BASIC_OVERTONES),
        ]
        result = compute_spectral_diagnostics(entries, l_mode=2, m_mode=2, C=0.1)
        ids = [d["geometry_id"] for d in result]
        assert ids == ["geo_A", "geo_B", "geo_C"]


# ── Test 6: Insufficient overtones ──────────────────────────────────────

class TestInsufficientOvertones:
    def test_two_overtones_marked_insufficient(self):
        """Geometry with <3 overtones is marked insufficient, not usable."""
        entry = _make_atlas_entry("geo_short", [(0, 0.5, -0.1), (1, 0.8, -0.15)])
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)
        diag = result[0]
        assert diag["insufficient_overtones"] is True
        assert diag["n_overtones"] == 2
        assert diag["kappa"] == []

    def test_no_qnm_data_marked_insufficient(self):
        """Geometry without qnm block is marked insufficient."""
        entry = {"geometry_id": "geo_empty"}
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)
        assert result[0]["insufficient_overtones"] is True
        assert result[0]["n_overtones"] == 0

    def test_boundary_min_overtones(self):
        """Exactly min_overtones (3) is usable; 2 is not."""
        entry_ok = _make_atlas_entry("geo_ok", BASIC_OVERTONES)
        entry_bad = _make_atlas_entry("geo_bad", [(0, 0.5, -0.1), (1, 0.8, -0.15)])
        result = compute_spectral_diagnostics(
            [entry_ok, entry_bad], l_mode=2, m_mode=2, C=0.1,
        )
        by_id = {d["geometry_id"]: d for d in result}
        assert "insufficient_overtones" not in by_id["geo_ok"]
        assert by_id["geo_bad"]["insufficient_overtones"] is True


# ── Test 7: Atlas format variants ────────────────────────────────────────

class TestAtlasFormats:
    def test_list_format(self, tmp_path):
        """Atlas as bare list of entries works."""
        entry = _make_atlas_entry("geo_list", BASIC_OVERTONES)
        atlas_path = tmp_path / "atlas_list.json"
        atlas_path.write_text(json.dumps([entry]), encoding="utf-8")

        from mvp.s4b_spectral_curvature import _load_atlas
        loaded = _load_atlas(atlas_path)
        assert len(loaded) == 1
        assert loaded[0]["geometry_id"] == "geo_list"

    def test_entries_dict_format(self, tmp_path):
        """Atlas as {"entries": [...]} works."""
        entry = _make_atlas_entry("geo_dict", BASIC_OVERTONES)
        atlas_path = tmp_path / "atlas_dict.json"
        atlas_path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")

        from mvp.s4b_spectral_curvature import _load_atlas
        loaded = _load_atlas(atlas_path)
        assert len(loaded) == 1
        assert loaded[0]["geometry_id"] == "geo_dict"


# ── Test 8: Omega format variants ───────────────────────────────────────

class TestOmegaFormats:
    def test_dict_omega_format(self):
        """Omega as {"re": ..., "im": ...} is parsed correctly."""
        omega = _parse_omega({"re": 0.5, "im": -0.1})
        assert omega == complex(0.5, -0.1)

    def test_array_omega_format(self):
        """Omega as [re, im] is parsed correctly."""
        omega = _parse_omega([0.5, -0.1])
        assert omega == complex(0.5, -0.1)

    def test_array_omega_in_atlas_entry(self):
        """Atlas with [re,im] omega produces same kappa as {"re","im"}."""
        entry_dict = _make_atlas_entry("geo_d", BASIC_OVERTONES)
        entry_arr = _make_atlas_entry_array_omega("geo_d", BASIC_OVERTONES)

        r1 = compute_spectral_diagnostics([entry_dict], l_mode=2, m_mode=2, C=0.1)
        r2 = compute_spectral_diagnostics([entry_arr], l_mode=2, m_mode=2, C=0.1)

        assert r1[0]["kappa"] == r2[0]["kappa"]


# ── Test 9: Non-finite omega triggers ValueError ─────────────────────────

class TestNonFiniteOmega:
    def test_nan_omega_raises(self):
        """Non-finite omega component raises ValueError."""
        with pytest.raises(ValueError, match="Non-finite"):
            _parse_omega({"re": float("nan"), "im": -0.1})

    def test_inf_omega_raises(self):
        """Infinite omega component raises ValueError."""
        with pytest.raises(ValueError, match="Non-finite"):
            _parse_omega([float("inf"), -0.1])

    def test_nan_in_atlas_aborts_via_subprocess(self, tmp_path):
        """Stage aborts with exit 2 when atlas has non-finite omega."""
        runs_root = tmp_path / "runs"
        run_id = "test_nan_abort"
        _create_run_valid(runs_root, run_id)
        _create_s3_estimates(runs_root, run_id)

        entry = {
            "geometry_id": "geo_nan",
            "qnm": {"(2,2)": [
                {"n": 0, "omega": {"re": 0.5, "im": -0.1}},
                {"n": 1, "omega": {"re": float("nan"), "im": -0.15}},
                {"n": 2, "omega": {"re": 1.2, "im": -0.2}},
            ]},
        }
        atlas_path = tmp_path / "atlas_nan.json"
        _write_atlas(atlas_path, [entry])

        result = _run_stage(
            "s4b_spectral_curvature.py",
            ["--run", run_id, "--atlas-path", str(atlas_path)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2


# ── Test 10: CLI end-to-end ──────────────────────────────────────────────

class TestCLIEndToEnd:
    def test_full_run_produces_contract(self, tmp_path):
        """CLI run produces spectral_diagnostics.json + contract files."""
        runs_root = tmp_path / "runs"
        run_id = "test_s4b_e2e"
        _create_run_valid(runs_root, run_id)
        _create_s3_estimates(runs_root, run_id)

        entries = [
            _make_atlas_entry("geo_001", BASIC_OVERTONES),
            _make_atlas_entry("geo_002", [
                (0, 1.0, -0.2), (1, 1.5, -0.3), (2, 2.1, -0.4),
            ]),
        ]
        atlas_path = tmp_path / "atlas_e2e.json"
        _write_atlas(atlas_path, entries)

        result = _run_stage(
            "s4b_spectral_curvature.py",
            ["--run", run_id, "--atlas-path", str(atlas_path)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s4b failed: {result.stderr}"

        stage_dir = runs_root / run_id / "s4b_spectral_curvature"
        assert (stage_dir / "stage_summary.json").exists()
        assert (stage_dir / "manifest.json").exists()
        assert (stage_dir / "outputs" / "spectral_diagnostics.json").exists()

        summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
        assert summary["verdict"] == "PASS"

        diag = json.loads(
            (stage_dir / "outputs" / "spectral_diagnostics.json").read_text(encoding="utf-8")
        )
        assert diag["schema_version"] == "mvp_spectral_diagnostics_v1"
        assert diag["run_id"] == run_id
        assert diag["event_id"] == "GW150914"
        assert diag["mode_family"] == [2, 2]
        assert diag["n_geometries_total"] == 2
        assert diag["n_geometries_usable"] == 2
        assert len(diag["diagnostics"]) == 2

    def test_cli_deterministic_hash(self, tmp_path):
        """Two identical CLI runs with same run_id produce identical spectral_diagnostics.json."""
        runs_root = tmp_path / "runs"
        run_id = "det_run"
        entries = [_make_atlas_entry("geo_det", BASIC_OVERTONES)]
        atlas_path = tmp_path / "atlas_det.json"
        _write_atlas(atlas_path, entries)

        _create_run_valid(runs_root, run_id)
        _create_s3_estimates(runs_root, run_id)

        hashes = []
        for i in range(2):
            result = _run_stage(
                "s4b_spectral_curvature.py",
                ["--run", run_id, "--atlas-path", str(atlas_path)],
                env={"BASURIN_RUNS_ROOT": str(runs_root)},
            )
            assert result.returncode == 0, f"Run {i} failed: {result.stderr}"

            out_path = runs_root / run_id / "s4b_spectral_curvature" / "outputs" / "spectral_diagnostics.json"
            hashes.append(_sha256(out_path))

        assert hashes[0] == hashes[1], f"Non-deterministic: {hashes[0]} != {hashes[1]}"


# ── Test 11: No writes outside runs/<run_id>/ ────────────────────────────

class TestOutputContainment:
    def test_outputs_under_runs_root(self, tmp_path):
        """All output files are under runs/<run_id>/."""
        runs_root = tmp_path / "runs"
        run_id = "test_containment"
        _create_run_valid(runs_root, run_id)
        _create_s3_estimates(runs_root, run_id)

        entries = [_make_atlas_entry("geo_c", BASIC_OVERTONES)]
        atlas_path = tmp_path / "atlas_c.json"
        _write_atlas(atlas_path, entries)

        result = _run_stage(
            "s4b_spectral_curvature.py",
            ["--run", run_id, "--atlas-path", str(atlas_path)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0

        stage_dir = runs_root / run_id / "s4b_spectral_curvature"
        for child in stage_dir.rglob("*"):
            assert str(child.resolve()).startswith(str(runs_root.resolve())), (
                f"File outside runs root: {child}"
            )


# ── Test 12: Contract integration ────────────────────────────────────────

class TestContractIntegration:
    def test_stage_registered_in_contracts(self):
        """CONTRACTS registry includes s4b_spectral_curvature."""
        from mvp.contracts import CONTRACTS
        assert "s4b_spectral_curvature" in CONTRACTS

    def test_produced_outputs_declared(self):
        """Contract declares spectral_diagnostics.json as output."""
        from mvp.contracts import CONTRACTS
        contract = CONTRACTS["s4b_spectral_curvature"]
        assert "outputs/spectral_diagnostics.json" in contract.produced_outputs

    def test_upstream_stages_correct(self):
        """Contract declares s3_ringdown_estimates as upstream."""
        from mvp.contracts import CONTRACTS
        contract = CONTRACTS["s4b_spectral_curvature"]
        assert "s3_ringdown_estimates" in contract.upstream_stages


# ── Test 13: Empty atlas triggers abort ──────────────────────────────────

class TestEmptyAtlas:
    def test_empty_atlas_aborts(self, tmp_path):
        """Stage aborts with exit 2 on empty atlas."""
        runs_root = tmp_path / "runs"
        run_id = "test_empty_atlas"
        _create_run_valid(runs_root, run_id)
        _create_s3_estimates(runs_root, run_id)

        atlas_path = tmp_path / "empty.json"
        atlas_path.write_text(json.dumps({"entries": []}), encoding="utf-8")

        result = _run_stage(
            "s4b_spectral_curvature.py",
            ["--run", run_id, "--atlas-path", str(atlas_path)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2


# ── Test 14: Missing geometry_id triggers abort ──────────────────────────

class TestMissingGeometryId:
    def test_missing_geometry_id_aborts(self, tmp_path):
        """Stage aborts when atlas entry lacks geometry_id."""
        runs_root = tmp_path / "runs"
        run_id = "test_no_gid"
        _create_run_valid(runs_root, run_id)
        _create_s3_estimates(runs_root, run_id)

        atlas_path = tmp_path / "bad_atlas.json"
        atlas_path.write_text(json.dumps({"entries": [
            {"qnm": {"(2,2)": [{"n": 0, "omega": {"re": 0.5, "im": -0.1}}]}},
        ]}), encoding="utf-8")

        result = _run_stage(
            "s4b_spectral_curvature.py",
            ["--run", run_id, "--atlas-path", str(atlas_path)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2


# ── Test 15: Boundary min_overtones (usable vs insufficient) ─────────────

class TestMinOvertonesBoundary:
    def test_exactly_three_is_usable(self):
        """3 overtones (exactly min_overtones) produces kappa and is usable."""
        entry = _make_atlas_entry("geo_three", BASIC_OVERTONES)
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)
        diag = result[0]
        assert "insufficient_overtones" not in diag
        assert len(diag["kappa"]) >= 1

    def test_two_is_insufficient(self):
        """2 overtones (< min_overtones) is marked insufficient."""
        entry = _make_atlas_entry("geo_two", [(0, 0.5, -0.1), (1, 0.8, -0.15)])
        result = compute_spectral_diagnostics([entry], l_mode=2, m_mode=2, C=0.1)
        diag = result[0]
        assert diag["insufficient_overtones"] is True
        assert diag["kappa"] == []
